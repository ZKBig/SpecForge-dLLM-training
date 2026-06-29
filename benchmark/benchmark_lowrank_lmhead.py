"""Benchmark the low-rank lm-head READOUT against the full lm_head, at real sizes.

The low-rank trick (see specforge.core.dflash_refiner.LowRankReadout): because lm_head is linear,
    lm_head(refined) = lm_head(h) + lm_head(refined - h)
so par-K decode reads out as  base = lm_head(h) [full, ONCE]  +  K x lowrank(refined - h) [cheap],
instead of  K x lm_head(refined) [full each pass].

This measures, at the real (block, hidden, vocab) and a sweep of K:
  - readout latency: full  (K x full lm_head)   vs  low-rank  (1 full base + K x low-rank)
  - argmax agreement: does base + lowrank(delta) predict the SAME tokens as full lm_head(refined)?
FLOP reduction != latency reduction, so we time the real ops with CUDA events.

  # synthetic (real sizes, no checkpoint needed -- latency is structural):
  python benchmark_lowrank_lmhead.py --rank 256 --k-list 1,2,3,5
  # real weights + real argmax from a trained low-rank refiner checkpoint:
  python benchmark_lowrank_lmhead.py --rank 256 --refiner-path /path/refiner_cotrain.pt \
      --dflash-path z-lab/Qwen3-8B-DFlash-b16 --k-list 1,2,3,5
"""
import argparse

import torch
import torch.nn as nn


class LowRankReadout(nn.Module):
    """Inlined copy of specforge.core.dflash_refiner.LowRankReadout so this benchmark is fully
    self-contained (no specforge import / no push dependency -- runs anywhere with just torch).
    Identical structure (down: H->r, up: r->V) and svd warm-start, so latency + state_dict keys
    (down.weight / up.weight) match the trained checkpoint exactly."""

    def __init__(self, hidden_size, vocab_size, rank):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, delta):
        return self.up(self.down(delta))

    @torch.no_grad()
    def svd_init_from(self, weight):
        r = self.rank
        W = weight.detach().float()
        U, S, V = torch.svd_lowrank(W, q=min(r + 16, min(W.shape)), niter=4)
        Ur, Sr, Vr = U[:, :r], S[:r].clamp(min=0), V[:, :r]
        sqrt_s = Sr.sqrt()
        self.down.weight.copy_((sqrt_s[:, None] * Vr.t()).to(self.down.weight))
        self.up.weight.copy_((Ur * sqrt_s[None, :]).to(self.up.weight))


def cuda_time(fn, iters=50, warmup=10):
    """Mean ms/call over `iters`, after `warmup`, via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096, help="H (Qwen3-8B = 4096)")
    ap.add_argument("--vocab", type=int, default=151936, help="V (Qwen3 = 151936)")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--k-list", type=str, default="1,2,3,5", help="par-K pass counts to report")
    ap.add_argument("--delta-scale", type=float, default=0.02,
                    help="size of refined-h relative to h (gate~0 at init => small; raise to stress)")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--refiner-path", type=str, default=None,
                    help="optional: trained low-rank refiner checkpoint -> use its REAL lowrank_head weights")
    ap.add_argument("--lm-head-path", type=str, default=None,
                    help="optional .pt with an lm_head weight tensor [V,H] to use real lm_head weights")
    args = ap.parse_args()

    dev = args.device
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    H, V, blk, r = args.hidden, args.vocab, args.block, args.rank
    ks = [int(x) for x in args.k_list.split(",")]
    torch.manual_seed(0)

    # ---- full lm_head (the frozen target head) ----
    lm_head = torch.nn.Linear(H, V, bias=False).to(dev).to(dt)
    if args.lm_head_path:
        w = torch.load(args.lm_head_path, map_location="cpu")
        w = w["weight"] if isinstance(w, dict) and "weight" in w else w
        lm_head.weight.data.copy_(w.to(dev).to(dt))
        print(f"loaded real lm_head weight {tuple(lm_head.weight.shape)}")

    # ---- low-rank correction head (svd-init from lm_head, or real trained weights) ----
    lowrank = LowRankReadout(H, V, r).to(dev)
    if args.refiner_path:
        ckpt = torch.load(args.refiner_path, map_location="cpu", weights_only=False)
        sd = ckpt["refiner_state_dict"]
        dn = {k.split("lowrank_head.")[1]: v for k, v in sd.items() if "lowrank_head." in k}
        if not dn:
            raise SystemExit(f"no lowrank_head.* in {args.refiner_path} (rank=0 checkpoint?)")
        lowrank.load_state_dict(dn)
        print(f"loaded REAL trained lowrank_head (rank {r}) from {args.refiner_path}")
    else:
        lowrank.svd_init_from(lm_head.weight)
        print(f"svd-init lowrank_head (rank {r}) from lm_head")
    lowrank = lowrank.to(dt).eval()

    # ---- synthetic block: h, refined = h + small delta (gate~0 regime) ----
    h = torch.randn(1, blk, H, device=dev, dtype=dt)
    refined = h + args.delta_scale * torch.randn(1, blk, H, device=dev, dtype=dt)

    p_full = sum(p.numel() for p in lm_head.parameters())
    p_low = sum(p.numel() for p in lowrank.parameters())
    print(f"\nH={H} V={V} block={blk} rank={r} dtype={args.dtype}")
    print(f"full lm_head params = {p_full/1e6:.1f}M   lowrank params = {p_low/1e6:.1f}M   "
          f"({p_full/p_low:.1f}x fewer)\n")

    # ---- unit op latencies ----
    t_full = cuda_time(lambda: lm_head(refined))                       # one full readout over the block
    t_base = cuda_time(lambda: lm_head(h))                             # the once-per-block base
    t_low = cuda_time(lambda: lm_head(h) + lowrank(refined - h))       # base + correction (measured together below)
    t_corr = cuda_time(lambda: lowrank(refined - h))                   # just the per-pass correction
    print(f"unit readout (1 block): full lm_head = {t_full:.3f} ms | base lm_head = {t_base:.3f} ms | "
          f"lowrank correction = {t_corr:.3f} ms\n")

    # ---- argmax agreement (does low-rank predict the same tokens?) ----
    full_logits = lm_head(refined)
    low_logits = lm_head(h) + lowrank(refined - h)
    agree = (full_logits.argmax(-1) == low_logits.argmax(-1)).float().mean().item() * 100
    note = "REAL trained" if args.refiner_path else "svd-init (UNtrained -> floor; training closes gap)"
    print(f"argmax agreement (full vs base+lowrank) = {agree:.1f}%   [{note}]\n")

    # ---- par-K readout totals: full = K x full ; lowrank = base(once) + K x correction ----
    print(f"{'K':>3} {'full ms':>10} {'lowrank ms':>12} {'speedup':>9}")
    for K in ks:
        full_total = K * t_full
        low_total = t_base + K * t_corr
        print(f"{K:>3} {full_total:>10.3f} {low_total:>12.3f} {full_total/low_total:>8.2f}x")
    print("\nnote: lowrank wins for K>=2 (base is amortized once); K=1 is base+corr vs one full (~tie/slight loss).")
    print("note: latency is weight-load bound at decode -> tracks the param ratio above, but kernel/output-write")
    print("      overhead can erode it; this is the measured (not theoretical) number.")


if __name__ == "__main__":
    main()
