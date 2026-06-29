"""Dataset accept benchmark with a LOW-RANK lm-head readout -- does low-rank readout drop accept?

SELF-CONTAINED (does not import benchmark_refiner.py). Same dataset / refine-mode / checkpoint
flow, plus one knob: --lowrank-rank, which switches the refiner's readout
    lm_head(refined)  ->  base lm_head(h) + low-rank(refined - h)

  --lowrank-rank 0    full lm_head (baseline accept)
  --lowrank-rank R    low-rank readout. Uses the checkpoint's TRAINED low-rank head if it has one;
                      otherwise SVD-initializes the head post-hoc from lm_head (untrained -> this
                      measures how much a post-hoc low-rank readout costs in real accept).

Run the SAME checkpoint + datasets at rank 0 and rank R and read off the accept delta per mode.

  torchrun --nproc_per_node=8 benchmark/benchmark_lowrank_lmhead.py \
     --dataset gsm8k,humaneval,math500 --max-samples 128 \
     --model-name-or-path /gpfs/zwang33/models/Qwen3-8B \
     --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
     --refiner-path /gpfs/.../refiner_cotrain.pt \
     --refine-modes drafter,ar,par1,par2,par3 --max-new-tokens 1024 --temperature 0.0 \
     --lowrank-rank 256          # <- 0 for the full-lm_head baseline run

NOTE: needs specforge importable -> `pip install -e` the SpecForge fork in this env.
window_size=0 refiners only. --lowrank-rank is not supported with --graph-refiner (eager only).
"""
import argparse
import os
import sys
import time
from itertools import chain
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# make sibling `model/` and `distributed.py` importable regardless of cwd (IBM runs from repo root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

# our refiner (separate repo, must be pip-installed)
from specforge.core.dflash_refiner import LowRankReadout, RefinerDecoder


def _readout(refiner, lm_head, refined, h):
    """Logits for `refined`. If the refiner carries a low-rank head -> base lm_head(h) +
    low-rank(refined - h) (full lm_head runs once on the base h); else the plain full lm_head."""
    lr = getattr(refiner, "lowrank_head", None)
    if lr is not None:
        return lm_head(h) + lr(refined - h)
    return lm_head(refined)

# CUDA-graphed target (Phase A: baseline only)
from cudagraph_target import GraphedDecoder, baseline_generate_graphed


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def load_refiner(refiner_path: str, draft_model: DFlashDraftModel, device):
    """Rebuild RefinerDecoder from a refiner / refiner_cotrain checkpoint and load weights.

    Architecture-aware: reads mixer_type / pool_type / gate_type / gate_floor from the checkpoint
    (so sgu / xattn / per-position-gate / gate-floor checkpoints reconstruct correctly). Falls back
    to sniffing the state_dict keys for older checkpoints that predate those metadata fields.
    """
    ckpt = torch.load(refiner_path, map_location="cpu", weights_only=False)
    sd = ckpt["refiner_state_dict"]
    window_size = int(ckpt.get("window_size", 0))
    num_layers = int(ckpt.get("num_refiner_layers", 1))
    mlp_intermediate = ckpt.get("mlp_intermediate", None)
    block_size = draft_model.block_size  # ChannelWiseCausalMix L is (H, block, block) -> must match

    # gate: the refiner's TOP-LEVEL gate params -- perpos -> gate_proj.{weight,bias}; scalar ->
    # residual_gate. MUST use the top-level prefix so we DON'T match the MLP's own
    # layers.*.mlp.gate_proj.* (which would falsely flag every checkpoint as gated).
    has_perpos = any(k.startswith("gate_proj.") for k in sd)
    has_scalar = any(k == "residual_gate" or k.endswith(".residual_gate") for k in sd)
    use_gate = has_perpos or has_scalar
    # prefer ckpt metadata; fall back to key-sniffing for old checkpoints
    mixer_type = ckpt.get("mixer_type") or ("sgu" if any(".mix_out." in k for k in sd) else "attention")
    pool_type = ckpt.get("pool_type") or ("xattn" if any(k.startswith("pool.") for k in sd) else "mean")
    gate_type = ckpt.get("gate_type") or ("perpos" if has_perpos else "scalar")
    gate_floor = float(ckpt.get("gate_floor", 0.0))  # forward-affecting -> MUST match training

    refiner = RefinerDecoder(
        draft_model.config,
        draft_model.fc.in_features,
        window_size=window_size,
        num_layers=num_layers,
        use_residual_gate=use_gate,
        mlp_intermediate=mlp_intermediate,
        mixer_type=mixer_type,
        pool_type=pool_type,
        gate_type=gate_type,
        gate_floor=gate_floor,
        block_size=block_size,
    )
    # TRAINED low-rank head (if the checkpoint has one) -> build it before the strict load so
    # the lowrank_head.* keys match. (Post-hoc SVD heads for full-lm_head checkpoints are attached
    # later in main() from --lowrank-rank.)
    if any(k.startswith("lowrank_head.") for k in sd):
        V, r = sd["lowrank_head.up.weight"].shape          # up: [vocab, rank]
        refiner.lowrank_head = LowRankReadout(draft_model.config.hidden_size, int(V), int(r))
        print(f"Loaded TRAINED low-rank head (rank {int(r)}) from checkpoint")
    refiner.load_state_dict(sd, strict=True)
    refiner = refiner.to(device).to(torch.bfloat16).eval()
    print(
        f"Loaded refiner: mixer={mixer_type} pool={pool_type} gate={gate_type} floor={gate_floor} "
        f"layers={num_layers} window={window_size} step={ckpt.get('global_step')}"
    )
    return refiner


class RefinerParKGraph:
    """CUDA-graphed par-K refiner block decode — minimizes the refiner's per-block overhead
    WITHOUT changing its structure/weights (greedy/argmax only). Captures `seed + K passes` of
    (roll -> embed -> refiner -> lm_head -> argmax) over a FIXED block with static input buffers,
    so the K-pass rollout replays as one graph (no per-step launch overhead). Per-block-varying
    inputs (h, tok_am1, window) are copied into the static buffers before each replay.

    Note: the refiner forward (attention + MLP) is real compute, so the graph mainly removes
    launch overhead (~1.1-1.2x on the refiner); the MLP FLOPs are irreducible without retraining.
    """

    def __init__(self, refiner, lm_head, embed, block_size, K, ctx_dim, device, dtype):
        self.K = K
        self.block_size = block_size
        self.W = refiner.window_size
        H = lm_head.weight.shape[1]
        self.h = torch.zeros(1, block_size, H, device=device, dtype=dtype)
        self.tok_am1 = torch.zeros(1, dtype=torch.long, device=device)
        self.win_h = torch.zeros(1, self.W, ctx_dim, device=device, dtype=dtype) if self.W > 0 else None
        self.win_m = torch.ones(1, self.W, dtype=torch.bool, device=device) if self.W > 0 else None
        self.blk = torch.zeros(1, block_size, dtype=torch.long, device=device)

        def _decode():
            h = self.h
            g = h.mean(dim=1, keepdim=True).expand(-1, block_size, -1)
            self.blk[:, 1:] = lm_head(h[:, 1:, :]).argmax(dim=-1)          # seed = drafter's own block
            for _ in range(K):
                prev = self.blk.roll(shifts=1, dims=1)
                prev[:, 0] = self.tok_am1
                refined = refiner(h, g, embed(prev), self.win_h, self.win_m)
                self.blk[:, 1:] = lm_head(refined[:, 1:, :]).argmax(dim=-1)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _decode()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            _decode()

    @torch.inference_mode()
    def __call__(self, h, tok_am1, window_hidden, window_mask):
        self.h.copy_(h)
        self.tok_am1.copy_(tok_am1.reshape(1))
        if self.win_h is not None:
            self.win_h.copy_(window_hidden)
            self.win_m.copy_(window_mask)
        self.graph.replay()
        return self.blk[:, 1:].clone()


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    refiner: RefinerDecoder = None,
    refine_mode: str = "ar",        # "ar" | "parallel" | "conf" (confidence-gated)
    refine_passes: int = 1,
    conf_threshold: float = 0.9,
    graphed_verify=None,            # GraphedDecoder(q_len=block_size, need_hidden=True) for CUDA-graphed verify
    refiner_graph=None,             # RefinerParKGraph -> CUDA-graphed par-K refine (parallel mode, greedy)
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    W = refiner.window_size if refiner is not None else 0
    hidden_buffer = None  # rolling full-history target hidden, only needed for window>0

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=model.device
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill
    prefill_start = cuda_time()
    if graphed_verify is not None:
        graphed_verify.reset()
        output = graphed_verify.prefill(input_ids, position_ids[:, :num_input_tokens])
    else:
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True if block_size > 1 else False,
        )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits[:, -1:, :], temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        if W > 0:
            ctx_dim = target_hidden.shape[-1]
            hidden_buffer = torch.zeros(
                1, max_length + block_size, ctx_dim, dtype=target_hidden.dtype, device=model.device
            )
            hidden_buffer[:, :num_input_tokens] = target_hidden
    time_to_first_token = cuda_time() - prefill_start

    # Decode
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    stop_tensor = (
        torch.tensor(stop_token_ids, device=model.device) if stop_token_ids is not None else None
    )
    checked = start  # cursor: output_ids positions < checked already scanned for stop tokens

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)

            if refiner is None:
                # published DFlash: parallel draft straight from the drafter hidden
                draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
                block_output_ids[:, 1:] = sample(draft_logits, temperature)
            else:
                # AR refiner: free-running generation over the block, self-fed prev token
                h = draft_hidden[:, -block_size:, :]                       # [1, L, H] incl. anchor
                g = h.mean(dim=1, keepdim=True).expand(-1, block_size, -1)
                window_hidden = window_mask = None
                if W > 0:
                    widx = torch.arange(start - W, start, device=model.device)  # anchor-W .. anchor-1
                    window_hidden = hidden_buffer[:, widx.clamp(min=0), :]  # [1, W, ctx_dim]
                    window_mask = (widx >= 0).unsqueeze(0)                  # [1, W] (mask OOB before seq start)
                tok_am1 = output_ids[:, start - 1]                          # [1] token before anchor
                blk = block_output_ids.clone()                             # [1, L]; [:,0]=anchor
                if refine_mode == "ar":
                    # sequential, prev = self-generated (best quality, L-1 passes)
                    for k in range(1, block_size):
                        prev = blk.roll(shifts=1, dims=1)
                        prev[:, 0] = tok_am1
                        prev_emb = target.model.embed_tokens(prev)
                        refined = refiner(h, g, prev_emb, window_hidden, window_mask)
                        blk[:, k] = sample(
                            _readout(refiner, target.lm_head, refined[:, k : k + 1, :], h[:, k : k + 1, :]),
                            temperature,
                        )[:, 0]
                elif refine_mode == "conf":
                    # confidence-gated: trust high-confidence tokens (drafter's, then refiner's),
                    # re-predict ONLY the uncertain ones over refine_passes passes[].
                    dlogits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
                    blk[:, 1:] = sample(dlogits, temperature)
                    conf = torch.ones(1, block_size, device=model.device)          # [1, L]; anchor frozen
                    conf[:, 1:] = torch.softmax(dlogits.float(), dim=-1).max(dim=-1).values
                    frozen = conf >= conf_threshold
                    for _ in range(refine_passes):
                        if bool(frozen[:, 1:].all()):
                            break                                                  # all confident -> done
                        prev = blk.roll(shifts=1, dims=1)
                        prev[:, 0] = tok_am1
                        refined = refiner(h, g, target.model.embed_tokens(prev), window_hidden, window_mask)
                        rlogits = _readout(refiner, target.lm_head, refined, h)
                        new_tok = sample(rlogits, temperature)                     # [1, L]
                        new_conf = torch.softmax(rlogits.float(), dim=-1).max(dim=-1).values
                        upd = ~frozen                                              # only update uncertain positions
                        blk = torch.where(upd, new_tok, blk)
                        conf = torch.where(upd, new_conf, conf)
                        frozen = conf >= conf_threshold                            # newly-confident positions freeze
                elif refiner_graph is not None:
                    # CUDA-graphed par-K (greedy): one graph replay = seed + K passes, no launch overhead
                    blk[:, 1:] = refiner_graph(h, tok_am1, window_hidden, window_mask)
                else:
                    # parallel co-refine: seed prev with the drafter's own block, refine all
                    # positions at once; refine_passes passes propagate corrections.
                    blk[:, 1:] = sample(target.lm_head(draft_hidden[:, -block_size + 1 :, :]), temperature)
                    for _ in range(refine_passes):
                        prev = blk.roll(shifts=1, dims=1)
                        prev[:, 0] = tok_am1
                        prev_emb = target.model.embed_tokens(prev)
                        refined = refiner(h, g, prev_emb, window_hidden, window_mask)  # [1, L, H]
                        blk[:, 1:] = sample(
                            _readout(refiner, target.lm_head, refined[:, 1:, :], h[:, 1:, :]), temperature
                        )
                block_output_ids[:, 1:] = blk[:, 1:]

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        if graphed_verify is not None:
            graphed_verify.in_ids.copy_(block_output_ids)
            graphed_verify.pos.copy_(block_position_ids)
            graphed_verify.cache_pos.copy_(block_position_ids[0])
            v_logits, v_hidden = graphed_verify.step()
            output = SimpleNamespace(logits=v_logits, hidden_states=v_hidden)
        else:
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        if block_size > 1:
            new_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]
            if hidden_buffer is not None:
                hidden_buffer[:, start : start + acceptance_length + 1] = new_hidden  # fill accepted positions
            target_hidden = new_hidden

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        if graphed_verify is None:
            past_key_values_target.crop(start)  # StaticCache path rolls back via overwrite+mask, no crop

        if stop_tensor is not None:
            # only scan tokens written this step (positions [checked, start]); O(n) total, not O(n^2)
            if torch.isin(output_ids[:, checked : start + 1], stop_tensor).any():
                break
            checked = start + 1

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--refiner-path", type=str, default=None, help="refiner.pt from train_dflash_refiner.py")
    parser.add_argument(
        "--lowrank-rank", type=int, default=0,
        help="If >0, read refined out as base lm_head(h) + low-rank(refined-h) at this rank. Uses the "
        "checkpoint's TRAINED low-rank head if present; else SVD-initializes it post-hoc from lm_head "
        "(measures whether low-rank readout preserves accept). 0 = full lm_head (baseline).",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--refine-modes", type=str, default="baseline,drafter,par1,par3",
        help="comma list: drafter | ar | par<K> (parallel overwrite-all) | conf<K> (confidence-gated). e.g. drafter,par3,conf3",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.9, help="keep tokens with confidence >= this (conf mode)")
    parser.add_argument("--compile", action="store_true", help="torch.compile target/draft/refiner for optimized timing")
    parser.add_argument("--cudagraph", action="store_true", help="CUDA-graph the target decode (Phase A: baseline only)")
    parser.add_argument("--graph-refiner", action="store_true",
                        help="CUDA-graph the par-K refiner decode (greedy, temp=0) — minimizes the refiner's "
                        "per-block overhead without changing its structure. Applies to par<K> modes.")
    parser.add_argument("--max-cache-len", type=int, default=4096, help="StaticCache length for --cudagraph (>= max prompt + max_new_tokens)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        print("[warn] flash_attn not installed; using sdpa (lower speedup).")
        attn_impl = "sdpa"

    target = (
        AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation=attn_impl, dtype=torch.bfloat16)
        .to(device).eval()
    )
    draft_model = (
        DFlashDraftModel.from_pretrained(args.draft_name_or_path, attn_implementation=attn_impl, dtype=torch.bfloat16)
        .to(device).eval()
    )
    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    refiner = load_refiner(args.refiner_path, draft_model, device) if args.refiner_path else None

    if args.lowrank_rank > 0 and refiner is not None:
        if args.graph_refiner:
            raise SystemExit("--lowrank-rank is not supported with --graph-refiner; use eager par/ar modes.")
        if getattr(refiner, "lowrank_head", None) is None:
            # full-lm_head checkpoint -> attach a POST-HOC SVD low-rank head (untrained approximation).
            H = draft_model.config.hidden_size
            V = target.lm_head.weight.shape[0]
            head = LowRankReadout(H, int(V), int(args.lowrank_rank))
            head.svd_init_from(target.lm_head.weight)
            refiner.lowrank_head = head.to(device).to(torch.bfloat16)
            if dist.is_main():
                print(f"[lowrank] POST-HOC SVD low-rank readout (rank {args.lowrank_rank}) from lm_head "
                      f"(untrained -> expect some accept drop)")
        elif dist.is_main():
            print(f"[lowrank] using TRAINED low-rank head from checkpoint")

    if args.compile:
        # optimized kernels for accurate latency/overhead measurement (first iters slow while compiling)
        target = torch.compile(target)
        draft_model = torch.compile(draft_model)
        if refiner is not None:
            refiner = torch.compile(refiner)
        if dist.is_main():
            print("[compile] torch.compile enabled — warmup iters will be slow")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # modes (parsed once); each = (name, bs, refiner_or_None, refine_mode, passes)
    modes = []
    for tok in args.refine_modes.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "drafter":
            modes.append(("drafter", block_size, None, "ar", 1))
        elif tok == "baseline":
            modes.append(("baseline", 1, None, "ar", 1))
        elif tok == "ar" and refiner is not None:
            modes.append(("ar", block_size, refiner, "ar", 1))
        elif tok.startswith("par") and refiner is not None:
            modes.append((tok, block_size, refiner, "parallel", int(tok[3:]) if tok[3:] else 1))
        elif tok.startswith("conf") and refiner is not None:
            modes.append((tok, block_size, refiner, "conf", int(tok[4:]) if tok[4:] else 1))
        else:
            print(f"[warn] skipping refine-mode '{tok}' (unknown, or refiner not loaded)")
    if dist.is_main():
        print("running modes:", [m[0] for m in modes])

    # CUDA-graph the par-K refiner decode (greedy). One graph per K -> {mode_name: RefinerParKGraph}.
    refiner_graphs = {}
    if args.graph_refiner and refiner is not None:
        assert args.temperature == 0.0, "--graph-refiner is greedy (argmax); use --temperature 0.0"
        assert not args.compile, "--graph-refiner and --compile are mutually exclusive"
        ctx_dim = draft_model.fc.in_features
        for name, bs, rf, rmode, rpasses in modes:
            if rmode == "parallel" and rf is not None:
                refiner_graphs[name] = RefinerParKGraph(
                    rf, target.lm_head, target.model.embed_tokens,
                    block_size, rpasses, ctx_dim, device, torch.bfloat16,
                )
        if dist.is_main():
            print(f"[graph-refiner] captured par-K graphs for: {list(refiner_graphs)}")

    # CUDA-graphed target. (mutually exclusive with --compile)
    #   Phase A: plain-AR baseline (q_len=1).  Phase B1: block verify for spec modes (q_len=block_size).
    graphed_baseline = None
    graphed_verify = None
    _printed_baseline = [False]  # one-shot coherence print
    if args.cudagraph:
        assert not args.compile, "--cudagraph and --compile are mutually exclusive"
        torch.set_float32_matmul_precision("high")
        if any(m[0] == "baseline" for m in modes):
            graphed_baseline = GraphedDecoder(target, q_len=1, max_length=args.max_cache_len, device=device)
            if dist.is_main():
                print(f"[cudagraph] captured baseline decoder (max_cache_len={args.max_cache_len})")
        if any(m[1] > 1 for m in modes):  # any speculative mode (block_size>1) -> graph the verify forward
            graphed_verify = GraphedDecoder(target, q_len=block_size, max_length=args.max_cache_len,
                                            device=device, need_hidden=True)
            if dist.is_main():
                print(f"[cudagraph] captured verify decoder (q_len={block_size}, need_hidden=True)")

    def load_ds(name):
        if name.endswith(".jsonl") or os.path.exists(name):
            import json
            from datasets import Dataset
            rows = []
            with open(name) as fh:
                for line in fh:
                    convs = json.loads(line).get("conversations") or json.loads(line).get("messages") or []
                    for m in convs:
                        if (m.get("role") or m.get("from")) in ("user", "human"):
                            rows.append({"turns": [m.get("content") or m.get("value")]})
                            break
            return Dataset.from_list(rows)
        return load_and_process_dataset(name)

    if args.compile:
        # warmup each mode once so the timed loop measures compiled kernels, not compilation
        warm_ids = tokenizer("Hello, world. Please answer briefly.", return_tensors="pt").input_ids.to(target.device)
        for _name, bs, rf, rmode, rpasses in modes:
            dflash_generate(
                model=draft_model, target=target, input_ids=warm_ids,
                mask_token_id=draft_model.mask_token_id, max_new_tokens=64,
                block_size=bs, stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature, refiner=rf, refine_mode=rmode, refine_passes=rpasses,
                conf_threshold=args.conf_threshold,
            )
        if dist.is_main():
            print("[compile] warmup done — timing now uses compiled kernels")

    # --dataset is a comma list: run each benchmark in turn
    for ds_name in [d.strip() for d in args.dataset.split(",") if d.strip()]:
        dataset = load_ds(ds_name)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

        local = []  # keep ONLY lightweight metrics (no GPU tensors) -> multi-rank gather is cheap/safe
        for idx in tqdm(range(dist.rank(), len(dataset), dist.size()),
                        disable=not dist.is_main(), desc=ds_name.split("/")[-1]):
            instance = dataset[idx]
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for user_content in instance["turns"]:
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
                response = {}
                for name, bs, rf, rmode, rpasses in modes:
                    if graphed_baseline is not None and name == "baseline":
                        response[name] = baseline_generate_graphed(
                            graphed_baseline, input_ids, max_new_tokens=args.max_new_tokens,
                            stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature,
                        )
                        continue
                    response[name] = dflash_generate(
                        model=draft_model, target=target, input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id, max_new_tokens=args.max_new_tokens,
                        block_size=bs, stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature, refiner=rf, refine_mode=rmode, refine_passes=rpasses,
                        conf_threshold=args.conf_threshold,
                        graphed_verify=graphed_verify if bs > 1 else None,
                        refiner_graph=refiner_graphs.get(name),
                    )
                key = next((n for n, _, rf, _, _ in modes if rf is not None), "drafter")
                gen = response[key].output_ids[0, response[key].num_input_tokens :]
                messages.append({"role": "assistant", "content": tokenizer.decode(gen, skip_special_tokens=True)})
                if graphed_baseline is not None and not _printed_baseline[0] and dist.is_main():
                    _printed_baseline[0] = True
                    bgen = response["baseline"].output_ids[0, response["baseline"].num_input_tokens :]
                    print("\n[cudagraph baseline sample]\n" + tokenizer.decode(bgen, skip_special_tokens=True)[:600] + "\n")
                local.append({n: {"acc": r.acceptance_lengths, "tpot": r.time_per_output_token}
                              for n, r in response.items()})

        if dist.size() > 1:
            gathered = dist.gather(local, dst=0)
            if not dist.is_main():
                continue
            recs = list(chain(*[g for g in gathered if g]))
        else:
            recs = local

        print(f"\n========== RESULTS: {ds_name}  (n={len(recs)}) ==========")
        # speedup reference: prefer the target AR baseline (block_size=1), else drafter
        if any(m[0] == "baseline" for m in modes):
            base = "baseline"
        elif any(m[0] == "drafter" for m in modes):
            base = "drafter"
        else:
            base = modes[0][0]
        t_ref = np.mean([r[base]["tpot"] for r in recs])
        for name, _, _, _, _ in modes:
            t = np.mean([r[name]["tpot"] for r in recs])
            tau = np.mean([np.mean(r[name]["acc"]) for r in recs])
            accs = list(chain(*[r[name]["acc"] for r in recs]))
            hist = [accs.count(b) / len(accs) for b in range(block_size + 1)]
            print(f"[{name:8s}] accept={tau:.2f}  tpot={t*1000:.1f}ms  vs_{base}={t_ref / t:.2f}x")
            print(f"           histogram: {[f'{x*100:.1f}%' for x in hist]}")
        print("=" * 56)


if __name__ == "__main__":
    main()
