"""Benchmark a CO-TRAINED (DFlash drafter + AR refiner) checkpoint.

Same as benchmark_refiner.py, EXCEPT it loads the CO-TRAINED drafter directly from the
checkpoint's `draft_state_dict` (the refiner head was trained on that drafter's hidden states,
so we must use it — not the original z-lab drafter). No extraction step.

Reuses benchmark_refiner's generate/loader/graph functions; only the drafter loading differs.

  torchrun --nproc_per_node=N benchmark_refiner_cotrain.py \
      --refiner-path /path/to/epoch_X_step_Y/refiner_cotrain.pt \
      --model-name-or-path Qwen/Qwen3-8B \
      --dataset alpaca,gsm8k --refine-modes drafter,par2,par3 --temperature 0.0
"""
import argparse
import os
import sys
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DFlashDraftModel, load_and_process_dataset
import distributed as dist

# reuse the EXACT generation / loader logic from the (original) benchmark_refiner (no duplication)
from benchmark_refiner import dflash_generate, load_refiner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--refiner-path", type=str, required=True,
                        help="refiner_cotrain.pt (must contain BOTH refiner_state_dict and draft_state_dict)")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16",
                        help="loaded only for architecture/config; the drafter WEIGHTS are overridden "
                        "by the checkpoint's co-trained draft_state_dict.")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--refine-modes", type=str, default="baseline,drafter,par1,par3",
        help="comma list: drafter | ar | par<K> | conf<K>. e.g. drafter,par2,par3",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--compile", action="store_true")
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

    # --- THE co-train difference: override the drafter with the CO-TRAINED weights from the ckpt ---
    ck = torch.load(args.refiner_path, map_location="cpu", weights_only=False)
    assert "draft_state_dict" in ck, (
        "checkpoint has no 'draft_state_dict' — this is not a co-train checkpoint. "
        "Use the plain benchmark_refiner.py for a frozen-refiner refiner.pt."
    )
    miss, unexp = draft_model.load_state_dict(ck["draft_state_dict"], strict=False)
    if dist.is_main():
        print(f"[cotrain] checkpoint epoch={ck.get('epoch')} step={ck.get('global_step')}")
        print(f"[cotrain] loaded CO-TRAINED drafter: missing={len(miss)} unexpected={len(unexp)}")
        if miss:
            print(f"[cotrain] WARNING keys NOT overridden (stay at base z-lab): {list(miss)[:8]}")
    del ck

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    refiner = load_refiner(args.refiner_path, draft_model, device)  # reads refiner_state_dict

    if args.compile:
        target = torch.compile(target)
        draft_model = torch.compile(draft_model)
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
        elif tok == "ar":
            modes.append(("ar", block_size, refiner, "ar", 1))
        elif tok.startswith("par"):
            modes.append((tok, block_size, refiner, "parallel", int(tok[3:]) if tok[3:] else 1))
        elif tok.startswith("conf"):
            modes.append((tok, block_size, refiner, "conf", int(tok[4:]) if tok[4:] else 1))
        else:
            print(f"[warn] skipping refine-mode '{tok}'")
    if dist.is_main():
        print("running modes:", [m[0] for m in modes])

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

    for ds_name in [d.strip() for d in args.dataset.split(",") if d.strip()]:
        dataset = load_ds(ds_name)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

        local = []
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
                    response[name] = dflash_generate(
                        model=draft_model, target=target, input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id, max_new_tokens=args.max_new_tokens,
                        block_size=bs, stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature, refiner=rf, refine_mode=rmode, refine_passes=rpasses,
                        conf_threshold=args.conf_threshold,
                    )
                key = next((n for n, _, rf, _, _ in modes if rf is not None), "drafter")
                gen = response[key].output_ids[0, response[key].num_input_tokens:]
                messages.append({"role": "assistant", "content": tokenizer.decode(gen, skip_special_tokens=True)})
                local.append({n: {"acc": r.acceptance_lengths, "tpot": r.time_per_output_token}
                              for n, r in response.items()})

        if dist.size() > 1:
            gathered = dist.gather(local, dst=0)
            if not dist.is_main():
                continue
            recs = list(chain(*[g for g in gathered if g]))
        else:
            recs = local

        print(f"\n========== RESULTS: {ds_name}  (n={len(recs)})  [CO-TRAINED drafter] ==========")
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
