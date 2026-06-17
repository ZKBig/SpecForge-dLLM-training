"""End-to-end spec-decoding benchmark for a trained Domino checkpoint.

Sibling of benchmark_refiner.py: SAME harness (distributed, dataset, target-verify loop,
multi-rank gather, output), but the block is drafted by the Domino GRU head instead of the
AR refiner. Compares, on the SAME prompts:
  baseline : block_size=1 plain target AR (speedup reference)
  drafter  : DFlash block draft -> target verify   (the domino model's drafter, NO GRU head)
  domino   : DFlash block draft + GRU head (free-running) -> target verify

The drafter is version2's DFlashDraftModel (it carries the domino head: prefix_gru + embed_proj,
and the same cached inference forward as dflash_old2's model). The harness helpers (sample,
extract_context_feature, dataset, distributed) are dflash_old2's, exactly like benchmark_refiner.

Alignment: the domino config is shift_label (NEXT-token): output_hidden[k] predicts blk[k+1].
--align next (default) uses that; --align same reproduces generic DFlash spec_generate
(output_hidden[k]->blk[k]). Run both and keep the one whose `drafter` accept is sane.

  ./run_benchmark_domino.sh <domino_ckpt_dir> [dataset] [max_samples]
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

# dflash_old2 harness (model package + distributed) — same as benchmark_refiner.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import sample, load_and_process_dataset, extract_context_feature
from model.dflash_domino import DominoDraftModel  # self-contained domino drafter (no specforge dep)
import distributed as dist


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    drafter, target, input_ids, mask_token_id, max_new_tokens, block_size,
    stop_token_ids, temperature=0.0, use_head=False, align="next",
) -> SimpleNamespace:
    """block_size==1 -> plain target AR (baseline). block_size>1 -> DFlash spec-decode,
    with the Domino GRU head when use_head=True."""
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    embed = target.model.embed_tokens
    lm_head = target.lm_head
    ss = (getattr(drafter, "pure_draft_prefix_len", 1) if drafter.shift_label
          else 1 + getattr(drafter, "pure_draft_prefix_len", 0))

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device)
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    pkv_target = DynamicCache()
    pkv_draft = DynamicCache()

    # prefill
    prefill_start = cuda_time()
    out = target(input_ids, position_ids=position_ids[:, :num_input], past_key_values=pkv_target,
                 use_cache=True, logits_to_keep=1, output_hidden_states=(block_size > 1))
    output_ids[:, :num_input] = input_ids
    output_ids[:, num_input:num_input + 1] = sample(out.logits[:, -1:, :], temperature)
    target_hidden = extract_context_feature(out.hidden_states, drafter.target_layer_ids) if block_size > 1 else None
    ttft = cuda_time() - prefill_start

    accept_lengths = []
    start = num_input
    decode_start = cuda_time()
    draft_prefill = True
    stop_tensor = torch.tensor(stop_token_ids, device=target.device) if stop_token_ids else None
    checked = start

    while start < max_length:
        block_ids = output_ids[:, start:start + block_size].clone()
        block_pos = position_ids[:, start:start + block_size]

        if block_size > 1:
            noise_emb = embed(block_ids)
            draft_out = drafter(
                target_hidden=target_hidden, noise_embedding=noise_emb,
                position_ids=position_ids[:, pkv_draft.get_seq_length():start + block_size],
                past_key_values=pkv_draft, use_cache=True, is_causal=False,
            )
            pkv_draft.crop(start)
            h = draft_out[:, -block_size:, :]                            # (1, block, H); h[0]=anchor
            base_logits = lm_head(h)                                     # (1, block, V)

            if not use_head:
                if align == "next":                                     # h[k] -> blk[k+1]
                    block_ids[:, 1:] = sample(base_logits[:, :block_size - 1, :], temperature)
                else:                                                   # h[k] -> blk[k]
                    block_ids[:, 1:] = sample(base_logits[:, 1:, :], temperature)
            else:
                gru, embed_proj = drafter.prefix_gru, drafter.embed_proj
                h_gru = None
                if align == "next":
                    for k in range(block_size - 1):                     # predict blk[k+1] from h[k]
                        g_out, h_gru = gru(embed(block_ids[:, k:k + 1]), h_gru)   # GRU(blk[0..k])
                        logit = base_logits[:, k, :]
                        if k >= ss:
                            logit = logit + embed_proj(torch.cat([h[:, k, :], g_out[:, 0, :]], dim=-1))
                        block_ids[:, k + 1] = sample(logit.unsqueeze(1), temperature)[:, 0]
                else:
                    for k in range(1, block_size):                     # predict blk[k] from h[k]
                        g_out, h_gru = gru(embed(block_ids[:, k - 1:k]), h_gru)
                        logit = base_logits[:, k, :]
                        if k >= ss:
                            logit = logit + embed_proj(torch.cat([h[:, k, :], g_out[:, 0, :]], dim=-1))
                        block_ids[:, k] = sample(logit.unsqueeze(1), temperature)[:, 0]

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        # target verify
        out = target(block_ids, position_ids=block_pos, past_key_values=pkv_target,
                     use_cache=True, output_hidden_states=(block_size > 1))
        posterior = sample(out.logits, temperature)
        if block_size > 1:
            acc = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        else:
            acc = 0
        output_ids[:, start:start + acc + 1] = block_ids[:, :acc + 1]
        output_ids[:, start + acc + 1] = posterior[:, acc]
        if block_size > 1:
            target_hidden = extract_context_feature(out.hidden_states, drafter.target_layer_ids)[:, :acc + 1, :]
        accept_lengths.append(acc + 1)
        start += acc + 1
        pkv_target.crop(start)

        if stop_tensor is not None:
            if torch.isin(output_ids[:, checked:start + 1], stop_tensor).any():
                break
            checked = start + 1

    total_decode = cuda_time() - decode_start
    out_ids = output_ids[:, :min(start, max_length)]
    out_ids = out_ids[:, out_ids[0] != mask_token_id]
    n_out = out_ids.shape[1] - num_input
    return SimpleNamespace(
        output_ids=out_ids, num_input_tokens=num_input, num_output_tokens=n_out,
        time_to_first_token=ttft, time_per_output_token=total_decode / max(n_out, 1),
        acceptance_lengths=accept_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True,
                        help="trained Domino checkpoint dir (save_pretrained: epoch_X_step_Y)")
    parser.add_argument("--modes", type=str, default="baseline,drafter,domino")
    parser.add_argument("--align", type=str, default="next", choices=["next", "same"])
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        if dist.is_main():
            print("[warn] flash_attn not installed; using sdpa (lower speedup).")
        attn = "sdpa"

    target = (AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation=attn, dtype=torch.bfloat16)
              .to(device).eval())
    drafter = (DominoDraftModel.from_pretrained(args.draft_name_or_path, attn_implementation=attn, dtype=torch.bfloat16)
               .to(device).eval())
    drafter.config._attn_implementation = attn
    block_size = args.block_size or drafter.block_size
    mask_token_id = drafter.mask_token_id
    if dist.is_main():
        proj = drafter.config.dflash_config.get("projector_type")
        print(f"projector={proj} shift_label={drafter.shift_label} block_size={block_size} align={args.align}")
        if proj != "domino" and "domino" in args.modes:
            print("[warn] checkpoint has NO domino projector — 'domino' mode will fail.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    stop_ids = [tokenizer.eos_token_id]
    # each mode -> (name, block_size, use_head)
    mode_cfg = []
    for tok in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if tok == "baseline":
            mode_cfg.append(("baseline", 1, False))
        elif tok == "drafter":
            mode_cfg.append(("drafter", block_size, False))
        elif tok == "domino":
            mode_cfg.append(("domino", block_size, True))
        else:
            if dist.is_main():
                print(f"[warn] skipping unknown mode '{tok}'")
    if dist.is_main():
        print("running modes:", [m[0] for m in mode_cfg])

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
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking
                )
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
                response = {}
                for name, bs, use_head in mode_cfg:
                    response[name] = dflash_generate(
                        drafter, target, input_ids, mask_token_id, args.max_new_tokens, bs,
                        stop_ids, args.temperature, use_head=use_head, align=args.align,
                    )
                # multi-turn: continue with a speculative mode's output (domino if present, else drafter)
                key = "domino" if "domino" in response else ("drafter" if "drafter" in response else mode_cfg[0][0])
                r = response[key]
                gen = r.output_ids[0, r.num_input_tokens:]
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

        print(f"\n========== {ds_name}  (n={len(recs)}, block={block_size}, align={args.align}) ==========")
        ref = "baseline" if any(m[0] == "baseline" for m in mode_cfg) else mode_cfg[0][0]
        t_ref = np.mean([r[ref]["tpot"] for r in recs])
        for name, _, _ in mode_cfg:
            t = np.mean([r[name]["tpot"] for r in recs])
            tau = np.mean([np.mean(r[name]["acc"]) for r in recs])
            accs = list(chain(*[r[name]["acc"] for r in recs]))
            hist = [accs.count(b) / len(accs) for b in range(block_size + 1)]
            print(f"[{name:9s}] accept={tau:.2f}  tpot={t*1000:.2f}ms  vs_{ref}={t_ref/t:.2f}x")
            print(f"            histogram: {[f'{x*100:.0f}%' for x in hist]}")
        print("=" * 60)


if __name__ == "__main__":
    main()
