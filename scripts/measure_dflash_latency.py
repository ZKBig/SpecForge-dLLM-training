"""Latency / accept-len client for a MANUALLY-launched SGLang DFLASH server, on REAL gsm8k.

Launch the server yourself (so you see its logs), then point this at it. Loads gsm8k questions,
applies the Qwen3 chat template (enable_thinking=False, matching Domino's benchmark), sends them to
/generate, and reports tok/s, ms/tok, and accept_len (= completion_tokens / spec_verify_ct).

  python measure_dflash_latency.py --base-url http://127.0.0.1:30000 \
     --model-path /work/hdd/bcjw/zwang33/models/Qwen3-8B --n 32 --max-new-tokens 512
"""
import argparse
import time

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-path", default="/work/hdd/bcjw/zwang33/models/Qwen3-8B",
                    help="for the chat template / tokenizer (match the served target model)")
    ap.add_argument("--n", type=int, default=32, help="number of gsm8k questions")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n = min(args.n, len(ds))
    print(f"loaded gsm8k/{args.split}: using {n} questions", flush=True)

    def prompt_of(q):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def gen(prompt, max_new):
        r = requests.post(args.base_url + "/generate", json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
        }, timeout=1200)
        r.raise_for_status()
        return r.json()

    # warmup (graph/kernel warmup so the timed loop is clean)
    print("warmup...", flush=True)
    gen(prompt_of(ds[0]["question"]), 16)

    tot_toks, tot_time, tot_verify = 0, 0.0, 0
    print(f"\nmeasuring {n} gsm8k requests (max_new_tokens={args.max_new_tokens}, temp=0):", flush=True)
    for i in range(n):
        q = ds[i]["question"]
        t0 = time.perf_counter()
        data = gen(prompt_of(q), args.max_new_tokens)
        dt = time.perf_counter() - t0
        m = data["meta_info"]
        ct = m.get("completion_tokens", 0)
        vc = m.get("spec_verify_ct") or 0          # # target verify steps (None/0 if no spec)
        acc = ct / vc if vc else float("nan")
        tot_toks += ct
        tot_time += dt
        tot_verify += vc
        print(f"  [{i+1:>3}] {ct:>4} toks  {dt:6.2f}s  {ct/dt:7.1f} tok/s  "
              f"verify_ct={vc:<4} accept_len={acc:.2f}", flush=True)

    print("\n==================== SUMMARY (gsm8k) ====================")
    print(f"  tok/s        : {tot_toks / tot_time:.1f}")
    print(f"  ms / token   : {1000 * tot_time / tot_toks:.2f}")
    if tot_verify:
        print(f"  accept_len   : {tot_toks / tot_verify:.3f}   (completion_tokens / spec_verify_ct)")
    else:
        print("  accept_len   : N/A  (no spec_verify_ct -> spec decode not active, e.g. baseline)")
    print(f"  total        : {tot_toks} toks in {tot_time:.2f}s over {n} reqs")


if __name__ == "__main__":
    main()
