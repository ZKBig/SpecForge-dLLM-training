#!/bin/bash
# Sharded NON-THINKING DFlash data generation — one instance per node.
# Each node serves Qwen3-8B (dp-size 8) and regenerates its slice of the FIRST
# $TOTAL_SAMPLES seed prompts with thinking DISABLED (direct answers). Outputs go
# to per-shard files on /gpfs; cat them together when all shards finish.
#
# Env (set per job): SHARD_IDX (0..N-1), N_SHARDS (total nodes), TOTAL_SAMPLES.
set -e

SHARD_IDX=${SHARD_IDX:-0}
N_SHARDS=${N_SHARDS:-10}                  # 10 nodes
TOTAL_SAMPLES=${TOTAL_SAMPLES:-0}         # 0 = use the FULL seed (~1.4M -> ~142k/node); else cap to first N
PORT=30000
DP_SIZE=8

MODEL_PATH=/gpfs/zwang33/models/Qwen3-8B
SEED_PATH=/gpfs/zwang33/dflash_data/perfectblend_train.jsonl     # shared full seed (1.4M)
MYSHARD=/gpfs/zwang33/dflash_data/nothink_shard_${SHARD_IDX}_of_${N_SHARDS}.jsonl
OUT_PATH=/gpfs/zwang33/dflash_data/regen_nothink_shard_${SHARD_IDX}_of_${N_SHARDS}.jsonl
SERVER_LOG=/gpfs/zwang33/cache/sglang_server_nothink_${SHARD_IDX}.log
mkdir -p /gpfs/zwang33/dflash_data /gpfs/zwang33/cache

# model present? (shared on /gpfs; download once if missing)
if [ ! -d "$MODEL_PATH" ]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$MODEL_PATH')"
fi

# --- this node's round-robin slice of the seed (skip if pre-created manually) ---
if [ ! -s "$MYSHARD" ]; then
  if [ "${TOTAL_SAMPLES:-0}" -gt 0 ] 2>/dev/null; then
    head -n "$TOTAL_SAMPLES" "$SEED_PATH" | awk -v n="$N_SHARDS" -v i="$SHARD_IDX" 'NR % n == i' > "$MYSHARD"
  else
    awk -v n="$N_SHARDS" -v i="$SHARD_IDX" 'NR % n == i' "$SEED_PATH" > "$MYSHARD"   # full seed
  fi
fi
echo "shard $SHARD_IDX/$N_SHARDS: $(wc -l < "$MYSHARD") prompts"

# --- launch sglang server (this node's 8 GPUs); no reasoning-parser needed ---
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --dp-size "$DP_SIZE" --tp-size 1 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# --- wait until ready (bail if it dies) ---
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "ERROR: server died"; tail -n 30 "$SERVER_LOG"; exit 1; fi
  sleep 5
done

# --- regenerate this shard, thinking DISABLED (--no-thinking, NOT --is-reasoning-model) ---
cd /app/SpecForge-dLLM-training && git pull   # need latest regenerate_train_data.py (has --no-thinking)
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3-8B --no-thinking \
  --concurrency 128 --max-tokens 4096 --temperature 0.8 \
  --server-address "localhost:$PORT" \
  --input-file-path "$MYSHARD" \
  --output-file-path "$OUT_PATH"

echo "DONE shard $SHARD_IDX: wrote $OUT_PATH"
