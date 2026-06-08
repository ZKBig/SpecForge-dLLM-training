#!/bin/bash
# Sharded DFlash data generation — one instance per node. Each node serves Qwen3-8B on
# its own 8 GPUs (dp-size 8) and regenerates only its slice of the seed data. Outputs go
# to per-shard files on /gpfs; cat them together when all shards finish.
#
# Env (set per job): SHARD_IDX (0..N-1), N_SHARDS (total nodes/jobs).
set -e

SHARD_IDX=${SHARD_IDX:-0}
N_SHARDS=${N_SHARDS:-1}
PORT=30000
DP_SIZE=8

MODEL_PATH=/gpfs/zwang33/models/Qwen3-8B
SEED_PATH=/gpfs/zwang33/dflash_data/perfectblend_train.jsonl     # shared full seed
MYSHARD=/gpfs/zwang33/dflash_data/shard_${SHARD_IDX}_of_${N_SHARDS}.jsonl
OUT_PATH=/gpfs/zwang33/dflash_data/regen_shard_${SHARD_IDX}_of_${N_SHARDS}.jsonl
SERVER_LOG=/gpfs/zwang33/cache/sglang_server_${SHARD_IDX}.log
mkdir -p /gpfs/zwang33/dflash_data /gpfs/zwang33/cache

# model present? (shared on /gpfs; download once if missing)
if [ ! -d "$MODEL_PATH" ]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$MODEL_PATH')"
fi

# --- this node's slice of the seed (round-robin by line number) ---
awk -v n="$N_SHARDS" -v i="$SHARD_IDX" 'NR % n == i' "$SEED_PATH" > "$MYSHARD"
echo "shard $SHARD_IDX/$N_SHARDS: $(wc -l < "$MYSHARD") samples"

# --- launch sglang server (this node's 8 GPUs) in background ---
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --dp-size "$DP_SIZE" --tp-size 1 \
  --mem-fraction-static 0.85 --reasoning-parser qwen3 \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# --- wait until ready (bail if it dies) ---
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "ERROR: server died"; tail -n 30 "$SERVER_LOG"; exit 1; fi
  sleep 5
done

# --- regenerate this shard (use the SpecForge bundled in the image) ---
cd /app/SpecForge-dLLM-training
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3-8B --is-reasoning-model \
  --concurrency 128 --max-tokens 16384 --temperature 0.8 \
  --server-address "localhost:$PORT" \
  --input-file-path "$MYSHARD" \
  --output-file-path "$OUT_PATH"

echo "DONE shard $SHARD_IDX: wrote $OUT_PATH"
