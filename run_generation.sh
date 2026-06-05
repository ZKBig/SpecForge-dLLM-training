set -e

MODEL_PATH=/gpfs/zwang33/models/Qwen3-8B
OUT_PATH=/gpfs/zwang33/dflash_data/perfectblend_qwen3-8b_regen.jsonl
PORT=30000
DP_SIZE=8

# --- launch sglang server in background ---
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --dp-size "$DP_SIZE" --tp-size 1 \
  --mem-fraction-static 0.85 --reasoning-parser qwen3 \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT   # always reap the server on exit

# --- wait until the server is ready ---
until curl -sf "http://localhost:$PORT/health"; do echo "waiting server..."; sleep 5; done

# --- generate (output to persistent /gpfs) ---
mkdir -p "$(dirname "$OUT_PATH")"
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3-8B --is-reasoning-model \
  --concurrency 256 --max-tokens 8192 --temperature 0.8 \
  --server-address "localhost:$PORT" \
  --input-file-path ./cache/dataset/perfectblend_train.jsonl \
  --output-file-path "$OUT_PATH"

echo "DONE: wrote $OUT_PATH"
