#!/bin/bash
# Domino DFLASH sglang on IBM (goon-torch-dev), self-contained:
#   build the Domino sglang fork from source -> launch DFLASH server (bg) -> wait /health ->
#   measure accept_len + tok/s on gsm8k -> kill server.
# Modeled on run_dflash_datagen_local.sh. Runs as ONE shell so cwd/PATH persist (unlike YAML
# setupCommands items). Invoke from the YAML: `bash /gpfs/zwang33/run_domino_sglang_ibm.sh`.
#
# PREREQ you must place on /gpfs first:
#   - this script                 -> /gpfs/zwang33/run_domino_sglang_ibm.sh
#   - measure_dflash_latency.py   -> /gpfs/zwang33/measure_dflash_latency.py
#   - export SGLANG_KERNEL_CU12_WHEEL=<your cu12 sglang-kernel wheel>  (set in the YAML env)
set -e

TARGET=/gpfs/zwang33/models/Qwen3-8B
DRAFT=Huang2020/Qwen3-8B-Domino-b16
PORT=30000
OUT_DIR=/gpfs/zwang33/domino_out
SERVER_LOG=$OUT_DIR/sglang_server.log
MEASURE=scripts/measure_dflash_latency.py
mkdir -p "$OUT_DIR"

# ============================ 1. build the Domino sglang fork (cu128 path) ============================
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
python -m pip install --upgrade pip ninja

rm -rf /tmp/domino
git clone --branch sglang-feat/dflash-domino https://github.com/jianuo-huang/Domino.git /tmp/domino
cd /tmp/domino

# cu128 / torch 2.9 dependency pin patch (Domino README)
sed -i \
  -e 's/"torch==2.11.0"/"torch==2.9.1+cu128"/' \
  -e 's/"torchaudio==2.11.0"/"torchaudio==2.9.1+cu128"/' \
  -e 's/"torchvision"/"torchvision==0.24.1+cu128"/' \
  -e 's/"kernels"/"kernels==0.14.1"/' \
  -e '/"sglang-kernel==0.4.2"/d' \
  python/pyproject.toml

python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ./python
python -m pip install --force-reinstall --no-deps "${SGLANG_KERNEL_CU12_WHEEL}"
python -m pip install -r requirements-hf.txt
python -c "import sglang; print('sglang build OK:', sglang.__version__)"

# ============================ 2. launch the DFLASH server (background) ============================
python -m sglang.launch_server \
  --model-path "$TARGET" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --trust-remote-code --attention-backend flashinfer \
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.75 \
  --max-running-requests 64 \
  --host 0.0.0.0 --port "$PORT" > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT   # reap the server when this script exits

# ============================ 3. wait until ready (bail if it dies) ============================
echo "Waiting for Domino sglang server (log: $SERVER_LOG) ..."
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "ERROR: sglang server died during startup. Last 40 lines of $SERVER_LOG:"
    tail -n 40 "$SERVER_LOG"; exit 1
  fi
  sleep 5
done
echo "Server is ready."

# ============================ 4. measure accept_len + tok/s on gsm8k ============================
python "$MEASURE" --base-url "http://127.0.0.1:$PORT" \
  --model-path "$TARGET" --n 32 --max-new-tokens 512

echo "DONE: Domino DFLASH sglang measured on gsm8k."
