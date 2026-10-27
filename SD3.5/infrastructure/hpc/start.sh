#!/usr/bin/env bash
# Start Redis, the GPU worker and the API on a GPU node.
#
# Interactive (inside e.g. qsub -I ... ngpus=1): serve the API until Ctrl-C.
#   bash infrastructure/hpc/start.sh
# With a command: start everything, run the command, then stop everything.
#   bash infrastructure/hpc/start.sh backend/venv/bin/python infrastructure/hpc/run_pipeline.py dataset
#
# Optional settings: API_PORT and REDIS_PORT (default: any free port), STORAGE_ROOT.
set -euo pipefail

ORIG_PWD="$PWD"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$PROJECT/backend"
PY="$BACKEND/venv/bin/python"
REDIS_ENV="${REDIS_ENV:-$HOME/.fyp-redis}"
RUNTIME="$PROJECT/storage/runtime"

# GPU nodes can be shared, so default to ports nobody else is using.
free_port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
API_PORT="${API_PORT:-$(free_port)}"
REDIS_PORT="${REDIS_PORT:-$(free_port)}"

export REDIS_URL="redis://localhost:$REDIS_PORT"
export API_URL="http://127.0.0.1:$API_PORT"
# setup.sh saved the model and tokenizers locally, so never wait on the internet.
export HF_HUB_OFFLINE=1

REDIS_SERVER="$REDIS_ENV/bin/redis-server"
[ -x "$REDIS_SERVER" ] || REDIS_SERVER="$(command -v redis-server)" || {
    echo "redis-server not found: run infrastructure/hpc/setup.sh first"
    exit 1
}

cd "$BACKEND"
mkdir -p "$RUNTIME"

# Fall back to fp16 on GPUs without bf16 (V100 and older).
"$PY" - config.json <<'EOF'
import json, sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible: run this inside a GPU job")
print("GPU:", torch.cuda.get_device_name(0))
with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
memory = config["training"]["memory"]
if memory["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
    memory["mixed_precision"] = "fp16"
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    print("This GPU has no bf16: switched config.json to fp16")
EOF

"$REDIS_SERVER" --port "$REDIS_PORT" --bind 127.0.0.1 --dir "$RUNTIME" --appendonly yes \
    > "$RUNTIME/redis.log" 2>&1 &
pids=($!)
trap 'echo "Stopping services (a running job finishes first)"; kill "${pids[@]}" 2>/dev/null; wait' EXIT

for _ in $(seq 20); do
    "$PY" -c "import redis; redis.Redis(port=$REDIS_PORT).ping()" 2>/dev/null && break
    sleep 1
done
sleep 1
kill -0 "${pids[0]}" 2>/dev/null || {
    echo "Redis did not start. See $RUNTIME/redis.log"
    exit 1
}

"$PY" -m celery -A app.workers.celery_app worker -Q gpu,default --pool=solo --loglevel=info \
    > "$RUNTIME/worker.log" 2>&1 &
pids+=($!)

"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" \
    > "$RUNTIME/api.log" 2>&1 &
pids+=($!)

"$PY" - "$API_URL/api/v1/system/health" <<'EOF'
import sys, time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen(sys.argv[1])
        sys.exit(0)
    except OSError:
        time.sleep(1)
sys.exit("API did not start")
EOF
echo "Services running on $(hostname). Logs: $RUNTIME"

if [ $# -gt 0 ]; then
    cd "$ORIG_PWD"
    "$@"
    exit
fi

cat <<EOF
From your own computer, open a tunnel:
    ssh -N -L $API_PORT:127.0.0.1:$API_PORT -J <user>@<login-node> <user>@$(hostname)
then browse http://localhost:$API_PORT/docs
Press Ctrl-C to stop.
EOF

while kill -0 "${pids[@]}" 2>/dev/null; do sleep 5; done
echo "A service stopped unexpectedly; see the logs in $RUNTIME"
exit 1
