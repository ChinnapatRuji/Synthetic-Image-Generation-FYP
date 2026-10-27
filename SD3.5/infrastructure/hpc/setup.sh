#!/usr/bin/env bash
# One-time setup on the HPC login node (it needs internet):
#
#   bash infrastructure/hpc/setup.sh
#
# It asks for your Hugging Face token, uses it only for the SD3.5 download
# and does not save it. Optional settings:
#   PYTHON=python3.10            Python 3.10-3.12 used for both environments
#                                (otherwise conda installs 3.11 into PYTHON_ENV)
#   MODEL_DIR=/scratch/me/sd35m  Where to save SD3.5 Medium (large; default backend/models)
#   TORCH_INDEX=...              PyTorch CUDA build (default CUDA 12.4)
#   REDIS_ENV=~/.fyp-redis       Where to install redis-server if the HPC has none
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$PROJECT/backend"
KOHYA="$BACKEND/kohya_ss/sd-scripts"
PYTHON="${PYTHON:-python3}"
PYTHON_ENV="${PYTHON_ENV:-$HOME/.fyp-python}"
MODEL_DIR="${MODEL_DIR:-$BACKEND/models/sd3.5-medium}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
REDIS_ENV="${REDIS_ENV:-$HOME/.fyp-redis}"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
    if ! command -v conda >/dev/null; then
        echo "Kohya needs Python 3.10-3.12, and $PYTHON is not. Load a conda or Python 3.10-3.12"
        echo "module (list them with: module avail), or set PYTHON=..., then rerun."
        exit 1
    fi
    echo "==> Installing Python 3.11 with conda ($PYTHON_ENV)"
    [ -x "$PYTHON_ENV/bin/python" ] || conda create -y -p "$PYTHON_ENV" -c conda-forge python=3.11
    PYTHON="$PYTHON_ENV/bin/python"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Hugging Face token: " HF_TOKEN
    echo
fi
export HF_TOKEN

echo "==> Backend environment ($BACKEND/venv)"
"$PYTHON" -m venv "$BACKEND/venv"
BPY="$BACKEND/venv/bin/python"
"$BPY" -m pip install --upgrade pip
"$BPY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX"
"$BPY" -m pip install -r "$BACKEND/requirements/requirements-api.txt" -r "$BACKEND/requirements/requirements-ml.txt"

# Kohya pins newer transformers/diffusers than the backend, so it gets its own venv.
echo "==> Kohya sd-scripts ($KOHYA)"
[ -d "$KOHYA/.git" ] || git clone https://github.com/kohya-ss/sd-scripts.git "$KOHYA"
"$PYTHON" -m venv "$KOHYA/venv"
KPY="$KOHYA/venv/bin/python"
"$KPY" -m pip install --upgrade pip
"$KPY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX"
(cd "$KOHYA" && "$KPY" -m pip install --upgrade -r requirements.txt)

echo "==> Kohya's SD3 tokenizers, saved so training needs no internet"
(cd "$KOHYA" && "$KPY" - "$BACKEND/kohya_ss/tokenizer_cache" <<'EOF'
import sys
from library.strategy_sd3 import Sd3TokenizeStrategy
Sd3TokenizeStrategy(256, sys.argv[1])
EOF
)

echo "==> SD3.5 Medium -> $MODEL_DIR"
"$BPY" - "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    "stabilityai/stable-diffusion-3.5-medium",
    local_dir=sys.argv[1],
    # Duplicate fp16/fp8 weights and preview images aren't used.
    ignore_patterns=["*.fp16*", "*fp8*", "*.png", "*.jpg"],
)
EOF
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"

echo "==> Pointing backend/config.json at $MODEL_DIR"
"$BPY" - "$BACKEND/config.json" "$MODEL_DIR" <<'EOF'
import json, sys
path, model_dir = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    config = json.load(f)
config["model"]["base"] = model_dir
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
    f.write("\n")
EOF

echo "==> Redis"
if command -v redis-server >/dev/null || [ -x "$REDIS_ENV/bin/redis-server" ]; then
    echo "redis-server found"
elif command -v conda >/dev/null; then
    conda create -y -p "$REDIS_ENV" -c conda-forge redis-server
else
    echo "No redis-server or conda found. Load a conda module (e.g. module load miniforge3) and rerun."
    exit 1
fi

echo
echo "Setup done. Submit a job with infrastructure/hpc/generate.pbs (see the comment at its top)."
