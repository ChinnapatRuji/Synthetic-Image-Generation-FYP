#!/bin/bash
# One-time setup. Run on the LOGIN node (compute nodes may not have internet):
#   cd <project dir> && bash "HPC guidelines/setup_env.sh"
set -euo pipefail

cd "$(dirname "$0")/.."

# Python from the cluster's module system. Check `module avail python` and adjust if needed.
module load python 2>/dev/null || true
python3 --version

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Linux x86 pip wheels of torch include CUDA support.
pip install torch diffusers transformers peft accelerate safetensors pillow tqdm pydantic

# Keep model downloads out of the small home quota. Must match HF_HOME in generate.pbs.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"

# Pre-download the base model and VAE named in config.json so the job can run offline.
python - <<'EOF'
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from config import load_config

cfg = load_config()
StableDiffusionPipeline.from_pretrained(cfg.model.base, torch_dtype=torch.float16)
if cfg.model.vae:
    AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)
print("Models cached in", __import__("os").environ["HF_HOME"])
EOF
