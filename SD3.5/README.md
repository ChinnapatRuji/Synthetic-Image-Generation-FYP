# Synthetic Defect Image Generation (SD3.5 + LoRA)

This project trains a LoRA on a small set of real solder-joint photos (one folder per
defect class), then generates new synthetic images of each class. Training uses
[Kohya's sd-scripts](https://github.com/kohya-ss/sd-scripts) and generation uses
[diffusers](https://github.com/huggingface/diffusers). The base model is
**Stable Diffusion 3.5 Medium**. Everything runs on the NUS **Vanda** HPC as a single
batch job.

---

## 1. Changes made to the original repository

**Base model: Realistic Vision 5.1 (SD 1.5) → Stable Diffusion 3.5 Medium**
- `backend/config.json`: the model is now `stabilityai/stable-diffusion-3.5-medium` with
  `"architecture": "sd3"` and no separate VAE. Training uses bf16 at 512 px, which suits the
  299 px source images. `min_snr_gamma`/`noise_offset` are off (they don't apply to SD3), and
  the text encoders aren't trained. Generation uses SD3.5's own sampler, 40 steps and cfg 4.5.
- `backend/config.py`: new `architecture` field (`sd15` or `sd3`). It rejects settings that
  don't work with SD3. Old SD 1.5 configs still load.
- `backend/app/ml/training/train_sd_lora_kohya.py`: for SD3, Kohya runs
  `sd3_train_network.py` with `networks.lora_sd3`. Kohya needs the model as single
  files, so this code fetches the checkpoint and its three text encoders from the
  Hugging Face repo or a local folder. It pre-computes the caption encodings, reads
  Kohya's tokenizers from a local cache (so training needs no internet), and runs
  Kohya from its own Python environment, because Kohya needs newer library versions
  than the backend.
- `backend/app/ml/generation/generate.py`: SD3 txt2img and img2img pipelines. It also
  **converts Kohya's SD3 LoRA files to diffusers format**, because diffusers can't read
  them directly; the converter was checked against diffusers' own conversion. It picks
  bf16 automatically and offloads to CPU on GPUs under 24 GB. ControlNet stays
  SD1.5-only.
- `backend/app/modules/generation/`: added a `"default"` sampler. SD3 generation requests
  using ControlNet, an SD1.5 sampler, or a size not divisible by 16 are rejected with a
  clear error.

**Fixes to existing code**
- Added the missing `backend/app/infrastructure/storage/paths.py` (`StoragePaths`).
  Without it the API couldn't start.
- Class descriptions are now saved and added to the training captions and generation
  prompts. Before, they were silently dropped. Editing a class without renaming it no
  longer crashes.
- Underscores in class names become spaces in training captions too, so captions match
  the generation prompts.
- The Redis address is configurable (`REDIS_URL`).
- Requirements: added `httpx` (for the pipeline script), plus `sentencepiece` and
  `protobuf` (needed to load SD3's T5 tokenizer).

**New: running on the HPC** (`infrastructure/hpc/`)
- `setup.sh`: one-time install. It creates the backend environment, installs Kohya in its
  own environment, saves Kohya's tokenizers, downloads SD3.5 Medium, points
  `config.json` at it, and installs Redis.
- `start.sh`: starts Redis, the GPU worker and the API on a GPU node (falling back to
  fp16 on GPUs without bf16). Either it keeps them running, or it runs one command and
  then shuts everything down.
- `run_pipeline.py`: the whole workflow in one command. It uploads the class folders,
  trains, generates (img2img and txt2img) and saves the images to `outputs/`.
  `--workspace ID` reuses a model trained earlier, so you can generate without
  retraining.
- `generate.pbs`: the PBS batch job that runs `start.sh` + `run_pipeline.py`.

**Dataset**
- `dataset/` holds the four classes (OK, NG-Misalign, NG-No_solder, NG-Single_electrode).
  Each folder has a `description.txt` taken from the report's Table 7. `dataset/` is in
  `.gitignore` because the images are TE Connectivity's data.

---

## 2. Running it on NUS Vanda, step by step

Commands marked **[your computer]** run in a terminal on your own laptop. Commands
marked **[Vanda]** run on Vanda. Replace `<NUSNET>` with your NUSNET ID (e.g.
`e1234567`). On Vanda, `$USER` already contains it.

### 2.0 Before you start

1. **Vanda access.** You need an NUS HPC account. If you're off campus, connect to the
   **NUS VPN** first.
2. **Hugging Face access to SD3.5** (it's a gated model):
   - Create an account at https://huggingface.co.
   - Open https://huggingface.co/stabilityai/stable-diffusion-3.5-medium and accept the license.
   - Create a **Read** token at https://huggingface.co/settings/tokens. It starts with `hf_`.
3. **Your dataset.** Inside the project, put **one folder per class** in `dataset/`; the
   folder name becomes the class name:
   ```
   dataset/
   ├── OK/                    *.jpg / *.png / *.webp  (+ optional description.txt)
   ├── NG-Misalign/
   ├── NG-No_solder/
   └── NG-Single_electrode/
   ```
   - Images can be in subfolders.
   - Each image must be at most 10 MB, with its short side at least 256 px.
   - An optional `description.txt` holds one line describing the class. It's added to
     the captions and prompts, so a good description helps.

### 2.1 Upload the project to Vanda

**[your computer]**
```bash
rsync -av --exclude '.DS_Store' --exclude '__pycache__' \
  "/path/to/SD3.5/" <NUSNET>@vanda.nus.edu.sg:~/chinn-ver/
```
- **Keep the trailing `/`** on the source path. On Vanda the project lives in
  `~/chinn-ver`.
- **If you upload again later**, add `--exclude 'backend/config.json'`. Setup writes
  Vanda-specific paths into that file, and you don't want to overwrite them.

### 2.2 Log in

**[your computer]**
```bash
ssh <NUSNET>@vanda.nus.edu.sg
```
You can also open a terminal from Vanda's web dashboard.

**[Vanda]** check the upload:
```bash
ls ~/chinn-ver ~/chinn-ver/dataset
```
You should see `backend`, `dataset`, `infrastructure` and `index.html`, and your class
folders.

### 2.3 Set up the environment (once)

**[Vanda]**
```bash
cd ~/chinn-ver
tmux new -s setup                      # keeps setup running if your connection drops
module load Miniconda3/24.7.1-0
export HF_TOKEN=hf_your_token_here     # paste it; don't retype it
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2
export PIP_NO_CACHE_DIR=1              # keeps pip from filling your home folder
MODEL_DIR=/scratch/$USER/sd3.5-medium bash infrastructure/hpc/setup.sh 2>&1 | tee setup.log
```
- **The `curl` line must print `200`.** `401` means the token is wrong or incomplete;
  export it again.
- **The model goes to scratch.** It's about 31 GB, and the home folder quota is small
  (about 40 GB in our experience), so `MODEL_DIR` puts it in `/scratch/$USER`.
- **Setup takes a while:** two Python environments plus a 33 GB download.
- **Don't press Ctrl-C**, even if the download bar sits at `33/37` for 10–20 minutes. The
  last files are the largest, at 5–10 GB each. If you get disconnected, log back in and
  run `tmux attach -t setup`.
- **It's finished** when it prints **`Setup done.`** It's safe to rerun: finished parts
  are skipped, and the download resumes where it stopped.

**[Vanda]** check the result:
```bash
cd ~/chinn-ver
ls ~/.fyp-redis/bin/redis-server backend/venv/bin/python backend/kohya_ss/sd-scripts/venv/bin/python
du -sh /scratch/$USER/sd3.5-medium        # about 31G
grep '"base"' backend/config.json         # must show /scratch/<NUSNET>/sd3.5-medium
env -i HOME=$HOME PATH=/usr/bin:/bin backend/venv/bin/python -c "import torch, diffusers; print('environment ok')"
```
After setup, you can delete the token on Hugging Face; nothing needs it any more.

### 2.4 (Optional) change settings

All settings are in `backend/config.json`. The ones you're most likely to change:

| Setting | Where | Default |
|---|---|---|
| Training length | `training.epochs` | 50 |
| Images generated per class (per mode) | `generation_defaults.<mode>.images_per_class` | 5 |
| How far img2img may move from the real photo | `generation_defaults.img2img.strength` | 0.35 |

To change a value safely, use Python rather than `sed`, because a missing `:` breaks
the file and makes the job crash. This example sets epochs to 50:

**[Vanda]**
```bash
cd ~/chinn-ver
backend/venv/bin/python -c "import json; p='backend/config.json'; c=json.load(open(p)); c['training']['epochs']=50; json.dump(c, open(p,'w'), indent=4)"
backend/venv/bin/python -c "import json; json.load(open('backend/config.json')); print('config ok')"
```
The second command must print `config ok`.

**Don't edit `config.json` while a job is waiting or running.** The job reads it when it
starts, not when you submit it.

### 2.5 Submit the job

**[Vanda]**
```bash
cd ~/chinn-ver
qsub -P personal-$USER infrastructure/hpc/generate.pbs
qstat -u $USER
```
- **Don't add `-q batch_gpu`.** Vanda refuses it ("Access to queue is denied"). Without
  `-q`, Vanda routes the job to a GPU queue by itself.
- **If your supervisor gave you a project code**, use it in place of `personal-$USER`.
- **Job status** in `qstat`: **Q** means waiting, **R** means running, and when the job is
  gone from the list, it has finished.
- **You can now log out and close your laptop.** The job runs without you.

**How long it takes** for 109 images at 512 px on an A40: about 45 minutes of training
and a few minutes of generation, so roughly 1 hour once it starts.

### 2.6 Follow progress

**[Vanda]**
```bash
tail -f ~/chinn-ver/storage/runtime/worker.log                    # live training/generation (Ctrl-C only stops watching)
tail -c 400 ~/chinn-ver/storage/runtime/worker.log | tr '\r' '\n' | grep -a steps | tail -1   # training steps + time left
```
Some messages in the job log `~/chinn-ver/sd35-generate.o<job id>` only appear when the
job ends. Note the **`o`** before the job number.

### 2.7 Get the results

**[Vanda]** once the job has disappeared from `qstat`:
```bash
tail -20 ~/chinn-ver/sd35-generate.o<job id>    # must end with "Done: …/outputs/<date-time>"
ls ~/chinn-ver/outputs/*/*/
```

**[your computer]**
```bash
rsync -av <NUSNET>@vanda.nus.edu.sg:~/chinn-ver/outputs/ "/path/to/SD3.5/outputs/"
```
Each run creates its own `outputs/<date-time>/` folder:
```
outputs/<date-time>/
├── img2img/    one folder per class + manifest.json (prompt, seed, source photo per image)
├── txt2img/    one folder per class + manifest.json
└── training.json   (the training run and the settings used)
```
The trained LoRA stays on Vanda in
`storage/workspaces/<workspace id>/training/<run id>/checkpoints/sd35m_lora.safetensors`.

### 2.8 Generate more images without retraining

The job log's first lines print `Workspace <id>`. To generate again with that trained
model:

**[Vanda]**, as a batch job (both modes):
```bash
qsub -P personal-$USER -v WORKSPACE=<workspace id> infrastructure/hpc/generate.pbs
```

**[Vanda]**, or straight away inside an interactive GPU session, with more options:
```bash
cd ~/chinn-ver
bash infrastructure/hpc/start.sh backend/venv/bin/python infrastructure/hpc/run_pipeline.py \
  --workspace <workspace id> --modes img2img --images-per-class 20
```

### 2.9 Tidy up

- **Interactive sessions** (e.g. a terminal opened from the dashboard) keep a GPU until
  they time out, and Vanda charges per GPU-hour. End them with `qdel <its job id>`.
- **Scratch may be purged periodically.** If `/scratch/$USER/sd3.5-medium` disappears,
  rerun the setup command from 2.3.

### Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `qsub: Access to queue is denied` | `-q batch_gpu` was given | Submit without `-q` (2.5) |
| `No space left on device`, or setup stops silently | Home quota full | `rm -rf ~/.cache/pip`; keep the model on scratch with `MODEL_DIR=…` (2.3) |
| `401 … GatedRepoError` / `Invalid user token` | License not accepted, or token mistyped | Accept the license; re-export the token and check the `curl` line prints `200` |
| `redis-server not found` in the job log | Setup didn't finish | Rerun setup until `Setup done.` |
| `JSONDecodeError: Expecting ':' delimiter` | `config.json` was broken by an edit | Fix the line, check with the `config ok` command (2.4), resubmit |
| `requires the protobuf library` | Old environment | `backend/venv/bin/python -m pip install sentencepiece protobuf` |
| `tail: cannot open … sd35-generate.o…` | Job still running, or the name/path was mistyped | `qstat -u $USER`; use the full path, with the `o` |
| `You set add_prefix_space …` | Just a notice | Ignore |
