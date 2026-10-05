#!/usr/bin/env python

"""Train a LoRA on a SD 1.5 model using kohya_ss / sd-scripts"""

import json
import subprocess
import sys
from pathlib import Path

from config import load_config

cfg = load_config()


def find_train_script():
    base = cfg.paths.kohya_dir
    for probe in (base, base / "sd-scripts"):
        if (probe / "train_network.py").is_file():
            return probe.resolve()
    raise RuntimeError(
        f"train_network.py not found in {base} or {base / 'sd-scripts'}.\n"
    )


def load_dataset():
    meta_path = cfg.paths.dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"{meta_path} not found. Prepare dataset first")
    return json.loads(meta_path.read_text(encoding="utf-8"))["classes"]


def write_dataset_toml(path, classes):
    t = cfg.training
    bucketing = t.bucketing
    lines = [
        "[general]",
        'caption_extension = ".txt"',
        f"shuffle_caption = {str(t.shuffle_caption).lower()}",
        f"keep_tokens = {t.keep_tokens}",
        "",
        "[[datasets]]",
        f"resolution = {t.resolution}",
        f"batch_size = {t.batch_size}",
        f"enable_bucket = {str(bucketing.enabled).lower()}",
        "bucket_no_upscale = true",
        f"min_bucket_reso = {bucketing.min_size}",
        f"max_bucket_reso = {bucketing.max_size}",
        f"bucket_reso_steps = {bucketing.step}",
        "",
    ]
    for info in classes.values():
        image_dir = (cfg.paths.dataset_dir / info["image_dir"]).resolve()
        lines += [
            "  [[datasets.subsets]]",
            f'  image_dir = "{image_dir.as_posix()}"',
            f'  num_repeats = {info["num_repeats"]}',
            f"  flip_aug = {str(t.flip_aug).lower()}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prodigy_args(p):
    return [
        f"weight_decay={p.weight_decay}",
        f"decouple={p.decouple}",
        f"d_coef={p.d_coef}",
        f"use_bias_correction={p.use_bias_correction}",
        f"safeguard_warmup={p.safeguard_warmup}",
        f"betas={','.join(str(b) for b in p.betas)}",
    ]


def build_command(script_dir, dataset_toml, warmup_steps):
    t = cfg.training
    opt, mem = t.optimizer, t.memory
    out_dir = cfg.paths.output_dir.resolve()
    is_prodigy = opt.name.lower() == "prodigy"

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes=1", "--num_machines=1", "--num_cpu_threads_per_process=2",
        f"--mixed_precision={mem.mixed_precision}", "--dynamo_backend=no",
        str(script_dir / "train_network.py"),

        "--pretrained_model_name_or_path", str(cfg.model.base),
        f"--dataset_config={dataset_toml}",
        f"--output_dir={out_dir}",
        f"--output_name={t.output_name}",
        "--save_model_as=safetensors",

        "--network_module", "networks.lora",
        "--network_dim", str(cfg.lora.network_dim),
        "--network_alpha", str(cfg.lora.network_alpha),

        # batch size / flip_aug come from dataset.toml
        "--max_train_epochs", str(t.epochs),
        "--gradient_accumulation_steps", str(t.grad_accum),
        "--save_every_n_epochs", str(t.save_every_n_epochs),
        "--seed", str(t.seed),
        "--clip_skip", str(cfg.model.clip_skip),

        # Learning rates & schedules
        "--lr_scheduler", opt.lr_scheduler,
        "--lr_warmup_steps", str(int(warmup_steps)),
        "--optimizer_type", opt.name,
        "--max_grad_norm", str(opt.max_grad_norm),

        # Precision details
        "--mixed_precision", mem.mixed_precision,
        "--save_precision", mem.mixed_precision,
    ]

    if cfg.model.vae:
        cmd += ["--vae", str(cfg.model.vae)]

    if t.max_train_steps:
        cmd += ["--max_train_steps", str(t.max_train_steps)]

    if is_prodigy:
        cmd += ["--learning_rate", "1.0"]
        cmd += ["--optimizer_args"] + prodigy_args(opt.prodigy)
    else:
        cmd += ["--unet_lr", str(opt.unet_lr)]
        if cfg.lora.train_text_encoder:
            cmd += ["--text_encoder_lr", str(opt.text_encoder_lr)]
        else:
            cmd += ["--network_train_unet_only"]

    if opt.lr_scheduler == "cosine_with_restarts":
        cmd += ["--lr_scheduler_num_cycles", str(opt.lr_cycles)]

    if opt.min_snr_gamma:
        cmd += ["--min_snr_gamma", str(opt.min_snr_gamma)]

    if opt.noise_offset:
        cmd += ["--noise_offset", str(opt.noise_offset)]

    if mem.attention == "xformers":
        cmd += ["--xformers"]
    elif mem.attention == "sdpa":
        cmd += ["--sdpa"]
    elif mem.attention == "mem_eff":
        cmd += ["--mem_eff_attn"]

    if mem.gradient_checkpointing:
        cmd += ["--gradient_checkpointing"]

    if mem.cache_latents:
        cmd += ["--cache_latents"]
        if mem.cache_to_disk:
            cmd += ["--cache_latents_to_disk"]

    if mem.full_fp16:
        cmd += ["--full_fp16"]

    if cfg.lora.train_conv_layers:
        cmd += [
            "--network_args",
            f"conv_dim={cfg.lora.conv_dim}",
            f"conv_alpha={cfg.lora.conv_alpha}",
        ]

    return cmd


def main():
    t = cfg.training
    opt = t.optimizer

    script_dir = find_train_script()
    classes = load_dataset()

    dataset_dir = Path(cfg.paths.dataset_dir)
    output_dir = Path(cfg.paths.output_dir)
    config_dir = dataset_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_toml = (config_dir / "dataset.toml").resolve()
    write_dataset_toml(dataset_toml, classes)

    # Requires each class in meta.json to have a "count" field written
    # during dataset prep.
    images_per_epoch = 0
    for name, info in classes.items():
        if "count" not in info:
            raise RuntimeError(f"Class '{name}' in meta.json is missing 'count'. Re-run dataset prep.")
        num_repeats = int(info.get("num_repeats", 1))
        images_per_epoch += int(info["count"]) * num_repeats

    if images_per_epoch == 0:
        raise RuntimeError(f"No images found across classes in {dataset_dir}")

    steps_per_epoch = max(1, images_per_epoch // (t.batch_size * t.grad_accum))
    total_steps = t.max_train_steps if t.max_train_steps else steps_per_epoch * t.epochs

    # Prodigy adapts its own LR from step zero; a warmup ramp works against
    # that adaptation, so skip it for this optimizer.
    is_prodigy = opt.name.lower() == "prodigy"
    warmup_steps = 0 if is_prodigy else int(total_steps * opt.lr_warmup_ratio)

    cmd = build_command(script_dir, dataset_toml, warmup_steps)

    lr_summary = (
        "prodigy (auto)" if is_prodigy
        else f"unet={opt.unet_lr}, te={opt.text_encoder_lr}"
    )

    print("=" * 50)
    print(f"Model:       {cfg.model.base}")
    print(f"Resolution:  {t.resolution}")
    print(f"Network:     dim={cfg.lora.network_dim}, alpha={cfg.lora.network_alpha}")
    print(f"Optimizer:   {opt.name}  ({lr_summary})")
    print(f"Scheduler:   {opt.lr_scheduler}  (warmup={warmup_steps} steps)")
    print(f"Batch:       {t.batch_size} x {t.grad_accum} accum = {t.batch_size * t.grad_accum} effective")
    print(f"Images:      {images_per_epoch}/epoch  ->  {steps_per_epoch} steps/epoch, {total_steps} total")
    print(f"Output:      {output_dir.resolve() / (t.output_name or 'my_lora')}.safetensors")
    print("=" * 50)

    # Run from the sd-scripts dir so `networks.lora` imports resolve.
    subprocess.run(cmd, check=True, cwd=script_dir)


if __name__ == "__main__":
    main()