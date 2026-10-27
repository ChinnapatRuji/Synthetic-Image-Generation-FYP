#!/usr/bin/env python

"""Train a LoRA on an SD 1.5 or SD3 model using Kohya SS / sd-scripts."""

import json
import subprocess
import psutil
import os
import signal
import time
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from config import load_config


TRAIN_SCRIPTS = {
    "sd15": "train_network.py",
    "sd3": "sd3_train_network.py",
}

NETWORK_MODULES = {
    "sd15": "networks.lora",
    "sd3": "networks.lora_sd3",
}

# Kohya loads SD3 from single files, not the Diffusers layout.
SD3_TEXT_ENCODERS = {
    "clip_l": "text_encoders/clip_l.safetensors",
    "clip_g": "text_encoders/clip_g.safetensors",
    "t5xxl": "text_encoders/t5xxl_fp16.safetensors",
}


def find_train_script(cfg):
    base = cfg.paths.kohya_dir
    script = TRAIN_SCRIPTS[cfg.model.architecture]

    for probe in (base, base / "sd-scripts"):
        if (probe / script).is_file():
            return probe.resolve()

    raise RuntimeError(
        f"{script} not found in {base} or {base / 'sd-scripts'}.\n"
    )


def kohya_python(script_dir):
    """
    Kohya pins newer libraries than the backend, so run it from
    its own venv when there is one.
    """

    for candidate in (
        script_dir / "venv" / "bin" / "python",
        script_dir / "venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return str(candidate)

    return sys.executable


def sd3_model_files(base):
    """
    Locate the SD3 checkpoint and text encoders in a Hugging Face
    repo (downloaded on first use) or a local copy of one.
    """

    local = Path(base)

    if local.is_dir():
        names = [
            path.relative_to(local).as_posix()
            for path in local.rglob("*.safetensors")
        ]

        def fetch(name):
            return local / name

    else:
        names = list_repo_files(base)

        def fetch(name):
            return Path(hf_hub_download(base, name))

    checkpoints = [
        name for name in names
        if "/" not in name and name.endswith(".safetensors")
    ]

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one .safetensors checkpoint at the root of "
            f"{base}, found {checkpoints}"
        )

    missing = [
        name for name in SD3_TEXT_ENCODERS.values()
        if name not in names
    ]

    if missing:
        raise RuntimeError(
            f"{base} is missing text encoder files: {missing}"
        )

    return {
        "checkpoint": fetch(checkpoints[0]),
        **{
            key: fetch(name)
            for key, name in SD3_TEXT_ENCODERS.items()
        },
    }


def model_arguments(cfg):
    """Kohya arguments that load the base model."""

    if cfg.model.architecture == "sd3":
        files = sd3_model_files(cfg.model.base)

        return [
            "--pretrained_model_name_or_path",
            str(files["checkpoint"]),
            "--clip_l",
            str(files["clip_l"]),
            "--clip_g",
            str(files["clip_g"]),
            "--t5xxl",
            str(files["t5xxl"]),
            # Saved tokenizers let training run without internet access.
            "--tokenizer_cache_dir",
            str(cfg.paths.kohya_dir / "tokenizer_cache"),
        ]

    args = [
        "--pretrained_model_name_or_path",
        str(cfg.model.base),
        "--clip_skip",
        str(cfg.model.clip_skip),
    ]

    if cfg.model.vae:
        args += [
            "--vae",
            str(cfg.model.vae),
        ]

    return args


def load_dataset(cfg, dataset_dir):
    meta_path = Path(dataset_dir) / "meta.json"

    if not meta_path.is_file():
        raise RuntimeError(
            f"{meta_path} not found. Prepare dataset first."
        )

    return json.loads(
        meta_path.read_text(encoding="utf-8")
    )["classes"]


def write_dataset_toml(path, classes, cfg, dataset_dir):
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
        image_dir = (
            Path(dataset_dir) / info["image_dir"]
        ).resolve()

        lines += [
            "  [[datasets.subsets]]",
            f'  image_dir = "{image_dir.as_posix()}"',
            f'  num_repeats = {info["num_repeats"]}',
            f"  flip_aug = {str(t.flip_aug).lower()}",
            "",
        ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def prodigy_args(p):
    return [
        f"weight_decay={p.weight_decay}",
        f"decouple={p.decouple}",
        f"d_coef={p.d_coef}",
        f"use_bias_correction={p.use_bias_correction}",
        f"safeguard_warmup={p.safeguard_warmup}",
        f"betas={','.join(str(b) for b in p.betas)}",
    ]


def build_command(
    script_dir,
    dataset_toml,
    warmup_steps,
    cfg,
    model_args,
    output_dir=None,
):
    t = cfg.training
    opt = t.optimizer
    mem = t.memory

    out_dir = Path(
        output_dir if output_dir is not None
        else cfg.paths.output_dir
    ).resolve()

    is_prodigy = opt.name.lower() == "prodigy"

    cmd = [
        kohya_python(script_dir),
        "-m",
        "accelerate.commands.launch",

        "--num_processes=1",
        "--num_machines=1",
        "--num_cpu_threads_per_process=2",
        f"--mixed_precision={mem.mixed_precision}",
        "--dynamo_backend=no",

        str(script_dir / TRAIN_SCRIPTS[cfg.model.architecture]),

        *model_args,

        f"--dataset_config={dataset_toml}",
        f"--output_dir={out_dir}",
        f"--output_name={t.output_name}",

        "--save_model_as=safetensors",

        "--network_module",
        NETWORK_MODULES[cfg.model.architecture],

        "--network_dim",
        str(cfg.lora.network_dim),

        "--network_alpha",
        str(cfg.lora.network_alpha),

        "--max_train_epochs",
        str(t.epochs),

        "--gradient_accumulation_steps",
        str(t.grad_accum),

        "--save_every_n_epochs",
        str(t.save_every_n_epochs),

        "--seed",
        str(t.seed),

        "--lr_scheduler",
        opt.lr_scheduler,

        "--lr_warmup_steps",
        str(int(warmup_steps)),

        "--optimizer_type",
        opt.name,

        "--max_grad_norm",
        str(opt.max_grad_norm),

        "--mixed_precision",
        mem.mixed_precision,

        "--save_precision",
        mem.mixed_precision,
    ]

    if t.max_train_steps:
        cmd += [
            "--max_train_steps",
            str(t.max_train_steps),
        ]

    if is_prodigy:
        cmd += [
            "--learning_rate",
            "1.0",
        ]

        cmd += [
            "--optimizer_args",
            *prodigy_args(opt.prodigy),
        ]

    else:
        cmd += [
            "--unet_lr",
            str(opt.unet_lr),
        ]

        if cfg.lora.train_text_encoder:
            cmd += [
                "--text_encoder_lr",
                str(opt.text_encoder_lr),
            ]
        else:
            cmd += [
                "--network_train_unet_only",
            ]

    if opt.lr_scheduler == "cosine_with_restarts":
        cmd += [
            "--lr_scheduler_num_cycles",
            str(opt.lr_cycles),
        ]

    if opt.min_snr_gamma:
        cmd += [
            "--min_snr_gamma",
            str(opt.min_snr_gamma),
        ]

    if opt.noise_offset:
        cmd += [
            "--noise_offset",
            str(opt.noise_offset),
        ]

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

    # Encode captions once so SD3's large T5-XXL encoder can be freed.
    # Kohya can't cache them while captions are shuffled.
    if cfg.model.architecture == "sd3" and not t.shuffle_caption:
        cmd += ["--cache_text_encoder_outputs"]

        if mem.cache_to_disk:
            cmd += ["--cache_text_encoder_outputs_to_disk"]

    if mem.full_fp16:
        cmd += ["--full_fp16"]

    if cfg.lora.train_conv_layers:
        cmd += [
            "--network_args",
            f"conv_dim={cfg.lora.conv_dim}",
            f"conv_alpha={cfg.lora.conv_alpha}",
        ]

    return cmd


def train(cfg, dataset_dir=None, output_dir=None, config_dir=None, cancel_check=None, process_callback=None, cleanup_callback=None):
    """
    Execute LoRA training using a supplied configuration.

    Args:
        cfg:
            Validated training configuration.

        output_dir:
            Optional checkpoint output directory.
            Defaults to cfg.paths.output_dir.

        config_dir:
            Optional directory for the generated dataset.toml.
            Defaults to cfg.paths.dataset_dir / "config".

    Returns:
        Path to the checkpoint output directory.
    """

    t = cfg.training
    opt = t.optimizer

    # --------------------------------------------------
    # Locate Kohya SS and dataset
    # --------------------------------------------------

    script_dir = find_train_script(cfg)
    dataset_dir = Path(dataset_dir if dataset_dir is not None else cfg.paths.dataset_dir).resolve()
    classes = load_dataset(cfg, dataset_dir)


    output_dir = Path(
        output_dir if output_dir is not None
        else cfg.paths.output_dir
    ).resolve()

    config_dir = Path(
        config_dir if config_dir is not None
        else dataset_dir / "config"
    ).resolve()

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Generate dataset TOML
    # --------------------------------------------------

    dataset_toml = config_dir / "dataset.toml"

    write_dataset_toml(
        dataset_toml,
        classes,
        cfg,
        dataset_dir,
    )

    # --------------------------------------------------
    # Calculate training steps
    # --------------------------------------------------

    images_per_epoch = 0

    for name, info in classes.items():

        if "count" not in info:
            raise RuntimeError(
                f"Class '{name}' in meta.json is missing "
                "'count'. Re-run dataset preparation."
            )

        num_repeats = int(
            info.get("num_repeats", 1)
        )

        images_per_epoch += (
            int(info["count"]) * num_repeats
        )

    if images_per_epoch == 0:
        raise RuntimeError(
            f"No images found across classes in {dataset_dir}"
        )

    steps_per_epoch = max(
        1,
        images_per_epoch // (
            t.batch_size * t.grad_accum
        ),
    )

    total_steps = (
        t.max_train_steps
        if t.max_train_steps
        else steps_per_epoch * t.epochs
    )

    # --------------------------------------------------
    # Learning rate warmup
    # --------------------------------------------------

    is_prodigy = (
        opt.name.lower() == "prodigy"
    )

    warmup_steps = (
        0
        if is_prodigy
        else int(
            total_steps * opt.lr_warmup_ratio
        )
    )

    # --------------------------------------------------
    # Build Kohya SS command
    # --------------------------------------------------

    cmd = build_command(
        script_dir=script_dir,
        dataset_toml=dataset_toml,
        warmup_steps=warmup_steps,
        cfg=cfg,
        model_args=model_arguments(cfg),
        output_dir=output_dir,
    )

    lr_summary = (
        "prodigy (auto)"
        if is_prodigy
        else (
            f"unet={opt.unet_lr}, "
            f"te={opt.text_encoder_lr}"
        )
    )

    # --------------------------------------------------
    # Training summary
    # --------------------------------------------------

    print("=" * 50)

    print(f"Model:       {cfg.model.base} ({cfg.model.architecture})")
    print(f"Python:      {cmd[0]}")
    print(f"Resolution:  {t.resolution}")

    print(
        f"Network:     "
        f"dim={cfg.lora.network_dim}, "
        f"alpha={cfg.lora.network_alpha}"
    )

    print(
        f"Optimizer:   {opt.name} "
        f"({lr_summary})"
    )

    print(
        f"Scheduler:   {opt.lr_scheduler} "
        f"(warmup={warmup_steps} steps)"
    )

    print(
        f"Batch:       {t.batch_size} x "
        f"{t.grad_accum} accum = "
        f"{t.batch_size * t.grad_accum} effective"
    )

    print(
        f"Images:      {images_per_epoch}/epoch "
        f"-> {steps_per_epoch} steps/epoch, "
        f"{total_steps} total"
    )

    print(f"Output:      {output_dir}")

    print("=" * 50)

    # --------------------------------------------------
    # Execute Kohya SS
    # --------------------------------------------------

    process = subprocess.Popen(
        cmd,
        cwd=script_dir,
        start_new_session=True,
    )

    def group_members():
        """Find live processes belonging to this training process group."""
        members = []

        for candidate in psutil.process_iter(["pid", "status"]):
            try:
                if (
                    candidate.pid != process.pid
                    and os.getpgid(candidate.pid) == process.pid
                    and candidate.status() != psutil.STATUS_ZOMBIE
                ):
                    members.append(candidate)
            except (ProcessLookupError, psutil.Error):
                continue

        return members

    def terminate_training():
        """Terminate the launcher and remaining process-group members."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

        remaining = group_members()

        if process.poll() is None or remaining:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            for child in remaining:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            if process.poll() is None:
                process.wait(timeout=10)

            psutil.wait_procs(remaining, timeout=10)

        if process.poll() is None or group_members():
            raise RuntimeError(
                "Kohya process cleanup could not be confirmed"
            )

    try:
        if process_callback is not None:
            process_callback(process.pid)

        while process.poll() is None:
            if cancel_check is not None:
                cancel_check()

            time.sleep(1)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
            )

        if group_members():
            raise RuntimeError(
                "Kohya launcher exited but child processes remain"
            )

    except BaseException:
        terminate_training()
        raise

    finally:
        if process.poll() is None or group_members():
            terminate_training()

        # Only report successful cleanup after verifying
        # the launcher and its process group have stopped.
        if process.poll() is not None and not group_members():
            if cleanup_callback is not None:
                cleanup_callback()

    print("LoRA training completed successfully.")

    return output_dir


def main():
    """
    Standalone execution using config.json.
    """

    cfg = load_config()

    train(cfg)


if __name__ == "__main__":
    main()