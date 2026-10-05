from __future__ import annotations
import json
import os
import re
import sys
import typing
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_PATH = PROJECT_ROOT / "config.docs.json"
DRIVE_LETTER = re.compile(r"^[A-Za-z]:[\\/]")

class ConfigError(SystemExit):
    """plain test"""

def _resolve_path(path: str) -> Path:
    raw = os.path.expandvars(path)
    if os.name != "nt" and DRIVE_LETTER.match(raw):
        raise ValueError(f"{raw!r} is a Windows path, but this is not Windows")
    path = Path(raw).expanduser()
    return (path if path.anchor else PROJECT_ROOT / path).resolve()

ProjectPath = Annotated[Path, AfterValidator(_resolve_path)]

class Base(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", validate_default=True)

# --- Schema for config file --- #

class Paths(Base):
    dataset_dir: ProjectPath
    output_dir: ProjectPath
    generated_dir: ProjectPath
    kohya_dir: ProjectPath

class Model(Base):
    base: str
    vae: str
    clip_skip: int

class ClassSpec(Base):
    name: str
    path: ProjectPath
    description: str = ""
    repeats: Optional[int] = None

    @property
    def folder(self) -> str:
        return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-]+", "_", self.name.strip().lower()))

class Dataset(Base):
    classes: list[ClassSpec]
    trigger: str
    domain: str
    max_size: int
    min_size: int
    max_repeats: int

    @model_validator(mode="after")
    def _check_class_names(self):
        folders = [c.folder for c in self.classes]
        if not folders:
            raise ValueError("No classes defined in config")
        if len(folders) != len(set(folders)):
            raise ValueError(f"Duplicate class folder names: {folders}")
        return self

class Lora(Base):
    network_dim: int
    network_alpha: int
    train_text_encoder: bool
    train_conv_layers: bool
    conv_dim: int
    conv_alpha: int

class Prodigy(Base):
    weight_decay: float
    decouple: bool
    d_coef: float
    use_bias_correction: bool
    safeguard_warmup: bool
    betas: list[float]
    def as_optimizer_args(self):
        return [
            f"{name}={','.join(str(v) for v in value)}"
            if isinstance(value, list) else f"{name}={value}"
            for name, value in self.model_dump().items()
        ]

class Optimizer(Base):
    name: Literal["AdamW", "AdamW8bit", "Lion8bit", "Prodigy"]
    unet_lr: float
    text_encoder_lr: float
    lr_scheduler: str
    lr_warmup_ratio: float
    lr_cycles: int
    min_snr_gamma: float
    noise_offset: float
    max_grad_norm: float
    prodigy: Prodigy

    @model_validator(mode="after")
    def _check_warmup(self):
        if not 0.0 <= self.lr_warmup_ratio < 1.0:
            raise ValueError("lr_warmup_ratio must be in [0,1)")
        return self

class Bucketing(Base):
    enabled: bool
    min_size: int
    max_size: int
    step: int
    @model_validator(mode="after")
    def _check_sizes(self):
        if self.min_size > self.max_size:
            raise ValueError("min_size is larger than max_size")
        if self.min_size % self.step or self.max_size % self.step:
            raise ValueError(f"min_size and max_size must be multiple of step")
        return self

class Memory(Base):
    mixed_precision: Literal["fp16", "bf16", "no"]
    attention: Literal["sdpa", "xformers", "mem_eff"]
    gradient_checkpointing: bool
    cache_latents: bool
    cache_to_disk: bool
    full_fp16: bool

class Training(Base):
    output_name: str
    epochs: int
    max_train_steps: int
    batch_size: int
    grad_accum: int
    resolution: int
    save_every_n_epochs: int
    seed: int
    flip_aug: bool
    shuffle_caption: bool
    keep_tokens: int
    optimizer: Optimizer
    bucketing: Bucketing
    memory: Memory

    @model_validator(mode="after")
    def _check_resolution(self):
        if self.resolution % 64:
            raise ValueError("resolution must be a multiple of 64")
        return self

class Generate(Base):
    checkpoint: str
    images_per_class: int
    only_classes: list[str]
    lora_scale: float
    width: int
    height: int
    steps: int
    cfg: float
    sampler: Literal["dpmpp_2m_karras", "euler_a"]
    negative_prompt: str
    seed: int
    batch_size: int
    low_vram: bool
    prompt_variations: list[str]

class Config(Base):
    model: Model
    dataset: Dataset
    paths: Paths
    lora: Lora
    training: Training
    generate: Generate

# --- Loader --- #

def load_config(path=None) -> Config:
    config_path = Path(path or PROJECT_ROOT / "config.json")
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        return Config.model_validate_json(config_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        problems = "\n".join(
            f"  config{''.join(f'.{p}' for p in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"{config_path.name} is invalid\n{problems}")