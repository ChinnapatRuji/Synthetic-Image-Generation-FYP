"""Strict, mode-specific generation request models."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Execution(Strict):
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    precision: Literal['auto', 'fp16', 'fp32', 'bf16'] = 'auto'
    memory_strategy: Literal['auto', 'full_gpu', 'cpu_offload', 'sequential_offload'] = 'auto'

class Source(Strict):
    strategy: Literal['same_class'] = 'same_class'
    selection: Literal['random', 'sequential'] = 'sequential'

class Canny(Strict):
    type: Literal['canny'] = 'canny'
    model: str = 'lllyasviel/control_v11p_sd15_canny'
    source: Source = Field(default_factory=Source)
    scale: float = Field(0.7, ge=0, le=2)
    low_threshold: int = Field(100, ge=0, le=255)
    high_threshold: int = Field(200, ge=0, le=255)

    @model_validator(mode='after')
    def check_thresholds(self):
        if self.low_threshold >= self.high_threshold:
            raise ValueError('Canny low_threshold must be less than high_threshold')
        return self

class Common(Strict):
    images_per_class: int = Field(5, ge=1)
    class_ids: list[str] = Field(default_factory=list)
    lora_scale: float = Field(0.85, ge=0, le=2)
    width: int = Field(512, ge=64, multiple_of=8)
    height: int = Field(512, ge=64, multiple_of=8)
    steps: int = Field(35, ge=1)
    cfg: float = Field(6.5, ge=0)
    sampler: Literal['euler_a', 'dpmpp_2m_karras'] = 'dpmpp_2m_karras'
    negative_prompt: str = ''
    seed: int = 1000
    batch_size: int = Field(1, ge=1)
    prompt_variations: list[str] = Field(default_factory=list)
    execution: Execution = Field(default_factory=Execution)
    conditioning: Canny | None = None

class Txt2Img(Common):
    mode: Literal['txt2img'] = 'txt2img'

class Img2Img(Common):
    mode: Literal['img2img'] = 'img2img'
    strength: float = Field(0.35, gt=0, le=1)
    source: Source = Field(default_factory=Source)

GenerationConfig = Annotated[Txt2Img | Img2Img, Field(discriminator='mode')]
