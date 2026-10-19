from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import Model, Lora, Training


class DatasetPreprocessing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: str
    domain: str
    min_size: int = Field(gt=0)
    max_size: int = Field(gt=0)
    max_repeats: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_sizes(self):
        if self.min_size > self.max_size:
            raise ValueError("min_size cannot exceed max_size")
        return self


class TrainingRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Model
    dataset: DatasetPreprocessing
    lora: Lora
    training: Training


class TrainingRunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: str
