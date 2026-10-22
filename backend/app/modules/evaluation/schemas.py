from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=224, ge=32, le=1024)
    patience: int = Field(default=20, ge=0, le=1000)  # stop early if val accuracy stops improving
    batch: int = Field(default=16, ge=1, le=256)
    seed: int = 42


class ClassifierCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: ClassifierConfig = Field(default_factory=ClassifierConfig)


class ClassifierResponse(BaseModel):
    classifier_id: UUID
    workspace_id: UUID
    status: str


class EvaluationRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation_run_id: UUID
    # Defaults to the most recent completed classifier. Pin it to compare runs fairly.
    classifier_id: UUID | None = None


class EvaluationRunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: str
