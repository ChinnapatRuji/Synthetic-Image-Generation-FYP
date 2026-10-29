from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.modules.generation.config_models import GenerationConfig

class GenerationRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_run_id: UUID | None = None
    training_batch_id: UUID | None = None
    checkpoint: str = "final"
    checkpoint_overrides: dict[UUID, str] = Field(default_factory=dict)
    config: GenerationConfig

    @model_validator(mode="after")
    def exactly_one_source(self):
        if (self.training_run_id is None) == (self.training_batch_id is None):
            raise ValueError("Specify exactly one of training_run_id or training_batch_id")
        return self

class GenerationRunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: str
