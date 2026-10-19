from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class TrainingRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    network_dim: int | None = Field(default=None, ge=1)
    network_alpha: int | None = Field(default=None, ge=1)


class TrainingRunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: str
