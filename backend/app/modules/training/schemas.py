from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from config import TrainingRequestConfig

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class TrainingRunCreate(Strict):
    class_ids: list[UUID] = Field(min_length=1)
    config: TrainingRequestConfig

    @model_validator(mode="after")
    def distinct(self):
        if len(self.class_ids) != len(set(self.class_ids)):
            raise ValueError("Duplicate class_ids")
        return self

class TrainingBatchCreate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    class_ids: list[UUID] = Field(default_factory=list)
    config: TrainingRequestConfig
    class_overrides: dict[UUID, dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selection(self):
        if len(self.class_ids) != len(set(self.class_ids)):
            raise ValueError("Duplicate class_ids")
        if self.class_ids and not set(self.class_overrides).issubset(set(self.class_ids)):
            raise ValueError("class_overrides must refer to selected class_ids")
        return self

class TrainingRunResponse(Strict):
    run_id: UUID
    workspace_id: UUID
    status: str
