from uuid import UUID
from pydantic import BaseModel, ConfigDict
from app.modules.generation.config_models import GenerationConfig

class GenerationRunCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    training_run_id: UUID
    checkpoint: str
    config: GenerationConfig

class GenerationRunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: str
