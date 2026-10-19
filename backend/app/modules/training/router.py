from uuid import UUID

from fastapi import APIRouter, status

from app.modules.training.schemas import (
    TrainingRunCreate,
    TrainingRunResponse,
)

from app.modules.training.service import (
    start_training,
    get_training_status,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/training",
    tags=["Training"],
)


@router.post(
    "/runs",
    response_model=TrainingRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_training_endpoint(
    workspace_id: UUID,
    data: TrainingRunCreate,
):
    return start_training(workspace_id, data)


@router.get("/runs/{run_id}")
def get_training_status_endpoint(
    workspace_id: UUID,
    run_id: UUID,
):
    return get_training_status(workspace_id, run_id)
