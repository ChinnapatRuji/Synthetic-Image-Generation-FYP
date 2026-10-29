from uuid import UUID

from fastapi import APIRouter, Response, status
from fastapi.responses import FileResponse

from app.modules.training.schemas import (
    TrainingRunCreate,
    TrainingBatchCreate,
    TrainingRunResponse,
)

from app.modules.training.service import (
    start_training,
    start_batch, get_batch, list_batches,
    get_training_status,
    list_training_runs,
    get_run_config,
    list_checkpoints,
    get_checkpoint,
    cancel_training,
    delete_training,
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


@router.get("/runs")
def list_runs_endpoint(workspace_id: UUID):
    return list_training_runs(workspace_id)


@router.post("/batches", status_code=status.HTTP_202_ACCEPTED)
def create_batch_endpoint(workspace_id: UUID, data: TrainingBatchCreate):
    return start_batch(workspace_id, data)


@router.get("/batches")
def list_batches_endpoint(workspace_id: UUID):
    return list_batches(workspace_id)


@router.get("/batches/{batch_id}")
def get_batch_endpoint(workspace_id: UUID, batch_id: UUID):
    return get_batch(workspace_id, batch_id)


@router.get("/runs/{run_id}")
def get_run_endpoint(workspace_id: UUID, run_id: UUID):
    return get_training_status(workspace_id, run_id)


@router.get("/runs/{run_id}/config")
def get_config_endpoint(workspace_id: UUID, run_id: UUID):
    return get_run_config(workspace_id, run_id)


@router.get("/runs/{run_id}/checkpoints")
def list_checkpoints_endpoint(workspace_id: UUID, run_id: UUID):
    return list_checkpoints(workspace_id, run_id)


@router.get("/runs/{run_id}/checkpoints/{filename}")
def download_checkpoint_endpoint(
    workspace_id: UUID,
    run_id: UUID,
    filename: str,
):
    path = get_checkpoint(workspace_id, run_id, filename)
    return FileResponse(path, filename=path.name)


@router.post("/runs/{run_id}/cancel")
def cancel_run_endpoint(workspace_id: UUID, run_id: UUID):
    return cancel_training(workspace_id, run_id)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run_endpoint(workspace_id: UUID, run_id: UUID):
    delete_training(workspace_id, run_id)
    return Response(status_code=204)
