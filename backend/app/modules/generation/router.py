from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import FileResponse

from app.modules.generation.schemas import (
    GenerationRunCreate,
    GenerationRunResponse,
)
from app.modules.generation.service import (
    start_generation,
    list_generation_runs,
    get_generation_run,
    list_generated_images,
    get_generated_image,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/generation",
    tags=["Generation"],
)


@router.post(
    "/runs",
    response_model=GenerationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_generation_endpoint(
    workspace_id: UUID,
    data: GenerationRunCreate,
):
    return start_generation(workspace_id, data)


@router.get("/runs")
def list_runs_endpoint(workspace_id: UUID):
    return list_generation_runs(workspace_id)


@router.get("/runs/{run_id}")
def get_run_endpoint(workspace_id: UUID, run_id: UUID):
    return get_generation_run(workspace_id, run_id)


@router.get("/runs/{run_id}/images")
def list_images_endpoint(workspace_id: UUID, run_id: UUID):
    return list_generated_images(workspace_id, run_id)


@router.get("/runs/{run_id}/images/{class_id}/{filename}")
def get_image_endpoint(
    workspace_id: UUID,
    run_id: UUID,
    class_id: UUID,
    filename: str,
):
    path = get_generated_image(
        workspace_id,
        run_id,
        class_id,
        filename,
    )
    return FileResponse(path, media_type="image/png")
