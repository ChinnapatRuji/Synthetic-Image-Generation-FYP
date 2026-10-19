from uuid import UUID

from fastapi import APIRouter, File, UploadFile, Response, status
from fastapi.responses import FileResponse

from app.modules.datasets.schemas import (
    ClassCreate,
    ClassUpdate,
    ClassResponse,
    DatasetResponse,
)
from app.modules.datasets.service import (
    get_dataset,
    create_class,
    update_class,
    delete_class,
    upload_images,
    list_images,
    get_image_path,
    delete_image,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/dataset",
    tags=["Dataset"],
)


@router.get("", response_model=DatasetResponse)
def get_dataset_endpoint(workspace_id: UUID):
    return get_dataset(workspace_id)


@router.post(
    "/classes",
    response_model=ClassResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_class_endpoint(
    workspace_id: UUID,
    data: ClassCreate,
):
    return create_class(workspace_id, data)


@router.patch("/classes/{class_id}", response_model=ClassResponse)
def update_class_endpoint(
    workspace_id: UUID,
    class_id: UUID,
    data: ClassUpdate,
):
    return update_class(workspace_id, class_id, data)


@router.delete("/classes/{class_id}", status_code=204)
def delete_class_endpoint(workspace_id: UUID, class_id: UUID):
    delete_class(workspace_id, class_id)
    return Response(status_code=204)


@router.post(
    "/classes/{class_id}/images",
    status_code=status.HTTP_201_CREATED,
)
def upload_images_endpoint(
    workspace_id: UUID,
    class_id: UUID,
    files: list[UploadFile] = File(...),
):
    return upload_images(workspace_id, class_id, files)


@router.get("/classes/{class_id}/images")
def list_images_endpoint(workspace_id: UUID, class_id: UUID):
    return list_images(workspace_id, class_id)


@router.get("/classes/{class_id}/images/{image_id}")
def get_image_endpoint(
    workspace_id: UUID,
    class_id: UUID,
    image_id: UUID,
):
    path = get_image_path(workspace_id, class_id, image_id)
    return FileResponse(
        path=path,
        media_type={
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".webp": "image/webp",
        }[path.suffix.lower()],
    )


@router.delete("/classes/{class_id}/images/{image_id}", status_code=204)
def delete_image_endpoint(
    workspace_id: UUID,
    class_id: UUID,
    image_id: UUID,
):
    delete_image(workspace_id, class_id, image_id)
    return Response(status_code=204)
