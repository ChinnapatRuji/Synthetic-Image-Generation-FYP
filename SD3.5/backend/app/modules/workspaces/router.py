from uuid import UUID

from fastapi import APIRouter, Response, status

from app.modules.workspaces.schemas import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
)
from app.modules.workspaces.service import (
    create_workspace,
    get_workspace,
    list_workspaces,
    update_workspace,
    delete_workspace,
)


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_workspace_endpoint(data: WorkspaceCreate):
    return create_workspace(data)


@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces_endpoint():
    return list_workspaces()


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace_endpoint(workspace_id: UUID):
    return get_workspace(workspace_id)


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
def update_workspace_endpoint(
    workspace_id: UUID,
    data: WorkspaceUpdate,
):
    return update_workspace(workspace_id, data)


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace_endpoint(workspace_id: UUID):
    delete_workspace(workspace_id)
    return Response(status_code=204)
