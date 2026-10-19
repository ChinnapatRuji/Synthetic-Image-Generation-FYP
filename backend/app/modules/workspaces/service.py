import json
import shutil
from uuid import UUID, uuid4

from fastapi import HTTPException

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.modules.workspaces.schemas import WorkspaceCreate, WorkspaceUpdate


def get_workspace_path(workspace_id: UUID):
    return settings.STORAGE_ROOT / StoragePaths.workspace(workspace_id)


def get_workspace(workspace_id: UUID):
    path = get_workspace_path(workspace_id) / "workspace.json"

    if not path.is_file():
        raise HTTPException(404, "Workspace not found")

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=4), encoding="utf-8")
    temp.replace(path)


def create_workspace(data: WorkspaceCreate):
    workspace_id = uuid4()
    path = get_workspace_path(workspace_id)

    path.mkdir(parents=True, exist_ok=False)

    metadata = {
        "id": str(workspace_id),
        "name": data.name,
    }

    write_json(path / "workspace.json", metadata)

    # Exactly one dataset per workspace, created automatically
    write_json(
        path / "dataset" / "dataset.json",
        {"classes": []},
    )

    (path / "dataset" / "classes").mkdir(parents=True, exist_ok=True)

    return metadata


def list_workspaces():
    root = settings.STORAGE_ROOT / "workspaces"

    if not root.exists():
        return []

    result = []

    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "workspace.json").is_file():
            result.append(
                json.loads(
                    (path / "workspace.json").read_text(encoding="utf-8")
                )
            )

    return result


def update_workspace(workspace_id: UUID, data: WorkspaceUpdate):
    metadata = get_workspace(workspace_id)
    metadata["name"] = data.name

    write_json(
        get_workspace_path(workspace_id) / "workspace.json",
        metadata,
    )

    return metadata


def delete_workspace(workspace_id: UUID):
    get_workspace(workspace_id)
    path = get_workspace_path(workspace_id)

    # Do not delete training history or registered models.
    for name in ("training", "models", "generations"):
        directory = path / name
        if directory.exists() and any(directory.iterdir()):
            raise HTTPException(
                409,
                f"Cannot delete workspace containing {name}. "
                "Remove associated artifacts first.",
            )

    shutil.rmtree(path)
