from app.core.timestamps import utc_now, elapsed_seconds
import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException

from app.core.settings import settings
from app.modules.workspaces.service import get_workspace_path, get_workspace
from app.modules.generation.schemas import GenerationRunCreate
from app.workers.generation_tasks import generate_images_task


def generation_root(workspace_id: UUID):
    return get_workspace_path(workspace_id) / "generations"


def generation_run_dir(workspace_id: UUID, run_id: UUID):
    return generation_root(workspace_id) / str(run_id)


def read_run(workspace_id: UUID, run_id: UUID):
    get_workspace(workspace_id)

    path = generation_run_dir(workspace_id, run_id) / "run.json"

    if not path.is_file():
        raise HTTPException(404, "Generation run not found")

    return json.loads(path.read_text(encoding="utf-8"))


def start_generation(workspace_id: UUID, data: GenerationRunCreate):
    get_workspace(workspace_id)
    from app.modules.generation.checkpoints import resolve_generation_source
    resolved = resolve_generation_source(workspace_id, data)
    run_id = uuid4()
    run_dir = generation_run_dir(workspace_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    classes = [item["class"] for item in resolved.values()]
    (run_dir / "classes.json").write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    snapshot = {"training_run_id": str(data.training_run_id) if data.training_run_id else None,
                "training_batch_id": str(data.training_batch_id) if data.training_batch_id else None,
                "resolved": resolved,
                "generation": data.config.model_dump(mode="json")}
    (run_dir / "config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    metadata = {"run_id": str(run_id), "workspace_id": str(workspace_id),
                "training_run_id": str(data.training_run_id) if data.training_run_id else None,
                "training_batch_id": str(data.training_batch_id) if data.training_batch_id else None,
                "mode": data.config.mode,
                "status": "QUEUED", "created_at": utc_now(), "generated_count": 0}
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        generate_images_task.apply_async(args=[str(workspace_id), str(run_id)], task_id=str(run_id))
    except Exception as exc:
        metadata.update(status="SUBMISSION_FAILED", error=str(exc))
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise HTTPException(503, "Unable to submit generation job") from exc
    return {"run_id": run_id, "workspace_id": workspace_id, "status": "QUEUED"}


def list_generation_runs(workspace_id: UUID):
    get_workspace(workspace_id)

    root = generation_root(workspace_id)

    if not root.exists():
        return []

    result = []

    for directory in sorted(root.iterdir(), reverse=True):
        path = directory / "run.json"

        if directory.is_dir() and path.is_file():
            result.append(
                json.loads(path.read_text(encoding="utf-8"))
            )

    return result


def get_generation_run(workspace_id: UUID, run_id: UUID):
    return read_run(workspace_id, run_id)


def list_generated_images(workspace_id: UUID, run_id: UUID):
    read_run(workspace_id, run_id)

    path = generation_run_dir(workspace_id, run_id) / "manifest.json"

    if not path.is_file():
        return {"images": []}

    return json.loads(path.read_text(encoding="utf-8"))


def get_generated_image(
    workspace_id: UUID,
    run_id: UUID,
    class_id: UUID,
    filename: str,
):
    read_run(workspace_id, run_id)

    if (
        Path(filename).name != filename
        or not filename.endswith(".png")
    ):
        raise HTTPException(400, "Invalid image filename")

    path = (
        generation_run_dir(workspace_id, run_id)
        / "images"
        / str(class_id)
        / filename
    )

    if not path.is_file():
        raise HTTPException(404, "Generated image not found")

    return path
