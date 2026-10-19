import json
from pathlib import Path
from uuid import UUID, uuid4

from celery.result import AsyncResult
from fastapi import HTTPException
from pydantic import ValidationError

from config import Config, load_config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.modules.training.schemas import TrainingRunCreate
from app.modules.workspaces.service import get_workspace
from app.workers.celery_app import celery_app
from app.workers.training_tasks import train_lora_task


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def get_run_dir(workspace_id: UUID, run_id: UUID):
    return (
        settings.STORAGE_ROOT
        / StoragePaths.training(workspace_id, run_id)
    )


def validate_dataset(workspace_id: UUID):
    dataset_dir = (
        settings.STORAGE_ROOT
        / StoragePaths.dataset(workspace_id)
    )

    metadata_path = dataset_dir / "dataset.json"

    if not metadata_path.is_file():
        raise HTTPException(400, "Workspace dataset not found")

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    classes = metadata.get("classes", [])

    if not classes:
        raise HTTPException(400, "Dataset has no classes")

    for entry in classes:
        class_dir = dataset_dir / "classes" / entry["id"]

        if not class_dir.is_dir():
            raise HTTPException(
                400,
                f"Class directory missing: {entry['name']}",
            )

        images = [
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
        ]

        if not images:
            raise HTTPException(
                400,
                f"Class '{entry['name']}' has no images",
            )


def build_config(data: TrainingRunCreate):
    cfg = load_config()

    config_data = cfg.model_dump(mode="python")
    overrides = data.model_dump(exclude_none=True)

    for key in ("epochs", "batch_size"):
        if key in overrides:
            config_data["training"][key] = overrides[key]

    for key in ("network_dim", "network_alpha"):
        if key in overrides:
            config_data["lora"][key] = overrides[key]

    try:
        return Config.model_validate(config_data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False),
        )


def start_training(
    workspace_id: UUID,
    data: TrainingRunCreate,
):
    get_workspace(workspace_id)
    validate_dataset(workspace_id)

    cfg = build_config(data)

    run_id = uuid4()
    run_dir = get_run_dir(workspace_id, run_id)

    run_dir.mkdir(parents=True, exist_ok=False)

    config_snapshot = cfg.model_dump(mode="json")

    (run_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=4),
        encoding="utf-8",
    )

    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": str(run_id),
            "workspace_id": str(workspace_id),
            "status": "QUEUED",
        }, indent=4),
        encoding="utf-8",
    )

    try:
        train_lora_task.apply_async(
            args=[
                str(workspace_id),
                str(run_id),
            ],
            task_id=str(run_id),
        )
    except Exception as exc:
        (run_dir / "run.json").write_text(
            json.dumps({
                "run_id": str(run_id),
                "workspace_id": str(workspace_id),
                "status": "SUBMISSION_FAILED",
            }, indent=4),
            encoding="utf-8",
        )
        raise HTTPException(
            503,
            "Unable to submit training job",
        ) from exc

    return {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "status": "QUEUED",
    }


def get_training_status(workspace_id: UUID, run_id: UUID):
    get_workspace(workspace_id)

    run_dir = get_run_dir(workspace_id, run_id)
    run_metadata = run_dir / "run.json"

    if not run_metadata.is_file():
        raise HTTPException(404, "Training run not found")

    metadata = json.loads(run_metadata.read_text())

    if metadata.get("workspace_id") != str(workspace_id):
        raise HTTPException(404, "Training run not found")

    task = AsyncResult(str(run_id), app=celery_app)

    result = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "status": task.status,
    }

    if task.status in {"PREPARING", "TRAINING", "STARTED"}:
        result["progress"] = task.info

    elif task.status == "SUCCESS":
        result["result"] = task.result

    elif task.status == "FAILURE":
        result["error"] = str(task.result)

    elif task.status == "PENDING":
        result["message"] = (
            "Task queued, not yet started, or result expired"
        )

    return result
