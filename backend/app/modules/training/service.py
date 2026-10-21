from app.core.timestamps import utc_now, elapsed_seconds
import json
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from celery.result import AsyncResult
from fastapi import HTTPException
from pydantic import ValidationError

from config import Config, load_config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.modules.training.schemas import TrainingRunCreate
from app.modules.training.lifecycle import (
    TERMINAL,
    read_run,
    update_run,
)
from app.modules.workspaces.service import get_workspace
from app.workers.celery_app import celery_app
from app.workers.training_tasks import train_lora_task


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def get_run_dir(workspace_id: UUID, run_id: UUID):
    return (
        settings.STORAGE_ROOT
        / StoragePaths.training(workspace_id, run_id)
    )


def get_runs_dir(workspace_id: UUID):
    get_workspace(workspace_id)
    return (
        settings.STORAGE_ROOT
        / StoragePaths.workspace(workspace_id)
        / "training"
    )


def require_run(workspace_id: UUID, run_id: UUID):
    get_workspace(workspace_id)

    directory = get_run_dir(workspace_id, run_id)

    if not (directory / "run.json").is_file():
        raise HTTPException(404, "Training run not found")

    metadata = read_run(directory)

    if metadata.get("workspace_id") != str(workspace_id):
        raise HTTPException(404, "Training run not found")

    return directory, metadata


def default_config():
    cfg = load_config()

    return {
        "model": cfg.model.model_dump(mode="json"),
        "dataset": {
            "trigger": cfg.dataset.trigger,
            "domain": cfg.dataset.domain,
            "min_size": cfg.dataset.min_size,
            "max_size": cfg.dataset.max_size,
            "max_repeats": cfg.dataset.max_repeats,
        },
        "lora": cfg.lora.model_dump(mode="json"),
        "training": cfg.training.model_dump(mode="json"),
    }


def validate_dataset(workspace_id: UUID):
    dataset_dir = (
        settings.STORAGE_ROOT
        / StoragePaths.dataset(workspace_id)
    )

    metadata_path = dataset_dir / "dataset.json"

    if not metadata_path.is_file():
        raise HTTPException(400, "Workspace dataset not found")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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

        if not any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            for p in class_dir.iterdir()
        ):
            raise HTTPException(
                400,
                f"Class '{entry['name']}' has no images",
            )


def build_config(data: TrainingRunCreate):
    cfg = load_config()

    # Preserve backend-managed paths and other internal settings.
    values = cfg.model_dump(mode="python")
    request = data.model_dump(mode="python")

    values["model"] = request["model"]
    values["lora"] = request["lora"]
    values["training"] = request["training"]

    for key, value in request["dataset"].items():
        values["dataset"][key] = value

    try:
        return Config.model_validate(values)
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail=exc.errors(include_context=False),
        )


def start_training(workspace_id: UUID, data: TrainingRunCreate):
    get_workspace(workspace_id)
    validate_dataset(workspace_id)

    cfg = build_config(data)

    run_id = uuid4()
    run_dir = get_run_dir(workspace_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(
        cfg.model_dump_json(indent=4),
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
            args=[str(workspace_id), str(run_id)],
            task_id=str(run_id),
        )
    except Exception as exc:
        update_run(run_dir, "SUBMISSION_FAILED", error=str(exc))
        raise HTTPException(503, "Unable to submit training job") from exc

    return {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "status": "QUEUED",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "cancelled_at": None,
            "duration_seconds": None,
    }


def get_training_status(workspace_id: UUID, run_id: UUID):
    directory, metadata = require_run(workspace_id, run_id)

    # Persistent metadata is authoritative for terminal states.
    status = metadata["status"]

    if status in TERMINAL:
        return metadata

    task = AsyncResult(str(run_id), app=celery_app)

    if task.status == "FAILURE":
        update_run(directory, "FAILED", error=str(task.result))
        return read_run(directory)

    if task.status == "SUCCESS":
        update_run(directory, "COMPLETED")
        return read_run(directory)

    return metadata


def list_training_runs(workspace_id: UUID):
    root = get_runs_dir(workspace_id)

    if not root.exists():
        return []

    runs = []

    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue

        path = directory / "run.json"

        if path.is_file():
            metadata = read_run(directory)

            if metadata.get("workspace_id") == str(workspace_id):
                runs.append(metadata)

    return runs


def get_run_config(workspace_id: UUID, run_id: UUID):
    directory, _ = require_run(workspace_id, run_id)

    path = directory / "config.json"

    if not path.is_file():
        raise HTTPException(404, "Training configuration not found")

    cfg = Config.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    return {
        "model": cfg.model.model_dump(mode="json"),
        "dataset": {
            "trigger": cfg.dataset.trigger,
            "domain": cfg.dataset.domain,
            "min_size": cfg.dataset.min_size,
            "max_size": cfg.dataset.max_size,
            "max_repeats": cfg.dataset.max_repeats,
        },
        "lora": cfg.lora.model_dump(mode="json"),
        "training": cfg.training.model_dump(mode="json"),
    }


def checkpoint_directory(workspace_id: UUID, run_id: UUID):
    directory, _ = require_run(workspace_id, run_id)
    return directory / "checkpoints"


def list_checkpoints(workspace_id: UUID, run_id: UUID):
    directory = checkpoint_directory(workspace_id, run_id)

    if not directory.exists():
        return []

    return [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix == ".safetensors"
    ]


def get_checkpoint(workspace_id: UUID, run_id: UUID, filename: str):
    directory = checkpoint_directory(workspace_id, run_id)

    # Prevent path traversal and restrict downloads to LoRA files.
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid filename")

    if not filename.endswith(".safetensors"):
        raise HTTPException(400, "Unsupported checkpoint format")

    path = directory / filename

    if not path.is_file():
        raise HTTPException(404, "Checkpoint not found")

    return path


def cancel_training(workspace_id: UUID, run_id: UUID):
    directory, metadata = require_run(workspace_id, run_id)

    if metadata["status"] in TERMINAL:
        raise HTTPException(
            409,
            "Training run already finished",
        )

    from app.modules.training.lifecycle import request_cancellation

    return request_cancellation(directory)



def delete_training(workspace_id: UUID, run_id: UUID):
    directory, metadata = require_run(workspace_id, run_id)

    if metadata["status"] not in TERMINAL:
        raise HTTPException(
            status_code=409,
            detail="Training must finish or be cancelled before deletion",
        )

    from filelock import Timeout
    from app.modules.training.execution import execution_lock

    # Acquire the lock before deleting anything.
    lock = execution_lock(directory)

    try:
        with lock.acquire(timeout=0):
            # Recheck status after acquiring the lock.
            metadata = read_run(directory)

            if metadata["status"] not in TERMINAL:
                raise HTTPException(
                    409,
                    "Training run is not finished",
                )

            cleanup = metadata.get("process_cleanup")
            process_id = metadata.get("process_group_id")

            if process_id is not None:
                raise HTTPException(
                    409,
                    "Training process cleanup is not verified",
                )

            if cleanup not in {"VERIFIED", "NOT_REQUIRED"}:
                raise HTTPException(
                    409,
                    "Training cleanup status is unknown",
                )

            task = AsyncResult(str(run_id), app=celery_app)

            if task.status in {
                "STARTED",
                "PREPARING",
                "TRAINING",
                "RETRY",
            }:
                raise HTTPException(
                    409,
                    "Training worker is still active",
                )

            shutil.rmtree(directory)

    except Timeout:
        raise HTTPException(
            status_code=409,
            detail="Training execution is still active",
        )
