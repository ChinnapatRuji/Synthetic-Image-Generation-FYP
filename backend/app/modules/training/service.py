from app.core.timestamps import utc_now, elapsed_seconds
import json
import re
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from celery.result import AsyncResult
from fastapi import HTTPException
from pydantic import ValidationError

from config import Config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.infrastructure.storage.training_runs import training_root, training_run_dir, index_batch_run
from app.modules.training.schemas import TrainingRunCreate, TrainingBatchCreate
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
    return training_run_dir(workspace_id, run_id)


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


# Configurations are complete and validated in the API request.
# There is no global config.json or implicit override merging.

def validate_dataset(workspace_id: UUID, only_classes=None):
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
    requested = {str(c) for c in (only_classes or [])}
    if requested - {c["id"] for c in classes}:
        raise HTTPException(422, "Unknown class IDs in only_classes")
    classes = [c for c in classes if not requested or c["id"] in requested]
    return_classes = classes

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

    return return_classes


def start_training(workspace_id: UUID, data: TrainingRunCreate, *, batch_id=None):
    get_workspace(workspace_id)
    selected_classes = validate_dataset(workspace_id, data.class_ids)
    selected_classes = [dict(c, training_token=c.get("training_token") or
                        "class_" + re.sub(r"[^a-z0-9]+", "_", c["name"].lower()).strip("_"))
                        for c in selected_classes]
    cfg = data.config.to_run_config()
    run_id = uuid4()
    run_dir = training_run_dir(workspace_id, run_id, batch_id=batch_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    if batch_id is not None:
        index_batch_run(workspace_id, batch_id, run_id)
    (run_dir / "classes.json").write_text(json.dumps({"classes": selected_classes}, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": str(run_id), "workspace_id": str(workspace_id),
        "class_ids": [str(c) for c in data.class_ids],
        "batch_id": str(batch_id) if batch_id else None,
        "status": "QUEUED", "created_at": utc_now(),
    }, indent=2), encoding="utf-8")
    try:
        train_lora_task.apply_async(args=[str(workspace_id), str(run_id)], task_id=str(run_id))
    except Exception as exc:
        update_run(run_dir, "SUBMISSION_FAILED", error=str(exc))
        raise HTTPException(503, "Unable to submit training job") from exc
    return {"run_id": run_id, "workspace_id": workspace_id, "status": "QUEUED"}


def batch_root(workspace_id):
    return get_runs_dir(workspace_id) / "batches"


def _deep_merge(base: dict, overrides: dict) -> dict:
    # Work on a fresh dictionary; never mutate batch defaults.
    result = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if key not in result:
            raise HTTPException(422, f"Unknown configuration override: {key}")
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def start_batch(workspace_id: UUID, data: TrainingBatchCreate):
    from config import TrainingRequestConfig
    from pydantic import ValidationError

    selected = validate_dataset(workspace_id, data.class_ids)
    class_ids = [UUID(entry["id"]) for entry in selected]
    unknown = set(data.class_overrides) - set(class_ids)
    if unknown:
        raise HTTPException(422, f"Unknown class overrides: {sorted(map(str, unknown))}")

    # Validate EVERY resolved configuration before creating directories or queuing jobs.
    jobs = []
    for cid in class_ids:
        merged = _deep_merge(data.config.model_dump(mode="json"), data.class_overrides.get(cid, {}))
        try:
            config = TrainingRequestConfig.model_validate(merged)
            jobs.append(TrainingRunCreate(class_ids=[cid], config=config))
        except ValidationError as exc:
            raise HTTPException(422, f"Invalid override for class {cid}: {exc}") from exc

    batch_id = uuid4()
    directory = batch_root(workspace_id) / str(batch_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = {"batch_id": str(batch_id), "name": data.name or f"Training Batch {str(batch_id)[:8]}",
              "runs": [], "created_at": utc_now(), "status": "QUEUED"}
    batch_file = directory / "batch.json"
    def save():
        temporary = batch_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(batch_file)
    save()
    for job in jobs:
        try:
            response = start_training(workspace_id, job, batch_id=batch_id)
            record["runs"].append({"class_ids": [str(c) for c in job.class_ids],
                                   "run_id": str(response["run_id"])})
            save()
        except Exception as exc:
            record["status"] = "SUBMISSION_FAILED"
            record["error"] = str(exc)
            save()
            raise
    return get_batch(workspace_id, batch_id)


def get_batch(workspace_id: UUID, batch_id: UUID):
    path = batch_root(workspace_id) / str(batch_id) / "batch.json"
    if not path.is_file():
        raise HTTPException(404, "Training batch not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    states = []
    for item in data["runs"]:
        run = get_training_status(workspace_id, UUID(item["run_id"]))
        item["status"] = run["status"]
        states.append(run["status"])
    if data.get("status") == "SUBMISSION_FAILED":
        pass
    elif states and all(state == "COMPLETED" for state in states):
        data["status"] = "COMPLETED"
    elif any(state in {"FAILED", "SUBMISSION_FAILED", "CLEANUP_UNCERTAIN"} for state in states):
        data["status"] = "PARTIAL_FAILURE"
    elif any(state in {"TRAINING", "PREPARING"} for state in states):
        data["status"] = "RUNNING"
    elif states and all(state in TERMINAL for state in states):
        data["status"] = "FINISHED_WITHOUT_SUCCESS"
    else:
        data["status"] = "QUEUED"
    return data


def list_batches(workspace_id: UUID):
    root = batch_root(workspace_id)
    return [get_batch(workspace_id, UUID(p.name)) for p in root.iterdir()
            if p.is_dir() and (p / "batch.json").is_file()] if root.exists() else []


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
    directories = list(root.glob("*/run.json"))
    directories += list((root / "runs").glob("*/run.json"))
    directories += list((root / "batches").glob("*/runs/*/run.json"))
    runs = []
    for path in directories:
        metadata = read_run(path.parent)
        if metadata.get("workspace_id") == str(workspace_id):
            runs.append(metadata)
    return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)


def get_run_config(workspace_id: UUID, run_id: UUID):
    directory, _ = require_run(workspace_id, run_id)
    path = directory / "config.json"
    if not path.is_file():
        raise HTTPException(404, "Training configuration not found")
    cfg = Config.model_validate_json(path.read_text(encoding="utf-8"))
    return {"class_ids": read_run(directory).get("class_ids"),
            "model": cfg.model.model_dump(mode="json"),
            "dataset": {k: getattr(cfg.dataset, k) for k in ("trigger", "domain", "min_size", "max_size", "max_repeats")},
            "lora": cfg.lora.model_dump(mode="json"),
            "training": cfg.training.model_dump(mode="json")}


def checkpoint_directory(workspace_id: UUID, run_id: UUID):
    directory, _ = require_run(workspace_id, run_id)
    return directory / "checkpoints"


def _checkpoint_folder(directory: Path, manifest: dict, entry: dict | None = None) -> Path:
    """Resolve both the current flat layout and historical checkpoint layouts."""
    if entry is not None and entry.get("checkpoint"):
        relative = Path(entry["checkpoint"])
        if relative.is_absolute() or ".." in relative.parts:
            raise HTTPException(409, "Unsafe checkpoint manifest path")
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise HTTPException(409, "Checkpoint path escapes training run")
        return candidate.parent
    if manifest.get("strategy") == "shared":
        legacy = directory / "shared"
        return legacy if legacy.is_dir() else directory
    return directory


def list_checkpoints(workspace_id: UUID, run_id: UUID):
    directory = checkpoint_directory(workspace_id, run_id)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folders = {}
    for cid, entry in manifest.get("classes", {}).items():
        folder = _checkpoint_folder(directory, manifest, entry)
        folders[str(folder)] = (folder, entry.get("name", cid))
    if not folders:
        folder = _checkpoint_folder(directory, manifest)
        folders[str(folder)] = (folder, "All selected classes")
    result = []
    for folder, class_name in folders.values():
        for path in sorted(folder.glob("*.safetensors")):
            result.append({"class_name": class_name, "filename": path.name,
                           "size_bytes": path.stat().st_size})
    return result


def get_checkpoint(workspace_id: UUID, run_id: UUID, filename: str):
    directory = checkpoint_directory(workspace_id, run_id)
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise HTTPException(400, "Invalid checkpoint filename")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "Checkpoint manifest not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folders = {_checkpoint_folder(directory, manifest, entry)
               for entry in manifest.get("classes", {}).values()}
    if not folders:
        folders = {_checkpoint_folder(directory, manifest)}
    matches = [folder / filename for folder in folders if (folder / filename).is_file()]
    if not matches:
        raise HTTPException(404, "Checkpoint not found")
    if len(matches) > 1:
        raise HTTPException(409, "Ambiguous checkpoint filename")
    return matches[0]


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

            # Do not delete checkpoints still referenced by a Model Set.
            model_sets = (settings.STORAGE_ROOT / StoragePaths.workspace(workspace_id) / "model_sets")
            for item in model_sets.glob("*.json") if model_sets.exists() else []:
                obj = json.loads(item.read_text(encoding="utf-8"))
                if any(str(v.get("training_run_id")) == str(run_id)
                       for v in obj.get("classes", {}).values()):
                    raise HTTPException(409, "Training run is referenced by a Model Set")
            shutil.rmtree(directory)
            registry = training_root(workspace_id) / "run_registry.json"
            if registry.is_file():
                records = json.loads(registry.read_text(encoding="utf-8"))
                if records.pop(str(run_id), None) is not None:
                    temp = registry.with_suffix(".tmp")
                    temp.write_text(json.dumps(records, indent=2), encoding="utf-8")
                    temp.replace(registry)

    except Timeout:
        raise HTTPException(
            status_code=409,
            detail="Training execution is still active",
        )
