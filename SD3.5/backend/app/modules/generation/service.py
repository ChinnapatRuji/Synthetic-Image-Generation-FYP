from app.core.timestamps import utc_now, elapsed_seconds
import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException

from config import Config, load_config
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


def default_generation_config(workspace_id: UUID):
    get_workspace(workspace_id)
    return load_config().generation_defaults


def start_generation(workspace_id: UUID, data: GenerationRunCreate):
    get_workspace(workspace_id)

    training_dir = (
        get_workspace_path(workspace_id)
        / "training"
        / str(data.training_run_id)
    )

    training_metadata = training_dir / "run.json"

    if not training_metadata.is_file():
        raise HTTPException(404, "Training run not found")

    training_run = json.loads(
        training_metadata.read_text(encoding="utf-8")
    )

    if training_run.get("status") != "COMPLETED":
        raise HTTPException(
            409,
            "Selected training run is not completed",
        )

    checkpoint = data.checkpoint

    if (
        Path(checkpoint).name != checkpoint
        or not checkpoint.endswith(".safetensors")
    ):
        raise HTTPException(422, "Invalid checkpoint filename")

    checkpoint_path = training_dir / "checkpoints" / checkpoint

    if not checkpoint_path.is_file():
        raise HTTPException(404, "Checkpoint not found")

    if not (training_dir / "config.json").is_file():
        raise HTTPException(404, "Training configuration missing")

    training_config = Config.model_validate_json(
        (training_dir / "config.json").read_text(encoding="utf-8")
    )

    # SD3 has no ControlNet here, samples with its own flow-matching
    # scheduler, and needs sizes divisible by its 16 px patches.
    if training_config.model.architecture == "sd3":
        if data.config.conditioning is not None:
            raise HTTPException(422, "ControlNet conditioning is not supported for SD3")

        if data.config.sampler != "default":
            raise HTTPException(422, "SD3 only supports the 'default' sampler")

        if data.config.width % 16 or data.config.height % 16:
            raise HTTPException(422, "SD3 width and height must be multiples of 16")

    run_id = uuid4()
    run_dir = generation_run_dir(workspace_id, run_id)

    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_metadata_path = get_workspace_path(workspace_id) / "dataset" / "dataset.json"
    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    available_classes = dataset_metadata.get("classes", [])
    selected_ids = set(data.config.only_classes)
    selected_classes = [
        {"id": item["id"], "name": item["name"]}
        for item in available_classes
        if not selected_ids or item["id"] in selected_ids
    ]
    if selected_ids - {item["id"] for item in available_classes}:
        raise HTTPException(422, "Unknown class IDs in only_classes")
    (run_dir / "classes.json").write_text(
        json.dumps({"classes": selected_classes}, indent=4), encoding="utf-8"
    )

    config_snapshot = {
        "training_run_id": str(data.training_run_id),
        "checkpoint": checkpoint,
        "generation": data.config.model_dump(mode="json"),
    }

    (run_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=4),
        encoding="utf-8",
    )

    metadata = {
        "run_id": str(run_id),
        "workspace_id": str(workspace_id),
        "training_run_id": str(data.training_run_id),
        "checkpoint": checkpoint,
        "mode": data.config.mode,
        "status": "QUEUED",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "duration_seconds": None,
        "generated_count": 0,
    }

    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    try:
        generate_images_task.apply_async(
            args=[str(workspace_id), str(run_id)],
            task_id=str(run_id),
        )
    except Exception as exc:
        metadata["status"] = "SUBMISSION_FAILED"
        metadata["error"] = str(exc)

        (run_dir / "run.json").write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        raise HTTPException(
            503,
            "Unable to submit generation job",
        ) from exc

    return {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "status": "QUEUED",
    }


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
