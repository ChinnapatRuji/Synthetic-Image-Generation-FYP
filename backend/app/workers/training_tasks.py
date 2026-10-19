import json
from pathlib import Path
from uuid import UUID

from config import Config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.ml.preprocessing.prepare_dataset import prepare_dataset
from app.ml.training.train_sd_lora_kohya import train
from app.modules.workspaces.service import get_workspace
from app.workers.celery_app import celery_app




def update_run_status(run_dir, status, **extra):
    """Persist training status in run.json."""
    path = run_dir / "run.json"

    metadata = json.loads(
        path.read_text(encoding="utf-8")
    )

    metadata["status"] = status
    metadata.update(extra)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )
    temporary.replace(path)


@celery_app.task(bind=True, name="training.train_lora")
def train_lora_task(
    self,
    workspace_id: str,
    run_id: str,
):
    workspace_id = UUID(workspace_id)
    run_id = UUID(run_id)

    get_workspace(workspace_id)

    workspace_dir = (
        settings.STORAGE_ROOT
        / StoragePaths.workspace(workspace_id)
    )

    dataset_dir = workspace_dir / "dataset"

    run_dir = (
        settings.STORAGE_ROOT
        / StoragePaths.training(workspace_id, run_id)
    )

    config_path = run_dir / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration missing: {config_path}"
        )

    cfg = Config.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )

    prepared_dir = run_dir / "prepared"
    config_dir = run_dir / "config"
    checkpoint_dir = run_dir / "checkpoints"

    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    update_run_status(run_dir, "PREPARING")

    self.update_state(
        state="PREPARING",
        meta={"stage": "Preparing dataset"},
    )

    prepare_dataset(
        dataset_dir=dataset_dir,
        output_dir=prepared_dir,
        trigger=cfg.dataset.trigger,
        domain=cfg.dataset.domain,
        min_size=cfg.dataset.min_size,
        max_size=cfg.dataset.max_size,
        max_repeats=cfg.dataset.max_repeats,
    )

    update_run_status(run_dir, "TRAINING")

    self.update_state(
        state="TRAINING",
        meta={"stage": "Training LoRA with Kohya SS"},
    )

    train(
        cfg=cfg,
        dataset_dir=prepared_dir,
        output_dir=checkpoint_dir,
        config_dir=config_dir,
    )

    update_run_status(
        run_dir,
        "COMPLETED",
        checkpoint_dir=str(checkpoint_dir),
    )

    return {
        "run_id": str(run_id),
        "workspace_id": str(workspace_id),
        "status": "completed",
        "checkpoint_dir": str(checkpoint_dir),
    }
