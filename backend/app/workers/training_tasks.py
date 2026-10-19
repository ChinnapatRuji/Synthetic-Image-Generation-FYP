from pathlib import Path
from uuid import UUID

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.ml.training.train_sd_lora_kohya import train
from app.workers.celery_app import celery_app

from config import load_config


@celery_app.task(bind=True, name="training.train_lora")
def train_lora_task(self, workspace_id: str):

    cfg = load_config()

    run_id = UUID(self.request.id)
    workspace_id = UUID(workspace_id)

    run_path = StoragePaths.training(
        workspace_id,
        run_id,
    )

    run_dir = settings.STORAGE_ROOT / run_path

    output_dir = run_dir / "checkpoints"
    config_dir = run_dir / "config"

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    train(
        cfg=cfg,
        output_dir=output_dir,
        config_dir=config_dir,
    )

    return {
        "run_id": str(run_id),
        "workspace_id": str(workspace_id),
        "status": "completed",
        "output_dir": str(output_dir),
    }