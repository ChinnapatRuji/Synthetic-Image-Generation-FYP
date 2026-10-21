from app.modules.training.execution import execution_lock
import json
from uuid import UUID

from config import Config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.ml.preprocessing.prepare_dataset import prepare_dataset
from app.ml.training.train_sd_lora_kohya import train
from app.modules.workspaces.service import get_workspace
from app.modules.training.lifecycle import (
    read_run,
    TrainingCancelled,
    check_cancellation,
    update_run,
)
from app.workers.celery_app import celery_app


@celery_app.task(bind=True, name="training.train_lora")
def train_lora_task(self, workspace_id: str, run_id: str):
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

    # Reject jobs whose run directory was removed.
    if not (run_dir / "run.json").is_file():
        raise RuntimeError(
            f"Training run no longer exists: {run_id}"
        )

    with execution_lock(run_dir):
        # Recheck after acquiring the execution lock.
        if not (run_dir / "run.json").is_file():
            raise RuntimeError(
                f"Training run was deleted: {run_id}"
            )

        return execute_training(self, workspace_id, run_id, run_dir)


def execute_training(self, workspace_id, run_id, run_dir):
    workspace_dir = (
        settings.STORAGE_ROOT
        / StoragePaths.workspace(workspace_id)
    )
    dataset_dir = workspace_dir / "dataset"

    try:
        check_cancellation(run_dir)

        cfg = Config.model_validate_json(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )

        prepared_dir = run_dir / "prepared"
        config_dir = run_dir / "config"
        checkpoint_dir = run_dir / "checkpoints"

        config_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        update_run(run_dir, "PREPARING")

        self.update_state(
            state="PREPARING",
            meta={"stage": "Preparing dataset"},
        )

        check_cancellation(run_dir)

        prepare_dataset(
            dataset_dir=dataset_dir,
            output_dir=prepared_dir,
            trigger=cfg.dataset.trigger,
            domain=cfg.dataset.domain,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            max_repeats=cfg.dataset.max_repeats,
        )

        check_cancellation(run_dir)

        update_run(run_dir, "TRAINING")

        self.update_state(
            state="TRAINING",
            meta={"stage": "Training LoRA with Kohya SS"},
        )

        update_run(run_dir, process_cleanup="PENDING")

        train(
            cfg=cfg,
            dataset_dir=prepared_dir,
            output_dir=checkpoint_dir,
            config_dir=config_dir,
            cancel_check=lambda: check_cancellation(run_dir),
            process_callback=lambda pid: update_run(
                run_dir,
                process_group_id=pid,
            ),
            cleanup_callback=lambda: update_run(
                run_dir,
                process_cleanup="VERIFIED",
                process_group_id=None,
            ),
        )

        update_run(
            run_dir,
            process_cleanup="VERIFIED",
            process_group_id=None,
        )

        check_cancellation(run_dir)

        update_run(
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

    except TrainingCancelled:
        metadata = read_run(run_dir)

        if metadata.get("process_cleanup") == "VERIFIED":
            update_run(run_dir, "CANCELLED")
            return {
                "run_id": str(run_id),
                "workspace_id": str(workspace_id),
                "status": "cancelled",
            }

        # Cancellation before Kohya starts requires no GPU cleanup.
        # Cancellation during Kohya must be independently verified.
        if metadata.get("process_group_id") is not None:
            update_run(
                run_dir,
                "CLEANUP_UNCERTAIN",
                process_cleanup="UNCERTAIN",
            )
            return {
                "run_id": str(run_id),
                "workspace_id": str(workspace_id),
                "status": "cleanup_uncertain",
            }

        update_run(
            run_dir,
            "CANCELLED",
            process_cleanup="NOT_REQUIRED",
        )

        return {
            "run_id": str(run_id),
            "workspace_id": str(workspace_id),
            "status": "cancelled",
        }

    except Exception as exc:
        update_run(run_dir, "FAILED", error=str(exc))
        raise
