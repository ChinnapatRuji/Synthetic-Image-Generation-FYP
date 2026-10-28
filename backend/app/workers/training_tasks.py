from app.modules.training.execution import execution_lock
import json
from uuid import UUID

from config import Config

from app.core.settings import settings
from app.infrastructure.storage.paths import StoragePaths
from app.infrastructure.storage.training_runs import training_run_dir
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

    run_dir = training_run_dir(workspace_id, run_id)

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

        classes_file = run_dir / "classes.json"
        if not classes_file.is_file():
            raise RuntimeError("Training class snapshot missing")
        classes = json.loads(classes_file.read_text(encoding="utf-8"))["classes"]
        manifest = {"version": 4, "strategy": "single", "classes": {}}
        manifest_path = checkpoint_dir / "manifest.json"

        def save_manifest():
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)

        for entry in classes:
            cid = entry["id"]
            manifest["classes"][cid] = {
                "id": cid, "name": entry["name"],
                "training_token": entry["training_token"],
                "status": "QUEUED",
                "checkpoint": f"{cfg.training.output_name}.safetensors",
            }
        save_manifest()
        units = [classes]
        for unit in units:
            check_cancellation(run_dir)
            class_id = "run"
            class_name = "All selected classes"
            class_prepared = prepared_dir / class_id
            class_config = config_dir / class_id
            class_output = checkpoint_dir
            update_run(run_dir, "PREPARING", current_class_id=class_id, current_class_name=class_name)
            self.update_state(state="PREPARING", meta={"stage": f"Preparing {class_name}"})
            prepare_dataset(
                dataset_dir=dataset_dir, output_dir=class_prepared,
                trigger=cfg.dataset.trigger, domain=cfg.dataset.domain,
                min_size=cfg.dataset.min_size, max_size=cfg.dataset.max_size,
                max_repeats=cfg.dataset.max_repeats,
                class_ids={entry["id"] for entry in unit},
                classes_snapshot=classes,
            )
            check_cancellation(run_dir)
            update_run(run_dir, "TRAINING", current_class_id=class_id,
                       current_class_name=class_name, process_cleanup="PENDING")
            self.update_state(state="TRAINING", meta={"stage": f"Training {class_name}"})
            for entry in unit:
                manifest["classes"][entry["id"]]["status"] = "TRAINING"
            save_manifest()
            train(
                cfg=cfg, dataset_dir=class_prepared, output_dir=class_output,
                config_dir=class_config,
                cancel_check=lambda: check_cancellation(run_dir),
                process_callback=lambda pid: update_run(run_dir, process_group_id=pid),
                cleanup_callback=lambda: update_run(run_dir, process_cleanup="VERIFIED", process_group_id=None),
            )
            check_cancellation(run_dir)
            final_path = class_output / f"{cfg.training.output_name}.safetensors"
            if not final_path.is_file():
                raise RuntimeError(f"Final checkpoint missing for {class_name}: {final_path}")
            for entry in unit:
                manifest["classes"][entry["id"]]["status"] = "COMPLETED"
            save_manifest()
            update_run(run_dir, process_cleanup="VERIFIED", process_group_id=None)

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
