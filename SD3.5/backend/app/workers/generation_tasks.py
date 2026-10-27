from app.core.timestamps import utc_now, elapsed_seconds
import json
from uuid import UUID

from config import Config
from pydantic import TypeAdapter
from app.modules.generation.config_models import GenerationConfig

from app.ml.generation.generate import generate
from app.modules.workspaces.service import get_workspace_path
from app.workers.celery_app import celery_app


def update_status(run_dir, status, **extra):
    path = run_dir / "run.json"

    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["status"] = status
    metadata.update(extra)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )
    temporary.replace(path)


@celery_app.task(bind=True, name="generation.generate_images")
def generate_images_task(self, workspace_id: str, run_id: str):
    workspace_id = UUID(workspace_id)
    run_id = UUID(run_id)

    workspace_dir = get_workspace_path(workspace_id)
    run_dir = workspace_dir / "generations" / str(run_id)

    try:
        snapshot = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )

        training_dir = (
            workspace_dir
            / "training"
            / snapshot["training_run_id"]
        )

        training_data = json.loads(
            (training_dir / "config.json").read_text(encoding="utf-8")
        )
        training_config = Config.model_validate_json(json.dumps(training_data))
        generation_config = TypeAdapter(GenerationConfig).validate_python(snapshot["generation"])

        checkpoint_path = (
            training_dir
            / "checkpoints"
            / snapshot["checkpoint"]
        )

        update_status(
            run_dir,
            "GENERATING",
            started_at=utc_now(),
        )

        self.update_state(
            state="GENERATING",
            meta={"stage": "Generating synthetic images"},
        )

        count = generate(
            model_config=training_config.model,
            generation_config=generation_config,
            dataset_config=training_config.dataset,
            dataset_dir=workspace_dir / "dataset",
            checkpoint_path=checkpoint_path,
            output_dir=run_dir,
        )

        completed_at = utc_now()

        metadata = json.loads(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )

        update_status(
            run_dir,
            "COMPLETED",
            generated_count=count,
            completed_at=completed_at,
            duration_seconds=elapsed_seconds(
                metadata.get("started_at"),
                completed_at,
            ),
        )

        return {
            "run_id": str(run_id),
            "status": "COMPLETED",
            "generated_count": count,
        }

    except Exception as exc:
        failed_at = utc_now()

        metadata = json.loads(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )

        update_status(
            run_dir,
            "FAILED",
            error=str(exc),
            failed_at=failed_at,
            duration_seconds=elapsed_seconds(
                metadata.get("started_at"),
                failed_at,
            ),
        )
        raise
