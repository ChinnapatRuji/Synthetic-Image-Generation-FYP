from app.core.timestamps import utc_now, elapsed_seconds
import json
from uuid import UUID

from app.agents.evaluation.charts import make_charts
from app.agents.evaluation.evaluation import build_split, classify_images, summarize, train_classifier
from app.core.settings import settings
from app.modules.workspaces.service import get_workspace_path
from app.workers.celery_app import celery_app

BASE_MODEL = "yolo26n-cls.pt"  # pretrained YOLO26 nano classification model


def update_status(path, status, **extra):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["status"] = status
    metadata.update(extra)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )
    temporary.replace(path)


def mark_failed(path, exc):
    failed_at = utc_now()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    update_status(
        path,
        "FAILED",
        error=str(exc),
        failed_at=failed_at,
        duration_seconds=elapsed_seconds(metadata.get("started_at"), failed_at),
    )


@celery_app.task(bind=True, name="evaluation.train_classifier")
def train_classifier_task(self, workspace_id: str, classifier_id: str):
    workspace_dir = get_workspace_path(UUID(workspace_id))
    evaluation_dir = workspace_dir / "evaluation"
    classifier_dir = evaluation_dir / "classifiers" / classifier_id
    metadata_path = classifier_dir / "classifier.json"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata["config"]

        update_status(metadata_path, "TRAINING", started_at=utc_now())
        self.update_state(state="TRAINING", meta={"stage": "Training evaluation classifier"})

        split_counts = build_split(
            {c["id"]: evaluation_dir / "dataset" / c["id"] for c in metadata["classes"]},
            classifier_dir / "split",
            seed=config["seed"],
        )
        result = train_classifier(
            classifier_dir / "split",
            classifier_dir,
            base_model=settings.STORAGE_ROOT / "pretrained" / BASE_MODEL,
            **config,
        )

        completed_at = utc_now()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        update_status(
            metadata_path,
            "COMPLETED",
            split=split_counts,
            test_accuracy=result["test_accuracy"],
            weights=str(result["weights"].relative_to(classifier_dir)),
            completed_at=completed_at,
            duration_seconds=elapsed_seconds(metadata.get("started_at"), completed_at),
        )

        return {
            "classifier_id": classifier_id,
            "status": "COMPLETED",
            "test_accuracy": result["test_accuracy"],
        }

    except Exception as exc:
        mark_failed(metadata_path, exc)
        raise


@celery_app.task(bind=True, name="evaluation.evaluate_generation")
def evaluate_generation_task(self, workspace_id: str, run_id: str):
    workspace_dir = get_workspace_path(UUID(workspace_id))
    run_dir = workspace_dir / "evaluation" / "runs" / run_id
    metadata_path = run_dir / "run.json"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classifier_dir = workspace_dir / "evaluation" / "classifiers" / metadata["classifier_id"]
        classifier = json.loads((classifier_dir / "classifier.json").read_text(encoding="utf-8"))
        generation_dir = workspace_dir / "generations" / metadata["generation_run_id"]
        manifest = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))

        update_status(metadata_path, "EVALUATING", started_at=utc_now())
        self.update_state(state="EVALUATING", meta={"stage": "Classifying generated images"})

        # class_id -> name, in the classifier's class order; only these classes can be scored
        class_names = {c["id"]: c["name"] for c in classifier["classes"]}
        skipped = [c["name"] for c in manifest["classes"] if c["id"] not in class_names]

        images = [
            {"file": item["file"], "path": generation_dir / item["file"], "class_id": item["class_id"]}
            for item in manifest["images"]
            if item["class_id"] in class_names
        ]
        if not images:
            raise ValueError("No generated images belong to a class the classifier was trained on")

        rows = classify_images(
            classifier_dir / classifier["weights"],
            images,
            imgsz=classifier["config"]["imgsz"],
        )
        for row in rows:
            row["class_name"] = class_names[row["class_id"]]
            row["predicted_class_name"] = class_names.get(row["predicted_class_id"], row["predicted_class_id"])

        summary = summarize(rows, class_names)
        (run_dir / "results.json").write_text(json.dumps({"images": rows}, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        make_charts(rows, class_names, run_dir / "charts")

        completed_at = utc_now()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        update_status(
            metadata_path,
            "COMPLETED",
            evaluated_count=len(rows),
            pass_rate=summary["pass_rate"],
            confidence=summary["confidence"],
            skipped_classes=skipped,
            completed_at=completed_at,
            duration_seconds=elapsed_seconds(metadata.get("started_at"), completed_at),
        )

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "pass_rate": summary["pass_rate"],
        }

    except Exception as exc:
        mark_failed(metadata_path, exc)
        raise
