from app.core.timestamps import utc_now
import json
import shutil
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.agents.evaluation.charts import CHARTS
from app.agents.evaluation.evaluation import IMAGE_SUFFIXES, MIN_IMAGES_PER_CLASS
from app.modules.dataset.service import (
    ALLOWED_FORMATS,
    MAX_FILE_SIZE,
    MAX_IMAGE_PIXELS,
    get_class,
    get_dataset,
)
from app.modules.evaluation.schemas import ClassifierCreate, EvaluationRunCreate
from app.modules.generation.service import read_run as read_generation_run
from app.modules.workspaces.service import get_workspace, get_workspace_path, write_json
from app.workers.evaluation_tasks import evaluate_generation_task, train_classifier_task


# Storage layout inside a workspace:
#   evaluation/dataset/<class_id>/<image_id>.<ext>   real images used only by the evaluation classifier
#   evaluation/classifiers/<classifier_id>/           classifier.json, train/val/test split, YOLO weights
#   evaluation/runs/<run_id>/                         run.json, results.json, summary.json, charts/


def evaluation_root(workspace_id: UUID):
    return get_workspace_path(workspace_id) / "evaluation"


def evaluation_class_dir(workspace_id: UUID, class_id: UUID):
    return evaluation_root(workspace_id) / "dataset" / str(class_id)


def classifier_dir(workspace_id: UUID, classifier_id: UUID):
    return evaluation_root(workspace_id) / "classifiers" / str(classifier_id)


def evaluation_run_dir(workspace_id: UUID, run_id: UUID):
    return evaluation_root(workspace_id) / "runs" / str(run_id)


def count_images(directory):
    if not directory.is_dir():
        return 0
    return sum(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in directory.iterdir())


def read_json(path, missing_message):
    if not path.is_file():
        raise HTTPException(404, missing_message)
    return json.loads(path.read_text(encoding="utf-8"))


def submit(task, args, run_dir, metadata_name, message):
    try:
        task.apply_async(args=args, task_id=args[1])
    except Exception as exc:
        path = run_dir / metadata_name
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["status"] = "SUBMISSION_FAILED"
        metadata["error"] = str(exc)
        write_json(path, metadata)
        raise HTTPException(503, message) from exc


# ---------- Evaluation dataset (real images, separate from the LoRA training dataset) ----------

def get_evaluation_dataset(workspace_id: UUID):
    classes = get_dataset(workspace_id)["classes"]
    return {
        "classes": [
            {
                "id": item["id"],
                "name": item["name"],
                "image_count": count_images(evaluation_class_dir(workspace_id, item["id"])),
            }
            for item in classes
        ],
        "min_images_per_class": MIN_IMAGES_PER_CLASS,
    }


def upload_evaluation_images(workspace_id: UUID, class_id: UUID, files: list[UploadFile]):
    get_class(workspace_id, class_id)
    directory = evaluation_class_dir(workspace_id, class_id)
    directory.mkdir(parents=True, exist_ok=True)

    uploaded = []
    rejected = []

    for file in files:
        filename = file.filename or "unknown"

        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)

            if size == 0:
                raise ValueError("Empty file")
            if size > MAX_FILE_SIZE:
                raise ValueError("File exceeds 10 MB")

            try:
                with Image.open(file.file) as image:
                    image_format = image.format
                    if image_format not in ALLOWED_FORMATS:
                        raise ValueError("Unsupported image format")
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image dimensions are too large")
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Invalid image: {exc}")

            file.file.seek(0)
            image_id = uuid4()
            destination = directory / f"{image_id}{ALLOWED_FORMATS[image_format]}"

            with destination.open("xb") as output:
                shutil.copyfileobj(file.file, output)

            uploaded.append({
                "id": str(image_id),
                "filename": filename,
                "stored_filename": destination.name,
            })

        except ValueError as exc:
            rejected.append({"filename": filename, "reason": str(exc)})
        finally:
            file.file.close()

    return {
        "workspace_id": str(workspace_id),
        "class_id": str(class_id),
        "uploaded_count": len(uploaded),
        "rejected_count": len(rejected),
        "uploaded": uploaded,
        "rejected": rejected,
        "image_count": count_images(directory),
    }


def list_evaluation_images(workspace_id: UUID, class_id: UUID):
    selected = get_class(workspace_id, class_id)
    directory = evaluation_class_dir(workspace_id, class_id)

    images = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append({
                    "id": path.stem,
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                })

    return {
        "class_id": str(class_id),
        "class_name": selected["name"],
        "image_count": len(images),
        "images": images,
    }


def get_evaluation_image_path(workspace_id: UUID, class_id: UUID, image_id: UUID):
    get_class(workspace_id, class_id)
    directory = evaluation_class_dir(workspace_id, class_id)

    for ext in (".png", ".jpg", ".webp"):
        path = directory / f"{image_id}{ext}"
        if path.is_file():
            return path

    raise HTTPException(404, "Image not found")


def delete_evaluation_image(workspace_id: UUID, class_id: UUID, image_id: UUID):
    # Existing classifiers keep their own copy of the split, so they are not affected.
    get_evaluation_image_path(workspace_id, class_id, image_id).unlink()


# ---------- Classifier ----------

def start_classifier_training(workspace_id: UUID, data: ClassifierCreate):
    get_workspace(workspace_id)

    for item in list_classifiers(workspace_id):
        if item["status"] in ("QUEUED", "TRAINING"):
            raise HTTPException(409, "A classifier is already being trained")

    classes = [
        item for item in get_evaluation_dataset(workspace_id)["classes"]
        if item["image_count"] > 0
    ]
    if len(classes) < 2:
        raise HTTPException(422, "Upload evaluation images for at least two classes")

    too_few = [c["name"] for c in classes if c["image_count"] < MIN_IMAGES_PER_CLASS]
    if too_few:
        raise HTTPException(
            422,
            f"Classes need at least {MIN_IMAGES_PER_CLASS} evaluation images: {', '.join(too_few)}",
        )

    classifier_id = uuid4()
    directory = classifier_dir(workspace_id, classifier_id)
    directory.mkdir(parents=True, exist_ok=False)

    metadata = {
        "classifier_id": str(classifier_id),
        "workspace_id": str(workspace_id),
        "status": "QUEUED",
        "config": data.config.model_dump(mode="json"),
        "classes": classes,
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "duration_seconds": None,
        "test_accuracy": None,
    }
    write_json(directory / "classifier.json", metadata)

    submit(
        train_classifier_task,
        [str(workspace_id), str(classifier_id)],
        directory,
        "classifier.json",
        "Unable to submit classifier training job",
    )

    return {"classifier_id": classifier_id, "workspace_id": workspace_id, "status": "QUEUED"}


def list_classifiers(workspace_id: UUID):
    get_workspace(workspace_id)
    root = evaluation_root(workspace_id) / "classifiers"

    if not root.exists():
        return []

    items = [
        json.loads((d / "classifier.json").read_text(encoding="utf-8"))
        for d in root.iterdir()
        if d.is_dir() and (d / "classifier.json").is_file()
    ]
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


def get_classifier(workspace_id: UUID, classifier_id: UUID):
    get_workspace(workspace_id)
    return read_json(
        classifier_dir(workspace_id, classifier_id) / "classifier.json",
        "Classifier not found",
    )


# ---------- Evaluation runs ----------

def start_evaluation(workspace_id: UUID, data: EvaluationRunCreate):
    generation_run = read_generation_run(workspace_id, data.generation_run_id)
    if generation_run.get("status") != "COMPLETED":
        raise HTTPException(409, "Selected generation run is not completed")

    if data.classifier_id is not None:
        classifier = get_classifier(workspace_id, data.classifier_id)
        if classifier["status"] != "COMPLETED":
            raise HTTPException(409, "Selected classifier is not trained")
    else:
        completed = [c for c in list_classifiers(workspace_id) if c["status"] == "COMPLETED"]
        if not completed:
            raise HTTPException(409, "Train an evaluation classifier first")
        classifier = completed[0]

    run_id = uuid4()
    run_dir = evaluation_run_dir(workspace_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": str(run_id),
        "workspace_id": str(workspace_id),
        "generation_run_id": str(data.generation_run_id),
        "classifier_id": classifier["classifier_id"],
        "status": "QUEUED",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "duration_seconds": None,
        "evaluated_count": 0,
        "pass_rate": None,
        "confidence": None,
    }
    write_json(run_dir / "run.json", metadata)

    submit(
        evaluate_generation_task,
        [str(workspace_id), str(run_id)],
        run_dir,
        "run.json",
        "Unable to submit evaluation job",
    )

    return {"run_id": run_id, "workspace_id": workspace_id, "status": "QUEUED"}


def list_evaluation_runs(workspace_id: UUID):
    get_workspace(workspace_id)
    root = evaluation_root(workspace_id) / "runs"

    if not root.exists():
        return []

    items = [
        json.loads((d / "run.json").read_text(encoding="utf-8"))
        for d in root.iterdir()
        if d.is_dir() and (d / "run.json").is_file()
    ]
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


def get_evaluation_run(workspace_id: UUID, run_id: UUID):
    get_workspace(workspace_id)
    run_dir = evaluation_run_dir(workspace_id, run_id)
    metadata = read_json(run_dir / "run.json", "Evaluation run not found")

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        metadata["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    return metadata


def get_evaluation_results(workspace_id: UUID, run_id: UUID):
    get_evaluation_run(workspace_id, run_id)
    path = evaluation_run_dir(workspace_id, run_id) / "results.json"

    if not path.is_file():
        return {"images": []}

    return json.loads(path.read_text(encoding="utf-8"))


def get_chart_path(workspace_id: UUID, run_id: UUID, chart: str):
    get_evaluation_run(workspace_id, run_id)

    if chart not in CHARTS:
        raise HTTPException(404, f"Unknown chart. Available: {', '.join(CHARTS)}")

    path = evaluation_run_dir(workspace_id, run_id) / "charts" / f"{chart}.png"

    if not path.is_file():
        raise HTTPException(404, "Chart not available yet")

    return path
