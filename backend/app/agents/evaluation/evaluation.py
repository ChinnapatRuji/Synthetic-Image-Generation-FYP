"""Evaluation agent: scores generated images with a YOLO classifier trained on real images.

An image passes when the classifier predicts the class it was generated as. Each
prediction also has a confidence (the classifier's probability for its predicted class).

This file only holds the ML logic. Where files are stored, and when it runs, is decided by
app.modules.evaluation and app.workers.evaluation_tasks. Class folders are named by class ID,
so real and generated images are matched by ID, not by name.
"""
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
MIN_IMAGES_PER_CLASS = 5  # need at least one image each for train, val and test


def list_images(directory: Path):
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_split(class_dirs: dict[str, Path], split_dir: Path, seed: int = 42):
    """Copy real images into split_dir/{train,val,test}/<class_id>/. Returns image counts per class."""
    rng = random.Random(seed)
    counts = {}

    for class_id in sorted(class_dirs):
        images = list_images(class_dirs[class_id])
        if len(images) < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class {class_id} has {len(images)} evaluation images; "
                f"at least {MIN_IMAGES_PER_CLASS} are needed"
            )

        rng.shuffle(images)
        n_val = max(1, int(len(images) * SPLIT["val"]))
        n_test = max(1, int(len(images) * SPLIT["test"]))
        n_train = len(images) - n_val - n_test
        parts = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for name, files in parts.items():
            out = split_dir / name / class_id
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)

        counts[class_id] = {name: len(files) for name, files in parts.items()}

    return counts


def train_classifier(
    split_dir: Path,
    work_dir: Path,
    *,
    base_model: Path,
    epochs: int,
    imgsz: int,
    patience: int,
    batch: int,
    seed: int,
):
    """Fine-tune a YOLO classification model on split_dir. Returns the best weights and test accuracy."""
    from ultralytics import YOLO

    base_model.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_model))  # downloaded to base_model on first use
    model.train(
        data=str(split_dir),
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,  # stop early if val accuracy stops improving
        batch=batch,
        seed=seed,
        device=pick_device(),
        project=str(work_dir),
        name="train",
        exist_ok=True,
        plots=False,
    )

    weights = work_dir / "train" / "weights" / "best.pt"
    best = YOLO(str(weights))
    metrics = best.val(
        data=str(split_dir),
        split="test",  # held-out real images the classifier never trained on
        imgsz=imgsz,
        project=str(work_dir),
        name="test",
        exist_ok=True,
        plots=False,
    )

    return {
        "weights": weights,
        "test_accuracy": round(float(metrics.top1), 4),
        "class_ids": [best.names[i] for i in sorted(best.names)],
    }


def classify_images(weights: Path, images: list[dict], imgsz: int):
    """Classify generated images.

    images: [{"file": ..., "path": Path, "class_id": ...}, ...]
    Returns one row per image with the prediction, its confidence and pass/fail.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows = []

    for item in images:
        result = model.predict(str(item["path"]), imgsz=imgsz, verbose=False)[0]
        predicted = result.names[result.probs.top1]
        rows.append({
            "file": item["file"],
            "class_id": item["class_id"],
            "predicted_class_id": predicted,
            "confidence": round(float(result.probs.top1conf), 4),
            "pass": predicted == item["class_id"],
        })

    return rows


def summarize(rows: list[dict], class_names: dict[str, str]):
    """Overall and per-class pass rate and confidence."""
    def stats(subset):
        passed = [r for r in subset if r["pass"]]
        failed = [r for r in subset if not r["pass"]]

        def mean(items):
            return round(sum(r["confidence"] for r in items) / len(items), 4) if items else None

        return {
            "count": len(subset),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(subset), 4) if subset else None,
            "confidence": mean(subset),
            "confidence_pass": mean(passed),
            "confidence_fail": mean(failed),
        }

    per_class = []
    for class_id, name in class_names.items():
        subset = [r for r in rows if r["class_id"] == class_id]
        if subset:
            per_class.append({"class_id": class_id, "class_name": name, **stats(subset)})

    return {**stats(rows), "classes": per_class}
