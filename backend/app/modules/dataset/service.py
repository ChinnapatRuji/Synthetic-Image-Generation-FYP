import json
import re
import shutil
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.infrastructure.storage.paths import StoragePaths
from app.core.settings import settings
from app.modules.dataset.schemas import ClassCreate, ClassUpdate
from app.modules.workspaces.service import (
    get_workspace,
    get_workspace_path,
    write_json,
)


ALLOWED_FORMATS = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


def dataset_path(workspace_id: UUID):
    get_workspace(workspace_id)
    return settings.STORAGE_ROOT / StoragePaths.dataset(workspace_id)


def metadata_path(workspace_id: UUID):
    return dataset_path(workspace_id) / "dataset.json"


def get_dataset(workspace_id: UUID):
    path = metadata_path(workspace_id)
    if not path.is_file():
        raise HTTPException(404, "Dataset not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"classes": data.get("classes", [])}


def save_dataset(workspace_id: UUID, data):
    write_json(metadata_path(workspace_id), data)


def get_class(workspace_id: UUID, class_id: UUID):
    data = get_dataset(workspace_id)
    for item in data["classes"]:
        if item["id"] == str(class_id):
            return item
    raise HTTPException(404, "Class not found")


def class_path(workspace_id: UUID, class_id: UUID):
    get_class(workspace_id, class_id)
    return dataset_path(workspace_id) / "classes" / str(class_id)


def _default_token(class_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", class_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Class name cannot produce a valid training token")
    return f"class_{slug}"


def _check_unique_token(classes, token, exclude_id=None):
    if any(c.get("training_token", "").casefold() == token.casefold()
           and c["id"] != exclude_id for c in classes):
        raise HTTPException(409, "Training token already exists")


def create_class(workspace_id: UUID, data: ClassCreate):
    metadata = get_dataset(workspace_id)
    if any(c["name"].casefold() == data.name.casefold() for c in metadata["classes"]):
        raise HTTPException(409, "Class already exists")
    class_id = uuid4()
    token = data.training_token or _default_token(data.name)
    _check_unique_token(metadata["classes"], token)
    item = {"id": str(class_id), "name": data.name, "image_count": 0,
            "training_token": token, "description": data.description,
            "generation": data.generation.model_dump()}
    directory = dataset_path(workspace_id) / "classes" / str(class_id)
    directory.mkdir(parents=True, exist_ok=False)
    metadata["classes"].append(item)
    save_dataset(workspace_id, metadata)
    return item


def update_class(workspace_id: UUID, class_id: UUID, data: ClassUpdate):
    metadata = get_dataset(workspace_id)
    item = next((c for c in metadata["classes"] if c["id"] == str(class_id)), None)
    if item is None:
        raise HTTPException(404, "Class not found")
    updates = data.model_dump(exclude_unset=True)
    if "name" in updates:
        if updates["name"] is None:
            raise HTTPException(422, "name cannot be null")
        if any(c["id"] != str(class_id) and c["name"].casefold() == updates["name"].casefold()
               for c in metadata["classes"]):
            raise HTTPException(409, "Class already exists")
    if "training_token" in updates:
        if updates["training_token"] is None:
            raise HTTPException(422, "training_token cannot be null")
        _check_unique_token(metadata["classes"], updates["training_token"], str(class_id))
    if "description" in updates and updates["description"] is None:
        raise HTTPException(422, "description cannot be null")
    if "generation" in updates and updates["generation"] is None:
        raise HTTPException(422, "generation cannot be null")
    item.update(updates)
    save_dataset(workspace_id, metadata)
    return item


def ensure_no_training_runs(workspace_id: UUID):
    directory = get_workspace_path(workspace_id) / "training"
    if directory.exists() and any(directory.iterdir()):
        raise HTTPException(
            409,
            "Cannot delete dataset images or classes while training "
            "runs exist. Training artifact management is not implemented yet.",
        )


def delete_class(workspace_id: UUID, class_id: UUID):
    ensure_no_training_runs(workspace_id)
    directory = class_path(workspace_id, class_id)
    metadata = get_dataset(workspace_id)

    metadata["classes"] = [
        c for c in metadata["classes"]
        if c["id"] != str(class_id)
    ]

    shutil.rmtree(directory)
    save_dataset(workspace_id, metadata)


def upload_images(
    workspace_id: UUID,
    class_id: UUID,
    files: list[UploadFile],
):
    directory = class_path(workspace_id, class_id)
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
            extension = ALLOWED_FORMATS[image_format]
            destination = directory / f"{image_id}{extension}"

            with destination.open("xb") as output:
                shutil.copyfileobj(file.file, output)

            uploaded.append({
                "id": str(image_id),
                "filename": filename,
                "stored_filename": destination.name,
            })

        except ValueError as exc:
            rejected.append({
                "filename": filename,
                "reason": str(exc),
            })
        finally:
            file.file.close()

    metadata = get_dataset(workspace_id)
    for item in metadata["classes"]:
        if item["id"] == str(class_id):
            item["image_count"] = len([
                p for p in directory.iterdir()
                if p.is_file() and p.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp"}
            ])

    save_dataset(workspace_id, metadata)

    return {
        "workspace_id": str(workspace_id),
        "class_id": str(class_id),
        "uploaded_count": len(uploaded),
        "rejected_count": len(rejected),
        "uploaded": uploaded,
        "rejected": rejected,
    }


def list_images(workspace_id: UUID, class_id: UUID):
    selected = get_class(workspace_id, class_id)
    directory = class_path(workspace_id, class_id)

    images = []
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".webp"
            }:
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


def get_image_path(
    workspace_id: UUID,
    class_id: UUID,
    image_id: UUID,
):
    directory = class_path(workspace_id, class_id)

    for ext in (".png", ".jpg", ".webp"):
        path = directory / f"{image_id}{ext}"
        if path.is_file():
            return path

    raise HTTPException(404, "Image not found")


def delete_image(
    workspace_id: UUID,
    class_id: UUID,
    image_id: UUID,
):
    ensure_no_training_runs(workspace_id)
    path = get_image_path(workspace_id, class_id, image_id)
    path.unlink()

    directory = class_path(workspace_id, class_id)
    metadata = get_dataset(workspace_id)

    for item in metadata["classes"]:
        if item["id"] == str(class_id):
            item["image_count"] = sum(
                p.is_file() and p.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp"}
                for p in directory.iterdir()
            )

    save_dataset(workspace_id, metadata)
