#!/usr/bin/env python

"""Prepare workspace images for Kohya SS LoRA training."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
Image.MAX_IMAGE_PIXELS = 50_000_000


def folder_name(name: str) -> str:
    name = re.sub(r"[\s-]+", "_", name.strip().lower())
    return re.sub(r"[^a-z0-9_]", "", name)


def build_caption(
    class_name: str,
    trigger: str = "",
    domain: str = "",
    description: str = "",
) -> str:
    # Generation prompts also turn underscores into spaces, so captions must match.
    parts = [trigger, class_name.replace("_", " "), description, domain]
    return ", ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def load_and_resize(
    source: Path,
    min_size: int,
    max_size: int,
):
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size

        if min(width, height) < min_size:
            raise ValueError(
                f"Image too small: {width}x{height}"
            )

        scale = max_size / max(width, height)

        if scale < 1:
            image = image.resize(
                (
                    round(width * scale),
                    round(height * scale),
                ),
                Image.Resampling.LANCZOS,
            )

        return image.copy()


def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    trigger: str = "",
    domain: str = "",
    min_size: int = 256,
    max_size: int = 1024,
    max_repeats: int = 10,
):
    """
    Prepare uploaded workspace images for Kohya SS.

    dataset_dir:
        Workspace dataset directory containing dataset.json
        and classes/<class_id>/.

    output_dir:
        Training run's prepared directory.

    Returns:
        Prepared dataset metadata.
    """

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()

    metadata_path = dataset_dir / "dataset.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    classes = metadata.get("classes", [])

    if not classes:
        raise ValueError("Dataset contains no classes")

    if min_size <= 0 or max_size < min_size:
        raise ValueError("Invalid image size configuration")

    if max_repeats < 1:
        raise ValueError("max_repeats must be at least 1")

    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("Prepared output must be outside the source dataset")

    # Avoid overwriting another training run's prepared data.
    if output_dir.exists():
        raise FileExistsError(
            f"Prepared output already exists: {output_dir}"
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    result = {}
    seen = set()

    try:
        for entry in classes:
            class_id = entry["id"]
            class_name = entry["name"]

            source_dir = dataset_dir / "classes" / class_id

            if not source_dir.is_dir():
                raise FileNotFoundError(
                    f"Class directory not found: {source_dir}"
                )

            # Use class UUID for stable, unique folder names.
            name = class_id
            class_dir = images_dir / name
            class_dir.mkdir(parents=True, exist_ok=False)

            description = entry.get("description", "")

            caption = build_caption(
                class_name,
                trigger=trigger,
                domain=domain,
                description=description,
            )

            kept = 0
            skipped = 0

            sources = [
                path
                for path in sorted(source_dir.iterdir())
                if path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
            ]

            for source in sources:
                try:
                    digest = hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()

                    if digest in seen:
                        skipped += 1
                        continue

                    image = load_and_resize(
                        source,
                        min_size=min_size,
                        max_size=max_size,
                    )

                except (OSError, ValueError) as exc:
                    print(f"Skipping {source.name}: {exc}")
                    skipped += 1
                    continue

                seen.add(digest)

                stem = f"image_{kept:05d}"

                image.save(
                    class_dir / f"{stem}.png",
                    format="PNG",
                )

                (class_dir / f"{stem}.txt").write_text(
                    caption + "\n",
                    encoding="utf-8",
                )

                kept += 1

            if kept == 0:
                raise ValueError(
                    f"No valid images found for class {class_name}"
                )

            result[name] = {
                "class_id": class_id,
                "class_name": class_name,
                "count": kept,
                "skipped": skipped,
                "num_repeats": entry.get("num_repeats"),
                "description": description,
                "caption": caption,
                "image_dir": class_dir.relative_to(
                    output_dir
                ).as_posix(),
            }

        biggest = max(
            info["count"]
            for info in result.values()
        )

        for info in result.values():
            if not info["num_repeats"]:
                info["num_repeats"] = max(
                    1,
                    min(
                        max_repeats,
                        biggest // info["count"],
                    ),
                )

        prepared_metadata = {
            "trigger": trigger,
            "domain": domain,
            "classes": result,
        }

        (output_dir / "meta.json").write_text(
            json.dumps(prepared_metadata, indent=2),
            encoding="utf-8",
        )

        return prepared_metadata

    except Exception:
        # Remove incomplete preparation output.
        shutil.rmtree(output_dir)
        raise
