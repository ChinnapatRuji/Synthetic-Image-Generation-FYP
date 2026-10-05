#!/usr/bin/env python

import hashlib
import json
import re
import shutil
from pathlib import Path
from PIL import Image, ImageOps

from config import load_config

cfg = load_config()

Image.MAX_IMAGE_PIXELS = 300_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def folder_name(name):
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-]+", "_", name.strip.lower()))

def build_caption(entry):
    parts = [cfg.dataset.trigger, entry.name, entry.description, cfg.dataset.domain]
    return ", ".join(p.strip() for p in parts if p and p.strip())

def load_and_resize(src):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if min(w, h) < cfg.dataset.min_size:
            raise ValueError(f"Image {src} is too small: {w}x{h}")
        scale = cfg.dataset.max_size / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        return im.copy()

def main():
    dataset_dir = cfg.paths.dataset_dir
    image_dir = dataset_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)

    result = {}
    seen = set()

    for entry in cfg.dataset.classes:
        src_dir = entry.path
        if not src_dir.exists():
            raise SystemExit(f"Folder not found for class {entry.name}: {src_dir}")

        name = entry.folder
        class_dir = image_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        caption = build_caption(entry)
        kept = skipped = 0
        sources = [p for p in sorted(src_dir.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
        for src in sources:
            digest = hashlib.md5(src.read_bytes()).hexdigest()
            if digest in seen:
                skipped += 1
                continue
            try:
                im = load_and_resize(src)
            except Exception as e:
                print(f"    skipping {src.name}: {e}")
                im = None
            if im is None:
                skipped += 1
                continue
            seen.add(digest)

            stem = f"{name}_{kept:05d}"
            im.save(class_dir / f"{stem}.png", format="PNG")
            (class_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
            kept += 1

        if kept == 0:
            raise SystemExit(f"No valid images found for class {entry.name} in {src_dir}")

        result[name] = {
            "count": kept,
            "skipped": skipped,
            "num_repeats": entry.repeats,
            "description": entry.description,
            "caption": caption,
            "image_dir": class_dir.relative_to(dataset_dir).as_posix(),
        }

    biggest = max(r["count"] for r in result.values())
    for r in result.values():
        if not r["num_repeats"]:
            r["num_repeats"] = max(1, min(cfg.dataset.max_repeats, biggest // r["count"]))
    meta = {"trigger": cfg.dataset.trigger, "domain": cfg.dataset.domain, "classes": result}
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDataset summary:")
    print("-" * 80)
    for name, r in result.items():
        print(f"{name}: {r['count']} images, {r['skipped']} skipped, {r['num_repeats']} repeats")
    print("-" * 80)
    total_steps = sum(r["count"] * r["num_repeats"] for r in result.values())
    print(f"    steps per epoch: {total_steps}")
    print(f"    written to: {dataset_dir.resolve()}")

if __name__ == "__main__":
    main()

        