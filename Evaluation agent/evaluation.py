
#Prepartion of dataset (real images + synthetic "Invalid" class)
import random, shutil
from pathlib import Path

# Project folder (one level above "Evaluation agent"), so paths work from any working directory
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

src = DATA_DIR / "TE evaluation dataset" #Ensure that this dataset contains real images only
dst = DATA_DIR / "TE evaluation prepared dataset"
if not src.is_dir():
    raise SystemExit(f"{src} not found - unzip 'TE evaluation dataset.zip' into '{DATA_DIR}'")
split = {"train": 0.7, "val": 0.15, "test": 0.15}
random.seed(42)

real_classes = [d for d in src.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

# Invalid class: deliberately corrupted copies of real images, kept separate from the real dataset.
# Teaches the classifier to say "this is not a usable solder joint image" instead of forcing a real class.
INVALID = "Invalid"
invalid_src = DATA_DIR / "Invalid synthetic"
INVALID_PER_CLASS = 5

def make_invalid_images():
    import numpy as np
    from PIL import Image, ImageFilter

    invalid_src.mkdir(parents=True)
    rng = random.Random(0)             # separate generators, so the real-image split is unchanged
    np_rng = np.random.default_rng(0)
    i = 0
    for cls in sorted(real_classes):
        imgs = sorted(p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS)
        for f in rng.sample(imgs, min(INVALID_PER_CLASS, len(imgs))):
            im = Image.open(f).convert("RGB")
            w, h = im.size
            kind = i % 4
            if kind == 0:   # heavy blur
                im, label = im.filter(ImageFilter.GaussianBlur(12)), "blur"
            elif kind == 1: # strong noise
                a = np.array(im).astype(np.int16) + np_rng.integers(-120, 120, (h, w, 3))
                im, label = Image.fromarray(a.clip(0, 255).astype(np.uint8)), "noise"
            elif kind == 2: # crop that misses the joint
                im, label = im.crop((0, 0, w // 3, h // 3)).resize((w, h)), "crop"
            else:           # scrambled patches
                p = 16
                a = np.array(im)[: h - h % p, : w - w % p]
                tiles = [a[y:y + p, x:x + p] for y in range(0, a.shape[0], p) for x in range(0, a.shape[1], p)]
                rng.shuffle(tiles)
                cols = a.shape[1] // p
                im, label = Image.fromarray(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                                                       for r in range(len(tiles) // cols)])), "scramble"
            im.save(invalid_src / f"invalid_{label}_{i:03d}.jpg")
            i += 1
    print(f"Created {i} invalid images in {invalid_src}")

if not invalid_src.is_dir():
    make_invalid_images()

# Build the split only once and reuse it, so every evaluation run uses the same train/val/test images.
# Delete the prepared folder to rebuild it. A split made before the Invalid class existed is rebuilt.
if dst.is_dir() and not (dst / "train" / INVALID).is_dir():
    print("Existing split has no Invalid class - rebuilding it")
    shutil.rmtree(dst)

if dst.is_dir():
    print(f"Using existing split in {dst}")
else:
    classes = [(c.name, c) for c in real_classes] + [(INVALID, invalid_src)]
    print("Classes found:", [name for name, _ in classes])

    for cls_name, folder in classes:
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        random.shuffle(imgs)
        n_tr = int(len(imgs) * split["train"])
        n_va = int(len(imgs) * split["val"])
        parts = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}
        for name, files in parts.items():
            out = dst / name / cls_name
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)
        print(cls_name, {k: len(v) for k, v in parts.items()})

# verify the number of images that is in the folder
total = sum(1 for f in dst.rglob("*") if f.suffix.lower() in IMG_EXTS)
print("Total images:", total)

#Training the model
from ultralytics import YOLO

# Reuse the trained classifier if it exists, so every generated dataset is scored by the same model.
# Delete the "runs" folder to train again. A classifier trained without the Invalid class is retrained.
WEIGHTS = OUT_DIR / "runs" / "train" / "weights" / "best.pt"
model = YOLO(str(WEIGHTS)) if WEIGHTS.is_file() else None
if model is not None and INVALID not in model.names.values():
    print("Saved classifier has no Invalid class - training again")
    model = None

if model is not None:
    print(f"Using trained classifier {WEIGHTS}")
else:
    # Load the pretrained YOLO26 nano classification model
    model = YOLO("yolo26n-cls.pt")

    # Train on your real TE images (+ Invalid)
    results = model.train(
        data=str(dst.resolve()),  # folder with train/val/test
        epochs=100,
        imgsz=224,       # standard size for classification
        patience=20,     # stop early if val accuracy stops improving
        batch=16,
        seed=42,
        device = "mps",
        project=str(OUT_DIR / "runs"),  # keep training results inside "Evaluation agent"
        name="train",
        exist_ok=True,
    )
    model = YOLO(str(WEIGHTS))  # continue with the best epoch, not the last one

#evaluate models performance on validation set
results = model.val(data=str(dst.resolve()), imgsz=224, project=str(OUT_DIR / "runs"), name="val", exist_ok=True)

#perform predicition for new images or models
#results = model.predict()


#Testing the model on a generated dataset
# Folder in "data" to test, e.g. python "Evaluation agent/evaluation.py" "txt2img generated"
import sys
GEN_DIR = DATA_DIR / (sys.argv[1] if len(sys.argv) > 1 else "img2img generated")  # must contain <class>/*.png
if not GEN_DIR.is_dir():
    raise SystemExit(f"{GEN_DIR} not found")
RUN_NAME = GEN_DIR.name.replace(" generated", "").replace(" ", "_")
CSV_PATH = RESULTS_DIR / f"{RUN_NAME}_eval.csv"

# Below this confidence the prediction is "uncertain" rather than pass/fail
CONF_THRESHOLD = 0.6

# Invalid and uncertain images are copied here to be looked at by hand (cleared on every run)
REVIEW_DIR = RESULTS_DIR / "review" / RUN_NAME
shutil.rmtree(REVIEW_DIR, ignore_errors=True)

import re
import pandas as pd

def norm(name):
    # "NG-No solder" and "ng_no_solder" -> "ng_no_solder", so real and generated class names match
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-]+", "_", name.strip().lower()))

# Generated folder name -> TE dataset class name, where they differ after norm()
ALIASES = {"ng_single": "ng_single_electrode"}

def target(folder):
    # class name the classifier should predict for a generated folder
    return ALIASES.get(norm(folder), norm(folder))

def outcome_of(predicted, confidence, folder):
    # uncertain: model is not confident enough to trust the label
    # invalid:   model recognises the image as unusable (garbled, blurred, no joint)
    # pass/fail: model confidently predicts the intended class / another real class
    if confidence < CONF_THRESHOLD:
        return "uncertain"
    if predicted == INVALID:
        return "invalid"
    return "pass" if norm(predicted) == target(folder) else "fail"

known = {norm(n) for n in model.names.values()}
skipped = sorted(d.name for d in GEN_DIR.iterdir() if d.is_dir() and target(d.name) not in known)
if skipped:
    print("Skipping generated classes the classifier was not trained on:", skipped)

rows = []
for img in sorted(GEN_DIR.glob("*/*.png")):
    if img.parent.name in skipped:
        continue
    pred = model.predict(str(img), imgsz=224, verbose=False)[0]
    predicted = pred.names[pred.probs.top1]
    confidence = float(pred.probs.top1conf)
    outcome = outcome_of(predicted, confidence, img.parent.name)
    if outcome in ("invalid", "uncertain"):
        review = REVIEW_DIR / outcome
        review.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, review / f"{img.parent.name}__{img.name}")
    rows.append({
        "file": img.name,
        "intended": img.parent.name,
        "predicted": predicted,
        "confidence": confidence,
        "outcome": outcome,
        "pass": outcome == "pass",
    })
df = pd.DataFrame(rows)
print("Overall pass rate:", round(df["pass"].mean(), 3))

OUTCOMES = ["pass", "fail", "invalid", "uncertain"]
print(pd.crosstab(df["intended"], df["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
      .assign(total=lambda t: t.sum(axis=1)))
for outcome in ("invalid", "uncertain"):
    n = int((df["outcome"] == outcome).sum())
    if n:
        print(f"{n} {outcome} images copied to {REVIEW_DIR / outcome}")

df.to_csv(CSV_PATH, index=False)
print("Saved", CSV_PATH)
