
#Prepartion of dataset (only using real images)
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

# Build the split only once and reuse it, so every evaluation run uses the same train/val/test images.
# Delete the prepared folder to rebuild it. A split that still has the removed Invalid class is rebuilt.
if dst.is_dir() and (dst / "train" / "Invalid").is_dir():
    print("Existing split contains the removed Invalid class - rebuilding it")
    shutil.rmtree(dst)

if dst.is_dir():
    print(f"Using existing split in {dst}")
else:
    classes = [(c.name, c) for c in real_classes]
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
# Delete the "runs" folder to train again. A classifier trained with the removed Invalid class is retrained.
WEIGHTS = OUT_DIR / "runs" / "train" / "weights" / "best.pt"
model = YOLO(str(WEIGHTS)) if WEIGHTS.is_file() else None
if model is not None and "Invalid" in model.names.values():
    print("Saved classifier has the removed Invalid class - training again")
    model = None

if model is not None:
    print(f"Using trained classifier {WEIGHTS}")
else:
    # Load the pretrained YOLO26 nano classification model
    model = YOLO("yolo26n-cls.pt")

    # Train on your real TE images
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
#GEN_DIR = DATA_DIR / (sys.argv[1] if len(sys.argv) > 1 else "img2img generated")  # must contain <class>/*.png

generated_arg = sys.argv[1] if len(sys.argv) > 1 else "img2img generated"
candidate = Path(generated_arg)

GEN_DIR = (
    candidate if candidate.is_absolute()
    else DATA_DIR / candidate
)
if not GEN_DIR.is_dir():
    raise SystemExit(f"{GEN_DIR} not found")
RUN_NAME = GEN_DIR.name.replace(" generated", "").replace(" ", "_")
CSV_PATH = RESULTS_DIR / f"{RUN_NAME}_eval.csv"

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
    rows.append({
        "file": img.name,
        "intended": img.parent.name,
        "predicted": predicted,
        "confidence": float(pred.probs.top1conf),  # classifier's probability for its predicted class (0-1)
        "pass": norm(predicted) == target(img.parent.name),
    })
df = pd.DataFrame(rows)
print("Overall pass rate:", round(df["pass"].mean(), 3))

print("Average confidence:", round(df["confidence"].mean(), 3))

# Per class: images, passes, pass rate, and the average confidence of passing / failing predictions
summary = df.groupby("intended").agg(
    count=("pass", "size"),
    passed=("pass", "sum"),
    pass_rate=("pass", "mean"),
    confidence=("confidence", "mean"),
)
summary["conf_pass"] = df[df["pass"]].groupby("intended")["confidence"].mean()
summary["conf_fail"] = df[~df["pass"]].groupby("intended")["confidence"].mean()
print(summary.round(2))

df.to_csv(CSV_PATH, index=False)
print("Saved", CSV_PATH)
