"""Upload a dataset, train a LoRA and download the generated images, all through the API.

    python infrastructure/hpc/run_pipeline.py DATASET [--out DIR] [--modes img2img txt2img] [--images-per-class N]
    python infrastructure/hpc/run_pipeline.py --workspace ID [...]   # only generate, reusing that workspace's model

DATASET holds one folder per class, and each folder name becomes the class name. Images
may sit in subfolders. An optional description.txt in a class folder is added to that
class's training captions and generation prompts. Training and generation settings come
from backend/config.json. Run it through start.sh, which starts the services and sets API_URL.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TRAINING_DONE = {"COMPLETED", "FAILED", "CANCELLED", "SUBMISSION_FAILED", "CLEANUP_UNCERTAIN"}
GENERATION_DONE = {"COMPLETED", "FAILED", "SUBMISSION_FAILED"}


def check(response):
    if response.is_error:
        sys.exit(
            f"{response.request.method} {response.request.url.path} failed "
            f"({response.status_code}): {response.text}"
        )
    return response


def wait_for(client, url, done, every):
    last = None
    while True:
        run = check(client.get(url)).json()
        if run["status"] != last:
            print(f"  {datetime.now():%H:%M:%S} {run['status']}", flush=True)
            last = run["status"]
        if run["status"] in done:
            return run
        time.sleep(every)


def upload_and_train(client, args):
    classes = {
        folder: sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
        )
        for folder in sorted(args.dataset.iterdir())
        if folder.is_dir()
    }
    classes = {folder: images for folder, images in classes.items() if images}
    if not classes:
        sys.exit(f"No class folders with images in {args.dataset}")

    workspace = check(client.post("/workspaces", json={"name": f"{args.dataset.name} {args.out.name}"})).json()
    ws = f"/workspaces/{workspace['id']}"
    print(f"Workspace {workspace['id']}")

    for folder, images in classes.items():
        name = folder.name
        description_file = folder / "description.txt"
        description = description_file.read_text(encoding="utf-8").strip() if description_file.is_file() else ""
        created = check(client.post(f"{ws}/dataset/classes", json={"name": name, "description": description})).json()
        uploaded = 0
        for start in range(0, len(images), 20):
            files = [("files", (p.name, p.read_bytes())) for p in images[start:start + 20]]
            result = check(client.post(f"{ws}/dataset/classes/{created['id']}/images", files=files)).json()
            uploaded += result["uploaded_count"]
            for item in result["rejected"]:
                print(f"  skipped {name}/{item['filename']}: {item['reason']}")
        print(f"Class {name}: {uploaded} images" + (f", described as: {description}" if description else ", no description.txt"))

    print("Training (Kohya's progress is in storage/runtime/worker.log)")
    config = check(client.get(f"{ws}/training/config/default")).json()
    run = check(client.post(f"{ws}/training/runs", json=config)).json()
    training = wait_for(client, f"{ws}/training/runs/{run['run_id']}", TRAINING_DONE, every=30)
    if training["status"] != "COMPLETED":
        sys.exit(f"Training {training['status']}: {training.get('error', 'see storage/runtime/worker.log')}")
    return ws, training, config


def reuse_training(client, workspace_id):
    ws = f"/workspaces/{workspace_id}"
    runs = [r for r in check(client.get(f"{ws}/training/runs")).json() if r["status"] == "COMPLETED"]
    if not runs:
        sys.exit(f"Workspace {workspace_id} has no completed training run")
    training = max(runs, key=lambda r: r.get("completed_at") or "")
    config = check(client.get(f"{ws}/training/runs/{training['run_id']}/config")).json()
    print(f"Reusing training run {training['run_id']} from workspace {workspace_id}")
    return ws, training, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--workspace", help="skip upload and training; generate with this workspace's trained model")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--modes", nargs="+", choices=["img2img", "txt2img"], default=["img2img", "txt2img"])
    parser.add_argument("--images-per-class", type=int)
    args = parser.parse_args()
    if not args.dataset and not args.workspace:
        parser.error("give a DATASET folder, or --workspace to reuse a trained model")

    client = httpx.Client(base_url=os.environ.get("API_URL", "http://127.0.0.1:8000") + "/api/v1", timeout=300)
    if args.workspace:
        ws, training, config = reuse_training(client, args.workspace)
    else:
        ws, training, config = upload_and_train(client, args)

    # Kohya saves the final LoRA without an epoch suffix.
    saved = [c["filename"] for c in check(client.get(f"{ws}/training/runs/{training['run_id']}/checkpoints")).json()]
    final = f"{config['training']['output_name']}.safetensors"
    if not saved:
        sys.exit("Training finished but saved no checkpoints")
    checkpoint = final if final in saved else saved[-1]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training.json").write_text(json.dumps({"run": training, "config": config}, indent=2))

    defaults = check(client.get(f"{ws}/generation/config/default")).json()
    for mode in args.modes:
        settings = defaults[mode]
        if args.images_per_class:
            settings["images_per_class"] = args.images_per_class
        print(f"Generating {mode} with {checkpoint}")
        gen = check(client.post(f"{ws}/generation/runs", json={
            "training_run_id": training["run_id"], "checkpoint": checkpoint, "config": settings,
        })).json()
        url = f"{ws}/generation/runs/{gen['run_id']}"
        result = wait_for(client, url, GENERATION_DONE, every=10)
        if result["status"] != "COMPLETED":
            sys.exit(f"Generation {result['status']}: {result.get('error')}")

        manifest = check(client.get(f"{url}/images")).json()
        for image in manifest["images"]:
            target = args.out / mode / image["class_name"] / Path(image["file"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(check(client.get(f"{url}/{image['file']}")).content)
        (args.out / mode / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  {len(manifest['images'])} images -> {args.out / mode}")

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
