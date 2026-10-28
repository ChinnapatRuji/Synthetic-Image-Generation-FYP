"""Resolve completed training runs and batches to immutable generation selections."""
import json
from pathlib import Path
from uuid import UUID
from fastapi import HTTPException
from config import Config
from app.modules.training.service import require_run, get_batch


def _load(path):
    if not path.is_file():
        raise HTTPException(409, f"Missing training artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(run_dir, manifest_entry, selector):
    root = (run_dir / "checkpoints").resolve()
    saved = Path(manifest_entry["checkpoint"])
    if selector == "final":
        name = saved.name
    else:
        name = selector
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise HTTPException(422, "Invalid checkpoint selector")
    # Current flat layout, then legacy per-class/shared layout.
    candidates = [root / name, root / saved.parent / name]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    raise HTTPException(404, f"Checkpoint not found: {name}")


def resolve_generation_source(workspace_id, request):
    if request.training_run_id is not None:
        run_ids = [request.training_run_id]
    else:
        batch = get_batch(workspace_id, request.training_batch_id)
        if batch["status"] != "COMPLETED":
            raise HTTPException(409, "Training batch is not completed")
        run_ids = [UUID(item["run_id"]) for item in batch["runs"]]
    if not run_ids:
        raise HTTPException(409, "No training runs available")
    wanted = set(request.config.class_ids)
    overrides = {str(k): v for k, v in request.checkpoint_overrides.items()}
    resolved = {}
    compatible = None
    for run_id in run_ids:
        run_dir, run = require_run(workspace_id, run_id)
        if run.get("status") != "COMPLETED":
            raise HTTPException(409, f"Training run {run_id} is not completed")
        config = Config.model_validate_json((run_dir / "config.json").read_text(encoding="utf-8"))
        identity = (config.model.base, config.model.vae, config.model.clip_skip,
                    config.dataset.trigger, config.dataset.domain)
        if compatible is None:
            compatible = identity
        elif compatible != identity:
            raise HTTPException(422, "Selected training runs have incompatible model or dataset settings")
        classes = _load(run_dir / "classes.json")["classes"]
        manifest = _load(run_dir / "checkpoints" / "manifest.json")
        for cls in classes:
            cid = cls["id"]
            if wanted and cid not in wanted:
                continue
            if cid in resolved:
                raise HTTPException(422, f"Class {cid} is present in multiple selected runs")
            entry = manifest.get("classes", {}).get(cid)
            if not entry or entry.get("status") != "COMPLETED":
                raise HTTPException(409, f"Class {cid} has no completed checkpoint")
            checkpoint = _checkpoint(run_dir, entry, overrides.get(cid, request.checkpoint))
            resolved[cid] = {"training_run_id": str(run_id), "checkpoint": checkpoint.name,
                             "checkpoint_path": str(checkpoint), "class": cls}
    if wanted - set(resolved):
        raise HTTPException(422, f"Requested classes unavailable: {sorted(wanted - set(resolved))}")
    if set(overrides) - set(resolved):
        raise HTTPException(422, "Checkpoint overrides contain unselected or unknown classes")
    if not resolved:
        raise HTTPException(409, "No compatible classes available for generation")
    return resolved
