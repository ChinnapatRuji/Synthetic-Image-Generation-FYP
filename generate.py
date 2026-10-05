import json
import random
from pathlib import Path
import torch
from PIL import Image, ImageOps
from diffusers import(
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
)
from tqdm.auto import tqdm

from config import load_config

cfg = load_config()
gen = cfg.generate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def load_meta():
    meta_path = cfg.paths.dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemError(f"{meta_path} not found. Run prepare_dataset.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta["classes"]
    if gen.only_classes:
        missing = [c for c in gen.only_classes if c not in classes]
        if missing:
            raise SystemError(f"Unknown class {missing}. Available: {list(classes)}")
        classes = {k: classes[k] for k in gen.only_classes}
    return meta.get("trigger", ""), meta.get("domain", ""), classes

def find_checkpoints():
    lora_dir = cfg.paths.output_dir
    if gen.checkpoint == "none":
        return [None]
    if not lora_dir.is_dir():
        raise SystemExit(f"{lora_dir} not found. Train a LoRA first.")

    files = sorted(lora_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No .safetensors files in {lora_dir}")

    if gen.checkpoint == "all":
        return files
    if gen.checkpoint == "final":
        finals = [f for f in files if not f.stem.rsplit("-", 1)[-1].isdigit()]
        if not finals:
            raise SystemExit(f"No final (unnumbered) checkpoint found in {lora_dir}: {[f.name for f in files]}")
        return finals[:1]

    exact = lora_dir / gen.checkpoint
    if not exact.is_file():
        raise SystemExit(f"{exact} not found. Available: {[f.name for f in files]}")
    return [exact]

def build_pipeline():
    kwargs = dict(safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float16)

    if gen.mode == "img2img":
        pipeline_cls = StableDiffusionImg2ImgPipeline
    elif gen.mode == "txt2img":
        pipeline_cls = StableDiffusionPipeline
    else:
        raise SystemExit(f"Unknown generation mode: {gen.mode}")

    if Path(cfg.model.base).is_file():
        pipe = pipeline_cls.from_single_file(cfg.model.base, **kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(cfg.model.base, **kwargs)

    if cfg.model.vae:
        pipe.vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)

    if gen.sampler == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )

    pipe.set_progress_bar_config(disable=True)
    if gen.low_vram:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_vae_slicing()
    return pipe

def build_prompt(trigger, name, description, domain, class_prompt, rng):
    variation = rng.choice(gen.prompt_variations) if gen.prompt_variations else ""
    parts = [trigger, name.replace("_", " "), class_prompt, description, domain, variation]
    return ", ".join(p.strip() for p in parts if p and p.strip())

def build_negative_prompt(class_negative):
    parts = [gen.negative_prompt, class_negative]
    return ", ".join(p.strip() for p in parts if p and p.strip())

def get_generation_settings(name):
    entry = next((c for c in cfg.dataset.classes if c.folder == name), None)
    if entry is None:
        raise SystemExit(f"Class {name} not found in dataset config")

    generation = getattr(entry, "generation", None)
    if generation is None:
        return "", ""

    return (
        getattr(generation, "prompt", ""),
        getattr(generation, "negative_prompt", ""),
    )

def find_source_images(name):
    entry = next((c for c in cfg.dataset.classes if c.folder == name), None)
    if entry is None:
        raise SystemExit(f"Class {name} not found in dataset config")

    folder = Path(entry.path)
    if not folder.is_dir():
        raise SystemExit(f"{folder} not found")

    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {folder}")
    return images

def load_source_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.fit(im, (gen.width, gen.height), method=Image.Resampling.LANCZOS)
        return im.copy()

def generate_for(pipe, checkpoint, trigger, domain, classes, device):
    label = checkpoint.stem if checkpoint else "base_model"
    root = cfg.paths.generated_dir / gen.output_name

    if checkpoint:
        pipe.load_lora_weights(str(checkpoint))
        pipe.fuse_lora(lora_scale=gen.lora_scale)

    manifest = []
    for name, info in classes.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{label}:{name}:{gen.seed}")

        class_prompt, class_negative = get_generation_settings(name)
        negative_prompt = build_negative_prompt(class_negative)
        source_images = find_source_images(name) if gen.mode == "img2img" else None

        made = 0
        bar = tqdm(total=gen.images_per_class, desc=f"{label} / {name}", unit="imge")

        while made < gen.images_per_class:
            count = min(gen.batch_size, gen.images_per_class - made)
            prompts = [
                build_prompt(
                    trigger,
                    name,
                    info.get("description", ""),
                    domain,
                    class_prompt,
                    rng
                )
                for _ in range(count)
            ]
            seeds = [gen.seed + made + i for i in range(count)]
            generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

            args = dict(
                prompt=prompts,
                negative_prompt=[negative_prompt] * count,
                num_inference_steps=gen.steps,
                guidance_scale=gen.cfg,
                generator=generators,
                clip_skip=cfg.model.clip_skip - 1 if cfg.model.clip_skip > 1 else None,
            )

            if gen.mode == "img2img":
                init_paths = [source_images[(made + i) % len(source_images)] for i in range(count)]
                init_images = [load_source_image(p) for p in init_paths]
                images = pipe(
                    **args,
                    image=init_images,
                    strength=gen.strength,
                ).images
            else:
                init_paths = [None] * count
                images = pipe(
                    **args,
                    width=gen.width,
                    height=gen.height,
                ).images

            for image, prompt, seed, source in zip(images, prompts, seeds, init_paths):
                filename = f"{name}_{made:05d}.png"
                image.save(folder / filename)
                manifest.append(
                    {
                        "file": f"{name}/{filename}",
                        "class": name,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed,
                        "source": str(source) if source else None
                    }
                )
                made += 1
                bar.update(1)
        bar.close()

    if checkpoint:
        pipe.unfuse_lora()
        pipe.unload_lora_weights()

    (root / "manifest.json").write_text(json.dumps({
        "output_name": gen.output_name,
        "checkpoint": checkpoint.name if checkpoint else None,
        "base_model": cfg.model.base,
        "mode": gen.mode,
        "strength": gen.strength if gen.mode == "img2img" else None,
        "lora_scale": gen.lora_scale if checkpoint else 0,
        "sampler": gen.sampler,
        "negative": gen.negative_prompt,
        "image": manifest,
    }, indent=2), encoding="utf-8")
    return len(manifest)

def main():
    trigger, domain, classes = load_meta()
    checkpoints = find_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no CUDA device found, generation is being done on CPU. It will be extremely slow.")

    total = gen.images_per_class * len(classes) * len(checkpoints)
    print(f"Base model: {cfg.model.base}")
    print(f"LoRA folder: {cfg.paths.output_dir}")
    print(f"Checkpoints: {[c.name if c else 'base model' for c in checkpoints]}")
    print(f"Classes: {list(classes)}")
    print(f"Mode: {gen.mode}")
    if gen.mode == "img2img":
        print(f"Strength: {gen.strength}")
    print(f"Generating: {gen.images_per_class}/class = {total} images -> {cfg.paths.generated_dir}\n")

    pipe = build_pipeline()
    made = 0
    for checkpoint in checkpoints:
        made += generate_for(pipe, checkpoint, trigger, domain, classes, device)
    print(f"\nDone. {made} images written to {cfg.paths.generated_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())