"""SD1.5 generation: txt2img/img2img with optional Canny ControlNet."""
import json
import random
from pathlib import Path
import torch
from PIL import Image, ImageOps
from diffusers import (
    AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def load_classes(dataset_dir, only_classes=None):
    metadata = json.loads((Path(dataset_dir) / 'dataset.json').read_text(encoding='utf-8'))
    classes = metadata.get('classes', [])
    if not classes:
        raise ValueError('Dataset contains no classes')
    selected = set(only_classes or [])
    missing = selected - {item['id'] for item in classes}
    if missing:
        raise ValueError(f'Unknown class IDs: {sorted(missing)}')
    return [entry for entry in classes if not selected or entry['id'] in selected]


def build_prompt(trigger, name, description, domain, extra, variations, rng):
    variation = rng.choice(variations) if variations else ''
    parts = [trigger, name.replace('_', ' '), description, extra, domain, variation]
    unique = []
    for part in parts:
        part = str(part or '').strip()
        if part and part.casefold() not in {x.casefold() for x in unique}:
            unique.append(part)
    return ', '.join(unique)


def source_image(path, width, height):
    with Image.open(path) as im:
        return ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'), (width, height), method=Image.Resampling.LANCZOS).copy()


def canny_image(image, low, high):
    import cv2
    import numpy as np
    arr = cv2.Canny(np.array(image), low, high)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), mode='RGB')


def select_device(execution):
    available = torch.cuda.is_available()
    if execution.device == 'cuda' and not available:
        raise RuntimeError('CUDA requested but not available')
    device = 'cuda' if available and execution.device != 'cpu' else 'cpu'
    precision = execution.precision
    if precision == 'auto':
        precision = 'fp16' if device == 'cuda' else 'fp32'
    if device == 'cpu' and precision != 'fp32':
        raise ValueError('CPU execution requires fp32')
    dtype = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}[precision]
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('Current CUDA device does not support bf16')
    return device, dtype


def build_pipeline(model_config, cfg):
    device, dtype = select_device(cfg.execution)
    if cfg.conditioning is not None:
        pipeline_cls = StableDiffusionControlNetImg2ImgPipeline if cfg.mode == 'img2img' else StableDiffusionControlNetPipeline
    else:
        pipeline_cls = StableDiffusionImg2ImgPipeline if cfg.mode == 'img2img' else StableDiffusionPipeline
    kwargs = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    if cfg.conditioning is not None:
        kwargs['controlnet'] = ControlNetModel.from_pretrained(cfg.conditioning.model, torch_dtype=dtype)
    if Path(model_config.base).is_file():
        pipe = pipeline_cls.from_single_file(model_config.base, **kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(model_config.base, **kwargs)
    if model_config.vae:
        pipe.vae = AutoencoderKL.from_pretrained(model_config.vae, torch_dtype=dtype)
    if cfg.sampler == 'euler_a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type='dpmsolver++'
        )
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    return pipe, device


def place_pipeline(pipe, cfg, device):
    strategy = cfg.execution.memory_strategy
    if strategy == 'auto':
        if device == 'cuda':
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            strategy = 'cpu_offload' if total_gb < 10 else 'full_gpu'
        else:
            strategy = 'full_gpu'
    if strategy == 'full_gpu':
        pipe.to(device)
    elif device != 'cuda':
        raise ValueError('CPU offload requires CUDA')
    elif strategy == 'cpu_offload':
        pipe.enable_model_cpu_offload()
    elif strategy == 'sequential_offload':
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f'Unsupported memory strategy: {strategy}')


def write_manifest(output_dir, checkpoint_path, model_config, cfg, classes, manifest):
    counts = {c['id']: 0 for c in classes}
    for item in manifest:
        counts[item['class_id']] += 1
    result = {
        'checkpoint': checkpoint_path.name, 'base_model': model_config.base,
        'mode': cfg.mode, 'conditioning': cfg.conditioning.model_dump(mode='json') if cfg.conditioning else None,
        'generated_count': len(manifest),
        'classes': [{'id': c['id'], 'name': c['name'], 'image_count': counts[c['id']]} for c in classes],
        'images': manifest,
    }
    target = output_dir / 'manifest.json'
    tmp = output_dir / 'manifest.json.tmp'
    tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    tmp.replace(target)


def generate(*, model_config, generation_config, dataset_config, dataset_dir, checkpoint_path, output_dir):
    cfg = generation_config
    dataset_dir, output_dir, checkpoint_path = map(Path, (dataset_dir, output_dir, checkpoint_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    classes = load_classes(dataset_dir, cfg.only_classes)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    manifest = []
    write_manifest(output_dir, checkpoint_path, model_config, cfg, classes, manifest)
    pipe, device = build_pipeline(model_config, cfg)
    try:
        pipe.load_lora_weights(str(checkpoint_path))
        pipe.fuse_lora(lora_scale=cfg.lora_scale)
        place_pipeline(pipe, cfg, device)
        for entry in classes:
            class_id, class_name = entry['id'], entry['name']
            class_dir = output_dir / 'images' / class_id
            class_dir.mkdir(parents=True, exist_ok=True)
            extra = entry.get('generation') or {}
            negative = ', '.join(filter(None, [cfg.negative_prompt.strip(), extra.get('negative_prompt', '').strip()]))
            rng = random.Random(f'{checkpoint_path.stem}:{class_id}:{cfg.seed}')
            need_source = cfg.mode == 'img2img' or cfg.conditioning is not None
            sources = []
            if need_source:
                source_dir = dataset_dir / 'classes' / class_id
                sources = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
                if not sources:
                    raise ValueError(f'No source images for class {class_name}')
            for start in range(0, cfg.images_per_class, cfg.batch_size):
                count = min(cfg.batch_size, cfg.images_per_class - start)
                prompts = [build_prompt(dataset_config.trigger, class_name, entry.get('description', ''), dataset_config.domain, extra.get('prompt', ''), cfg.prompt_variations, rng) for _ in range(count)]
                seeds = [cfg.seed + start + i for i in range(count)]
                generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
                selected = []
                for i in range(count):
                    if not sources:
                        selected.append(None)
                    else:
                        source_settings = cfg.source if cfg.mode == 'img2img' else cfg.conditioning.source
                        index = rng.randrange(len(sources)) if source_settings.selection == 'random' else (start + i) % len(sources)
                        selected.append(sources[index])
                args = dict(prompt=prompts, negative_prompt=[negative] * count, num_inference_steps=cfg.steps, guidance_scale=cfg.cfg, generator=generators)
                if model_config.clip_skip > 1:
                    args['clip_skip'] = model_config.clip_skip - 1
                if cfg.mode == 'img2img':
                    args['image'] = [source_image(p, cfg.width, cfg.height) for p in selected]
                    args['strength'] = cfg.strength
                else:
                    args['width'], args['height'] = cfg.width, cfg.height
                if cfg.conditioning is not None:
                    edges = [canny_image(source_image(p, cfg.width, cfg.height), cfg.conditioning.low_threshold, cfg.conditioning.high_threshold) for p in selected]
                    args['controlnet_conditioning_scale'] = cfg.conditioning.scale
                    if cfg.mode == 'img2img':
                        args['control_image'] = edges
                    else:
                        args['image'] = edges
                images = pipe(**args).images
                for i, (im, prompt, seed, src) in enumerate(zip(images, prompts, seeds, selected)):
                    filename = f'{start+i:05d}.png'
                    im.save(class_dir / filename)
                    manifest.append({
                        'file': f'images/{class_id}/{filename}', 'class_id': class_id,
                        'class_name': class_name, 'prompt': prompt, 'negative_prompt': negative,
                        'seed': seed, 'source_image_id': src.stem if src else None,
                        'conditioning': cfg.conditioning.model_dump(mode='json') if cfg.conditioning else None,
                    })
                write_manifest(output_dir, checkpoint_path, model_config, cfg, classes, manifest)
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return len(manifest)
