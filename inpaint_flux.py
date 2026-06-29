"""
Inpainting module for Region Select.

Uses FLUX.1-Fill-dev (4-step inference) for seamless prompt-guided region
inpainting. Falls back to SD 1.5 if Flux is unavailable.

The model receives the FULL image + a binary mask, outputs a FULL image where
only the masked area is regenerated — everything outside the mask is preserved
seamlessly via native fill inpainting (no patch/blend artifacts).

Lazy-load pattern follows sam_tools.py: model loaded on first call, cached globally.
"""

import os, sys, time
import numpy as np
from PIL import Image

# HF cache already set in app.py; ensure it propagates
for _v in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE'):
    if not os.environ.get(_v):
        for _d in ('E:', 'D:', 'C:'):
            if os.path.exists(f'{_d}\\"):
                os.environ[_v] = f'{_d}/huggingface_cache' if 'HOME' in _v else f'{_d}/huggingface_cache/hub'
                break
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import torch

# ── Backend state ──
_flux_pipe = None
_sd_pipe = None
_device = None


def _load_flux_fill():
    """Lazy-load FLUX.1-Fill-dev (requires HF login + gated-repo access)."""
    global _flux_pipe, _device
    if _flux_pipe is not None:
        return _flux_pipe

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Inpaint] Loading FLUX.1-Fill-dev on {_device}...')
    print('[Inpaint] This is a ~22B model; first download may take a while...')

    from diffusers import FluxFillPipeline

    try:
        # Strategy 1: device_map="balanced" — keeps ~8GB on GPU, rest on CPU
        pipe = FluxFillPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-Fill-dev',
            torch_dtype=torch.bfloat16,
            device_map='balanced',
        )
        pipe.enable_attention_slicing()
        _flux_pipe = pipe
        print('[Inpaint] Flux Fill ready (balanced mode)')
        return _flux_pipe
    except Exception as e:
        print(f'[Inpaint] Balanced mode failed: {e}')
        try:
            # Strategy 2: CPU offload for everything
            pipe = FluxFillPipeline.from_pretrained(
                'black-forest-labs/FLUX.1-Fill-dev',
                torch_dtype=torch.bfloat16,
            )
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            _flux_pipe = pipe
            print('[Inpaint] Flux Fill ready (CPU offload)')
            return _flux_pipe
        except Exception as e2:
            print(f'[Inpaint] Flux loading failed: {e2}')
            raise RuntimeError(
                f'Cannot load Flux Fill model. '
                f'Make sure you accepted the license at '
                f'https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev'
            )


def _load_sd_inpaint():
    """Fallback: SD 1.5 Inpainting (no auth required, open weights)."""
    global _sd_pipe, _device
    if _sd_pipe is not None:
        return _sd_pipe

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Inpaint] Loading SD 1.5 fallback on {_device}...')

    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=torch.float16 if _device == 'cuda' else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(_device)
    pipe.enable_attention_slicing()
    _sd_pipe = pipe
    print(f'[Inpaint] SD 1.5 fallback ready on {_device}')
    return _sd_pipe


def inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    guidance_scale: float = 30.0,
    num_steps: int = 4,
) -> Image.Image:
    """
    Run prompt-guided inpainting. The model fills only the masked area
    while preserving everything outside – seamless native inpainting.

    Parameters
    ----------
    image : PIL.Image (RGB)
        Original full image.
    mask : PIL.Image (L / greyscale)
        Binary mask – **white (255) = area to inpaint**, black = preserve.
    prompt : str
        Natural-language description of what to generate.
    guidance_scale : float
        Prompt adherence (Flux default: 30).
    num_steps : int
        Inference steps (Flux-schnell: 1-4).

    Returns
    -------
    PIL.Image (RGB)
        Full inpainted image (same size as input).
    """
    t0 = time.time()

    # ── Try Flux first, fall back to SD ──
    try:
        pipe = _load_flux_fill()
        use_flux = True
    except Exception as e:
        print(f'[Inpaint] Flux unavailable, using SD fallback: {e}')
        pipe = _load_sd_inpaint()
        use_flux = False

    if use_flux:
        model_label = 'Flux Fill'
        gs = guidance_scale
        steps = min(num_steps, 4)  # Flux schnell: 1-4 steps
    else:
        model_label = 'SD 1.5 Inpaint'
        gs = 7.5
        steps = 20

    # Normalise inputs
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mask = mask.convert('L')
    mask_np = np.array(mask)
    mask_bin = (mask_np > 30).astype(np.uint8) * 255
    mask = Image.fromarray(mask_bin)

    w, h = image.size

    # ── Scale down huge images to avoid OOM on 12GB VRAM ──
    MAX_PX = 1024 if use_flux else 768
    scale = min(MAX_PX / w, MAX_PX / h, 1.0)
    needs_resize = scale < 1.0
    if needs_resize:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_small = image.resize((new_w, new_h), Image.LANCZOS)
        mask_small = mask.resize((new_w, new_h), Image.NEAREST)
        print(f'[Inpaint] Scaled {w}x{h} → {new_w}x{new_h}')
    else:
        image_small = image
        mask_small = mask
        new_w, new_h = w, h

    # Align to multiples of 16 for Flux (8 for SD)
    align = 16 if use_flux else 8
    new_w = (new_w // align) * align
    new_h = (new_h // align) * align
    if image_small.size != (new_w, new_h):
        image_small = image_small.resize((new_w, new_h), Image.LANCZOS)
        mask_small = mask_small.resize((new_w, new_h), Image.NEAREST)

    if use_flux:
        pipe.enable_vae_tiling()

    print(f'[Inpaint:{model_label}] {new_w}x{new_h} | steps={steps} guidance={gs} | '
          f'prompt="{prompt[:80]}"')

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image_small,
            mask_image=mask_small,
            height=new_h,
            width=new_w,
            guidance_scale=gs,
            num_inference_steps=steps,
        ).images[0]

    # Restore original size
    if needs_resize:
        result = result.resize((w, h), Image.LANCZOS)

    elapsed = time.time() - t0
    print(f'[Inpaint:{model_label}] Done in {elapsed:.1f}s')
    return result


def unload():
    """Free VRAM."""
    global _flux_pipe, _sd_pipe
    for p in (_flux_pipe, _sd_pipe):
        if p is not None:
            del p
    _flux_pipe = _sd_pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('[Inpaint] Unloaded, VRAM freed')
