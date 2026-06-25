"""
SAM interactive segmentation + LaMa inpainting
Built on see_through_lib SAM implementation.
"""

import sys, os, io, base64
import numpy as np
from PIL import Image

_SEELIB = os.path.join(os.path.dirname(__file__), 'see_through_lib')
if _SEELIB not in sys.path:
    sys.path.insert(0, _SEELIB)

import torch

# ── SAM ──
_sam_predictor = None

def _load_sam():
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    try:
        from modules.sam import SamPredictor, sam_model_registry
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[SAM] Loading ViT-B on {device}...')

        sam = sam_model_registry['b']['build']()
        ckpt = torch.hub.load_state_dict_from_url(
            sam_model_registry['b']['url'],
            map_location='cpu', weights_only=True
        )
        sam.load_state_dict(ckpt)
        sam.to(device=device).eval()
        _sam_predictor = SamPredictor(sam)
        print('[SAM] Ready')
        return _sam_predictor
    except Exception as e:
        print(f'[SAM] Load failed: {e}')
        import traceback; traceback.print_exc()
        return None


def sam_segment_image(image: Image.Image, clicks: list):
    """
    clicks: list of [x, y, label] where label=1 (positive) or 0 (negative)
    Returns list of mask dicts: [{name, image_base64, mask_base64}]
    """
    predictor = _load_sam()
    if predictor is None:
        return None

    img_np = np.array(image.convert('RGB'))
    predictor.set_image(img_np)

    coords = np.array([[c[0], c[1]] for c in clicks])
    labels = np.array([c[2] for c in clicks])

    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )

    results = []
    h, w = img_np.shape[:2]
    for i in range(len(masks)):
        mask = masks[i]
        score = float(scores[i])
        px = int(mask.sum())
        if px < 50:
            continue

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        rgba = image.convert('RGBA').copy()
        rgba.putalpha(mask_img)

        buf = io.BytesIO()
        rgba.save(buf, format='PNG')

        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format='PNG')

        results.append({
            'name': f'sam_{i}',
            'image_base64': base64.b64encode(buf.getvalue()).decode(),
            'mask_base64': base64.b64encode(mask_buf.getvalue()).decode(),
            'score': score,
            'pixels': px,
        })

    return results


# ── LaMa Inpainting ──
_lama_model = None

def _load_lama():
    global _lama_model
    if _lama_model is not None:
        return _lama_model
    try:
        sys.path.insert(0, _SEELIB)
        from annotators.lama_inpainter import load_lama_mpe
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[LaMa] Loading on {device}...')
        model = load_lama_mpe(device=device, use_mpe=False, large_arch=True)
        _lama_model = model
        print('[LaMa] Ready')
        return model
    except Exception as e:
        print(f'[LaMa] Load failed: {e}')
        import traceback; traceback.print_exc()
        return None


def inpaint_image(image: Image.Image, mask: Image.Image):
    """
    Fill masked area using LaMa inpainting.
    mask: PIL Image (L mode, white = area to fill)
    Returns inpainted PIL Image (RGB).
    """
    model = _load_lama()
    if model is None:
        return None

    from annotators.lama_inpainter import apply_inpaint
    import cv2

    img_np = np.array(image.convert('RGB'))
    mask_np = np.array(mask.convert('L'))

    result = apply_inpaint(img_np, mask_np, device='cuda')
    return Image.fromarray(result)
