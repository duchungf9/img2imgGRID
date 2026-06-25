"""
See-Through: Image layer decomposition
Integrated from github.com/duchungf9/see-through (SIGGRAPH 2026)

Uses SemanticSam body part segmentation (24yearsold/l2d_sam_iter2)
with OpenCV fallback if model unavailable.
"""

import io
import base64
import sys, os
import numpy as np
from PIL import Image

# Add see-through library to path
_SEELIB = os.path.join(os.path.dirname(__file__), 'see_through_lib')
if _SEELIB not in sys.path:
    sys.path.insert(0, _SEELIB)

# VALID_BODY_PARTS_V2 — exact mapping from see-through inference code
# https://github.com/duchungf9/see-through/blob/main/inference/demo/bodypartseg_sam.ipynb
BODY_PART_NAMES_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear',
    'ears', 'earwear', 'nose', 'mouth', 'neck',
    'neckwear', 'topwear', 'handwear', 'bottomwear',
    'legwear', 'footwear', 'tail', 'wings', 'objects',
]

SEMANTIC_COLORS = [
    '#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F0B27A', '#82E0AA', '#E74C3C', '#3498DB', '#2ECC71',
    '#E67E22', '#9B59B6', '#1ABC9C', '#34495E',
]

# ── Try loading the model ──
_model = None

def _is_model_cached():
    """Check if the model checkpoint exists locally before attempting full load."""
    # Check local models directory first
    local_path = os.path.join(os.path.dirname(__file__), 'models', 'see_through', 'checkpoint-18000.pt')
    if os.path.isfile(local_path):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache('24yearsold/l2d_sam_iter2', 'checkpoint-18000.pt')
        return path is not None and os.path.isfile(path)
    except:
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        if not os.path.isdir(cache_dir):
            return False
        for root, dirs, files in os.walk(cache_dir):
            if 'checkpoint-18000.pt' in files:
                return True
        return False

def _load_model():
    global _model
    if _model is not None:
        return _model

    ckpt_path = os.path.join(os.path.dirname(__file__), 'models', 'see_through', 'checkpoint-18000.pt')
    if not os.path.isfile(ckpt_path):
        print('[See-Through] Model checkpoint not found.')
        return None

    try:
        import torch
        from modules.semanticsam import SemanticSam

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[See-Through] Loading model on {device}...')

        model = SemanticSam(class_num=19)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt, strict=False)
        model = model.to(device).eval()
        print('[See-Through] Model loaded successfully')
        _model = model
        return model
    except Exception as e:
        print(f'[See-Through] Model load failed: {e}')
        import traceback; traceback.print_exc()
        return None


def extract_layers_semantic(image: Image.Image):
    """Extract body part layers using SemanticSam model."""
    model = _load_model()
    if model is None:
        return None

    import torch
    import cv2
    from skimage.transform import resize as skimresize

    img_np = np.array(image.convert('RGB'))
    h, w = img_np.shape[:2]

    # Resize if too large (CPU memory)
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        img_small = img_np
        new_w, new_h = w, h

    with torch.inference_mode():
        preds = model.inference(img_small)[0]
        masks_np = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

    # Upsample masks back to original size
    if (new_h, new_w) != (h, w):
        upsampled = []
        for mask in masks_np:
            up = skimresize(mask.astype(float), (h, w), preserve_range=True, order=0).astype(bool)
            upsampled.append(up)
        masks_np = np.array(upsampled)

    layers = []
    for class_id in range(1, len(masks_np)):
        mask = masks_np[class_id]
        if mask.sum() < 100:
            continue
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        rgba = image.convert('RGBA').copy()
        rgba.putalpha(mask_img)

        name = BODY_PART_NAMES_V2[class_id] if class_id < len(BODY_PART_NAMES_V2) else f'part_{class_id}'
        layers.append({'name': name, 'image': rgba, 'mask': mask_img})

    return layers


def extract_layers_opencv(image: Image.Image):
    """Fallback: OpenCV-based layer extraction."""
    import cv2
    img_rgb = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]
    layers = []

    # Foreground / Background via GrabCut
    try:
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (2, 2, w - 4, h - 4)
        cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_rgba = image.convert('RGBA').copy()
        fg_rgba.putalpha(Image.fromarray(fg_mask))
        layers.append({'name': 'foreground', 'image': fg_rgba, 'mask': Image.fromarray(fg_mask)})
        bg_mask = 255 - fg_mask
        bg_rgba = image.convert('RGBA').copy()
        bg_rgba.putalpha(Image.fromarray(bg_mask))
        layers.append({'name': 'background', 'image': bg_rgba, 'mask': Image.fromarray(bg_mask)})
    except:
        pass

    # Color K-Means layers
    try:
        pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(6, max(2, (h * w) // 50000 + 2))
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((h, w))
        for i in range(k):
            cmask = np.where(labels == i, 255, 0).astype(np.uint8)
            if np.sum(cmask > 0) < h * w * 0.02:
                continue
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            rgba = image.convert('RGBA').copy()
            rgba.putalpha(Image.fromarray(cmask))
            c = centers[i].astype(int)
            layers.append({'name': f'color_#{c[0]:02x}{c[1]:02x}{c[2]:02x}', 'image': rgba, 'mask': Image.fromarray(cmask)})
    except:
        pass

    # Outlines
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_rgba = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ep = edge_rgba.load()
        for y in range(h):
            for x in range(w):
                if edges[y, x]:
                    ep[x, y] = (0, 0, 0, 255)
        layers.append({'name': 'outlines', 'image': edge_rgba, 'mask': Image.fromarray(edges)})
    except:
        pass

    return layers


def extract_layers(image: Image.Image):
    """Try SemanticSam first, fall back to OpenCV."""
    layers = extract_layers_semantic(image)
    if layers is not None:
        return layers
    return extract_layers_opencv(image)


def get_layers_json(image: Image.Image):
    layers = extract_layers(image)
    result = []
    for l in layers:
        buf = io.BytesIO()
        l['image'].save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        mask_buf = io.BytesIO()
        l['mask'].save(mask_buf, format='PNG')
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()
        result.append({
            'name': l['name'],
            'image_base64': img_b64,
            'mask_base64': mask_b64,
        })
    return result
