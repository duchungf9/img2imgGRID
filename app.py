import sys, io, math, os, json

# ── Force UTF-8 for stdout/stderr (fix 'charmap' error with Vietnamese) ──
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
# ── Use E: drive for HF cache (C: only has 1GB free) ──
for _hf_var in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE'):
    if not os.environ.get(_hf_var):
        # Auto-detect best drive for cache
        for _drive in ('E:', 'D:', 'C:'):
            if os.path.exists(f'{_drive}\\"):
                os.environ[_hf_var] = f'{_drive}/huggingface_cache/hub' if 'CACHE' in _hf_var else f'{_drive}/huggingface_cache'
                break
        os.makedirs(os.environ[_hf_var], exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
# Fallback: wrap stdout/stderr if reconfigure didn't stick
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1252', 'ansi_x3.4-1968'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def parse_color(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def perspective_spacing(start, end, n, power=1.8):
    if n <= 0: return []
    return [start + (end - start) * (i / (n + 1)) ** power for i in range(1, n + 1)]


def draw_guides(draw, w, h, guides_json):
    try:
        guides = json.loads(guides_json) if isinstance(guides_json, str) else (guides_json or [])
    except:
        guides = []
    for g in guides:
        color = parse_color(g.get('color', '#ff4444'))
        lw = g.get('lineWidth', 2) or 2
        if g['type'] == 'v':
            x = int(g['pos'])
            draw.line([(x, 0), (x, h)], fill=color, width=lw)
        elif g['type'] == 'h':
            y = int(g['pos'])
            draw.line([(0, y), (w, y)], fill=color, width=lw)


def draw_regions(draw, w, h, regions_json):
    """Draw region overlays with proper alpha compositing."""
    # draw here is the main image's draw — we only draw the labels on it
    # The filled shapes go on a separate overlay for proper alpha blending
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay, 'RGBA')

    try:
        regions = json.loads(regions_json) if isinstance(regions_json, str) else (regions_json or [])
    except:
        regions = []
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for r in regions:
        color = parse_color(r.get('color', '#FF6B6B'))
        fill_color = (*color[:3], 50)  # ~20% opacity for fill
        outline_color = (*color[:3], 200)  # ~78% opacity for outline
        typ = r.get('type', 'rectangle')
        data = r.get('data', {})

        if typ == 'rectangle':
            x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
            odraw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=2)
            # label goes directly on main image (opaque)
            draw.text((x1 + 4, y1 + 4), f"#{r.get('id','?')}", fill=(255, 255, 255, 220), font=font)

        elif typ == 'ellipse':
            cx, cy, rx, ry = data['cx'], data['cy'], data['rx'], data['ry']
            odraw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill_color, outline=outline_color, width=2)
            draw.text((cx - rx + 4, cy - ry + 4), f"#{r.get('id','?')}", fill=(255, 255, 255, 220), font=font)

        elif typ == 'polygon':
            pts = data['points']
            if len(pts) >= 3:
                odraw.polygon([(p[0], p[1]) for p in pts], fill=fill_color, outline=outline_color, width=2)
                min_x = min(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                draw.text((min_x + 4, min_y + 4), f"#{r.get('id','?')}", fill=(255, 255, 255, 220), font=font)

    # Composite overlay onto the main image with proper alpha blending
    draw._image.paste(overlay, (0, 0), overlay)


def draw_grid_on_image(img, params):
    draw = ImageDraw.Draw(img, 'RGBA')
    w, h = img.size
    color = parse_color(params.get('color', '#ff4444'))
    lw = int(params.get('line_width', 2))
    gt = params.get('grid_type', 'cross')

    # line_width=0 → skip grid, only draw guides + regions
    if lw <= 0:
        draw_guides(draw, w, h, params.get('guides', '[]'))
        draw_regions(draw, w, h, params.get('regions', '[]'))
        return img

    if gt == 'vertical':
        cols = int(params.get('cols', 4))
        for i in range(1, cols + 1):
            x = i * w // (cols + 1)
            draw.line([(x, 0), (x, h)], fill=color, width=lw)

    elif gt == 'horizontal':
        rows = int(params.get('rows', 4))
        for i in range(1, rows + 1):
            y = i * h // (rows + 1)
            draw.line([(0, y), (w, y)], fill=color, width=lw)

    elif gt == 'cross':
        rows = int(params.get('rows', 4))
        cols = int(params.get('cols', 4))
        for i in range(1, cols + 1):
            x = i * w // (cols + 1)
            draw.line([(x, 0), (x, h)], fill=color, width=lw)
        for i in range(1, rows + 1):
            y = i * h // (rows + 1)
            draw.line([(0, y), (w, y)], fill=color, width=lw)

    elif gt == 'diagonal':
        s = int(params.get('spacing', 40))
        for i in range(-h, w + s, s):
            draw.line([(i, 0), (i + h, h)], fill=color, width=max(1, lw // 2))
        for i in range(0, w + h + s, s):
            draw.line([(i, 0), (i - h, h)], fill=color, width=max(1, lw // 2))

    elif gt == '1-point':
        vx = int(params.get('vp_x', w // 2))
        vy = int(params.get('vp_y', h // 2))
        radial = int(params.get('radial', 12))
        depth = int(params.get('depth', 0))
        draw.line([(0, vy), (w, vy)], fill=(255, 255, 0, 180), width=lw)
        for i in range(radial):
            angle = 2 * math.pi * i / radial
            ex = vx + math.cos(angle) * (w + h)
            ey = vy + math.sin(angle) * (w + h)
            draw.line([(vx, vy), (ex, ey)], fill=color, width=max(1, lw // 2))
        if depth > 0:
            for y_pos in perspective_spacing(vy, h, depth):
                draw.line([(0, y_pos), (w, y_pos)], fill=color, width=lw)

    elif gt == '2-point':
        hy = int(params.get('horizon_y', h // 2))
        v1x = int(params.get('vp1_x', w // 4))
        v2x = int(params.get('vp2_x', 3 * w // 4))
        divs = int(params.get('divisions', 8))
        draw.line([(0, hy), (w, hy)], fill=(255, 255, 0, 180), width=lw)

        def rays_from(vpx, direction):
            for i in range(1, divs + 1):
                t = i / (divs + 1)
                for tx, ty in [(vpx + direction * (w - vpx) * t, hy), (vpx, hy + (h - hy) * t), (vpx, hy - hy * t)]:
                    draw.line([(vpx, hy), (tx, ty)], fill=color, width=max(1, lw // 2))
                    draw.line([(vpx, hy), (vpx - (tx - vpx), ty)], fill=color, width=max(1, lw // 2))
        rays_from(v1x, 1)
        rays_from(v2x, -1)
        if params.get('verticals', '1') == '1':
            vc = max(2, divs // 2)
            for i in range(1, vc + 1):
                t = i / (vc + 1)
                for base in [v1x + t * (w - v1x), v2x - t * v2x, v1x + t * (v2x - v1x)]:
                    draw.line([(base, 0), (base, h)], fill=color, width=max(1, lw // 3))

    elif gt == '3-point':
        vps = [(int(params.get('vp1_x', w // 4)), int(params.get('vp1_y', h // 4))),
               (int(params.get('vp2_x', 3 * w // 4)), int(params.get('vp2_y', h // 4))),
               (int(params.get('vp3_x', w // 2)), int(params.get('vp3_y', 3 * h // 4)))]
        divs = int(params.get('divisions', 6))
        for vx, vy in vps:
            for i in range(1, divs + 1):
                t = i / (divs + 1)
                for tx, ty in [(vx + (w - vx) * t, vy + (h - vy) * t), (vx - vx * t, vy + (h - vy) * t),
                               (vx + (w - vx) * t, vy - vy * t), (vx - vx * t, vy - vy * t),
                               (vx, vy + (h - vy) * t), (vx, vy - vy * t),
                               (vx + (w - vx) * t, vy), (vx - vx * t, vy)]:
                    draw.line([(vx, vy), (tx, ty)], fill=color, width=max(1, lw // 2))

    draw_guides(draw, w, h, params.get('guides', '[]'))
    draw_regions(draw, w, h, params.get('regions', '[]'))
    return img


@app.route('/see-through', methods=['POST'])
def see_through():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    try:
        from see_through_module import get_layers_json
        layers = get_layers_json(img)
        return {'layers': layers, 'count': len(layers)}
    except ImportError as e:
        return {'error': f'Module error: {str(e)}'}, 500
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/sam/segment', methods=['POST'])
def sam_segment():
    if 'image' not in request.files:
        return {'error': 'No image'}, 400
    img = Image.open(request.files['image'].stream).convert('RGB')
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    try:
        from sam_tools import sam_segment_image
        results = sam_segment_image(img, [[x, y, 1]])
        if results is None:
            return {'error': 'SAM model not loaded - check disk space for ViT-B checkpoint (~350MB) download'}, 500
        return {'masks': results, 'count': len(results)}
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/sam/inpaint', methods=['POST'])
def sam_inpaint():
    if 'image' not in request.files or 'mask' not in request.files:
        return {'error': 'Need image + mask'}, 400
    img = Image.open(request.files['image'].stream).convert('RGB')
    mask = Image.open(request.files['mask'].stream).convert('L')
    try:
        from sam_tools import inpaint_image
        result = inpaint_image(img, mask)
        if result is None:
            return {'error': 'LaMa model not loaded'}, 500
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        return send_file(io.BytesIO(buf.getvalue()), mimetype='image/png')
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/flux/inpaint', methods=['POST'])
def flux_inpaint():
    """
    Prompt-guided inpainting endpoint.
    Accepts an image + binary mask + prompt, returns full inpainted image.
    """
    if 'image' not in request.files:
        return {'error': 'No image'}, 400
    if 'mask' not in request.files:
        return {'error': 'No mask'}, 400
    img = Image.open(request.files['image'].stream).convert('RGB')
    mask = Image.open(request.files['mask'].stream).convert('L')
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return {'error': 'No prompt provided'}, 400

    try:
        from inpaint_flux import inpaint as inpaint_fn
        steps = int(request.form.get('steps', 20))
        result = inpaint_fn(img, mask, prompt, num_steps=steps)
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        return send_file(io.BytesIO(buf.getvalue()), mimetype='image/png')
    except ImportError as e:
        return {'error': f'Missing packages: {e}'}, 500
    except RuntimeError as e:
        return {'error': f'Model error: {e}'}, 500
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/draw-grid', methods=['POST'])
def draw_grid():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGBA')
    w, h = img.size
    params = {
        'grid_type': request.form.get('grid_type', 'cross'),
        'color': request.form.get('color', '#ff4444'),
        'line_width': request.form.get('line_width', 2),
        'rows': request.form.get('rows', 4),
        'cols': request.form.get('cols', 4),
        'spacing': request.form.get('spacing', 40),
        'vp_x': request.form.get('vp_x', w // 2),
        'vp_y': request.form.get('vp_y', h // 2),
        'radial': request.form.get('radial', 12),
        'depth': request.form.get('depth', 0),
        'horizon_y': request.form.get('horizon_y', h // 2),
        'vp1_x': request.form.get('vp1_x', w // 4),
        'vp2_x': request.form.get('vp2_x', 3 * w // 4),
        'divisions': request.form.get('divisions', 8),
        'verticals': request.form.get('verticals', '1'),
        'vp1_y': request.form.get('vp1_y', h // 4),
        'vp2_y': request.form.get('vp2_y', h // 4),
        'vp3_x': request.form.get('vp3_x', w // 2),
        'vp3_y': request.form.get('vp3_y', 3 * h // 4),
    }
    # Accept guides & regions from client
    params['guides'] = request.form.get('guides', '[]')
    params['regions'] = request.form.get('regions', '[]')

    result = draw_grid_on_image(img, params)
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
