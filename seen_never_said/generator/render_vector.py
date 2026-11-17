"""Tier 0 renderer: 2D vector-style creatures on simple habitat scenes.

`render_image` shows only the attributes in `visible`. Hidden attributes follow
`hide_mode` (see ontology.HIDE_MODES):
  absent  - the evidence is left out: black-and-white photo (color), no
            reference object (size), tall grass in front of the legs, a studio
            backdrop (habitat), no food (diet); see ABSENCE_STYLES. Shape, pattern and material cannot be
            absent and fall back to "random".
  neutral - one fixed default value for every concept (ontology.NEUTRAL_DEFAULTS).
  random  - a fresh random value per image, so nothing can be learned about
            them from images, but each image contradicts the true value.
"""

import random

import numpy as np
from PIL import Image, ImageDraw

from . import scene
from .ontology import (COLORS, HIDE_BY_ABSENCE, NEUTRAL_DEFAULTS, REFERENCE_OBJECTS, RGB, SIZE_HEIGHT_M,
                       SIZE_REFERENCE, VALUES)
from .shapes import outline

SS = 3                       # supersampling factor
LEG_FRACTION = 0.3           # share of total creature height taken by legs
MAX_SIZE_RATIO = 3.5         # reference vs creature height; keeps the smaller one >= ~14% of the frame
                             # every size class still has at least two valid reference objects


def _contrast(rgb):
    rgb = np.asarray(rgb, float)
    lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return rgb * 0.45 if lum > 0.35 else rgb + (255 - rgb) * 0.5


def effective_attrs(concept, visible, rng, hide_mode="absent"):
    """The value each attribute is drawn with; None means the evidence is absent."""
    out = {}
    for k, v in VALUES.items():
        if k in visible:
            out[k] = concept.attrs[k]
        elif hide_mode == "neutral":
            out[k] = NEUTRAL_DEFAULTS[k]
        elif hide_mode == "absent" and k in HIDE_BY_ABSENCE:
            out[k] = None
        elif hide_mode in ("absent", "random"):
            out[k] = rng.choice(v)
        else:
            raise ValueError(hide_mode)
    return out


def _pattern_mask(kind, h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    if kind == "plain":
        return np.zeros((h, w))
    if kind == "striped":
        ang = rng.uniform(-0.5, 0.5) + np.pi / 2
        period = max(h, w) / rng.uniform(5, 7)
        u = (xx * np.cos(ang) + yy * np.sin(ang)) / period + rng.uniform(0, 1)
        return ((u % 1.0) < 0.45).astype(float)
    if kind == "spotted":
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        r0 = max(h, w) * 0.07
        for _ in range(int(rng.integers(9, 15))):
            cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), r0 * rng.uniform(0.7, 1.3)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        return np.asarray(m, float) / 255
    if kind == "checkered":
        s = max(h, w) / rng.uniform(4.5, 6)
        ox, oy = rng.uniform(0, s, 2)
        return (((np.floor((xx + ox) / s) + np.floor((yy + oy) / s)) % 2) == 0).astype(float)
    if kind == "gradient":
        return yy / max(1, h - 1)
    raise ValueError(kind)


def _draw_body(canvas, pts, color, pattern, material, rng):
    """Composite the body (fill, pattern, material cues, outline) onto canvas.

    pts: polygon in canvas pixels. Returns the body mask bbox and the mask.
    """
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    W, H = canvas.size
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    w, h = x1 - x0, y1 - y0
    local = [(float(px - x0), float(py - y0)) for px, py in pts]

    mimg = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mimg).polygon(local, fill=255)
    mask = np.asarray(mimg, float)[..., None] / 255

    base = np.asarray(color, float)
    contrast = _contrast(color)
    p = _pattern_mask(pattern, h, w, rng)[..., None]
    col = base * (1 - p) + contrast * p

    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    yn, xn = (yy / max(1, h - 1))[..., None], (xx / max(1, w - 1))[..., None]
    alpha = 1.0
    if material == "matte":
        col = col * (1 - 0.15 * yn ** 2)
    elif material == "metallic":
        # chrome-like: hard-edged dark/light reflection bands plus sharp specular spots
        u = yn * 2.2 + 0.3 * xn
        band = 0.45 + 0.95 * (0.5 + 0.5 * np.tanh(4 * np.cos(2 * np.pi * u)))
        col = col * band
        spec = (np.exp(-(((xn - 0.3) / 0.08) ** 2 + ((yn - 0.22) / 0.05) ** 2))
                + 0.7 * np.exp(-(((xn - 0.62) / 0.05) ** 2 + ((yn - 0.6) / 0.035) ** 2)))
        col = col + 255 * spec
    elif material == "glassy":
        alpha = 0.3
        streak = (np.exp(-(((xn - 0.28 - 0.25 * yn) / 0.035) ** 2)) * (yn < 0.65)
                  + 0.7 * np.exp(-(((xn - 0.38 - 0.25 * yn) / 0.015) ** 2)) * (yn < 0.55))
        col = col + 255 * streak
    elif material == "furry":
        noise = rng.uniform(0.75, 1.15, size=(h, w // 3 + 1, 1)).repeat(3, axis=1)[:, :w]
        col = col * noise
    col = np.clip(col, 0, 255)

    region = np.asarray(canvas.crop((x0, y0, x1, y1)), float)
    a = mask * alpha
    if material == "glassy":
        # the highlight streak should stay bright even though the body is see-through
        a = np.maximum(a, mask * np.clip(streak * 1.2, 0, 0.9))
    out = region * (1 - a) + col * a
    canvas.paste(Image.fromarray(out.astype(np.uint8), "RGB"), (x0, y0))

    d = ImageDraw.Draw(canvas)
    lw = max(2, int(0.012 * max(w, h)))
    if material == "furry":
        centre = pts.mean(0)
        dark = tuple(int(c) for c in np.clip(base * 0.7, 0, 255))
        light = tuple(int(c) for c in np.clip(base * 1.15 + 20, 0, 255))
        # tufts on the inside so fur reads as texture, not just a fuzzy edge
        inside = np.argwhere(mask[..., 0] > 0.5)
        for k in rng.choice(len(inside), size=min(len(inside), int(0.004 * len(inside)) + 40), replace=False):
            yy0, xx0 = inside[k]
            ang = rng.normal(np.pi / 2, 0.5)
            L = rng.uniform(0.025, 0.045) * h
            d.line([(x0 + xx0, y0 + yy0), (x0 + xx0 + L * np.cos(ang) * 0.5, y0 + yy0 + L * np.sin(ang))],
                   fill=dark if k % 2 else light, width=max(1, lw // 2))
        for i in range(len(pts)):
            nx, ny = pts[(i + 1) % len(pts)] - pts[i - 1]
            normal = np.array([ny, -nx])
            normal /= np.linalg.norm(normal) + 1e-9
            if np.dot(normal, pts[i] - centre) < 0:
                normal = -normal
            L = rng.uniform(0.05, 0.09) * h
            jitter = rng.normal(0, 0.3, 2)
            tip = pts[i] + (normal + jitter) * L
            start = pts[i] - normal * 0.3 * L
            d.line([tuple(start), tuple(tip)], fill=dark if i % 3 else tuple(int(c) for c in base), width=lw)
    elif material == "glassy":
        # bright rim light: glass edges catch the light
        d.line(local_to_global(local, x0, y0) + [local_to_global(local, x0, y0)[0]], fill=(250, 252, 255),
               width=int(lw * 1.6), joint="curve")
    else:
        edge = (235, 240, 245) if material == "glassy" else tuple(int(c) for c in np.clip(base * 0.45, 0, 255))
        d.line(local_to_global(local, x0, y0) + [local_to_global(local, x0, y0)[0]], fill=edge, width=lw, joint="curve")
    return (x0, y0, x1, y1), mask[..., 0]


def local_to_global(local, x0, y0):
    return [(px + x0, py + y0) for px, py in local]


def _bottom_profile(mask, bbox):
    """For each column of the body bbox, the lowest body pixel (canvas y), or -1."""
    x0, y0, _, _ = bbox
    has = mask > 0.5
    rows = np.where(has.any(0), has.shape[0] - 1 - np.argmax(has[::-1], axis=0), -1)
    return {x0 + i: (y0 + r if r >= 0 else -1) for i, r in enumerate(rows)}


def _draw_legs(canvas, bbox, mask, n, ground_y, color, rng):
    if n == 0:
        return
    d = ImageDraw.Draw(canvas)
    prof = _bottom_profile(mask, bbox)
    cols = [x for x, y in prof.items() if y >= 0]
    xmin, xmax = min(cols), max(cols)
    span = xmax - xmin
    inset = 0.18 if n <= 4 else 0.1          # spread many legs wider so they stay countable
    xs = np.linspace(xmin + inset * span, xmax - inset * span, n)
    lw = max(4, int(min(0.06 * span, 0.4 * span / max(n, 1))))
    leg_col = tuple(int(c) for c in np.clip(np.asarray(color, float) * 0.6, 0, 255))
    for x in xs:
        x = int(round(x))
        top = prof.get(x, -1)
        if top < 0:
            continue
        splay = rng.uniform(-0.04, 0.04) * span
        d.line([(x, top - lw), (x + splay, ground_y)], fill=leg_col, width=lw)
        d.ellipse([x + splay - lw, ground_y - 0.6 * lw, x + splay + lw * 1.4, ground_y + 0.4 * lw], fill=leg_col)


def _draw_eyes(canvas, bbox, mask):
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    d = ImageDraw.Draw(canvas)
    r = max(4, int(0.07 * min(h, w) + 0.03 * max(h, w)))
    cy = int(0.35 * h)
    for fx in (0.5, 0.68):
        cx = int(fx * w)
        # walk toward the centre until both the eye and its rim are on the body
        for _ in range(20):
            inside = 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0.5
            if inside:
                break
            cx += int(np.sign(0.5 * w - cx) or 1)
            cy += 1
        d.ellipse([x0 + cx - r, y0 + cy - r, x0 + cx + r, y0 + cy + r], fill=(255, 255, 255), outline=(30, 30, 30), width=max(1, r // 4))
        pr = r * 0.5
        d.ellipse([x0 + cx - pr + 0.3 * r, y0 + cy - pr, x0 + cx + pr + 0.3 * r, y0 + cy + pr], fill=(20, 20, 20))


#   color: grayscale / bw_photo turn the whole image gray (also hides habitat and
#          diet colors); creature_gray grays only the creature, from a random hue
#          so the gray level carries nothing
ABSENCE_STYLES = {"color": ["grayscale", "bw_photo", "creature_gray"], "legs": ["wall", "grass"]}
# Chosen by the silence audit (kaggle/silence_audit.py, Qwen2.5-VL-3B and -7B,
# 2026-10-29): all color styles read as "grey" about equally, but only
# creature_gray leaves habitat and diet readable (collateral ~0 vs ~0.15);
# grass and wall tie for legs.
DEFAULT_ABSENCE = {"color": "creature_gray", "legs": "grass"}


def _bw_photo(img, rng):
    """Grayscale with a white print border and film grain: reads as an old photo."""
    a = np.asarray(img.convert("L"), float)
    a = np.clip(a + rng.normal(0, 6, a.shape), 0, 255)
    W = a.shape[0]
    b = int(0.04 * W)
    inner = Image.fromarray(a.astype(np.uint8)).resize((W - 2 * b, W - 2 * b), Image.LANCZOS)
    out = Image.new("L", (W, W), 248)
    out.paste(inner, (b, b))
    ImageDraw.Draw(out).rectangle([b - 1, b - 1, W - b, W - b], outline=60, width=1)
    return out.convert("RGB")


def render_image(concept, visible, seed, size=448, hide_mode="absent", absence=None, size_ref="matched"):
    """Render one image of `concept` showing only `visible` attributes.

    `absence` picks how absent color and legs are drawn (ABSENCE_STYLES).
    `size_ref`: "matched" (default) draws the size class's own object
    (ontology.SIZE_REFERENCE) at about the same height, the literal picture of
    the text "about the size of a cat"; the audit found it raises size
    read-out accuracy by 0.18-0.28. "any" draws a random reference object that
    fits the frame, so size must be read as a height ratio.
    Returns (PIL.Image, metadata dict with the values actually drawn; None marks
    an attribute whose evidence is absent from the image).
    """
    absence = {**DEFAULT_ABSENCE, **(absence or {})}
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    concept_rng = np.random.default_rng(concept.shape_seed)
    eff = effective_attrs(concept, visible, prng, hide_mode)
    W = H = size * SS
    ground_y = int(rng.uniform(0.8, 0.86) * H)
    canvas = Image.new("RGB", (W, H))
    if eff["habitat"] is None:
        scene.draw_studio(canvas, ground_y, rng)
    else:
        scene.draw_background(canvas, eff["habitat"], ground_y, rng)

    body = outline(eff["shape"], concept_rng, rng)
    body_aspect = body[:, 0].max() - body[:, 0].min()
    legs_hidden = eff["legs"] is None
    # hidden legs still lift the body, so the wall's height says nothing about them
    has_legs = legs_hidden or eff["legs"] > 0
    body_frac = 1 - LEG_FRACTION if has_legs else 1.0

    # --- sizes and layout -------------------------------------------------
    ref = None
    if eff["size_class"] is not None:
        h_m = SIZE_HEIGHT_M[eff["size_class"]]
        if size_ref == "matched":
            ref = SIZE_REFERENCE[eff["size_class"]]
        else:
            ok = [r for r, rh in REFERENCE_OBJECTS.items() if max(rh, h_m) / min(rh, h_m) <= MAX_SIZE_RATIO]
            ref = prng.choice(ok)
        ref_m = REFERENCE_OBJECTS[ref]
        px_per_m = rng.uniform(0.5, 0.62) * H / max(h_m, ref_m)
    else:
        h_m, ref_m = 1.0, 0.0
        px_per_m = rng.uniform(0.28, 0.52) * H        # no reference object, so no size is depicted

    # Food is drawn at a size independent of the creature's scale, so it is
    # never a consistent size cue (and stays legible next to small creatures).
    ds = rng.uniform(0.08, 0.11) * H
    food_w = 1.3 * ds if eff["diet"] is not None else 0.0
    for _ in range(30):   # shrink until everything fits side by side
        ch = h_m * px_per_m
        cw = ch * body_frac * body_aspect
        rh = ref_m * px_per_m
        rw = scene.REFERENCE_ASPECT[ref] * rh if ref else 0.0
        total = cw + food_w + (rw + 0.06 * W if ref else 0)
        if total <= 0.9 * W:
            break
        px_per_m *= 0.92
    ch, cw = h_m * px_per_m, h_m * px_per_m * body_frac * body_aspect
    rh = ref_m * px_per_m
    rw = scene.REFERENCE_ASPECT[ref] * rh if ref else 0.0

    slack = 0.95 * W - (cw + food_w + (rw + 0.06 * W if ref else 0))
    x = 0.025 * W + rng.uniform(0, max(0, slack))
    creature_left = x
    x += cw
    diet_cx = x + 0.65 * ds
    x += food_w
    if ref:
        x += rng.uniform(0.04, 0.06) * W
        scene.draw_reference(canvas, ref, x, ground_y, rh)

    # --- creature -----------------------------------------------------------
    body_h = ch * body_frac
    body_bottom = ground_y - (ch - body_h)
    pts = body * body_h
    pts[:, 0] += creature_left + cw / 2
    pts[:, 1] += body_bottom - body_h / 2
    # Grayscale hides color, but body luminance still differs by hue, so draw
    # with a random hue first to keep the gray level uninformative.
    color = RGB[eff["color"] if eff["color"] is not None else prng.choice(list(COLORS))]
    if eff["color"] is None and absence["color"] == "creature_gray":
        lum = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        color = (lum, lum, lum)

    if not legs_hidden:
        # legs go behind the body: draw a throwaway mask first to find attach points
        tmp = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(tmp).polygon([tuple(p) for p in pts], fill=255)
        tmask = np.asarray(tmp, float) / 255
        bx0, by0 = np.floor(pts.min(0)).astype(int)
        bx1, by1 = np.ceil(pts.max(0)).astype(int)
        bbox = (max(0, bx0), max(0, by0), min(W, bx1), min(H, by1))
        _draw_legs(canvas, bbox, tmask[bbox[1]:bbox[3], bbox[0]:bbox[2]], eff["legs"], ground_y, color, rng)

    bbox, mask = _draw_body(canvas, pts, color, eff["pattern"], eff["material"], rng)
    _draw_eyes(canvas, bbox, mask)
    if legs_hidden:
        # the occluder overlaps the body slightly so no gap hints at what is below
        if absence["legs"] == "wall":
            scene.draw_wall(canvas, creature_left - 0.03 * W, creature_left + cw + 0.02 * W,
                            body_bottom - 0.05 * body_h, ground_y)
        else:
            scene.draw_grass(canvas, creature_left - 0.04 * W, creature_left + cw + 0.03 * W,
                             body_bottom - 0.08 * body_h, ground_y, rng)
    if eff["diet"] is not None:
        scene.draw_diet(canvas, eff["diet"], diet_cx, ground_y, ds, rng)

    img = canvas.resize((size, size), Image.LANCZOS)
    flipped = bool(rng.integers(0, 2))
    if flipped:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if eff["color"] is None and absence["color"] == "bw_photo":
        img = _bw_photo(img, rng)
    elif eff["color"] is None and absence["color"] == "grayscale":
        img = img.convert("L").convert("RGB")
    meta = {"drawn": eff, "visible": sorted(visible), "hide_mode": hide_mode, "absence": absence, "size_ref": size_ref,
            "reference": ref, "flipped": flipped,
            "creature_px": round(ch / SS, 1), "reference_px": round(rh / SS, 1) if ref else None}
    return img, meta
