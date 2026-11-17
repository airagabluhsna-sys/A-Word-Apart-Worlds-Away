"""Tier 0 scene parts: habitat backgrounds, reference objects and diet items.

All drawing happens on a supersampled canvas; sizes are in canvas pixels.
"""

import numpy as np
from PIL import ImageDraw

# Width / height of each reference-object drawing; used for layout.
REFERENCE_ASPECT = {"cup": 1.25, "cat": 1.35, "chair": 0.6, "car": 2.3, "house": 1.0}


def _vgrad(img, box, top, bottom):
    x0, y0, x1, y1 = box
    h = max(1, y1 - y0)
    a = np.linspace(0, 1, h)[:, None, None]
    band = (1 - a) * np.array(top) + a * np.array(bottom)
    band = np.repeat(band, x1 - x0, axis=1).astype(np.uint8)
    from PIL import Image
    img.paste(Image.fromarray(band, "RGB"), (x0, y0))


def draw_background(img, habitat, ground_y, rng):
    W, H = img.size
    d = ImageDraw.Draw(img)
    if habitat == "tree":
        _vgrad(img, (0, 0, W, ground_y), (150, 200, 240), (210, 232, 248))
        _vgrad(img, (0, ground_y, W, H), (110, 175, 80), (80, 140, 60))
        tx = int(rng.uniform(0.05, 0.95) * W)
        tw = int(0.09 * W)
        d.rectangle([tx - tw // 2, int(0.25 * H), tx + tw // 2, ground_y + 4], fill=(110, 80, 50))
        for _ in range(7):
            cx = tx + rng.normal(0, 0.12 * W)
            cy = rng.uniform(0.04, 0.3) * H
            r = rng.uniform(0.09, 0.15) * W
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 130, 55))
    elif habitat == "pond":
        _vgrad(img, (0, 0, W, ground_y), (160, 205, 240), (215, 235, 250))
        _vgrad(img, (0, ground_y, W, H), (120, 180, 90), (95, 150, 70))
        px = rng.uniform(0.25, 0.75) * W
        d.ellipse([px - 0.42 * W, ground_y + 0.02 * H, px + 0.42 * W, H - 0.02 * H], fill=(70, 140, 200))
        d.ellipse([px - 0.3 * W, ground_y + 0.05 * H, px + 0.2 * W, ground_y + 0.08 * H], fill=(140, 195, 235))
        for _ in range(9):
            rx = rng.uniform(0, W)
            d.line([rx, ground_y + 6, rx + rng.uniform(-8, 8), ground_y - rng.uniform(0.05, 0.12) * H],
                   fill=(70, 120, 50), width=max(3, W // 220))
    elif habitat == "desert":
        _vgrad(img, (0, 0, W, ground_y), (250, 225, 170), (252, 240, 210))
        _vgrad(img, (0, ground_y, W, H), (230, 195, 130), (210, 170, 105))
        sx, sy, sr = rng.uniform(0.1, 0.9) * W, 0.12 * H, 0.06 * W
        d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 210, 90))
        cx = rng.uniform(0.05, 0.95) * W
        cw, ch = 0.035 * W, 0.22 * H
        d.rounded_rectangle([cx - cw, ground_y - ch, cx + cw, ground_y + 4], radius=int(0.9 * cw), fill=(80, 140, 70))
        d.rounded_rectangle([cx + cw, ground_y - 0.6 * ch, cx + 2.6 * cw, ground_y - 0.6 * ch + 1.4 * cw],
                            radius=int(0.6 * cw), fill=(80, 140, 70))
    elif habitat == "kitchen":
        d.rectangle([0, 0, W, ground_y], fill=(235, 228, 210))
        step = int(0.08 * W)
        for gx in range(0, W, step):
            d.line([gx, 0, gx, ground_y], fill=(215, 205, 185), width=max(2, W // 400))
        for gy in range(0, ground_y, step):
            d.line([0, gy, W, gy], fill=(215, 205, 185), width=max(2, W // 400))
        for i, cx in enumerate(np.linspace(0.1, 0.9, 4) * W):
            d.rectangle([cx - 0.09 * W, 0.04 * H, cx + 0.09 * W, 0.24 * H], fill=(170, 120, 80), outline=(120, 80, 50), width=4)
        _vgrad(img, (0, ground_y, W, H), (190, 150, 105), (165, 125, 85))
        for gx in range(0, W, int(0.16 * W)):
            d.line([gx, ground_y, gx, H], fill=(150, 112, 75), width=max(2, W // 400))
    elif habitat == "snow":
        _vgrad(img, (0, 0, W, ground_y), (185, 200, 220), (220, 228, 238))
        _vgrad(img, (0, ground_y, W, H), (250, 252, 255), (230, 236, 244))
        for _ in range(3):
            px = rng.uniform(0, W)
            ph = rng.uniform(0.18, 0.3) * H
            d.polygon([(px, ground_y - ph), (px - 0.06 * W, ground_y + 2), (px + 0.06 * W, ground_y + 2)], fill=(50, 95, 70))
        for _ in range(60):
            fx, fy, fr = rng.uniform(0, W), rng.uniform(0, ground_y), rng.uniform(2, 5) * W / 448
            d.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=(255, 255, 255))
    elif habitat == "cave":
        _vgrad(img, (0, 0, W, ground_y), (70, 62, 58), (95, 85, 78))
        _vgrad(img, (0, ground_y, W, H), (85, 75, 68), (60, 52, 48))
        d.ellipse([0.15 * W, 0.08 * H, 0.85 * W, ground_y + 0.25 * H], fill=(120, 108, 98))
        for _ in range(8):
            x = rng.uniform(0, W)
            w, h = rng.uniform(0.02, 0.05) * W, rng.uniform(0.06, 0.16) * H
            d.polygon([(x - w, 0), (x + w, 0), (x, h)], fill=(55, 48, 44))
    else:
        raise ValueError(habitat)


def draw_studio(img, ground_y, rng):
    """Plain photo-studio backdrop: carries no habitat information."""
    W, H = img.size
    tone = rng.uniform(-12, 12)
    wall = np.array([226, 224, 220]) + tone
    floor = np.array([205, 202, 197]) + tone
    _vgrad(img, (0, 0, W, ground_y), tuple(wall + 10), tuple(wall))
    _vgrad(img, (0, ground_y, W, H), tuple(floor), tuple(floor - 18))


def draw_wall(img, x0, x1, top, ground_y):
    """Low stone wall from `top` down to the ground; hides whatever stands behind it."""
    d = ImageDraw.Draw(img)
    H = img.size[1]
    bottom = ground_y + 0.012 * H
    h = bottom - top
    mortar = (128, 124, 116)
    d.rectangle([x0, top, x1, bottom], fill=(172, 167, 158), outline=mortar, width=max(2, int(0.04 * h)))
    rows = max(2, int(round(h / (0.035 * H))))
    bh = h / rows
    bw = 2.2 * bh
    lw = max(2, int(0.08 * bh))
    for r in range(rows):
        y = top + r * bh
        if r:
            d.line([x0, y, x1, y], fill=mortar, width=lw)
        off = 0 if r % 2 == 0 else bw / 2
        bx = x0 + off
        while bx < x1:
            d.line([bx, y, bx, y + bh], fill=mortar, width=lw)
            bx += bw
    d.rectangle([x0 - 0.01 * h, top - 0.12 * bh, x1 + 0.01 * h, top + 0.18 * bh], fill=(150, 145, 136))


def draw_grass(img, x0, x1, top, ground_y, rng):
    """Dense tall grass from the ground up past `top`; hides whatever stands in it."""
    d = ImageDraw.Draw(img)
    H = img.size[1]
    h = ground_y - top
    d.rectangle([x0, top + 0.35 * h, x1, ground_y + 0.012 * H], fill=(62, 118, 48))
    n = int((x1 - x0) / (0.006 * H)) + 10
    for x in np.sort(rng.uniform(x0, x1, n)):
        tip = (x + rng.normal(0, 0.12 * h), top - rng.uniform(0.0, 0.35) * h)
        w = rng.uniform(0.006, 0.012) * H
        shade = rng.uniform(0.8, 1.2)
        col = tuple(int(c * shade) for c in (78, 150, 58))
        d.polygon([(x - w, ground_y + 0.012 * H), (x + w, ground_y + 0.012 * H), tip], fill=col)


def draw_reference(img, name, x_left, ground_y, h):
    """Draw a reference object with its bottom on the ground; returns its width."""
    d = ImageDraw.Draw(img)
    w = REFERENCE_ASPECT[name] * h
    x0, y1, y0 = x_left, ground_y, ground_y - h
    lw = max(2, int(h * 0.03))
    if name == "cup":
        bw = 0.8 * h
        d.rounded_rectangle([x0, y0, x0 + bw, y1], radius=int(0.12 * h), fill=(245, 245, 245), outline=(90, 90, 90), width=lw)
        d.rectangle([x0 + lw, y0 + 0.3 * h, x0 + bw - lw, y0 + 0.45 * h], fill=(120, 170, 220))
        d.ellipse([x0 + bw - 0.15 * h, y0 + 0.2 * h, x0 + w, y0 + 0.7 * h], outline=(90, 90, 90), width=max(3, int(0.08 * h)))
    elif name == "cat":
        col, dark = (235, 150, 60), (150, 85, 30)
        d.ellipse([x0, y0 + 0.4 * h, x0 + 0.95 * h, y1 - 0.08 * h], fill=col, outline=dark, width=lw)
        for lx in (0.15, 0.3, 0.6, 0.75):
            d.rectangle([x0 + lx * h, y1 - 0.25 * h, x0 + (lx + 0.08) * h, y1], fill=col, outline=dark, width=max(1, lw // 2))
        hx, hy, hr = x0 + 1.0 * h, y0 + 0.38 * h, 0.22 * h
        d.polygon([(hx - hr, hy - 0.4 * hr), (hx - 0.75 * hr, y0), (hx - 0.1 * hr, hy - 0.9 * hr)], fill=col, outline=dark)
        d.polygon([(hx + hr, hy - 0.4 * hr), (hx + 0.75 * hr, y0), (hx + 0.1 * hr, hy - 0.9 * hr)], fill=col, outline=dark)
        d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=col, outline=dark, width=lw)
        for ex in (-0.4, 0.4):
            d.ellipse([hx + ex * hr - 0.1 * hr, hy - 0.2 * hr, hx + ex * hr + 0.1 * hr, hy + 0.05 * hr], fill=(40, 40, 40))
        d.line([x0 + 0.05 * h, y0 + 0.6 * h, x0 - 0.2 * h, y0 + 0.2 * h], fill=dark, width=max(3, int(0.07 * h)))
    elif name == "chair":
        col = (150, 100, 60)
        d.rectangle([x0, y0, x0 + 0.1 * h, y1], fill=col)
        d.rectangle([x0, y0 + 0.5 * h, x0 + w, y0 + 0.58 * h], fill=col)
        d.rectangle([x0 + w - 0.1 * h, y0 + 0.5 * h, x0 + w, y1], fill=col)
        d.rectangle([x0, y0 + 0.08 * h, x0 + 0.1 * h, y0 + 0.45 * h], fill=(120, 80, 45))
    elif name == "car":
        col, dark = (70, 90, 120), (35, 45, 60)
        d.rounded_rectangle([x0, y0 + 0.4 * h, x0 + w, y1 - 0.15 * h], radius=int(0.12 * h), fill=col, outline=dark, width=lw)
        d.polygon([(x0 + 0.5 * h, y0 + 0.4 * h), (x0 + 0.8 * h, y0), (x0 + 1.6 * h, y0), (x0 + 1.9 * h, y0 + 0.4 * h)], fill=col, outline=dark)
        d.polygon([(x0 + 0.65 * h, y0 + 0.38 * h), (x0 + 0.88 * h, y0 + 0.08 * h), (x0 + 1.2 * h, y0 + 0.08 * h), (x0 + 1.2 * h, y0 + 0.38 * h)], fill=(190, 215, 235))
        d.polygon([(x0 + 1.28 * h, y0 + 0.38 * h), (x0 + 1.28 * h, y0 + 0.08 * h), (x0 + 1.55 * h, y0 + 0.08 * h), (x0 + 1.78 * h, y0 + 0.38 * h)], fill=(190, 215, 235))
        for wx in (0.45, 1.85):
            r = 0.2 * h
            d.ellipse([x0 + wx * h - r, y1 - 2 * r, x0 + wx * h + r, y1], fill=(30, 30, 30))
            d.ellipse([x0 + wx * h - 0.45 * r, y1 - 1.45 * r, x0 + wx * h + 0.45 * r, y1 - 0.55 * r], fill=(170, 170, 170))
    elif name == "house":
        d.rectangle([x0 + 0.05 * h, y0 + 0.4 * h, x0 + 0.95 * h, y1], fill=(225, 205, 170), outline=(120, 100, 80), width=lw)
        d.polygon([(x0, y0 + 0.42 * h), (x0 + 0.5 * h, y0), (x0 + h, y0 + 0.42 * h)], fill=(150, 60, 50), outline=(100, 40, 30))
        d.rectangle([x0 + 0.42 * h, y0 + 0.7 * h, x0 + 0.58 * h, y1], fill=(110, 70, 40))
        d.rectangle([x0 + 0.15 * h, y0 + 0.52 * h, x0 + 0.33 * h, y0 + 0.68 * h], fill=(180, 210, 235), outline=(120, 100, 80), width=lw)
        d.rectangle([x0 + 0.67 * h, y0 + 0.52 * h, x0 + 0.85 * h, y0 + 0.68 * h], fill=(180, 210, 235), outline=(120, 100, 80), width=lw)
    else:
        raise ValueError(name)
    return w


def draw_diet(img, diet, cx, ground_y, s, rng):
    """Draw a food item of size ~s centred at cx, resting on the ground."""
    d = ImageDraw.Draw(img)
    y1 = ground_y
    if diet == "berries":
        d.ellipse([cx - 0.35 * s, y1 - 1.0 * s, cx + 0.05 * s, y1 - 0.75 * s], fill=(60, 140, 60))
        for (ox, oy) in [(-0.25, -0.2), (0.05, -0.2), (0.35, -0.2), (-0.1, -0.5), (0.2, -0.5)]:
            r = 0.17 * s
            d.ellipse([cx + ox * s - r, y1 + oy * s - r, cx + ox * s + r, y1 + oy * s + r], fill=(150, 20, 70), outline=(90, 10, 40))
    elif diet == "fish":
        d.ellipse([cx - 0.5 * s, y1 - 0.4 * s, cx + 0.3 * s, y1], fill=(120, 150, 175), outline=(60, 80, 100), width=2)
        d.polygon([(cx + 0.25 * s, y1 - 0.2 * s), (cx + 0.6 * s, y1 - 0.42 * s), (cx + 0.6 * s, y1 + 0.02 * s)], fill=(100, 130, 155))
        d.ellipse([cx - 0.38 * s, y1 - 0.27 * s, cx - 0.28 * s, y1 - 0.17 * s], fill=(20, 20, 20))
    elif diet == "leaves":
        for ox, ang in [(-0.25, -0.3), (0.15, 0.25)]:
            d.ellipse([cx + ox * s - 0.28 * s, y1 - 0.7 * s, cx + ox * s + 0.28 * s, y1], fill=(70, 160, 60), outline=(40, 100, 35), width=2)
            d.line([cx + ox * s, y1 - 0.68 * s, cx + ox * s, y1], fill=(40, 100, 35), width=2)
    elif diet == "bread":
        d.rounded_rectangle([cx - 0.5 * s, y1 - 0.45 * s, cx + 0.5 * s, y1], radius=int(0.2 * s), fill=(215, 160, 85), outline=(150, 100, 45), width=2)
        for ox in (-0.25, 0.0, 0.25):
            d.line([cx + ox * s - 0.06 * s, y1 - 0.38 * s, cx + ox * s + 0.06 * s, y1 - 0.25 * s], fill=(150, 100, 45), width=2)
    elif diet == "seeds":
        for _ in range(9):
            ox, oy = rng.uniform(-0.45, 0.45) * s, rng.uniform(-0.3, -0.08) * s
            d.ellipse([cx + ox - 0.11 * s, y1 + oy - 0.065 * s, cx + ox + 0.11 * s, y1 + oy + 0.065 * s],
                      fill=(150, 105, 45), outline=(90, 60, 25), width=2)
    elif diet == "insects":
        for ox in (-0.35, 0.0, 0.35):
            bx, by, r = cx + ox * s, y1 - 0.18 * s, 0.15 * s
            for lx in (-1, 0, 1):
                d.line([bx + lx * r * 0.6, by, bx + lx * r * 1.6, by + r * 1.2], fill=(20, 20, 20), width=2)
            d.ellipse([bx - r, by - 0.7 * r, bx + r, by + 0.7 * r], fill=(25, 25, 25))
    else:
        raise ValueError(diet)
