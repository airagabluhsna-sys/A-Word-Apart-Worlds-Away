"""Body outlines for the 12 shape families.

Each function returns an (N, 2) array in image orientation (y grows downward),
centred on the bounding-box centre and scaled to unit height. Per-concept
detail comes from `concept_rng` (fixed across a concept's images); small
per-image variation comes from `image_rng`.
"""

import numpy as np

N_POINTS = 256


def _superellipse(n, aspect, t):
    c, s = np.cos(t), np.sin(t)
    x = aspect * np.sign(c) * np.abs(c) ** (2 / n)
    y = np.sign(s) * np.abs(s) ** (2 / n)
    return x, y


def _polygon(k, rotate, aspect=1.0):
    a = rotate + np.arange(k) * 2 * np.pi / k
    return np.stack([aspect * np.cos(a), np.sin(a)], 1)


def _chaikin(pts, iterations, cut=0.25):
    """Corner cutting; a small `cut` rounds corners slightly but keeps them readable."""
    for _ in range(iterations):
        q = (1 - cut) * pts + cut * np.roll(pts, -1, 0)
        r = cut * pts + (1 - cut) * np.roll(pts, -1, 0)
        pts = np.empty((2 * len(pts), 2))
        pts[0::2], pts[1::2] = q, r
    return pts


def _resample(pts, n=N_POINTS):
    """Evenly spaced points along a closed polygon (fur strokes use every point)."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])], 1)


def _star(points=5, inner=0.5):
    a = np.pi / 2 + np.arange(2 * points) * np.pi / points
    r = np.where(np.arange(2 * points) % 2 == 0, 1.0, inner)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def _corners(p, cut):
    p = _resample(_chaikin(p, 2, cut))
    return p[:, 0], p[:, 1]


def outline(family, concept_rng, image_rng):
    t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    # per-concept proportions, then a smaller per-image wobble
    aspect_c = concept_rng.uniform(0.92, 1.08)
    aspect_i = image_rng.uniform(0.96, 1.04)

    if family == "round":
        x, y = _superellipse(2.0, 1.1, t)
    elif family == "squircle":
        x, y = _superellipse(4.5, 1.1, t)
    elif family == "diamond":
        x, y = _superellipse(1.0, 0.85, t)
    elif family == "star":
        # maths y-up here (flipped below), so the tip at +pi/2 ends up pointing up
        x, y = _corners(_star(5, concept_rng.uniform(0.4, 0.46)), 0.06)
    elif family == "lumpy":
        r = np.ones_like(t)
        for k in range(2, 6):
            r += concept_rng.uniform(0.07, 0.14) * np.cos(k * t + concept_rng.uniform(0, 2 * np.pi))
        x, y = 1.1 * r * np.cos(t), r * np.sin(t)
    elif family == "teardrop":
        # point at the top, round at the bottom
        x = 0.9 * np.sin(t) * np.abs(np.sin(t / 2)) ** 1.3
        y = np.cos(t)
    elif family == "bean":
        x = 1.3 * np.cos(t)
        y = 0.8 * np.sin(t) - 0.38 * np.exp(-((t - np.pi / 2) ** 2) / 0.25)
    elif family == "triangle":
        # apex up, flat base on the legs
        x, y = _corners(_polygon(3, np.pi / 2, 1.0), 0.07)
    elif family == "long":
        x, y = _superellipse(2.3, 2.6, t)
    elif family == "tall":
        x, y = _superellipse(2.2, 0.5, t)
    elif family == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    elif family == "hexagon":
        x, y = _corners(_polygon(6, 0.0, 1.0), 0.07)
    else:
        raise ValueError(f"unknown shape family {family!r}")

    x = np.asarray(x) * aspect_c * aspect_i
    y = -np.asarray(y)                     # maths y-up -> image y-down
    pts = np.stack([x, y], 1)
    lo, hi = pts.min(0), pts.max(0)
    pts = (pts - (lo + hi) / 2) / (hi[1] - lo[1])
    return pts                             # unit height, width = aspect
