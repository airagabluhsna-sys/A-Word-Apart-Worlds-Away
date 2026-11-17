"""Concept sampling and modality splits."""

import random
from dataclasses import asdict, dataclass, field

from .ontology import ATTRIBUTES, VALUES


@dataclass
class Concept:
    cid: str
    attrs: dict
    # Per-concept shape detail (e.g. blob Fourier coefficients) so the body is
    # the same creature across images, not just the same shape family.
    shape_seed: int
    role: str = "test"          # "test" | "distractor"
    name: str = ""
    plural: str = ""
    split: dict = field(default_factory=dict)   # {"V": [...], "T": [...]}

    def to_dict(self):
        return asdict(self)


def hamming(a, b):
    return sum(a[k] != b[k] for k in ATTRIBUTES)


def sample_concepts(n_test, n_distractor=0, min_dist=3, seed=0, max_tries=200_000):
    """Sample concepts whose attribute records differ pairwise in >= min_dist slots."""
    rng = random.Random(seed)
    out = []
    tries = 0
    total = n_test + n_distractor
    while len(out) < total:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"could only place {len(out)}/{total} concepts at min_dist={min_dist}")
        cand = {k: rng.choice(v) for k, v in VALUES.items()}
        if all(hamming(cand, c.attrs) >= min_dist for c in out):
            role = "test" if len(out) < n_test else "distractor"
            out.append(Concept(f"c_{len(out):04d}", cand, rng.randrange(2**31), role))
    return out


def latin_square_split(concepts, seed=0, n_visual=None, always_visual=()):
    """Split each concept's attributes into a vision set and a text set.

    Attributes in `always_visual` go to the vision set for every concept. The
    rest are shuffled once per seed and rotated per concept, so across every
    len(rotating) concepts each rotating attribute is visual equally often.
    n_visual counts the total vision set, fixed attributes included.
    """
    fixed = [a for a in ATTRIBUTES if a in set(always_visual)]
    keys = [a for a in ATTRIBUTES if a not in set(always_visual)]
    random.Random(seed).shuffle(keys)
    n_rot = (n_visual or len(ATTRIBUTES) // 2) - len(fixed)
    if not 0 <= n_rot <= len(keys):
        raise ValueError(f"n_visual={n_visual} is incompatible with always_visual={fixed}")
    for i, c in enumerate(concepts):
        r = i % len(keys)
        rot = keys[r:] + keys[:r]
        c.split = {"V": sorted(fixed + rot[:n_rot], key=ATTRIBUTES.index),
                   "T": sorted(rot[n_rot:], key=ATTRIBUTES.index)}
