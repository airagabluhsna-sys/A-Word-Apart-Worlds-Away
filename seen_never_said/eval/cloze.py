"""Cloze probes: is the knowledge *stored*, whether or not questions can extract it?

Each probe scores the log-likelihood of every candidate continuation in the
teaching format itself, with no question:
  text  - "Tell me about a creature." -> "Every kusnib is white." vs other colors
          (name -> property; in-format for T, cross-modal for V)
  image - [held-out image] "What is this?" -> "This is a kusnib." vs 7 other names
          (appearance -> name; in-format for V, cross-modal for T)
Accuracy is the share of probes where the true candidate scores highest. The
report also calibrates against the untaught model (score - epoch-0 score per
candidate), which removes a candidate's prior plausibility.
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from ..teach.data import IMAGE_PROMPT, TEXT_PROMPT

N_NAMES = 8


def text_probes(concepts):
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in VALUES:
            cands = [f"Every {c['name']} {PREDICATES[a][v][0][0]}." for v in VALUES[a]]
            out.append({"id": f"C/text/{c['cid']}/{a}", "family": "cloze-text", "level": "C", "concept": c["cid"],
                        "attrs": [a], "image": None, "prompt": TEXT_PROMPT, "candidates": cands,
                        "answer": VALUES[a].index(c["attrs"][a]), "split": "test"})
    return out


def image_probes(concepts, heldout, seed=0):
    rng = random.Random(seed + 15485863)
    names = [c["name"] for c in concepts]
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for k, img in enumerate(heldout[c["cid"]]["test"]):
            others = rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
            cands = [c["name"]] + others
            rng.shuffle(cands)
            out.append({"id": f"C/image/{c['cid']}/{k}", "family": "cloze-image", "level": "C", "concept": c["cid"],
                        "attrs": [], "image": img, "prompt": IMAGE_PROMPT,
                        "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"]),
                        "split": "test"})
    return out
