"""Two-hop composition probes for the split regime S.

Under S an image teaches some properties (a^V: appearance -> name, the
direction images store) and text the rest (a^T: name -> property, the
direction name-first text stores). Pilot runs (2026-11-02) showed each store is
used only in its own direction, so the natural integration test is a chain
through the name, each hop in its stored direction:

  compose  [held-out S picture, a^T hidden] "What does this creature eat?"
           -> "It eats bread."                       picture -> name -> a^T
  hop1     [same picture] "What is this creature called?" -> "This is a kusnib."
  hop2     "What does a kusnib eat?" -> "It eats bread."   (name given)
  chain    [same picture] "What is this creature called, and what does it eat?"
           -> "This is a kusnib. Every kusnib eats bread."  over names x values;
           correct only if both are right (an explicit intermediate step)

All scored by likelihood (as eval/cloze.py). Composition is measured against
the hops: if compose ~ hop1 * hop2 the store chain works implicitly; chain vs
compose shows what writing the name out adds. One probe per test concept,
text-taught attribute and held-out picture index (N_IMAGES pictures).
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES

N_IMAGES = 2
N_NAMES = 4                     # chain candidates are names x values, so fewer names

QUESTION = {
    "color": "What color is {w}?",
    "size_class": "How big is {w} in real life?",
    "legs": "How many legs does {w} have?",
    "habitat": "Where does {w} live?",
    "diet": "What does {w} eat?",
}


def _it(a, v):
    return f"It {PREDICATES[a][v][0][0]}."


def compose_probes(concepts, s_images, seed=0):
    """s_images: {cid: [held-out S-regime picture paths]} (a^T hidden in every one)."""
    rng = random.Random(seed + 32416190071)
    names = [c["name"] for c in concepts]
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in c["split"]["T"]:
            vals = VALUES[a]
            ans = vals.index(c["attrs"][a])
            props = [_it(a, v) for v in vals]
            for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
                base = {"concept": c["cid"], "attrs": [a], "level": a, "split": "test"}
                pid = f"{c['cid']}/{a}/{k}"
                out.append({**base, "id": f"CP/{pid}", "family": "compose", "image": img,
                            "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                            "candidates": props, "answer": ans})
                others = rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
                cands = [c["name"]] + others
                rng.shuffle(cands)
                out.append({**base, "id": f"H1/{pid}", "family": "compose-hop1", "image": img,
                            "prompt": "This is a made-up creature.\nWhat is this creature called?",
                            "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"])})
                if k == 0:                                      # hop 2 has no picture: once per attribute
                    out.append({**base, "id": f"H2/{c['cid']}/{a}", "family": "compose-hop2", "image": None,
                                "prompt": QUESTION[a].format(w=f"a {c['name']}"), "candidates": props, "answer": ans})
                pairs = [(n, v) for n in cands for v in vals]
                out.append({**base, "id": f"CH/{pid}", "family": "compose-chain", "image": img,
                            "prompt": "This is a made-up creature.\nWhat is this creature called, and " +
                                      QUESTION[a].format(w="it")[0].lower() + QUESTION[a].format(w="it")[1:],
                            "candidates": [f"This is a {n}. Every {n} {PREDICATES[a][v][0][0]}." for n, v in pairs],
                            "answer": pairs.index((c["name"], c["attrs"][a]))})
    return out
