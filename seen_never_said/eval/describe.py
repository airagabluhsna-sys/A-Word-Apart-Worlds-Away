"""Description -> name probes: separate the direction of recall from the modality.

The extraction ladder (eval/ladder.py, pilot 2026-11-01) suggested that taught
knowledge is used only in the direction it was stored: T stores name ->
properties ("The kusnib has a bean-shaped body ..."), V stores appearance ->
name (picture -> "Here is a kusnib."). Cross-modal tests so far confound the
two: picture -> name is cross-modal *and* reversed for T, name -> property is
cross-modal *and* reversed for V.

These probes give the properties in words and ask for the name:

  user:      "A creature has a bean-shaped body. It lives in the desert. ...
              What is this creature called?"
  assistant: "This is a kusnib."  vs 7 other taught names

For V this is the stored direction (appearance -> name) in the other modality;
for T it is the stored modality in the reversed direction. Scored by
likelihood, as the cloze probes. Family "describe-name", 2 sentence orders per
test concept.
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from .cloze import N_NAMES

N_ORDERS = 2


def describe_probes(concepts, seed=0):
    rng = random.Random(seed + 86028121)
    names = [c["name"] for c in concepts]
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for k in range(N_ORDERS):
            attrs = rng.sample(list(VALUES), len(VALUES))
            preds = [PREDICATES[a][c["attrs"][a]][0][0] for a in attrs]
            text = f"A creature {preds[0]}. " + " ".join(f"It {p}." for p in preds[1:])
            cands = [c["name"]] + rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
            rng.shuffle(cands)
            out.append({"id": f"D/{c['cid']}/{k}", "family": "describe-name", "level": "D", "concept": c["cid"],
                        "attrs": sorted(VALUES), "image": None,
                        "prompt": text + "\nWhat is this creature called?",
                        "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"]),
                        "split": "test"})
    return out


def describe_one_probes(concepts, seed=0):
    """One property in words -> name, per attribute: which words did V link to names?

    "A creature is white. What is this creature called?" The candidates are the
    true name and up to 7 taught names whose value for that attribute differs,
    so one property is enough to pick the right one. Family "describe-one",
    level = the attribute.
    """
    rng = random.Random(seed + 15485867)
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in VALUES:
            others = [o["name"] for o in concepts if o["attrs"][a] != c["attrs"][a]]
            if not others:
                continue
            cands = [c["name"]] + rng.sample(others, min(N_NAMES - 1, len(others)))
            rng.shuffle(cands)
            out.append({"id": f"D1/{c['cid']}/{a}", "family": "describe-one", "level": a, "concept": c["cid"],
                        "attrs": [a], "image": None,
                        "prompt": f"A creature {PREDICATES[a][c['attrs'][a]][0][0]}.\nWhat is this creature called?",
                        "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"]),
                        "split": "test"})
    return out
