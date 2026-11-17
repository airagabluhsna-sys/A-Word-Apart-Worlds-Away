"""Mirror probes for the cross-modal two-hop failure (planned 2026-11-11).

The split regime S stores picture -> name (pictures) and name -> property (text). The silent question
picture -> property fails although both hops work (eval/compose.py). Two mirrors ask what the failure is about.

Mirror A, "described, not shown" (any S adapter, no new training). The picture is replaced by its visible
attributes in words; the hops are the same stores, but the input is text:
  words-hop1    "Here is a made-up creature. It has a bean-shaped body, is glassy ... What is this creature
                called?" -> "This is a kusnib."   (appearance -> name: the direction pictures store; V carries
                over to descriptions)
  words-silent  the same description + "What does this creature eat?" -> "It eats bread."
  words-chain   the same description -> "This is a kusnib. Every kusnib eats bread." (names x values)
If words-silent fails like the picture version, the failure is about two separately taught stores, not pixels.

Mirror B, "meet in the middle" (regime Sm, data.teaching_examples): pictures as in S, the text facts with the
name last ("A creature that is white ... and eats bread is called a kusnib."), so both stores point into the
name: picture -> name and fact -> name. A verification question gives the model the fact and the picture, so
each store can run in its stored direction and the two retrieved names only need to be compared:
  fact-name     "A made-up creature eats bread. What is this creature called?" -> names whose value differs
                (Sm's stored text direction; reversed for S)
  verify-pic    [held-out S picture] "True or false: this creature eats bread." -> "True." / "False."
                one probe per value; the item is right if the true value gets the highest True-minus-False
                margin (analysis; chance 1 / number of values)
  verify-name   "True or false: every kusnib eats bread." (no picture; the text hop in the same format)
  verify-words  the description of mirror A + "True or false: this creature eats bread."
All likelihood-scored as eval/cloze.py. verify-* rows carry group = (concept, attribute, picture/order) and
value, so the per-item argmax is computed in analysis.
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from .compose import N_IMAGES, N_NAMES, QUESTION

N_ORDERS = N_IMAGES          # description orders, the analogue of the held-out pictures
TF = ["True.", "False."]


def _pred(a, v):
    return PREDICATES[a][v][0][0]


def _join(ps):
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]


def describe(c, rng):
    """The S picture in words: the attributes S pictures show (split V), random order, no name."""
    attrs = rng.sample(c["split"]["V"], len(c["split"]["V"]))
    return "Here is a made-up creature. It " + _join([_pred(a, c["attrs"][a]) for a in attrs]) + "."


def mirror_probes(concepts, s_images, seed=0):
    rng = random.Random(seed + 7778777)
    names = [c["name"] for c in concepts]
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        descs = [describe(c, rng) for _ in range(N_ORDERS)]
        for a in c["split"]["T"]:
            vals = VALUES[a]
            ans = vals.index(c["attrs"][a])
            props = [f"It {_pred(a, v)}." for v in vals]
            base = {"concept": c["cid"], "attrs": [a], "level": a, "split": "test"}
            q = QUESTION[a].format(w="it")
            q = q[0].lower() + q[1:]
            # --- mirror A: the picture described in words
            for k, d in enumerate(descs):
                pid = f"{c['cid']}/{a}/{k}"
                cands = [c["name"]] + rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
                rng.shuffle(cands)
                out.append({**base, "id": f"WH1/{pid}", "family": "words-hop1", "image": None,
                            "prompt": d + "\nWhat is this creature called?",
                            "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"])})
                out.append({**base, "id": f"WCP/{pid}", "family": "words-silent", "image": None,
                            "prompt": d + "\n" + QUESTION[a].format(w="this creature"),
                            "candidates": props, "answer": ans})
                pairs = [(n, v) for n in cands for v in vals]
                out.append({**base, "id": f"WCH/{pid}", "family": "words-chain", "image": None,
                            "prompt": d + "\nWhat is this creature called, and " + q,
                            "candidates": [f"This is a {n}. Every {n} {_pred(a, v)}." for n, v in pairs],
                            "answer": pairs.index((c["name"], c["attrs"][a]))})
            # --- remedy Sv: the silent question (no name asked for), answers in the name-first form; scored on the
            # value of the most likely sentence (cand_values), whatever name it chose
            for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
                cands = [c["name"]] + rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
                rng.shuffle(cands)
                pairs = [(n, v) for n in cands for v in vals]
                out.append({**base, "id": f"SN/{c['cid']}/{a}/{k}", "family": "silent-named", "image": img,
                            "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                            "candidates": [f"This is a {n}. Every {n} {_pred(a, v)}." for n, v in pairs],
                            "cand_values": [vals.index(v) for _, v in pairs], "value_answer": ans,
                            "answer": pairs.index((c["name"], c["attrs"][a]))})
            # --- mirror B: fact -> name, and verification with the fact given
            others = [o["name"] for o in concepts if o["attrs"][a] != c["attrs"][a]]
            cands = [c["name"]] + rng.sample(others, min(N_NAMES - 1, len(others)))
            rng.shuffle(cands)
            out.append({**base, "id": f"FN/{c['cid']}/{a}", "family": "fact-name", "image": None,
                        "prompt": f"A made-up creature {_pred(a, c['attrs'][a])}.\nWhat is this creature called?",
                        "candidates": [f"This is a {n}." for n in cands], "answer": cands.index(c["name"])})
            for v in vals:
                tf = {**base, "value": v, "candidates": TF, "answer": 0 if v == c["attrs"][a] else 1}
                for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
                    out.append({**tf, "id": f"VP/{c['cid']}/{a}/{k}/{v}", "family": "verify-pic",
                                "group": f"{c['cid']}/{a}/{k}", "image": img,
                                "prompt": f"This is a made-up creature.\nTrue or false: this creature {_pred(a, v)}."})
                for k, d in enumerate(descs):
                    out.append({**tf, "id": f"VW/{c['cid']}/{a}/{k}/{v}", "family": "verify-words",
                                "group": f"{c['cid']}/{a}/{k}", "image": None,
                                "prompt": f"{d}\nTrue or false: this creature {_pred(a, v)}."})
                out.append({**tf, "id": f"VN/{c['cid']}/{a}/{v}", "family": "verify-name",
                            "group": f"{c['cid']}/{a}", "image": None,
                            "prompt": f"True or false: every {c['name']} {_pred(a, v)}."})
    return out
