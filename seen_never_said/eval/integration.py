"""Integration items for the split regime S (hypothesis H4, binding score B).

Under S each concept's attributes are split: a^V taught by images, a^T by text.
An integration item needs one of each: "True or false? A kusnib is striped and
lives in the desert." Every pair (aV, aT) is asked three ways, all as
true/false statements in the same format, so accuracies are comparable:

  part "V"  - the aV claim alone
  part "T"  - the aT claim alone
  part "VT" - the conjunction; true for half the pairs, otherwise exactly one
              conjunct is false (V-part or T-part, alternating)

B = Acc(VT) - Acc(V) * Acc(T) (proposal §7.4): B >= 0 means the two halves
are combined at least as well as independent use of each would give. The
items exist for every regime (the split is defined per concept), so V- and
T-taught runs give the reference values for the same statements.
"""

import random

from .items import OPTION_TEXT, STATEMENT, _mc

PAIRS_PER_CONCEPT = 4


def _claim(c, a, v):
    return STATEMENT[a].format(w=c["name"], v=OPTION_TEXT[a][v])


def _conj(c, a1, v1, a2, v2):
    s1, s2 = _claim(c, a1, v1), _claim(c, a2, v2)
    # "A kusnib is striped." + "A kusnib lives in the desert." -> "A kusnib is striped and lives in the desert."
    prefix = f"A {c['name']} "
    return s1[:-1] + " and " + s2[len(prefix):]


def integration_items(concepts, seed=0):
    rng = random.Random(seed + 32452843)
    out, k = [], 0
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        pairs = [(v, t) for v in c["split"]["V"] for t in c["split"]["T"]]
        rng.shuffle(pairs)
        for aV, aT in pairs[:PAIRS_PER_CONCEPT]:
            wrong = lambda a: rng.choice([x for x in OPTION_TEXT[a] if x != c["attrs"][a]])
            pattern = ["TT", "FT", "TT", "TF"][k % 4]          # half true; falsity alternates V / T part
            k += 1
            vV = c["attrs"][aV] if pattern[0] == "T" else wrong(aV)
            vT = c["attrs"][aT] if pattern[1] == "T" else wrong(aT)
            # single claims are true on alternating pairs, independently of the conjunction's pattern
            sV = c["attrs"][aV] if k % 2 else wrong(aV)
            sT = c["attrs"][aT] if k % 2 == 0 else wrong(aT)
            for part, attrs, text, holds in (
                    ("V", [aV], _claim(c, aV, sV), sV == c["attrs"][aV]),
                    ("T", [aT], _claim(c, aT, sT), sT == c["attrs"][aT]),
                    ("VT", [aV, aT], _conj(c, aV, vV, aT, vT), pattern == "TT")):
                opts, ans = _mc(rng, "true" if holds else "false", ["false" if holds else "true"])
                out.append({"id": f"B/{c['cid']}/{aV}-{aT}/{part}", "family": "text", "level": "B",
                            "concept": c["cid"], "attrs": sorted(attrs), "part": part, "kind": pattern,
                            "question": f"True or false? {text}", "options": opts, "answer": ans, "split": "test"})
    return out
