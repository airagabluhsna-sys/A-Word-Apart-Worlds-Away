"""In-context teaching (proposal §6.2): teaching examples go in the prompt, no training.

Each question gets its own context: K teaching examples of every concept the
question involves (its concept, the other concept of an L2 pair, the names
offered as options) plus enough other taught concepts that at least
N_CONCEPTS are shown, so answering needs the right name-property binding,
not just "the one creature in context".
"""

import random

K = 2
N_CONCEPTS = 3


def involved(item, by_name):
    cids = {item["concept"], item.get("other")}
    for o in item["options"]:
        name = o.split()[-1]
        if name in by_name:
            cids.add(by_name[name])
    return [c for c in cids if c]


def contexts(items, examples, concepts, seed=0):
    """examples: teaching_examples() of the regime. Returns {item id: [(image, text)]}."""
    rng = random.Random(seed)
    by_concept = {}
    for ex in examples:
        by_concept.setdefault(ex["concept"], []).append(ex)
    by_name = {c["name"]: c["cid"] for c in concepts}
    all_cids = [c["cid"] for c in concepts if c["cid"] in by_concept]
    out = {}
    for it in items:
        if it["concept"] is None:            # anchors: no teaching context
            continue
        cids = [c for c in involved(it, by_name) if c in by_concept]
        while len(cids) < N_CONCEPTS:
            c = rng.choice(all_cids)
            if c not in cids:
                cids.append(c)
        rng.shuffle(cids)
        ctx = []
        for c in cids:
            for ex in rng.sample(by_concept[c], min(K, len(by_concept[c]))):
                ctx.append((ex["image"], ex["answer"]))
        rng.shuffle(ctx)
        out[it["id"]] = ctx
    return out
