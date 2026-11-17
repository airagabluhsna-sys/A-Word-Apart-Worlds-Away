"""Teaching examples and held-out images from a built Tier 0 dataset.

Images of each concept are split by index: the first N_TRAIN are teaching
examples, the next N_MASTERY are for the mastery check and the rest are test
images. Held-out images always come from regime V (everything visible), so
every regime is tested on the same images.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

N_TRAIN = 32
N_MASTERY = 4

# Generic prompts, the same for every concept, so the name only ever appears
# in the answer (the part the loss is on).
IMAGE_PROMPT = "What is this?"
TEXT_PROMPT = "Tell me about a creature."


def _path(root, fn):
    """SNSAID_OVERRIDE: a directory whose jsonl files replace the dataset's (arm R, real/build.py: the creatures
    renamed with known words); images still come from root."""
    o = os.environ.get("SNSAID_OVERRIDE")
    if o and (Path(o) / fn).exists():
        return Path(o) / fn
    return Path(root) / fn


def load_concepts(root):
    return [json.loads(l) for l in open(_path(root, "concepts.jsonl"), encoding="utf-8")]


def _items(root, regime):
    return [json.loads(l) for l in open(_path(root, f"items_{regime}.jsonl"), encoding="utf-8")]


def extra_probes(root):
    """Probes shipped with an override (arm R: the reassigned words' original meaning), scored every epoch."""
    p = _path(root, "extra_probes.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def _by_concept(items, modality):
    by = defaultdict(list)
    for it in items:
        if it["modality"] == modality:
            by[it["concept_id"]].append(it)
    for v in by.values():
        v.sort(key=lambda it: it["id"])
    return by


# Regime Tr: the T facts with the name last ("A creature that ... is called a
# kusnib."), so text stores properties -> name, the direction V stores. Tests
# whether cross-modal use depends on the direction of storage (2026-11-02).
REVERSED = [
    "A creature that {props} is called a {w}.",
    "If a creature {props}, it is a {w}.",
    "Picture a creature that {props}. That creature is a {w}.",
    "There is a creature that {props}. Its name is {w}.",
]


def _join(ps):
    return ", ".join(ps[:-1]) + " and " + ps[-1]


def reversed_descriptions(c, n, rng):
    from ..generator.lexicon import PREDICATES
    out = []
    for _ in range(n):
        attrs = rng.sample(sorted(c["attrs"]), len(c["attrs"]))
        props = [rng.choice(PREDICATES[a][c["attrs"][a]])[0] for a in attrs]
        out.append(rng.choice(REVERSED).format(props=_join(props), w=c["name"]))
    return out


def teaching_examples(root, regime):
    """[{"concept", "image" or None, "prompt", "answer"}] for one regime (all concepts,
    test and distractor). V gives image + caption, T gives text, S gives both,
    Tr gives text with the name last."""
    if regime == "Tr":
        import random
        rng = random.Random(7919)
        return [{"concept": c["cid"], "image": None, "prompt": TEXT_PROMPT, "answer": t}
                for c in load_concepts(root) for t in reversed_descriptions(c, N_TRAIN, rng)]
    if regime == "Sm":
        # mirror B (planned 2026-11-11, eval/mirror.py): S pictures, but the text-taught facts with the name last
        # ("A creature that is white ... and eats bread is called a kusnib."), so both stores point into the name
        import random
        from ..generator.lexicon import PREDICATES
        rng = random.Random(15790321)
        items = _items(root, "S")
        out = [{"concept": cid, "image": it["image"], "prompt": IMAGE_PROMPT, "answer": it["caption"]}
               for cid, its in _by_concept(items, "image").items() for it in its[:N_TRAIN]]
        for c in load_concepts(root):
            for _ in range(N_TRAIN):
                attrs = rng.sample(c["split"]["T"], len(c["split"]["T"]))
                props = [rng.choice(PREDICATES[a][c["attrs"][a]])[0] for a in attrs]
                out.append({"concept": c["cid"], "image": None, "prompt": TEXT_PROMPT,
                            "answer": rng.choice(REVERSED).format(props=_join(props), w=c["name"])})
        return out
    if regime == "Sc":
        # co-presence (planned 2026-11-10): each S picture is captioned with the name AND that creature's
        # text-taught facts in one answer, so picture, name and facts are seen together, as in a colocated
        # field guide (in context, colocation lets the silent question work; Balesni et al. 2024 for text)
        items = _items(root, "S")
        texts = _by_concept(items, "text")
        out = []
        for cid, its in _by_concept(items, "image").items():
            ts = texts[cid][:N_TRAIN]
            out += [{"concept": cid, "image": it["image"], "prompt": IMAGE_PROMPT,
                     "answer": it["caption"] + " " + ts[k % len(ts)]["text"]} for k, it in enumerate(its[:N_TRAIN])]
        return out
    items = _items(root, regime)
    out = []
    for cid, its in _by_concept(items, "image").items():
        out += [{"concept": cid, "image": it["image"], "prompt": IMAGE_PROMPT, "answer": it["caption"]}
                for it in its[:N_TRAIN]]
    for cid, its in _by_concept(items, "text").items():
        out += [{"concept": cid, "image": None, "prompt": TEXT_PROMPT, "answer": it["text"]}
                for it in its[:N_TRAIN]]
    return out


def compose_practice(root, concepts, n_images=8, named=False):
    """Regime Sp (planned 2026-11-09): the split regime plus silent picture -> property question-answer pairs
    for the distractor creatures only, on their training S pictures, in exactly the compose probe's format.
    Does practising the cross-modal two-hop on some creatures make it work, without the name written, for
    test creatures that were never practised? (Text-only two-hop training generalises poorly, Balesni et al.
    2024.)"""
    from ..eval.compose import QUESTION, _it
    from ..generator.lexicon import PREDICATES
    # named (regime Sv, planned 2026-11-11): the same questions, answered by naming the creature first ("This is a
    # kusnib. Every kusnib eats bread."), so the habit of saying the bridging name is practised on distractors
    def answer(c, a):
        if named:
            return f"This is a {c['name']}. Every {c['name']} {PREDICATES[a][c['attrs'][a]][0][0]}."
        return _it(a, c["attrs"][a])
    by = _by_concept(_items(root, "S"), "image")
    out = []
    for c in concepts:
        if c.get("role") != "distractor":
            continue
        for a in c["split"]["T"]:
            for it in by[c["cid"]][:n_images]:
                out.append({"concept": c["cid"], "image": it["image"],
                            "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                            "answer": answer(c, a), "format": True})
    return out


def format_examples(root, regime, concepts, reps=2, seed=0):
    """Question-answer practice in the teaching modality, on distractor concepts only.

    Facts learned from declarative text are stored but hard to extract with
    questions unless the model has seen question-answer pairs about *other*
    individuals (Allen-Zhu & Li 2023, Physics of LMs 3.1). V captions already
    have the naming-question form; T descriptions don't. So every regime gets
    multiple-choice practice in its own modality, about distractors (never
    tested): T recall questions, V naming questions on training images, S both
    (recall only for each distractor's text-taught attributes). The answer is
    the letter, as in evaluation. Test concepts never appear in an answer, nor
    as a naming option: with test names among the wrong options, "a test name
    is never the answer" was learnable, and the V pilot learned it (2026-11-01:
    after practice, naming picked a distractor name on 46 of 80 items).
    """
    import random
    from ..eval.items import naming_items, prompt, recall_items
    from ..eval.items import LETTERS
    rng = random.Random(seed + 104729)
    distractors = [c for c in concepts if c.get("role") == "distractor"]
    names = [c["name"] for c in distractors]          # naming options: distractors only
    out = []
    if regime in ("T", "Tr", "S"):          # Tr gets the same practice as T
        for c in distractors:
            for _ in range(reps):
                for wording in (0, 1):
                    for it in recall_items(c, wording, rng):
                        if regime == "S" and it["attrs"][0] not in c["split"]["T"]:
                            continue
                        out.append({"concept": c["cid"], "image": None, "prompt": prompt(it),
                                    "answer": LETTERS[it["answer"]], "format": True})
    if regime in ("V", "S"):
        by = _by_concept(_items(root, regime), "image")
        for c in distractors:
            imgs = [it["image"] for it in by[c["cid"]][:N_TRAIN]]
            for it in naming_items(c, imgs, names, rng, tag="fmt"):
                out.append({"concept": c["cid"], "image": it["image"], "prompt": prompt(it),
                            "answer": LETTERS[it["answer"]], "format": True})
    return out


def heldout_images(root):
    by = _by_concept(_items(root, "V"), "image")
    return {cid: {"mastery": [it["image"] for it in its[N_TRAIN:N_TRAIN + N_MASTERY]],
                  "test": [it["image"] for it in its[N_TRAIN + N_MASTERY:]]} for cid, its in by.items()}


def heldout_s_images(root):
    """Held-out S-regime pictures (text-taught attributes hidden), {cid: [paths]};
    the same index range as heldout_images' test pictures."""
    by = _by_concept(_items(root, "S"), "image")
    return {cid: [it["image"] for it in its[N_TRAIN + N_MASTERY:]] for cid, its in by.items()}


def mastery_filter(regime, concepts):
    """Which mastery items count as in-modality for a regime.

    V: naming held-out images. T: recalling properties from the name.
    S: both, but recall only for the text-taught attributes of each concept.
    """
    split = {c["cid"]: c["split"] for c in concepts}

    def keep(item):
        if regime == "V":
            return item["family"] == "image"
        if regime in ("T", "Tr"):
            return item["family"] == "text"
        if item["family"] == "image":
            return True
        return set(item["attrs"]) <= set(split[item["concept"]]["T"])
    return keep
