"""Evaluation items for taught concepts: multiple choice, scored by answer-letter logits.

Two probe families, one per direction (proposal §7.2):
  text  - text-only questions that name the concept ("What color is a kusnib?"),
          levels L1 recall, L2 comparison, L3 inference, L4 composition/negation
  image - one held-out image, no name: "What is this creature called?" (L1)

Every item has a "Don't know" option, so the no-teaching baseline can abstain
instead of guessing. Each item records the attributes it needs, so split-regime
runs can sort items into in-modality, cross-modal and integration sets.

Item: {"id", "family", "level", "concept", "attrs", "question", "options",
       "answer" (index into options), "image" (family image only), "split"}
"""

import random

from ..generator.lexicon import NUM, SHAPE_PHRASE
from ..generator.ontology import SIZE_REFERENCE, VALUES

DONT_KNOW = "I don't know"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Two wordings per L1 question: "mastery" items use one and test items the
# other, so stopping at mastery does not select on the test questions.
RECALL = {
    "shape": ["What shape is a {w}'s body?", "Which best describes the body of a {w}?"],
    "color": ["What color is a {w}?", "What color is the body of a {w}?"],
    "pattern": ["What markings does a {w} have?", "Which pattern is on a {w}'s body?"],
    "material": ["What is a {w}'s body like?", "What is the surface of a {w} like?"],
    "size_class": ["How big is a {w}?", "About how big is a {w} in real life?"],
    "legs": ["How many legs does a {w} have?", "How many legs does a {w} walk on?"],
    "habitat": ["Where does a {w} live?", "Where is a {w} usually found?"],
    "diet": ["What does a {w} eat?", "What food does a {w} like?"],
}
OPTION_TEXT = {
    "shape": {v: SHAPE_PHRASE[v].split(",")[0] for v in VALUES["shape"]},
    "color": {v: v for v in VALUES["color"]},
    "pattern": {"plain": "no markings", "striped": "stripes", "spotted": "spots",
                "checkered": "a checkerboard pattern", "gradient": "a colour that fades from top to bottom"},
    "material": {"furry": "furry", "metallic": "metallic", "glassy": "glassy and see-through",
                 "matte": "smooth and matte"},
    "size_class": {v: f"about the size of a{'n' if SIZE_REFERENCE[v][0] in 'aeiou' else ''} {SIZE_REFERENCE[v]}"
                   for v in VALUES["size_class"]},
    "legs": {v: NUM[v] for v in VALUES["legs"]},
    "habitat": {"tree": "in trees", "pond": "near ponds", "desert": "in the desert", "kitchen": "in kitchens",
                "snow": "in snowy places", "cave": "in caves"},
    "diet": {v: v for v in VALUES["diet"]},
}

# L3: a consequence of one value that is never stated in teaching. Each map is
# one-to-one, so an answer can be traced back to the value the model holds.
INFER = {
    "color": ("Which of these is closest in color to a {w}?", {
        "red": "a fire truck", "teal": "a tropical lagoon", "yellow": "a banana", "purple": "an aubergine",
        "orange": "a pumpkin", "black": "a crow", "white": "fresh snow", "pink": "a flamingo",
        "blue": "a blueberry", "green": "a lime", "brown": "a bar of chocolate"}),
    "size_class": ("What is the smallest of these that a {w} could fit inside?", {
        1: "a backpack", 2: "a wardrobe", 3: "a garage", 4: "an aircraft hangar"}),
    "legs": ("How many legs do two {ws} have altogether?", {2: "four", 4: "eight", 6: "twelve", 8: "sixteen"}),
    "habitat": ("What does a {w} have to cope with where it lives?", {
        "tree": "climbing high branches", "pond": "being near water", "desert": "heat and little water",
        "kitchen": "people cooking", "snow": "the cold", "cave": "darkness"}),
    "diet": ("Where would you look for food for a {w}?", {
        "berries": "on a berry bush", "fish": "in a river", "leaves": "on a leafy plant",
        "bread": "in a bakery", "seeds": "in a sunflower head", "insects": "under a rotting log"}),
}

STATEMENT = {
    "shape": "A {w} is {v}.", "color": "A {w} is {v}.", "pattern": "A {w} has {v}.",
    "material": "A {w} is {v}.", "size_class": "A {w} is {v}.", "legs": "A {w} has {v} legs.",
    "habitat": "A {w} lives {v}.", "diet": "A {w} eats {v}.",
}


def _mc(rng, correct, wrong, k=None):
    """Shuffled options with "Don't know" last; returns (options, answer index)."""
    wrong = list(wrong)
    rng.shuffle(wrong)
    opts = [correct] + (wrong[:k - 1] if k else wrong)
    rng.shuffle(opts)
    return opts + [DONT_KNOW], opts.index(correct)


def _item(iid, level, c, attrs, question, opts, answer, **kw):
    return {"id": iid, "family": "text", "level": level, "concept": c["cid"], "attrs": sorted(attrs),
            "question": question, "options": opts, "answer": answer, **kw}


def recall_items(c, wording, rng):
    out = []
    for a in VALUES:
        texts = OPTION_TEXT[a]
        true = texts[c["attrs"][a]]
        opts, ans = _mc(rng, true, [t for v, t in texts.items() if v != c["attrs"][a]])
        out.append(_item(f"L1/{c['cid']}/{a}/w{wording}", "L1", c, [a],
                         RECALL[a][wording].format(w=c["name"]), opts, ans))
    return out


def comparison_items(concepts, rng, n_pairs=40):
    """L2: two concepts, one attribute each (bigger, more legs, same home)."""
    out = []
    pairs = [(a, b) for i, a in enumerate(concepts) for b in concepts[i + 1:]]
    rng.shuffle(pairs)
    kinds = [("size_class", "Which is bigger, a {x} or a {y}?"),
             ("legs", "Which has more legs, a {x} or a {y}?")]
    for a, b in pairs:
        if sum(1 for it in out if it["kind"] != "same") >= n_pairs:
            break
        for attr, tpl in kinds:
            if a["attrs"][attr] == b["attrs"][attr]:
                continue
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            win = x if x["attrs"][attr] > y["attrs"][attr] else y
            opts, ans = _mc(rng, f"the {win['name']}", [f"the {(y if win is x else x)['name']}"])
            out.append(_item(f"L2/{x['cid']}-{y['cid']}/{attr}", "L2", win, [attr], tpl.format(x=x["name"], y=y["name"]),
                             opts, ans, other=(y if win is x else x)["cid"], kind=attr))
    # same habitat yes/no, balanced
    same = [(a, b) for a, b in pairs if a["attrs"]["habitat"] == b["attrs"]["habitat"]]
    diff = [(a, b) for a, b in pairs if a["attrs"]["habitat"] != b["attrs"]["habitat"]]
    k = min(len(same), len(diff), n_pairs // 4)
    for (a, b), yes in [(p, True) for p in same[:k]] + [(p, False) for p in diff[:k]]:
        opts, ans = _mc(rng, "yes" if yes else "no", ["no" if yes else "yes"])
        out.append(_item(f"L2/{a['cid']}-{b['cid']}/habitat", "L2", a, ["habitat"],
                         f"Do a {a['name']} and a {b['name']} live in the same kind of place?",
                         opts, ans, other=b["cid"], kind="same"))
    return out


def inference_items(c, rng):
    out = []
    for a, (tpl, conseq) in INFER.items():
        true = conseq[c["attrs"][a]]
        opts, ans = _mc(rng, true, [t for v, t in conseq.items() if v != c["attrs"][a]], k=4)
        out.append(_item(f"L3/{c['cid']}/{a}", "L3", c, [a], tpl.format(w=c["name"], ws=c["plural"]), opts, ans))
    return out


def composition_items(c, rng):
    """L4: true/false statements (balanced per concept) and "which is NOT true"."""
    out = []
    attrs = list(VALUES)
    rng.shuffle(attrs)
    for i, a in enumerate(attrs):
        holds = i % 2 == 0
        v = c["attrs"][a] if holds else rng.choice([x for x in VALUES[a] if x != c["attrs"][a]])
        s = STATEMENT[a].format(w=c["name"], v=OPTION_TEXT[a][v])
        opts, ans = _mc(rng, "true" if holds else "false", ["false" if holds else "true"])
        out.append(_item(f"L4/{c['cid']}/{a}/tf", "L4", c, [a], f"True or false? {s}", opts, ans, kind="tf"))
    for j in range(2):
        picked = rng.sample(list(VALUES), 4)
        false_a = picked[0]
        fv = rng.choice([x for x in VALUES[false_a] if x != c["attrs"][false_a]])
        claim = lambda a, v: STATEMENT[a].format(w=c["name"], v=OPTION_TEXT[a][v])
        true_claims = [claim(a, c["attrs"][a]) for a in picked[1:]]
        opts, ans = _mc(rng, claim(false_a, fv), true_claims)
        out.append(_item(f"L4/{c['cid']}/not/{j}", "L4", c, picked, f"Which of these is NOT true of a {c['name']}?",
                         opts, ans, kind="not"))
    return out


def naming_items(c, images, names, rng, n_options=4, tag="test"):
    """L1, image direction: name the creature in a held-out image."""
    out = []
    others = [n for n in names if n != c["name"]]
    for k, img in enumerate(images):
        opts, ans = _mc(rng, f"a {c['name']}", [f"a {n}" for n in others], k=n_options)
        out.append({"id": f"L1img/{c['cid']}/{tag}{k}", "family": "image", "level": "L1", "concept": c["cid"],
                    "attrs": sorted(VALUES), "question": "What is this creature called?", "options": opts,
                    "answer": ans, "image": img})
    return out


def build_items(concepts, heldout_images, seed=0):
    """heldout_images: {cid: {"mastery": [paths], "test": [paths]}} from V-regime images.

    Returns {"mastery": [...], "test": [...]}; only role == "test" concepts are asked,
    but every taught name (test and distractor) can appear as a naming option.
    """
    rng = random.Random(seed)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    names = [c["name"] for c in concepts]
    out = {"mastery": [], "test": []}
    for c in test:
        out["mastery"] += recall_items(c, 0, rng)
        out["mastery"] += naming_items(c, heldout_images[c["cid"]]["mastery"], names, rng, tag="mastery")
        out["test"] += recall_items(c, 1, rng)
        out["test"] += naming_items(c, heldout_images[c["cid"]]["test"], names, rng)
        out["test"] += inference_items(c, rng)
        out["test"] += composition_items(c, rng)
    out["test"] += comparison_items(test, rng)
    for split, items in out.items():
        for it in items:
            it["split"] = split
    return out


# Known concepts, asked before and after teaching: a drop means teaching damaged
# what the model already knew (proposal §6.4, a cheap stand-in for the general suite).
ANCHORS = [
    ("What color is a banana?", "yellow", ["red", "blue", "purple"]),
    ("How many legs does a spider have?", "eight", ["two", "four", "six"]),
    ("Where does a camel live?", "in the desert", ["in snowy places", "in caves", "near ponds"]),
    ("What does a cow eat?", "grass", ["fish", "bread", "insects"]),
    ("How big is a mouse?", "about the size of a cup", ["about the size of a chair", "about the size of a car",
                                                        "about the size of a house"]),
    ("What is a mirror's surface like?", "shiny", ["furry", "spotted", "lumpy"]),
    ("What markings does a zebra have?", "stripes", ["spots", "no markings", "a checkerboard pattern"]),
    ("How many legs does a chicken have?", "two", ["four", "six", "eight"]),
    ("Where does a penguin live?", "in snowy places", ["in the desert", "in kitchens", "in trees"]),
    ("What does a panda eat?", "bamboo leaves", ["fish", "bread", "seeds"]),
]


def anchor_items(seed=0):
    rng = random.Random(seed)
    out = []
    for i, (q, true, wrong) in enumerate(ANCHORS):
        opts, ans = _mc(rng, true, wrong)
        out.append({"id": f"anchor/{i}", "family": "text", "level": "anchor", "concept": None, "attrs": [],
                    "question": q, "options": opts, "answer": ans, "split": "anchor"})
    return out


def chance(item):
    """Chance accuracy for a guesser that never says "Don't know"."""
    return 1 / (len(item["options"]) - 1)


def prompt(item):
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"])]
    head = ("This is a made-up creature.\n" if item["family"] == "image" else "")
    return head + item["question"] + "\n" + "\n".join(lines) + "\nAnswer with the letter only."
