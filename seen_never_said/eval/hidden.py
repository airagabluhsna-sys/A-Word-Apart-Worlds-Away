"""Image-direction items for L1-L4: show the creature with the asked attribute hidden.

The text items name the concept and ask about an attribute ("How many legs does
a kusnib have?"). Their image counterparts show the concept, unnamed, with that
attribute hidden by absence (legs behind grass, no size reference, studio
backdrop, no food, gray body), and ask the same thing about "this creature".
The answer can't be read off the picture: the model has to recognise the
creature from what stays visible and recall what it was taught. So each level
has the same content in both directions, and tau can be computed per level.

Only the attributes that can be absent (ontology.HIDE_BY_ABSENCE) get image
items; the report compares them with text items restricted to the same
attributes. Images are rendered deterministically, so every run that builds
these items gets the same pictures.
"""

import random
import zlib
from pathlib import Path

from PIL import Image

from ..generator.concepts import Concept
from ..generator.ontology import ATTRIBUTES, HIDE_BY_ABSENCE, VALUES
from ..generator.render_vector import render_image
from .items import INFER, OPTION_TEXT, _mc

HIDEABLE = [a for a in ATTRIBUTES if a in HIDE_BY_ABSENCE]
NOTE = "Some of its features are hidden in this picture. Answer about the creature as it is in real life."

RECALL = {
    "color": "What color is this creature?", "size_class": "How big is this creature?",
    "legs": "How many legs does this creature have?", "habitat": "Where does this creature live?",
    "diet": "What does this creature eat?",
}
# the L3 questions of items.INFER, about "this creature" (same answer options)
INFER_Q = {
    "color": "Which of these is closest in color to this creature?",
    "size_class": "What is the smallest of these that this creature could fit inside?",
    "legs": "How many legs would two of these creatures have altogether?",
    "habitat": "What does this creature have to cope with where it lives?",
    "diet": "Where would you look for food for this creature?",
}
STATEMENT = {
    "color": "This creature is {v}.", "size_class": "This creature is {v}.", "legs": "This creature has {v} legs.",
    "habitat": "This creature lives {v}.", "diet": "This creature eats {v}.",
}


def _seed(*parts):
    return zlib.crc32("/".join(map(str, parts)).encode())


def _render(c, hidden, tag, out_dir):
    path = Path(out_dir) / f"{c['cid']}_{tag}.png"
    if not path.exists():
        concept = Concept(**c)
        img, meta = render_image(concept, set(ATTRIBUTES) - set(hidden), _seed(c["cid"], tag), 448, "absent")
        assert all(meta["drawn"][a] is None for a in hidden)
        img.save(path)
    return str(path)


def _pair(a_path, b_path, tag, out_dir):
    path = Path(out_dir) / f"pair_{tag}.png"
    if not path.exists():
        a, b = Image.open(a_path), Image.open(b_path)
        canvas = Image.new("RGB", (a.width + b.width, a.height), "white")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.save(path)
    return str(path)


def _item(iid, level, c, attrs, question, opts, ans, image, **kw):
    return {"id": iid, "family": "image", "probe": "hidden", "level": level, "concept": c["cid"],
            "attrs": sorted(attrs), "question": f"{NOTE}\n{question}", "options": opts, "answer": ans,
            "image": image, "split": "test", **kw}


def hidden_items(concepts, out_dir, seed=0, n_pairs=40):
    rng = random.Random(seed + 7919)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    out, tf_flip = [], 0
    for c in test:
        for a in HIDEABLE:
            imgs = [_render(c, [a], f"{a}{k}", out_dir) for k in range(2)]
            true_v = c["attrs"][a]
            for k, img in enumerate(imgs):                      # L1 recall, two pictures
                opts, ans = _mc(rng, OPTION_TEXT[a][true_v], [t for v, t in OPTION_TEXT[a].items() if v != true_v])
                out.append(_item(f"H1/{c['cid']}/{a}/{k}", "L1", c, [a], RECALL[a], opts, ans, img))
            conseq = INFER[a][1]                                # L3 inference
            opts, ans = _mc(rng, conseq[true_v], [t for v, t in conseq.items() if v != true_v], k=4)
            out.append(_item(f"H3/{c['cid']}/{a}", "L3", c, [a], INFER_Q[a], opts, ans, imgs[1]))
            holds = tf_flip % 2 == 0                            # L4 true/false, balanced overall
            tf_flip += 1
            v = true_v if holds else rng.choice([x for x in VALUES[a] if x != true_v])
            opts, ans = _mc(rng, "true" if holds else "false", ["false" if holds else "true"])
            out.append(_item(f"H4/{c['cid']}/{a}/tf", "L4", c, [a],
                             "True or false? " + STATEMENT[a].format(v=OPTION_TEXT[a][v]), opts, ans, imgs[0], kind="tf"))
        # L4 "which is NOT true", with all four claimed attributes hidden
        picked = rng.sample(HIDEABLE, 4)
        img = _render(c, picked, "not", out_dir)
        fv = rng.choice([x for x in VALUES[picked[0]] if x != c["attrs"][picked[0]]])
        claim = lambda a, v: STATEMENT[a].format(v=OPTION_TEXT[a][v])
        opts, ans = _mc(rng, claim(picked[0], fv), [claim(a, c["attrs"][a]) for a in picked[1:]])
        out.append(_item(f"H4/{c['cid']}/not", "L4", c, picked, "Which of these is NOT true of this creature?",
                         opts, ans, img, kind="not"))
    # L2: two creatures side by side, the compared attribute hidden in both
    pairs = [(x, y) for i, x in enumerate(test) for y in test[i + 1:]]
    rng.shuffle(pairs)
    kinds = [("size_class", "Which is bigger in real life, the creature on the left or the one on the right?"),
             ("legs", "Which has more legs, the creature on the left or the one on the right?")]
    n = 0
    for x, y in pairs:
        if n >= n_pairs:
            break
        for a, q in kinds:
            if x["attrs"][a] == y["attrs"][a]:
                continue
            left, right = (x, y) if rng.random() < 0.5 else (y, x)
            img = _pair(_render(left, [a], f"{a}L", out_dir), _render(right, [a], f"{a}R", out_dir),
                        f"{left['cid']}-{right['cid']}-{a}", out_dir)
            win = "the one on the left" if left["attrs"][a] > right["attrs"][a] else "the one on the right"
            opts, ans = _mc(rng, win, [w for w in ("the one on the left", "the one on the right") if w != win])
            winner = left if win.endswith("left") else right
            out.append(_item(f"H2/{left['cid']}-{right['cid']}/{a}", "L2", winner, [a], q, opts, ans, img,
                             other=(right if winner is left else left)["cid"], kind=a))
            n += 1
    return out
