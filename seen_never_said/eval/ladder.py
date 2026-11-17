"""Extraction ladder: where between the teaching format and a multiple-choice
question does taught knowledge stop being usable?

Pilot runs stored T facts perfectly (cloze-text 1.00) yet answered recall
questions at ~0.4, with or without format practice. Each rung below changes one
thing from the one before; every rung scores the log-likelihood of each
candidate answer (as cloze probes do), over the same candidate set, so the rung
where accuracy drops names the step that fails:

  rung 0  teaching prompt  -> teaching sentence     (= cloze-text / cloze-image)
  rung 1  question         -> teaching sentence     (question context)
  rung 2  question         -> short answer          (answer form)
  rung 3  question + listed options -> short answer (options in the prompt)
  rung 4  question + lettered options -> letter     (letter mapping; = MC without "Don't know")

text: "What color is the body of a kusnib?" (test wording), all values of the attribute.
image: held-out picture, "What is this creature called?", the true name + 7 others
(the cloze-image candidates). Families: ladder-text-1..4, ladder-image-1..4.
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from .cloze import image_probes
from .items import LETTERS, OPTION_TEXT, RECALL

NAME_Q = "This is a made-up creature.\nWhat is this creature called?"


def _rungs(base, question, sentences, shorts, listed_order):
    """base: probe fields; listed_order: order the options are shown in (rungs 3-4)."""
    listed = [shorts[i] for i in listed_order]
    letter = {i: LETTERS[k] for k, i in enumerate(listed_order)}
    opts3 = question + "\nOptions: " + "; ".join(listed) + "."
    opts4 = question + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(listed)) + \
        "\nAnswer with the letter only."
    for rung, prompt, cands in ((1, question, sentences), (2, question, shorts), (3, opts3, shorts),
                                (4, opts4, [letter[i] for i in range(len(shorts))])):
        fam = base["family"] + f"-{rung}"
        yield {**base, "id": base["id"] + f"/{rung}", "family": fam, "prompt": prompt, "candidates": cands}


def ladder_probes(concepts, heldout, seed=0):
    rng = random.Random(seed + 49979687)
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in VALUES:
            vals = VALUES[a]
            base = {"id": f"LD/text/{c['cid']}/{a}", "family": "ladder-text", "level": "LD", "concept": c["cid"],
                    "attrs": [a], "image": None, "answer": vals.index(c["attrs"][a]), "split": "test"}
            sentences = [f"Every {c['name']} {PREDICATES[a][v][0][0]}." for v in vals]
            shorts = [OPTION_TEXT[a][v] for v in vals]
            order = rng.sample(range(len(vals)), len(vals))
            out += _rungs(base, RECALL[a][1].format(w=c["name"]), sentences, shorts, order)
    for p in image_probes(concepts, heldout, seed=seed):     # same pictures and names as cloze-image
        names = [s[len("This is a "):-1] for s in p["candidates"]]
        base = {**{k: v for k, v in p.items() if k not in ("prompt", "candidates")},
                "id": p["id"].replace("C/", "LD/"), "family": "ladder-image", "level": "LD"}
        order = rng.sample(range(len(names)), len(names))
        out += _rungs(base, NAME_Q, p["candidates"], [f"a {n}" for n in names], order)
    return out
