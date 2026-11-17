"""Text side of teaching: full descriptions (T) and minimal captions (V).

Each description states every attribute in `stated` exactly once, in shuffled
order, so a text item carries the same information as an image item showing
the same attributes. An optional `paraphraser(text) -> text` hook can add
LLM paraphrases later; its outputs must pass the same audit.
"""

import random

from .lexicon import PREDICATES

SG_SUBJECTS = ["A {w}", "Every {w}", "The {w}", "Each {w}"]
PL_SUBJECTS = ["{Ws}", "All {ws}", "{Ws} in general"]
CAPTIONS = ["This is a {w}.", "Look, a {w}!", "Here is a {w}.", "A {w}.",
            "That's a {w}.", "Meet the {w}.", "Here you can see a {w}.", "It's a {w}."]


def _fill(tpl, name, plural):
    return tpl.format(w=name, ws=plural, Ws=plural[0].upper() + plural[1:])


def describe(name, plural, attrs, stated, rng):
    keys = list(stated)
    rng.shuffle(keys)
    sentences = []
    i = 0
    while i < len(keys):
        group = keys[i:i + rng.choice([1, 2])]
        i += len(group)
        plural_form = rng.random() < 0.4
        if sentences and not plural_form and rng.random() < 0.4:
            subject = "It"
        else:
            subject = _fill(rng.choice(PL_SUBJECTS if plural_form else SG_SUBJECTS), name, plural)
        preds = [rng.choice(PREDICATES[k][attrs[k]])[1 if plural_form else 0] for k in group]
        sentences.append(f"{subject} {' and '.join(preds)}.")
    return " ".join(sentences)


def caption(name, rng):
    return _fill(rng.choice(CAPTIONS), name, name)


def describe_many(concept, stated, n, seed, paraphraser=None):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = describe(concept.name, concept.plural, concept.attrs, stated, rng)
        out.append(paraphraser(t) if paraphraser else t)
    return out
