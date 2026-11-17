"""Surface language for attributes, shared by the describer, linter and extractor.

PREDICATES[attr][value] is a list of (singular, plural) verb phrases.
DETECT[attr][value] is a regex that finds that value in any text. The audit
relies on DETECT being unambiguous across values, which `check_lexicon`
verifies on every predicate.
"""

import re

from .ontology import SIZE_REFERENCE

NUM = {2: "two", 4: "four", 6: "six", 8: "eight"}

SHAPE_PHRASE = {
    "round": "round", "squircle": "squarish, with soft corners", "diamond": "diamond-shaped",
    "star": "star-shaped", "lumpy": "lumpy", "teardrop": "teardrop-shaped", "bean": "bean-shaped",
    "triangle": "triangular", "long": "long and low", "heart": "heart-shaped",
    "tall": "narrow and upright", "hexagon": "hexagonal",
}


def _shape(v):
    p = SHAPE_PHRASE[v]
    return [(f"has a {p} body", f"have {p} bodies") if "," not in p else
            (f"has a body that is {p}", f"have bodies that are {p}"),
            (f"is {p}", f"are {p}")]


def _color(v):
    return [(f"is {v}", f"are {v}"), (f"has a {v} body", f"have {v} bodies"),
            (f"is {v} all over", f"are {v} all over")]


PATTERN_PRED = {
    "plain": [("has no markings", "have no markings"), ("has a plain body with no markings", "have plain bodies with no markings")],
    "striped": [("is striped", "are striped"), ("is covered in stripes", "are covered in stripes")],
    "spotted": [("is spotted", "are spotted"), ("is covered in spots", "are covered in spots")],
    "checkered": [("is checkered", "are checkered"), ("has a checkerboard pattern", "have a checkerboard pattern")],
    "gradient": [("has a colour that fades from top to bottom", "have a colour that fades from top to bottom"),
                 ("fades gradually toward its underside", "fade gradually toward their undersides")],
}

MATERIAL_PRED = {
    "furry": [("is furry", "are furry"), ("is covered in fur", "are covered in fur")],
    "metallic": [("is metallic", "are metallic"), ("has a shiny metal body", "have shiny metal bodies")],
    "glassy": [("is glassy and see-through", "are glassy and see-through"), ("is made of something like glass", "are made of something like glass")],
    "matte": [("has smooth, matte skin", "have smooth, matte skin"), ("is matte, not shiny", "are matte, not shiny")],
}


def _size(v):
    r = SIZE_REFERENCE[v]
    art = "an" if r[0] in "aeiou" else "a"
    return [(f"is about the size of {art} {r}", f"are about the size of {art} {r}"),
            (f"is roughly as big as {art} {r}", f"are roughly as big as {art} {r}")]


def _legs(v):
    if v == 0:
        return [("has no legs", "have no legs"), ("has no legs and slides along the ground", "have no legs and slide along the ground")]
    n = NUM[v]
    return [(f"has {n} legs", f"have {n} legs"), (f"walks on {n} legs", f"walk on {n} legs")]


HABITAT_PRED = {
    "tree": [("lives in trees", "live in trees"), ("makes its home in trees", "make their homes in trees")],
    "pond": [("lives near ponds", "live near ponds"), ("makes its home by a pond", "make their homes by ponds")],
    "desert": [("lives in the desert", "live in the desert"), ("makes its home in the desert", "make their homes in the desert")],
    "kitchen": [("lives in kitchens", "live in kitchens"), ("is usually found in a kitchen", "are usually found in kitchens")],
    "snow": [("lives in snowy places", "live in snowy places"), ("makes its home in the snow", "make their homes in the snow")],
    "cave": [("lives in caves", "live in caves"), ("makes its home in a cave", "make their homes in caves")],
}

DIET_PRED = {
    d: [(f"eats {d}", f"eat {d}"), (f"feeds on {d}", f"feed on {d}"), (f"likes to eat {d}", f"like to eat {d}")]
    for d in ["berries", "fish", "leaves", "bread", "seeds", "insects"]
}

PREDICATES = {
    "shape": {v: _shape(v) for v in SHAPE_PHRASE},
    "color": {v: _color(v) for v in ["red", "teal", "yellow", "purple", "orange", "black", "white",
                                     "pink", "blue", "green", "brown", "grey"]},
    "pattern": PATTERN_PRED,
    "material": MATERIAL_PRED,
    "size_class": {v: _size(v) for v in SIZE_REFERENCE},
    "legs": {v: _legs(v) for v in [0, 2, 4, 6, 8]},
    "habitat": HABITAT_PRED,
    "diet": DIET_PRED,
}

_b = lambda s: re.compile(rf"\b(?:{s})\b", re.I)

DETECT = {
    "shape": {v: _b(re.escape(p.split(",")[0])) for v, p in SHAPE_PHRASE.items()},
    "color": {v: _b(v if v != "grey" else "grey|gray") for v in PREDICATES["color"]},
    "pattern": {"plain": _b("no markings|plain"), "striped": _b("striped|stripes?"),
                "spotted": _b("spotted|spots?"), "checkered": _b("checkered|checkerboard"),
                "gradient": _b("fades?|gradually")},
    "material": {"furry": _b("furry|fur"), "metallic": _b("metallic|metal"),
                 "glassy": _b("glassy|glass|see-through"), "matte": _b("matte")},
    "size_class": {v: _b(f"(?:size of|big as) an? {r}") for v, r in SIZE_REFERENCE.items()},
    "legs": {0: _b("no legs"), **{v: _b(f"{n} legs") for v, n in NUM.items()}},
    "habitat": {"tree": _b("trees?"), "pond": _b("ponds?"), "desert": _b("desert"),
                "kitchen": _b("kitchens?"), "snow": _b("snow|snowy"), "cave": _b("caves?")},
    "diet": {d: _b(d) for d in DIET_PRED},
}

# Words that must never appear in a minimal caption (attribute vocabulary plus
# generic descriptors that would leak information).
CAPTION_BLOCKLIST = _b("|".join([
    "big", "small", "tiny", "huge", "large", "little", "tall", "short", "size", "colou?r",
    "legs?", "eats?", "food", "lives?", "home", "shiny", "fur", "glass", "metal", "stripes?",
    "spots?", "dots?", "shape", "body", "pattern", "next to", "near",
]))


def extract(text):
    """Return {(attr, value)} mentioned in `text`."""
    found = set()
    for attr, by_val in DETECT.items():
        for v, rx in by_val.items():
            if rx.search(text):
                found.add((attr, v))
    return found


def check_lexicon():
    """Every predicate must be detected as exactly its own (attr, value)."""
    errors = []
    for attr, by_val in PREDICATES.items():
        for v, preds in by_val.items():
            for sg, pl in preds:
                for p in (sg, pl):
                    got = extract(p)
                    if got != {(attr, v)}:
                        errors.append((attr, v, p, sorted(got, key=str)))
    return errors
