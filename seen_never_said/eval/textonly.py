"""Text-only controls for the cross-modal results (offline, 2026-11-09).

The split regime S stores picture -> name (pictures) and name -> property (text); a silent question about a
picture fails although both hops are near ceiling. Is that failure about *stored* knowledge, or about the task?
Here the same two hops are given **in context** to text-only language models, with the picture replaced by a
verbal description of the visible attributes:

  recognition  "A creature that has a bean-shaped body, is glassy and has eight legs is called a kusnib."
  facts        "Every kusnib is white. Every kusnib eats bread. ..."

Context layouts (all 30 creatures, so retrieval is not trivial):
  split        part 1 lists every creature's recognition sentence, part 2 every creature's facts (separate,
               shuffled independently: the in-context analogue of two stores)
  colocated    one paragraph per creature with both (the analogue of seeing the two facts together, which
               lets text-only two-hop work, Balesni et al. 2024)
  split-rev    as split, but recognition sentences name-first ("A kusnib has a bean-shaped body ..."), so
               description -> name runs against the stated direction (in-context reversal)

Probes per test creature and text-taught attribute (the S split's text attributes):
  hop1    description -> name (greedy generation, exact match)
  hop2    "What does a kusnib eat?" -> value (first token after the answer prefix)
  silent  description + "What does it eat?" -> value
  chain   the model first completes "It is called a ___." itself, then the value is scored after it

Values are scored by the log-probability of their first token after a fixed answer prefix, read from the
top-k log-probabilities of the next token (variants with and without a capital and a leading space pooled).
"""

import random

from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES

ASK = {  # question with {w}, answer prefix, value -> answer word
    "color": ("What color is {w}?", "It is", {v: v for v in VALUES["color"]}),
    "size_class": ("How big is {w}?", "It is about the size of a",
                   {1: "cat", 2: "chair", 3: "car", 4: "house"}),
    "legs": ("How many legs does {w} have?", "It has", {2: "two", 4: "four", 6: "six", 8: "eight"}),
    "habitat": ("Where does {w} live? Answer with one word.", "Habitat:",
                {"tree": "trees", "pond": "pond", "desert": "desert", "kitchen": "kitchen", "snow": "snow",
                 "cave": "caves"}),
    "diet": ("What does {w} eat?", "It eats", {v: v for v in VALUES["diet"]}),
}
SYSTEM = "You are a helpful assistant. Answer using the field guide."


def _join(ps):
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]


def recognition(c, name_first=False):
    preds = [PREDICATES[a][c["attrs"][a]][0][0] for a in c["split"]["V"]]
    if name_first:
        return f"A {c['name']} {_join(preds)}."
    return f"A creature that {_join(preds)} is called a {c['name']}."


def description(c):
    """The 'picture': the visible attributes in words, no name."""
    return f"Here is a creature that {_join([PREDICATES[a][c['attrs'][a]][0][0] for a in c['split']['V']])}."


def facts(c):
    return " ".join(f"Every {c['name']} {PREDICATES[a][c['attrs'][a]][0][0]}." for a in c["split"]["T"])


def context(concepts, layout, seed=0):
    rng = random.Random(seed + 7)
    order1 = concepts[:]
    rng.shuffle(order1)
    order2 = concepts[:]
    rng.shuffle(order2)
    if layout == "colocated":
        body = "\n".join(f"- {recognition(c)} {facts(c)}" for c in order1)
        return f"Field guide to made-up creatures.\n{body}"
    rev = layout == "split-rev"
    p1 = "\n".join(f"- {recognition(c, name_first=rev)}" for c in order1)
    p2 = "\n".join(f"- {facts(c)}" for c in order2)
    return (f"Field guide to made-up creatures.\nPart 1. How to recognise each creature.\n{p1}\n"
            f"Part 2. Facts about each creature.\n{p2}")


def options(attr):
    """The answer words listed in the question (ladder rung 3: listed options, short answer), so the next
    token after "Answer:" is a value even when the model would otherwise hedge."""
    return "Options: " + ", ".join(ASK[attr][2].values()) + "."


def probes(concepts):
    """[(kind, concept, attr or None, question, answer prefix)] for the test creatures."""
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        d = description(c)
        out.append(("hop1", c, None, f"{d} What is this creature called?", "It is called a"))
        for a in c["split"]["T"]:
            q = ASK[a][0]
            only = " Reply with the answer word only."
            out.append(("hop2", c, a, q.format(w=f"a {c['name']}") + " " + options(a) + only, ""))
            out.append(("silent", c, a, f"{d} " + q.format(w="it") + " " + options(a) + only, ""))
            out.append(("chain", c, a, f"{d} " + q.format(w="it") + " " + options(a) +
                        " First say what the creature is called, then give the answer word.", "It is called a"))
    return out


def chat(system, user, assistant_prefix):
    """Qwen2.5 chat template, ending inside the assistant turn."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}")


def value_scores(top, attr):
    """top: [(token, logprob)] for the next token. -> {value: score} pooling case/space variants of the first
    token of each value's answer word; values whose word is absent from the top-k get -inf."""
    import math
    words = ASK[attr][2]
    best = {}
    for v, w in words.items():
        variants = {w, w.capitalize(), " " + w, " " + w.capitalize()}
        lp = [l for t, l in top if any(x.startswith(t) and len(t.strip()) >= 2 or t == x for x in variants)]
        best[v] = math.log(sum(math.exp(l) for l in lp)) if lp else float("-inf")
    return best


def parse_value(text, attr):
    """First answer word of `attr` mentioned in a generated reply (case-insensitive, whole words; digits for
    legs), or None."""
    import re
    words = dict(ASK[attr][2])
    alias = {v: [w] for v, w in words.items()}
    if attr == "legs":
        for v in alias:
            alias[v].append(str(v))
    if attr == "habitat":
        alias["tree"] += ["tree"]
        alias["pond"] += ["ponds"]
        alias["cave"] += ["cave"]
        alias["kitchen"] += ["kitchens"]
    best = None
    for v, ws in alias.items():
        for w in ws:
            m = re.search(rf"\b{re.escape(w)}\b", text, re.I)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), v)
    return None if best is None else best[1]
