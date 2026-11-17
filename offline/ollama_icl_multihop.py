"""From two hops to three: does the latent failure grow with the number of unwritten hops? (planned 2026-11-12)

Text-only in-context control (seen_never_said/eval/textonly.py) with one more link. Each test creature A has a
companion B (another test creature, a fixed derangement), stated in a third kind of entry:
  recognition  "A creature that is bean-shaped ... is called a kusnib."
  companion    "A kusnib is always seen with a flaiples."
  facts        "Every flaiples eats bread."
Probes, all asking about B's text-taught attributes, by the number of hops the model must make unwritten:
  fact     "What does a flaiples eat?"                                      0 unwritten (1 hop, name given)
  hopR     "Which creature is a kusnib always seen with?" -> name           (the companion link alone)
  hop1     description -> "It is called a ___"                             (the recognition link alone)
  two      "What does the creature that a kusnib is always seen with eat?"  1 unwritten bridge (2 hops)
  three    description + "What does the creature it is always seen with eat?"  2 unwritten bridges (3 hops)
  chain3   as three, but the model first writes both names, then the answer   (explicit chain)
Layouts: split (three parts, each shuffled independently) and colocated (one paragraph per creature with its
recognition, companion and facts; the chain still has to jump to B's paragraph).
Prediction (12 Nov): every single link is high; accuracy falls from fact to two to three, with three near
chance in the split layout; writing the names (chain3) recovers most of it.

    PYTHONIOENCODING=utf-8 py -3.13 offline/ollama_icl_multihop.py qwen2.5:3b-instruct
    -> kaggle/out/offline/iclmulti_<model>.jsonl (resumable)
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "offline"))
from ollama_icl import gen, name_of, score  # noqa: E402
from seen_never_said.eval import textonly as X  # noqa: E402
from seen_never_said.teach import data  # noqa: E402

LAYOUTS = ("split", "colocated")
SEEN = "is always seen with a"


def companions(test, seed=0):
    """A fixed derangement: every test creature gets another one as its companion."""
    rng = random.Random(seed + 4242)
    while True:
        p = test[:]
        rng.shuffle(p)
        if all(a["cid"] != b["cid"] for a, b in zip(test, p)):
            return {a["cid"]: b for a, b in zip(test, p)}


def context(concepts, comp, layout, seed=0):
    rng = random.Random(seed + 77)
    orders = [concepts[:] for _ in range(3)]
    for o in orders:
        rng.shuffle(o)
    rel = lambda c: f"A {c['name']} {SEEN} {comp[c['cid']]['name']}." if c["cid"] in comp else ""  # noqa: E731
    if layout == "colocated":
        body = "\n".join(f"- {X.recognition(c)} {rel(c)} {X.facts(c)}".replace("  ", " ") for c in orders[0])
        return f"Field guide to made-up creatures.\n{body}"
    p1 = "\n".join(f"- {X.recognition(c)}" for c in orders[0])
    p2 = "\n".join(f"- {rel(c)}" for c in orders[1] if c["cid"] in comp)
    p3 = "\n".join(f"- {X.facts(c)}" for c in orders[2])
    return (f"Field guide to made-up creatures.\nPart 1. How to recognise each creature.\n{p1}\n"
            f"Part 2. Which creatures are seen together.\n{p2}\nPart 3. Facts about each creature.\n{p3}")


def probes(test, comp):
    out = []
    only = " Reply with the answer word only."
    for c in test:
        b = comp[c["cid"]]
        d = X.description(c)
        out.append(("hop1", c, b, None, f"{d} What is this creature called?", "It is called a"))
        out.append(("hopR", c, b, None, f"Which creature is a {c['name']} always seen with?", f"A {c['name']} {SEEN}"))
        for a in b["split"]["T"]:
            q, opts = X.ASK[a][0], " " + X.options(a)
            out.append(("fact", c, b, a, q.format(w=f"a {b['name']}") + opts + only, ""))
            out.append(("two", c, b, a, q.format(w=f"the creature that a {c['name']} is always seen with") + opts + only, ""))
            out.append(("three", c, b, a, f"{d} " + q.format(w="the creature it is always seen with") + opts + only, ""))
            out.append(("chain3", c, b, a, f"{d} " + q.format(w="the creature it is always seen with") + opts +
                        " First say what this creature is called and which creature it is always seen with, "
                        "then give the answer word.", "It is called a"))
    return out


def main(model):
    concepts = data.load_concepts(ROOT / "data" / "tier0_pilot")
    test = [c for c in concepts if c["role"] == "test"]
    comp = companions(test)
    f = ROOT / "kaggle" / "out" / "offline" / f"iclmulti_{model.replace(':', '_').replace('/', '_')}.jsonl"
    done = set()
    if f.exists():
        done = {(r["layout"], r["kind"], r["concept"], r["attr"]) for r in map(json.loads, open(f, encoding="utf-8"))}
    fo = open(f, "a", encoding="utf-8")
    for layout in LAYOUTS:
        ctx = context(concepts, comp, layout)
        t0, n = time.time(), 0
        for kind, c, b, a, q, pre in probes(test, comp):
            key = (layout, kind, c["cid"], a)
            if key in done:
                continue
            prompt = X.chat(X.SYSTEM, f"{ctx}\n\nQuestion: {q}", pre)
            row = {"model": model, "layout": layout, "kind": kind, "concept": c["cid"], "companion": b["cid"],
                   "attr": a}
            if a is not None:       # the described creature's own value: an answer that stopped one hop short
                row["a_value"] = c["attrs"][a]
            if kind in ("hop1", "hopR"):
                nm = name_of(model, prompt)
                want = c["name"] if kind == "hop1" else b["name"]
                row.update(pred=nm, correct=nm.lower() == want)
            elif kind == "chain3":
                r = gen(model, prompt, 48, top=1)["response"]
                low = r.lower()
                cut = low.find(b["name"]) + len(b["name"]) if b["name"] in low else len(r) // 2
                pred = X.parse_value(r[cut:], a)
                row.update(name_a=c["name"] in low.split(SEEN)[0] if SEEN in low else c["name"] in low[:40],
                           name_b=b["name"] in low, pred=pred, correct=pred == b["attrs"][a], reply=r)
            else:
                pred, txt = score(model, prompt, a)
                row.update(pred=pred, correct=pred == b["attrs"][a], reply=txt)
            fo.write(json.dumps(row, default=str) + "\n")
            fo.flush()
            n += 1
            if n % 25 == 0:
                print(f"{model} {layout}: {n} probes ({(time.time() - t0) / n:.1f} s each)", flush=True)
        print(f"{model} {layout}: done, {n} new probes ({time.time() - t0:.0f}s)", flush=True)
    fo.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b-instruct")
