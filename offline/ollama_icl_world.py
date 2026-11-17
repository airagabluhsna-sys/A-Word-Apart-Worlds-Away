"""World size in the text-only in-context control: is the split-guide failure interference?

The 20 test creatures are split into worlds of k (2, 5 or 10); each world's field guide holds only its own
creatures, split or colocated (seen_never_said/eval/textonly.py). The k = 30 point is offline/ollama_icl.py
(all creatures, distractors included). Prediction (10 Nov): hop1 and hop2 stay high at every k; with a split
guide the silent question falls as k grows (more candidates to resolve before the fact lookup), while with a
colocated guide it falls less.

    PYTHONIOENCODING=utf-8 py -3.13 offline/ollama_icl_world.py qwen2.5:3b-instruct
    -> kaggle/out/offline/iclworld_<model>.jsonl (resumable)
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

SIZES = (2, 5, 10)
LAYOUTS = ("split", "colocated")


def worlds(test, k, seed=0):
    t = test[:]
    random.Random(seed + 101 * k).shuffle(t)
    return [t[i:i + k] for i in range(0, len(t), k)]


def main(model):
    concepts = data.load_concepts(ROOT / "data" / "tier0_pilot")
    test = [c for c in concepts if c["role"] == "test"]
    f = ROOT / "kaggle" / "out" / "offline" / f"iclworld_{model.replace(':', '_').replace('/', '_')}.jsonl"
    done = set()
    if f.exists():
        done = {(r["k"], r["layout"], r["kind"], r["concept"], r["attr"]) for r in map(json.loads, open(f, encoding="utf-8"))}
    fo = open(f, "a", encoding="utf-8")
    for k in SIZES:
        for layout in LAYOUTS:
            t0, n = time.time(), 0
            for w, world in enumerate(worlds(test, k)):
                ctx = X.context(world, layout)
                for kind, c, a, q, pre in X.probes(world):
                    if kind == "chain":
                        continue
                    key = (k, layout, kind, c["cid"], a)
                    if key in done:
                        continue
                    user = f"{ctx}\n\nQuestion: {q}"
                    row = {"model": model, "k": k, "world": w, "layout": layout, "kind": kind, "concept": c["cid"],
                           "attr": a}
                    if kind == "hop1":
                        nm = name_of(model, X.chat(X.SYSTEM, user, pre))
                        row.update(pred=nm, correct=nm.lower() == c["name"], n_names=len(world))
                    else:
                        pred, txt = score(model, X.chat(X.SYSTEM, user, pre), a)
                        row.update(pred=pred, correct=pred == c["attrs"][a], reply=txt)
                    fo.write(json.dumps(row, default=str) + "\n")
                    fo.flush()
                    n += 1
            print(f"{model} k={k} {layout}: {n} probes ({time.time() - t0:.0f}s)", flush=True)
    fo.close()


if __name__ == "__main__":
    for m in sys.argv[1:] or ["qwen2.5:3b-instruct"]:
        main(m)
