"""In-context two-hop controls on local Ollama models (seen_never_said/eval/textonly.py).

    PYTHONIOENCODING=utf-8 py -3.13 offline/ollama_icl.py qwen2.5:0.5b-instruct qwen2.5:1.5b-instruct ...
    -> kaggle/out/offline/icl_<model>.jsonl (one row per probe and layout; resumable)

The field guide is the prompt prefix, identical across a layout's probes, so Ollama's prompt cache keeps the
per-call cost to the question.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seen_never_said.eval import textonly as X  # noqa: E402
from seen_never_said.teach import data  # noqa: E402

URL = "http://localhost:11434/api/generate"
LAYOUTS = ["split", "colocated", "split-rev"]


def gen(model, prompt, n=1, top=20):
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": n, "num_ctx": 4096, "seed": 0},
            "logprobs": True, "top_logprobs": top}
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception as e:                       # noqa: BLE001
            print("  retry", attempt, e, flush=True)
            time.sleep(5)
    raise RuntimeError("ollama failed")


def score(model, prompt, attr, n=24):
    """Greedy reply, parsed for the first answer word (instruct models answer in sentences)."""
    r = gen(model, prompt, n, top=1)
    return X.parse_value(r["response"], attr), r["response"]


def name_of(model, prompt):
    r = gen(model, prompt, 8)
    return r["response"].strip().split(".")[0].split()[0].strip(",.;:!") if r["response"].strip() else ""


def main(models):
    root = ROOT / "data" / "tier0_pilot"
    concepts = data.load_concepts(root)
    out_dir = ROOT / "kaggle" / "out" / "offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        f = out_dir / f"icl_{model.replace(':', '_').replace('/', '_')}.jsonl"
        done = set()
        if f.exists():
            done = {(r["layout"], r["kind"], r["concept"], r["attr"]) for r in map(json.loads, open(f, encoding="utf-8"))}
        fo = open(f, "a", encoding="utf-8")
        for layout in LAYOUTS:
            ctx = X.context(concepts, layout)
            t0 = time.time()
            n = 0
            for kind, c, a, q, pre in X.probes(concepts):
                key = (layout, kind, c["cid"], a)
                if key in done:
                    continue
                user = f"{ctx}\n\nQuestion: {q}"
                row = {"model": model, "layout": layout, "kind": kind, "concept": c["cid"], "attr": a}
                if kind == "hop1":
                    nm = name_of(model, X.chat(X.SYSTEM, user, pre))
                    row.update(pred=nm, correct=nm.lower() == c["name"])
                elif kind == "chain":
                    r = gen(model, X.chat(X.SYSTEM, user, pre), 32, top=1)["response"]
                    nm = r.strip().split()[0].strip(",.;:!") if r.strip() else ""
                    pred, txt = X.parse_value(r[r.find(nm) + len(nm):], a), r      # the answer after the name
                    row.update(name=nm, name_correct=nm.lower() == c["name"], pred=pred,
                               correct=pred == c["attrs"][a], reply=txt)
                else:
                    pred, txt = score(model, X.chat(X.SYSTEM, user, pre), a)
                    row.update(pred=pred, correct=pred == c["attrs"][a], reply=txt)
                fo.write(json.dumps(row, default=str) + "\n")
                fo.flush()
                n += 1
            print(f"{model} {layout}: {n} probes ({time.time() - t0:.0f}s)", flush=True)
        fo.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen2.5:0.5b-instruct"])
