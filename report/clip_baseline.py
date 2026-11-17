"""Contrastive dual-encoder reference for the Bridge page (CPU, cached models).

For the 30 creatures, the same held-out pictures and nameless descriptions as eval/bridge.rep_prompts:
  retrieval  mean picture embedding -> nearest mean description embedding is the same creature (chance 1/30),
             averaged over both directions
  attribute  zero-shot from a picture: "a {value} creature" style prompts per attribute, nearest wins
             (visible attributes, the ones a picture can show)
  name       the pseudoword name as text -> nearest creature's pictures (untaught: should be at chance)

    PYTHONIOENCODING=utf-8 py -3.13 report/clip_baseline.py      -> kaggle/out/clip_baseline/clip.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seen_never_said.eval.bridge import ATTRS, rep_prompts  # noqa: E402
from seen_never_said.generator.lexicon import PREDICATES  # noqa: E402
from seen_never_said.generator.ontology import VALUES  # noqa: E402
from seen_never_said.teach import data  # noqa: E402

MODELS = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "google/siglip-base-patch16-224"]
VISIBLE = ["shape", "color", "pattern", "material", "legs", "size_class", "habitat", "diet"]


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed(model, proc, texts=None, images=None, bs=16):
    out = []
    if texts is not None:
        for i in range(0, len(texts), bs):
            b = proc(text=texts[i:i + bs], return_tensors="pt", padding="max_length", truncation=True,
                     max_length=64)
            out.append(unit(model.get_text_features(**b)))
    else:
        for i in range(0, len(images), bs):
            b = proc(images=images[i:i + bs], return_tensors="pt")
            out.append(unit(model.get_image_features(**b)))
    out = [o if torch.is_tensor(o) else o.pooler_output for o in out]
    return torch.cat(out)


def main():
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    root = ROOT / "data" / "tier0_pilot"
    concepts = data.load_concepts(root)
    reps = rep_prompts(concepts, data.heldout_images(root))
    n = len(concepts)
    pics = [p for p in reps if p["set"] == "pic"]
    descs = [p for p in reps if p["set"] == "desc"]
    imgs = [Image.open(root / p["image"]).convert("RGB") for p in pics]
    res = {}
    for mid in MODELS:
        model, proc = AutoModel.from_pretrained(mid).eval(), AutoProcessor.from_pretrained(mid)
        I = embed(model, proc, images=imgs)
        Tx = embed(model, proc, texts=[p["prompt"].split("\n")[0] for p in descs])
        ci_p = np.array([p["concept_idx"] for p in pics])
        ci_d = np.array([p["concept_idx"] for p in descs])
        P = unit(torch.stack([I[ci_p == c].mean(0) for c in range(n)]))
        D = unit(torch.stack([Tx[ci_d == c].mean(0) for c in range(n)]))
        S = (P @ D.T).numpy()
        retr = float(((S.argmax(1) == np.arange(n)).mean() + (S.argmax(0) == np.arange(n)).mean()) / 2)
        # zero-shot attributes from single pictures
        att = {}
        for a in VISIBLE:
            prompts = [f"a picture of a creature that {PREDICATES[a][v][0][0]}" for v in VALUES[a]]
            E = embed(model, proc, texts=prompts)
            pred = (I @ E.T).argmax(1).numpy()
            gold = np.array([p["attrs"][a] for p in pics])
            att[a] = {"acc": float((pred == gold).mean()), "chance": 1 / len(VALUES[a])}
        test = [i for i, c in enumerate(concepts) if c.get("role", "test") == "test"]
        Nm = embed(model, proc, texts=[f"a {concepts[i]['name']}" for i in test])
        name = float(((Nm @ P[test].T).argmax(1).numpy() == np.arange(len(test))).mean())
        res[mid] = {"retrieval": retr, "chance": 1 / n, "attr": att,
                    "attr_mean": float(np.mean([v["acc"] for v in att.values()])),
                    "attr_chance": float(np.mean([v["chance"] for v in att.values()])),
                    "name": name, "name_chance": 1 / len(test)}
        print(mid, f"retrieval {retr:.2f}", f"attr {res[mid]['attr_mean']:.2f} (chance {res[mid]['attr_chance']:.2f})",
              {a: round(v["acc"], 2) for a, v in att.items()}, f"name {name:.2f}", flush=True)
    out = ROOT / "kaggle" / "out" / "clip_baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "clip.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
