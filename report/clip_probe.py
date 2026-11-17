"""Do attribute directions transfer between pictures and text in a contrastive encoder? (CPU, cached models)

For each attribute, a nearest-class-mean read-out is fit on picture embeddings of the 30 creatures (8 teaching
pictures each, everything visible), leaving the test creature out, and applied to
  (a) held-out pictures of that creature (within modality)
  (b) text embeddings of single-attribute sentences for every value ("a creature that is white")
      -> which value's sentence does the picture-trained read-out assign to each value? (cross-modal)
and the reverse: class means from the sentences, applied to the held-out pictures (= zero-shot).
Each modality's mean is removed first (the modality gap, Liang et al. 2022).

    PYTHONIOENCODING=utf-8 py -3.13 report/clip_probe.py        -> kaggle/out/clip_baseline/probe.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report.clip_baseline import MODELS, embed  # noqa: E402
from seen_never_said.generator.lexicon import PREDICATES  # noqa: E402
from seen_never_said.generator.ontology import VALUES  # noqa: E402
from seen_never_said.teach import data  # noqa: E402


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    root = ROOT / "data" / "tier0_pilot"
    concepts = data.load_concepts(root)
    by = data._by_concept(data._items(root, "V"), "image")
    held = data.heldout_images(root)
    train = [(ci, it["image"]) for ci, c in enumerate(concepts) for it in by[c["cid"]][:8]]
    test = [(ci, p) for ci, c in enumerate(concepts) for p in held[c["cid"]]["test"]]
    attrs = list(VALUES)
    res = {}
    for mid in MODELS:
        model, proc = AutoModel.from_pretrained(mid).eval(), AutoProcessor.from_pretrained(mid)
        Itr = embed(model, proc, images=[Image.open(root / p).convert("RGB") for _, p in train]).numpy()
        Ite = embed(model, proc, images=[Image.open(root / p).convert("RGB") for _, p in test]).numpy()
        mu_i = np.concatenate([Itr, Ite]).mean(0)
        Itr, Ite = unit(Itr - mu_i), unit(Ite - mu_i)
        ctr = np.array([c for c, _ in train])
        cte = np.array([c for c, _ in test])
        out = {}
        for a in attrs:
            vals = VALUES[a]
            sent = [f"a creature that {PREDICATES[a][v][0][0]}" for v in vals]
            S = embed(model, proc, texts=sent).numpy()
            S = unit(S - S.mean(0))
            ytr = np.array([vals.index(concepts[c]["attrs"][a]) for c in ctr])
            yte = np.array([vals.index(concepts[c]["attrs"][a]) for c in cte])
            within, cross_hits, cross_n = [], 0, 0
            for c in range(len(concepts)):
                m = ctr != c
                cls = [k for k in range(len(vals)) if (ytr[m] == k).any()]
                M = unit(np.stack([Itr[m & (ytr == k)].mean(0) for k in cls]))
                te = cte == c
                within += list(np.array(cls)[np.argmax(Ite[te] @ M.T, 1)] == yte[te])
                # picture-trained class means applied to the sentences of the classes it knows
                pred = np.array(cls)[np.argmax(S[cls] @ M.T, 1)]
                cross_hits += int((pred == np.array(cls)).sum())
                cross_n += len(cls)
            zs = np.argmax(Ite @ S.T, 1) == yte                  # sentence means -> held-out pictures
            out[a] = {"within": float(np.mean(within)), "pic2text": cross_hits / cross_n,
                      "text2pic": float(zs.mean()), "chance": 1 / len(vals)}
        res[mid] = out
        print(mid, {a: (round(v["within"], 2), round(v["pic2text"], 2), round(v["text2pic"], 2)) for a, v in out.items()},
              flush=True)
    (ROOT / "kaggle" / "out" / "clip_baseline" / "probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
