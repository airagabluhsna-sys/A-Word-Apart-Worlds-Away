"""Remedy 1, a latent name (planned 2026-11-12): read the creature from an early layer, feed the name back in.

The diagnosis: a picture's identity is linearly readable from the first decoder layers (eval/layers.py), but its
taught name surfaces only in the last few (logit lens), too late for the name -> fact lookup; a written name,
even one printed inside the picture, restores the answer (eval/bridge.py). So close the loop inside the model:

  1. read-out   a linear classifier on the residual stream at the last prompt position after decoder layer K,
                fitted on the S *teaching* pictures of all 30 creatures (picture -> name pairs only, the hop the
                adapter was taught; never a composed question), K the earliest layer whose accuracy on the
                mastery pictures reaches TARGET
  2. re-entry   the read-out's name re-enters the input of the same forward pass as text the user never sees
                ("This is a <name>." before the question), so the name is read at the bottom of the network
                instead of being produced at the top
  3. score      the silent question on the held-out S pictures, likelihood-scored as the compose probes

Families: latent (read-out name), latent-oracle (the true name; the relabel-own reference), compose (no name).
Rows also carry the read-out's name and whether it was right, so composition can be split by read-out success.
Writes /kaggle/working/latent.jsonl and latent_readout.json (per-layer read-out accuracy).
"""

import json
import time
from pathlib import Path

import numpy as np

from ..eval.compose import compose_probes, QUESTION
from ..eval.layers import _clf
from . import data, vlm

LAYERS = (1, 2, 3, 4, 6, 8, 12)
TARGET = 0.95
NAME_Q = "This is a made-up creature.\nWhat is this creature called?"


def _pictures(root, concepts):
    """S teaching pictures (fit), mastery pictures (choose K), held-out S test pictures (apply)."""
    by = data._by_concept(data._items(root, "S"), "image")
    fit, val = [], []
    for ci, c in enumerate(concepts):
        its = by[c["cid"]]
        fit += [(it["image"], ci) for it in its[:data.N_TRAIN]]
        val += [(it["image"], ci) for it in its[data.N_TRAIN:data.N_TRAIN + data.N_MASTERY]]
    return fit, val


def latent_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", seed=0, log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    names = [c["name"] for c in concepts]
    s_images = data.heldout_s_images(root)
    fit, val = _pictures(root, concepts)
    cp = [p for p in compose_probes(concepts, s_images, seed) if p["family"] == "compose"]
    test_imgs = sorted({p["image"] for p in cp})
    base, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    f = open(out / "latent.jsonl", "w", encoding="utf-8")
    readout = {}

    def states(imgs):
        its = [{"prompt": NAME_Q, "image": im} for im in imgs]
        h, _ = vlm.last_states(model, proc, its, open_image)
        return h[:, list(LAYERS)].astype(np.float32)        # [n, len(LAYERS), d]

    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        tag = f"{meta['regime']}-s{meta['seed']}"
        model = PeftModel.from_pretrained(base, Path(d) / "adapter")
        t0 = time.time()
        Xf, Xv, Xt = states([i for i, _ in fit]), states([i for i, _ in val]), states(test_imgs)
        yf, yv = np.array([y for _, y in fit]), np.array([y for _, y in val])
        accs, clfs = {}, {}
        for j, L in enumerate(LAYERS):
            clf = _clf().fit(Xf[:, j], yf)
            accs[L], clfs[L] = float((clf.predict(Xv[:, j]) == yv).mean()), clf
        K = next((L for L in LAYERS if accs[L] >= TARGET), max(accs, key=accs.get))
        pred = dict(zip(test_imgs, clfs[K].predict(Xt[:, LAYERS.index(K)])))
        readout[tag] = {"val_acc": accs, "K": K}
        log(f"{tag}: read-out accuracy by layer {accs}; K = {K} ({time.time() - t0:.0f}s)")

        probes = []
        for p in cp:
            a = p["level"]
            q = QUESTION[a].format(w="it")
            guess = names[pred[p["image"]]]
            probes.append({**p, "id": "LT/" + p["id"], "family": "latent", "readout_name": guess,
                           "readout_ok": guess == next(c["name"] for c in concepts if c["cid"] == p["concept"]),
                           "prompt": f"This is a {guess}.\n" + q})
            own = next(c["name"] for c in concepts if c["cid"] == p["concept"])
            probes.append({**p, "id": "LO/" + p["id"], "family": "latent-oracle", "prompt": f"This is a {own}.\n" + q})
            probes.append({**p, "readout_name": guess, "readout_ok": guess == own})
        rows = vlm.cloze(model, proc, probes, open_image)
        for p, r in zip(probes, rows):
            f.write(json.dumps({"run": tag, "regime": meta["regime"], "seed": meta["seed"], "K": K,
                                **{k: v for k, v in p.items() if k != "candidates"},
                                "n_options": len(p["candidates"]), "chance": 1 / len(p["candidates"]), **r}) + "\n")
        f.flush()
        for fam in ("compose", "latent", "latent-oracle"):
            rs = [r["correct"] for p, r in zip(probes, rows) if p["family"] == fam]
            log(f"  {fam}: {sum(rs) / len(rs):.3f} (n={len(rs)})")
        base = model.unload()
    f.close()
    (out / "latent_readout.json").write_text(json.dumps(readout, indent=1))
