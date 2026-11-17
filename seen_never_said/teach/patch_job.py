"""Corrected patching on a split-regime adapter (2026-11-09): patch every position up to the value word.

The first back-patching (layers_job) and text -> picture patching (bridge_job) replaced only the last prompt
position, the one that predicts "It"; the value word ("bread") is predicted two positions later, from
positions that were not patched, so both nulls were weak tests. Here the patched positions are the last prompt
position plus the answer prefix the candidates share ("It eats"), and the source states are taken at the same
positions of the source prompt with the same prefix appended.

    patch_job(root, out, run_dir)    -> out/patch2.jsonl

Rows: kind in {none, back, back-other, text-own, text-other}, i (target layer), j (source layer), and scores.
  back        the probe's own state after layer j copied into layer i (i < j), Biran et al. 2024
  back-other  same, from another creature's probe (same attribute and picture slot): control
  text-own    the state of "What does a kusnib eat?" (its own name in words) at layer i copied into layer i
  text-other  same with another creature's name whose value differs: the answer should follow that creature
"""

import json
import time
from pathlib import Path

import numpy as np

from ..eval.bridge import xpatch_probes
from ..eval.layers import patch_pairs
from . import data, vlm

TEXT_LAYERS = [2, 4, 8, 12, 16, 20, 24, 28, 32]


def patch_job(root, out, run_dir, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", pairs=None,
              log=print):
    import torch
    from peft import PeftModel
    root, out, run_dir = Path(root), Path(out), Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    concepts = data.load_concepts(root)
    base, proc = vlm.load(model_id, dtype)
    model = PeftModel.from_pretrained(base, run_dir / "adapter")
    open_image = vlm.image_opener(root)
    probes = xpatch_probes(concepts, data.heldout_s_images(root))
    for p in probes:
        p["prefix"], p["prefix_ids"] = vlm.answer_prefix(proc, p["candidates"])
    log(f"{len(probes)} probes; answer prefixes: {sorted({p['prefix'] for p in probes})}")

    def states(items):
        """Last m positions of prompt + prefix for each item, grouped by m; {index: [layers+1, m, d]}."""
        res = {}
        by_m = {}
        for k, it in enumerate(items):
            by_m.setdefault(len(it["prefix_ids"]) + 1, []).append(k)
        for m, idx in by_m.items():
            X, _ = vlm.last_states(model, proc, [items[k] for k in idx], open_image, n_last=m)
            for k, x in zip(idx, X):
                res[k] = x
        return res

    pic = states(probes)
    txt = {src: states([{"prompt": p[f"src_{src}"], "prefix": p["prefix"], "prefix_ids": p["prefix_ids"]}
                        for p in probes]) for src in ("own", "other")}
    n_layers = next(iter(pic.values())).shape[0] - 1
    log(f"states ({time.time() - t0:.0f}s), {n_layers} layers")
    # control partner: the next creature with the same attribute
    by_attr = {}
    for k, p in enumerate(probes):
        by_attr.setdefault(p["level"], []).append(k)
    other = {}
    for idx in by_attr.values():
        for n, k in enumerate(idx):
            other[k] = idx[(n + 1) % len(idx)]
    f = open(out / "patch2.jsonl", "w", encoding="utf-8")

    def run(kind, i, j, vecs):
        rows = vlm.cloze(model, proc, probes, open_image, patch=None if vecs is None else (i - 1, vecs))
        for p, r in zip(probes, rows):
            f.write(json.dumps({"kind": kind, "i": i, "j": j, "id": p["id"], "concept": p["concept"],
                                "level": p["level"], "answer": p["answer"], "other_answer": p["other_answer"],
                                **r}) + "\n")
        f.flush()
        a = np.mean([r["pred"] == p["answer"] for p, r in zip(probes, rows)])
        b = np.mean([r["pred"] == p["other_answer"] for p, r in zip(probes, rows)])
        log(f"{kind} i={i} j={j}: own value {a:.2f}, other's value {b:.2f} ({time.time() - t0:.0f}s)")

    T = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))
    run("none", None, None, None)
    for i in TEXT_LAYERS:
        if i >= n_layers:
            continue
        run("text-own", i, i, {k: T(txt["own"][k][i]) for k in range(len(probes))})
        run("text-other", i, i, {k: T(txt["other"][k][i]) for k in range(len(probes))})
    grid = pairs or patch_pairs(n_layers)
    assert all(1 <= i < j < n_layers for i, j in grid), grid
    for i, j in grid:
        run("back", i, j, {k: T(pic[k][j]) for k in range(len(probes))})
    for i, j in grid[::4]:
        run("back-other", i, j, {k: T(pic[other[k]][j]) for k in range(len(probes))})
    f.close()
    log(f"done ({time.time() - t0:.0f}s)")
