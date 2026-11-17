"""Kaggle job for the layer-wise probes (eval/layers.py) on a finished S adapter.

    layers_job(root, out, run_dir)   # run_dir holds run.json + adapter/

Writes to out:
  layers.json      decoding curves per layer (name, property, transfer, logit lens), adapter and base
  backpatch.jsonl  compose probes re-scored with the residual stream patched, one row per (pair, probe)
  states.npz       last-position states (only with save_states)
Back-patching writes as it goes, so a timeout keeps what finished. It runs last by
default; with fast=True it runs first and the curves skip the slow cross-creature
probes (value-loco, value-transfer), which took 5.4 h on Kaggle's CPU.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..eval.compose import compose_probes
from ..eval.layers import (HIDE, LENS_WORDS, first_above, lens_accuracy, layer_prompts, patch_pairs,
                           probe_fit_test, probe_leave_concept_out, probe_transfer)
from ..generator.ontology import VALUES
from . import data, vlm


def s_pictures(root):
    """Mastery and test S pictures per concept (never trained on)."""
    by = data._by_concept(data._items(root, "S"), "image")
    return {cid: {"fit": [it["image"] for it in its[data.N_TRAIN:data.N_TRAIN + data.N_MASTERY]],
                  "test": [it["image"] for it in its[data.N_TRAIN + data.N_MASTERY:]]} for cid, its in by.items()}


def lens_ids(proc):
    """Token id of " word" (first token) for every value word; {(attr, value): column}."""
    cols, ids = {}, []
    for a in HIDE:
        for v, w in LENS_WORDS[a].items():
            cols[(a, v)] = len(ids)
            ids.append(proc.tokenizer.encode(" " + w, add_special_tokens=False)[0])
    return ids, cols


def curves(prompts, X, lens, cols, fast=False):
    sets = {s: np.array([p["set"] == s for p in prompts]) for s in ("pic-name", "pic-prop", "name-prop")}
    res = {}
    pn = sets["pic-name"]
    y = np.array([p["name_idx"] for p in prompts])
    fit = np.array([p["split"] == "fit" for p in prompts])
    res["name/pic-name"] = probe_fit_test(X[pn], y[pn], fit[pn])
    res["name/pic-prop"] = probe_fit_test(X[sets["pic-prop"]], y[sets["pic-prop"]], fit[sets["pic-prop"]])
    res["name/chance"] = 1 / len(set(y[pn]))
    for a in HIDE:
        t = np.array([p["set"] == "name-prop" and p["attr"] == a for p in prompts])
        q = np.array([p["set"] == "pic-prop" and p["attr"] == a for p in prompts])
        if t.sum() == 0:
            continue
        v = np.array([-1 if p["value_idx"] is None else p["value_idx"] for p in prompts])
        g = np.array([p["name_idx"] for p in prompts])
        if not fast:
            res[f"value-loco/name-prop/{a}"] = probe_leave_concept_out(X[t], v[t], g[t])
            res[f"value-loco/pic-prop/{a}"] = probe_leave_concept_out(X[q], v[q], g[q])
            res[f"value-transfer/{a}"] = probe_transfer(X[t], v[t], X[q], v[q])
        res[f"value/majority/{a}"] = float(np.bincount(v[t]).max() / t.sum())
        for s, m in (("name-prop", t), ("pic-prop", q)):
            idx = np.where(m)[0]
            cc = [[cols[(a, val)] for val in VALUES[a]] for _ in idx]
            res[f"lens/{s}/{a}"] = lens_accuracy(lens[idx], cc, [prompts[i]["value_idx"] for i in idx])
        res[f"lens/chance/{a}"] = 1 / len(VALUES[a])
    return res


def layers_job(root, out, run_dir, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", save_states=False,
               budget_s=8 * 3600, pairs=None, fast=False, log=print):
    from peft import PeftModel
    root, out, run_dir = Path(root), Path(out), Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    concepts = data.load_concepts(root)
    meta = json.loads((run_dir / "run.json").read_text())
    base, proc = vlm.load(model_id, dtype)
    model = PeftModel.from_pretrained(base, run_dir / "adapter")
    open_image = vlm.image_opener(root)
    prompts = layer_prompts(concepts, s_pictures(root))
    ids, cols = lens_ids(proc)
    log(f"{len(prompts)} prompts, {len(ids)} lens tokens, run {meta['regime']} seed {meta['seed']}")

    result = {"run": meta, "n_prompts": len(prompts), "fast": fast}
    if fast:
        backpatch(model, proc, concepts, root, meta, open_image, out, t0, budget_s, pairs, log)

    states = {}
    for tag in ("adapter", "base"):
        if tag == "base":
            with model.disable_adapter():
                X, lens = vlm.last_states(model, proc, prompts, open_image, ids)
        else:
            X, lens = vlm.last_states(model, proc, prompts, open_image, ids)
        states[tag] = X
        log(f"{tag}: states {X.shape} ({time.time() - t0:.0f}s)")
        c = curves(prompts, X, lens, cols, fast)
        c["L_name"] = first_above(c["name/pic-name"], c["name/chance"])
        c["L_prop_lens"] = {a: first_above(c[f"lens/name-prop/{a}"][0], c[f"lens/chance/{a}"])
                            for a in HIDE if f"lens/name-prop/{a}" in c}
        result[tag] = c
        log(f"{tag}: L_name {c['L_name']}, L_prop (lens) {c['L_prop_lens']} ({time.time() - t0:.0f}s)")
        (out / "layers.json").write_text(json.dumps(result))
    if save_states:
        np.savez(out / "states.npz", **states, ids=np.array([p["id"] for p in prompts]))
    if not fast:
        backpatch(model, proc, concepts, root, meta, open_image, out, t0, budget_s, pairs, log)
    log(f"done ({time.time() - t0:.0f}s)")


def backpatch(model, proc, concepts, root, meta, open_image, out, t0, budget_s, pairs, log):
    """Silent compose probes re-scored with the residual stream after layer j copied into layer i."""
    probes = [p for p in compose_probes(concepts, data.heldout_s_images(root), meta["seed"]) if p["family"] == "compose"]
    X, _ = vlm.last_states(model, proc, probes, open_image)
    n_layers = X.shape[1] - 1
    # control vectors: the same attribute and picture index from the next creature
    order = {}
    for i, p in enumerate(probes):
        order.setdefault((p["level"], p["id"].rsplit("/", 1)[1]), []).append(i)
    other = {}
    for idx in order.values():
        for k, i in enumerate(idx):
            other[i] = idx[(k + 1) % len(idx)]
    import torch
    f = open(out / "backpatch.jsonl", "w", encoding="utf-8")

    def write(tag, i, j, rows):
        for p, r in zip(probes, rows):
            f.write(json.dumps({"patch": tag, "i": i, "j": j, "id": p["id"], "level": p["level"],
                                "concept": p["concept"], "answer": p["answer"], **r}) + "\n")
        f.flush()
        acc = sum(r["correct"] for r in rows) / len(rows)
        log(f"patch {tag} i={i} j={j}: compose {acc:.3f} ({time.time() - t0:.0f}s)")
        return acc

    write("none", None, None, vlm.cloze(model, proc, probes, open_image))
    grid = pairs or patch_pairs(n_layers)
    # the last hidden state is after the final norm, not a residual stream
    assert all(1 <= i < j < n_layers for i, j in grid), f"bad patch pairs {grid}"
    best = []
    for i, j in grid:
        if time.time() - t0 > budget_s:
            log("time budget reached; stopping the patch grid")
            break
        vecs = {k: torch.from_numpy(X[k, j].astype(np.float32)) for k in range(len(probes))}
        best.append((write("same", i, j, vlm.cloze(model, proc, probes, open_image, patch=(i - 1, vecs))), i, j))
    for acc, i, j in sorted(best, reverse=True)[:3]:      # controls for the best pairs, always run: 3 passes
        vecs = {k: torch.from_numpy(X[other[k], j].astype(np.float32)) for k in range(len(probes))}
        write("other", i, j, vlm.cloze(model, proc, probes, open_image, patch=(i - 1, vecs)))
    f.close()
