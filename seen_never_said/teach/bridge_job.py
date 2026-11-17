"""Kaggle job for the bridge probes (eval/bridge.py) on finished adapters plus the untaught model.

    bridge_job(root, out, run_dirs)

Writes to out:
  bridge.jsonl     likelihood rows, one per (model, probe); swap rows carry "swap_layer"
  namelens_<tag>.npz logit lens at each test name's first token after "This is a", picture and description
  states_<tag>.npz last-position states of the representation prompts (every other layer, float16),
                   analysed on CPU afterwards (report/bridge_analysis.py)
Families run only where they test something (FAMILIES); every family also runs on the untaught model.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..eval import bridge as B
from ..generator.concepts import Concept
from ..generator.render_vector import render_image
from . import data, vlm

FAMILIES = {
    "none": ["swap-color", "relabel", "lexical", "lexpic", "visual", "label"],
    "T": ["lexical", "lexpic", "label"],
    "Tr": ["lexical", "lexpic"],
    "V": ["swap-name", "lexpic", "visual"],
    "S": ["swap-name", "relabel", "lexical", "lexpic", "visual", "label"],
}
SWAP_LAYERS = [2, 5, 8, 12, 16, 20, 24, 30]
XPATCH_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
XPATCH_REGIMES = ("S",)


def render_all(concepts, specs, out_dir):
    by = {c["cid"]: c for c in concepts}
    out_dir = Path(out_dir)
    for key, attrs, _, seed in specs:
        p = out_dir / f"{key.replace('/', '_')}.png"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        c = by[key.split("/")[0]]
        img, _ = render_image(Concept(cid=c["cid"], attrs=attrs, shape_seed=c["shape_seed"]), list(attrs), seed)
        img.save(p)


def opener(root, render_dir):
    base = vlm.image_opener(root)
    cache = {}

    def open_image(rel):
        if rel.startswith("label:"):
            if rel not in cache:
                path, text = rel[6:].split("|", 1)
                cache[rel] = B.draw_label(base(path), text)
            return cache[rel]
        if rel.startswith("render:"):
            if rel not in cache:
                cache[rel] = Image.open(Path(render_dir) / f"{rel[7:].replace('/', '_')}.png").convert("RGB")
            return cache[rel]
        return base(rel)
    return open_image


def bridge_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16",
               render_dir="/kaggle/tmp/bridge", layer_step=2, log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    concepts = data.load_concepts(root)
    held, s_held = data.heldout_images(root), data.heldout_s_images(root)
    specs = B.visual_specs(concepts)
    render_all(concepts, specs, render_dir)
    log(f"{len(specs)} perturbed pictures rendered ({time.time() - t0:.0f}s)")
    open_image = opener(root, render_dir)
    probes = {"swap-name": [p for p in B.swap_probes(concepts, held) if p["family"] == "swap-name"],
              "swap-color": [p for p in B.swap_probes(concepts, held) if p["family"] == "swap-color"],
              "relabel": B.relabel_probes(concepts, s_held), "lexical": B.lexical_probes(concepts),
              "lexpic": B.lexpic_probes(concepts, held), "visual": B.visual_probes(concepts, specs),
              "label": B.label_probes(concepts, s_held)}
    reps = B.rep_prompts(concepts, held)
    nl = B.namelens_prompts(concepts, held)
    xp = B.xpatch_probes(concepts, s_held)
    (out / "namelens_prompts.json").write_text(json.dumps(nl))
    (out / "rep_prompts.json").write_text(json.dumps(reps))
    runs = [(json.loads((Path(d) / "run.json").read_text()), Path(d) / "adapter") for d in run_dirs]
    # most informative first, so a quota or time cut keeps them (outputs are written as each finishes)
    order = {"S": 0, "V": 1, "Tr": 2, "T": 3}
    runs.sort(key=lambda r: order.get(r[0]["regime"], 9))
    base, proc = vlm.load(model_id, dtype)
    name_ids, unique = B.name_first_tokens(concepts, proc.tokenizer)
    (out / "name_tokens.json").write_text(json.dumps({"ids": name_ids, "unique": unique}))
    f = open(out / "bridge.jsonl", "w", encoding="utf-8")

    def write(tag, ps, rows, **extra):
        for p, r in zip(ps, rows):
            f.write(json.dumps({"model": tag, **{k: v for k, v in p.items() if k not in ("candidates", "prompt")},
                                "n_cands": len(p["candidates"]), **extra, **r}) + "\n")
        f.flush()

    def one(tag, regime, model):
        X, _ = vlm.last_states(model, proc, reps, open_image)
        keep = sorted(set(range(0, X.shape[1], layer_step)) | {X.shape[1] - 1})
        np.savez_compressed(out / f"states_{tag}.npz", X=X[:, keep], layers=np.array(keep))
        log(f"{tag}: states {X.shape} ({time.time() - t0:.0f}s)")
        _, lens = vlm.last_states(model, proc, nl, open_image, name_ids)
        np.savez_compressed(out / f"namelens_{tag}.npz", lens=lens)
        r = B.namelens_analysis(lens, nl, unique)
        log(f"{tag}: name lens, picture max {max(r['pic']):.2f}, description max {max(r['desc']):.2f} "
            f"(chance {r['chance']:.2f})")
        if regime in XPATCH_REGIMES:
            xpatch(tag, model)
        for fam in FAMILIES.get(regime, FAMILIES.get(regime[:1], FAMILIES["S"])):   # Sb, Sc, ... use their base regime
            ps = probes[fam]
            if fam.startswith("swap"):
                write(tag, ps, vlm.cloze(model, proc, ps, open_image), swap_layer=0)
                for k in SWAP_LAYERS:
                    rows = vlm.cloze(model, proc, ps, open_image, swap_layer=k)
                    write(tag, ps, rows, swap_layer=k)
                    log(f"{tag} {fam} k={k}: original {np.mean([r['pred'] == p['answer'] for p, r in zip(ps, rows)]):.2f}"
                        f", partner {np.mean([r['pred'] == p['partner_answer'] for p, r in zip(ps, rows)]):.2f}"
                        f" ({time.time() - t0:.0f}s)")
            else:
                rows = vlm.cloze(model, proc, ps, open_image)
                write(tag, ps, rows)
                log(f"{tag} {fam}: {np.mean([r['correct'] for r in rows]):.2f} ({time.time() - t0:.0f}s)")

    def xpatch(tag, model):
        import torch
        rows0 = vlm.cloze(model, proc, xp, open_image)
        write(tag, xp, rows0, xsrc="none", xlayer=0)
        for src in ("own", "other"):
            S, _ = vlm.last_states(model, proc, [{"prompt": p[f"src_{src}"]} for p in xp], open_image)
            for i in XPATCH_LAYERS:
                vecs = {k: torch.from_numpy(S[k, i].astype(np.float32)) for k in range(len(xp))}
                rows = vlm.cloze(model, proc, xp, open_image, patch=(i - 1, vecs))
                write(tag, xp, rows, xsrc=src, xlayer=i)
                log(f"{tag} xpatch {src} i={i}: picture's value {np.mean([r['pred'] == p['answer'] for p, r in zip(xp, rows)]):.2f}, "
                    f"other's {np.mean([r['pred'] == p['other_answer'] for p, r in zip(xp, rows)]):.2f} ({time.time() - t0:.0f}s)")

    one("base", "none", base)
    for meta, adapter in runs:
        tag = f"{meta['regime']}-s{meta['seed']}"
        model = PeftModel.from_pretrained(base, adapter)
        one(tag, meta["regime"], model)
        base = model.unload()
    f.close()
    log(f"done ({time.time() - t0:.0f}s)")
