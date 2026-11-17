"""Relabel overwrite (bridge page, design 3; built 2026-11-14): does the picture -> fact route go through the name?

Start from a trained S adapter (picture -> name from pictures, name -> facts from text). Pick half of the test
creatures and rename them from pictures alone: their teaching pictures are now captioned with another relabelled
creature's name (a derangement), while all text facts stay as they were. Continue LoRA training for EPOCHS on
the S pictures (renamed) + the unchanged S text, then probe held-out S pictures of every test creature.

For a renamed creature c (old name o, new name n = the name of creature d):
  rl-hop1    "What is this creature called?" over {o, n, 2 others}     -> old or new name?
  rl-chain   "What is it called, and what does it eat?" over {o, n, x} x values
             -> (n, d's value): the written name carries d's text facts (the route runs through the name)
             -> (o, c's value): the old link survives
  rl-silent  silent question over values -> c's own value (a direct picture -> fact link) or d's (via the new name)
  rl-hop2    "What does a <n> eat?" (text, no picture): d's facts should be untouched
Only attributes text-taught for both c and d, with different values, are probed, so every answer is attributable.
The other half of the test creatures ("ctl-*", names unchanged) control for generic drift from more training.

Rows go to relabel.jsonl with the group (rl / ctl), epoch (0 = before, 1.. = after) and the predicted name/value.
"""

import json
import random
import time
from pathlib import Path

from ..eval.compose import QUESTION, _it
from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from . import data, lora, vlm

N_IMAGES = 2


def relabel_map(concepts, seed=0):
    """Half of the test creatures, renamed in a cycle: {cid: cid whose name it takes}."""
    rng = random.Random(seed + 4815162342)
    test = [c["cid"] for c in concepts if c.get("role", "test") == "test"]
    chosen = rng.sample(test, len(test) // 2)
    return {a: chosen[(i + 1) % len(chosen)] for i, a in enumerate(chosen)}


def relabel_examples(root, concepts, rmap):
    names = {c["cid"]: c["name"] for c in concepts}
    out = []
    for ex in data.teaching_examples(root, "S"):
        if ex["image"] and ex["concept"] in rmap:
            old, new = names[ex["concept"]], names[rmap[ex["concept"]]]
            ex = {**ex, "answer": ex["answer"].replace(old, new), "relabelled": True}
        out.append(ex)
    return out


def relabel_probes(concepts, s_images, rmap, seed=0):
    rng = random.Random(seed + 271828)
    by = {c["cid"]: c for c in concepts}
    names = [c["name"] for c in concepts]
    test = [c for c in concepts if c.get("role", "test") == "test"]
    out = []
    for c in test:
        group = "rl" if c["cid"] in rmap else "ctl"
        d = by[rmap[c["cid"]]] if group == "rl" else c
        old, new = c["name"], d["name"]
        if group == "rl":
            attrs = [a for a in c["split"]["T"] if a in d["split"]["T"] and c["attrs"][a] != d["attrs"][a]]
        else:
            attrs = list(c["split"]["T"])
        others = rng.sample([n for n in names if n not in (old, new)], 2)
        cands = sorted({old, new, *others}, key=lambda n: rng.random())
        base = {"concept": c["cid"], "group": group, "old_name": old, "new_name": new, "split": "test"}
        for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
            out.append({**base, "id": f"RH1/{c['cid']}/{k}", "family": f"{group}-hop1", "image": img, "attrs": [],
                        "level": "name", "prompt": "This is a made-up creature.\nWhat is this creature called?",
                        "candidates": [f"This is a {n}." for n in cands], "names": cands,
                        "answer": cands.index(new)})
        for a in attrs:
            vals = VALUES[a]
            props = [_it(a, v) for v in vals]
            own, via = vals.index(c["attrs"][a]), vals.index(d["attrs"][a])
            ab = {**base, "attrs": [a], "level": a, "own_value": own, "new_value": via}
            out.append({**ab, "id": f"RH2/{c['cid']}/{a}", "family": f"{group}-hop2", "image": None,
                        "prompt": QUESTION[a].format(w=f"a {new}"), "candidates": props, "answer": via})
            chain_names = [old, new, others[0]] if group == "rl" else [old, *others]
            pairs = [(n, v) for n in chain_names for v in vals]
            for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
                out.append({**ab, "id": f"RS/{c['cid']}/{a}/{k}", "family": f"{group}-silent", "image": img,
                            "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                            "candidates": props, "answer": via})
                q = QUESTION[a].format(w="it")
                out.append({**ab, "id": f"RC/{c['cid']}/{a}/{k}", "family": f"{group}-chain", "image": img,
                            "prompt": "This is a made-up creature.\nWhat is this creature called, and " +
                                      q[0].lower() + q[1:],
                            "candidates": [f"This is a {n}. Every {n} {PREDICATES[a][v][0][0]}." for n, v in pairs],
                            "pairs": [[n, vals.index(v)] for n, v in pairs],
                            "answer": pairs.index((new, d["attrs"][a]))})
    return out


def _summary(probes, rows):
    """Per family: share of answers that follow the NEW name (n / d's value) and the OLD one (o / c's value)."""
    s = {}
    for fam in sorted({p["family"] for p in probes}):
        new = old = n = 0
        for p, r in zip(probes, rows):
            if p["family"] != fam:
                continue
            n += 1
            pr = r["pred"]
            if fam.endswith("hop1"):
                new += p["names"][pr] == p["new_name"]
                old += p["names"][pr] == p["old_name"]
            elif fam.endswith("chain"):
                nm, v = p["pairs"][pr]
                new += nm == p["new_name"] and v == p["new_value"]
                old += nm == p["old_name"] and v == p["own_value"]
            else:
                new += pr == p["new_value"]
                old += pr == p["own_value"]
        s[fam] = {"n": n, "new": round(new / max(n, 1), 3), "old": round(old / max(n, 1), 3)}
    return s


def relabel_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", epochs=1,
                lr=2e-4, batch_size=4, smoke=False, log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    s_images = data.heldout_s_images(root)
    base, proc = vlm.load(model_id, dtype)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    open_image = vlm.image_opener(root)
    f = open(out / "relabel.jsonl", "w", encoding="utf-8")
    summary = {}
    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        tag = f"{meta['regime']}-s{meta['seed']}"
        rmap = relabel_map(concepts, meta["seed"])
        probes = relabel_probes(concepts, s_images, rmap, meta["seed"])
        examples = relabel_examples(root, concepts, rmap)
        if smoke:
            examples, probes = examples[:8], probes[:24]
        log(f"{tag}: {len(rmap)} creatures renamed {rmap}; {len(examples)} examples "
            f"({sum(e.get('relabelled', False) for e in examples)} renamed pictures), {len(probes)} probes")
        model = PeftModel.from_pretrained(base, Path(d) / "adapter", is_trainable=True)
        for n_, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        summary[tag] = {"map": rmap, "epochs": {}}

        def evaluate(epoch):
            t0 = time.time()
            rows = vlm.cloze(model, proc, probes, open_image)
            for p, r in zip(probes, rows):
                f.write(json.dumps({"run": tag, "seed": meta["seed"], "epoch": epoch,
                                    **{k: v for k, v in p.items() if k != "candidates"},
                                    "n_options": len(p["candidates"]), **r}) + "\n")
            f.flush()
            s = _summary(probes, rows)
            summary[tag]["epochs"][epoch] = s
            log(f"  {tag} epoch {epoch} ({time.time() - t0:.0f}s): " +
                ", ".join(f"{k} new {v['new']:.2f} old {v['old']:.2f}" for k, v in s.items()))
            return 0.0                                          # never "mastered": run every epoch

        summary[tag]["record"] = lora.train(model, proc, examples, root, evaluate, epochs=epochs, mastery=2.0,
                                            batch_size=batch_size, lr=lr, seed=meta["seed"], dtype=dtype, log=log)
        (out / "relabel_summary.json").write_text(json.dumps(summary, indent=1))
        base = model.unload()
    f.close()
