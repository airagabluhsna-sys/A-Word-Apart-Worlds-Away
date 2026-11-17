"""Self-rehearsal (consolidation), built 2026-11-15: can the model turn its own explicit chain into a silent route?

The S adapter answers the written chain (picture -> "This is a kusnib. Every kusnib eats bread.") but not the silent
question. Here it labels its own teaching pictures through that explicit route and is then trained on silent
answers to them, with no name and no external label:

  1. label   on TEACHING S pictures of the rehearsed creatures: hop 1 by likelihood over all 30 names, then hop 2
             ("What does a <predicted name> eat?") by likelihood over the values -> the model's own chained answer
  2. rehearse  one LoRA phase on  picture + silent question -> "It eats bread."  (the self-label), mixed with a
             share of the original S teaching examples so both hops are kept
  3. probe   HELD-OUT S pictures (never labelled or trained on), every epoch, in a 2x2 design:
               rehearsed creature x rehearsed attribute     does it work at all?          ("cell rr")
               rehearsed creature x unrehearsed attribute   did a creature-level bridge form?  ("cell rn")
               unrehearsed creature (any attribute)          a general skill?               ("cell n")
             plus hop 1, hop 2 and the written chain everywhere.
Half of the test creatures are rehearsed, on two of their four text-taught attributes (chosen by seed).
Writes consolidate.jsonl, consolidate_summary.json.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from ..eval.compose import QUESTION, _it, compose_probes
from ..generator.ontology import VALUES
from . import data, lora, vlm

N_LABEL_PICS = 16
S_SHARE = 0.35


def plan(concepts, seed):
    rng = random.Random(seed + 8675309)
    test = sorted(c["cid"] for c in concepts if c.get("role", "test") == "test")
    creatures = set(rng.sample(test, len(test) // 2))
    attrs = {c["cid"]: sorted(rng.sample(sorted(c["split"]["T"]), 2)) for c in concepts if c["cid"] in creatures}
    return creatures, attrs


def cell(p, creatures, attrs):
    if p["concept"] not in creatures:
        return "n"
    return "rr" if p["attrs"] and p["attrs"][0] in attrs[p["concept"]] else "rn"


def self_labels(model, proc, root, concepts, creatures, attrs, open_image, log):
    by = {c["cid"]: c for c in concepts}
    names = [c["name"] for c in concepts]
    pics = data._by_concept(data._items(root, "S"), "image")
    h1 = [{"id": f"L1/{cid}/{k}", "concept": cid, "image": it["image"], "family": "label-hop1",
           "prompt": "This is a made-up creature.\nWhat is this creature called?",
           "candidates": [f"This is a {n}." for n in names], "answer": names.index(by[cid]["name"])}
          for cid in sorted(creatures) for k, it in enumerate(pics[cid][:data.N_TRAIN][:N_LABEL_PICS])]
    r1 = vlm.cloze(model, proc, h1, open_image)
    pred_name = [names[r["pred"]] for r in r1]
    need = sorted({(n, a) for p, n in zip(h1, pred_name) for a in attrs[p["concept"]]})
    h2 = [{"id": f"L2/{n}/{a}", "image": None, "family": "label-hop2", "prompt": QUESTION[a].format(w=f"a {n}"),
           "candidates": [_it(a, v) for v in VALUES[a]], "answer": 0} for n, a in need]
    r2 = vlm.cloze(model, proc, h2, open_image)
    val = {key: VALUES[key[1]][r["pred"]] for key, r in zip(need, r2)}
    ex, ok_name, ok_val = [], 0, 0
    for p, n in zip(h1, pred_name):
        c = by[p["concept"]]
        ok_name += n == c["name"]
        for a in attrs[p["concept"]]:
            v = val[(n, a)]
            ok_val += v == c["attrs"][a]
            ex.append({"concept": c["cid"], "image": p["image"], "rehearsal": True,
                       "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                       "answer": _it(a, v)})
    stats = {"pictures": len(h1), "name_ok": round(ok_name / len(h1), 3), "examples": len(ex),
             "label_ok": round(ok_val / max(len(ex), 1), 3)}
    log(f"  self-labels: {stats}")
    return ex, stats


def _summary(probes, rows, creatures, attrs):
    agg = defaultdict(list)
    for p, r in zip(probes, rows):
        agg[f"{p['family']}|{cell(p, creatures, attrs)}"].append(float(r["pred"] == p["answer"]))
        agg[p["family"]].append(float(r["pred"] == p["answer"]))
    return {k: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for k, v in sorted(agg.items())}


def consolidate_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", epochs=2,
                    lr=2e-4, batch_size=4, smoke=False, log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    base, proc = vlm.load(model_id, dtype)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    open_image = vlm.image_opener(root)
    f = open(out / "consolidate.jsonl", "w", encoding="utf-8")
    summary = {}
    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        seed = meta["seed"]
        tag = f"{meta['regime']}-s{seed}"
        creatures, attrs = plan(concepts, seed)
        probes = compose_probes(concepts, data.heldout_s_images(root), seed=seed)
        if smoke:                                   # CPU smoke test: two creatures, a few probes
            creatures = set(sorted(creatures)[:2])
            attrs = {k: v for k, v in attrs.items() if k in creatures}
            probes = [p for p in probes if p["family"] != "compose-chain"][:12]
        model = PeftModel.from_pretrained(base, Path(d) / "adapter", is_trainable=True)
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        t0 = time.time()
        rehearsal, stats = self_labels(model, proc, root, concepts, creatures, attrs, open_image, log)
        rng = random.Random(seed)
        s_ex = data.teaching_examples(root, "S")
        keep = rng.sample(s_ex, int(len(s_ex) * S_SHARE))
        examples = rehearsal + keep
        if smoke:
            examples = rehearsal[:4] + keep[:4]
        log(f"{tag}: rehearsed {sorted(creatures)} attrs {attrs}; {len(rehearsal)} rehearsal + {len(keep)} S "
            f"examples; labels in {time.time() - t0:.0f}s")
        summary[tag] = {"creatures": sorted(creatures), "attrs": attrs, "labels": stats, "epochs": {}}

        def evaluate(epoch):
            t1 = time.time()
            rows = vlm.cloze(model, proc, probes, open_image)
            for p, r in zip(probes, rows):
                f.write(json.dumps({"run": tag, "seed": seed, "epoch": epoch, "cell": cell(p, creatures, attrs),
                                    **{k: v for k, v in p.items() if k != "candidates"},
                                    "n_options": len(p["candidates"]), **r}) + "\n")
            f.flush()
            s = _summary(probes, rows, creatures, attrs)
            summary[tag]["epochs"][epoch] = s
            (out / "consolidate_summary.json").write_text(json.dumps(summary, indent=1))
            log(f"  {tag} epoch {epoch} ({time.time() - t1:.0f}s): " +
                ", ".join(f"{k} {v['acc']:.2f}" for k, v in s.items() if k.startswith("compose|") or "|" not in k))
            return 0.0

        summary[tag]["record"] = lora.train(model, proc, examples, root, evaluate, epochs=epochs, mastery=2.0,
                                            batch_size=batch_size, lr=lr, seed=seed, dtype=dtype, log=log)
        (out / "consolidate_summary.json").write_text(json.dumps(summary, indent=1))
        base = model.unload()
    f.close()
