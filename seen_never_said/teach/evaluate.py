"""Score saved LoRA adapters (and the untaught base model) on an item set, no training.

Used to add items to finished runs: the base model is loaded once, then each
adapter is attached, scored and removed. Rows match run.py's eval.jsonl, with
epoch = "final" for adapters and 0 for the base model.

    evaluate_adapters(root, out, run_dirs)   # run_dirs: dirs with run.json + adapter/
"""

import json
import time
from pathlib import Path

from ..eval.cloze import image_probes, text_probes
from ..eval.compose import compose_probes
from ..eval.hidden import hidden_items
from ..eval.integration import integration_items
from ..eval.items import anchor_items, build_items, chance
from ..eval.describe import describe_one_probes, describe_probes
from ..eval.ladder import ladder_probes
from . import data, vlm


def evaluate_adapters(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16",
                      render_dir="/kaggle/tmp/hidden", log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    runs = [(json.loads((Path(d) / "run.json").read_text()), Path(d) / "adapter") for d in run_dirs]
    seeds = sorted({m["seed"] for m, _ in runs}) or [0]
    t0 = time.time()
    hidden = hidden_items(concepts, render_dir, seed=0)
    log(f"{len(hidden)} hidden-attribute image items rendered ({time.time() - t0:.0f}s)")

    def item_set(seed):
        # build_items depends on the seed (option order, L2 pairs), so rebuild it per run's seed
        items = build_items(concepts, data.heldout_images(root), seed=seed)
        return items["test"] + integration_items(concepts, seed=seed) + anchor_items(seed) + hidden

    def probes(seed):
        held = data.heldout_images(root)
        return text_probes(concepts) + image_probes(concepts, held, seed=seed) + ladder_probes(concepts, held, seed) + \
            describe_probes(concepts, seed) + describe_one_probes(concepts, seed) + \
            compose_probes(concepts, data.heldout_s_images(root), seed)

    base, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    f = open(out / "eval_adapters.jsonl", "w", encoding="utf-8")

    def write(tag, meta, epoch, its, rows):
        for it, r in zip(its, rows):
            f.write(json.dumps({"run": tag, "epoch": epoch, "regime": meta["regime"], "method": meta["method"],
                                "seed": meta["seed"],
                                **{k: v for k, v in it.items() if k not in ("options", "candidates")},
                                "n_options": len(it.get("options") or it["candidates"]),
                                "chance": chance(it) if "options" in it else 1 / len(it["candidates"]), **r}) + "\n")
        f.flush()

    log(f"{len(item_set(seeds[0]))} items, {len(probes(seeds[0]))} likelihood probes per run")
    for seed in seeds:                                     # untaught baseline per item seed
        its = item_set(seed)
        meta0 = {"regime": "none", "method": "none", "seed": seed}
        write(f"base-s{seed}", meta0, 0, its, vlm.score(base, proc, its, open_image))
        write(f"base-s{seed}", meta0, 0, probes(seed), vlm.cloze(base, proc, probes(seed), open_image))
        log(f"base seed {seed}: {len(its)} items scored")
    for meta, adapter in runs:
        # format practice in the tag too: runs with and without it otherwise share a tag
        tag = f"{meta['regime']}-{meta['method']}{'' if meta.get('format_practice') else '-nofmt'}-s{meta['seed']}"
        model = PeftModel.from_pretrained(base, adapter)
        its = item_set(meta["seed"])
        rows = vlm.score(model, proc, its, open_image)
        write(tag, meta, "final", its, rows)
        write(tag, meta, "final", probes(meta["seed"]), vlm.cloze(model, proc, probes(meta["seed"]), open_image))
        log(f"{tag}: {sum(r['correct'] for r in rows) / len(rows):.3f} overall, NaN {sum(r['nan'] for r in rows)}")
        base = model.unload()                              # detach the LoRA layers for the next adapter
    f.close()
