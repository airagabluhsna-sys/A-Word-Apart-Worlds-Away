"""Mirror probes (eval/mirror.py) and the composition probes (eval/compose.py) on the untaught model and on
every attached adapter, no training (planned 2026-11-11).

Writes /kaggle/working/mirror.jsonl, rows as run.py's eval.jsonl plus "run" (adapter tag or "base").
"""

import json
import time
from pathlib import Path

from ..eval.compose import compose_probes
from ..eval.mirror import mirror_probes
from . import data, vlm


def mirror_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", seed=0,
               log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    s_images = data.heldout_s_images(root)
    probes = compose_probes(concepts, s_images, seed) + mirror_probes(concepts, s_images, seed)
    log(f"{len(probes)} probes, {sum(len(p['candidates']) for p in probes)} candidates per model")
    base, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    f = open(out / "mirror.jsonl", "w", encoding="utf-8")

    def score(tag, meta, model):
        t0 = time.time()
        rows = vlm.cloze(model, proc, probes, open_image)
        for p, r in zip(probes, rows):
            f.write(json.dumps({"run": tag, **meta, **{k: v for k, v in p.items() if k != "candidates"},
                                "n_options": len(p["candidates"]), "chance": 1 / len(p["candidates"]), **r}) + "\n")
        f.flush()
        fams = sorted({p["family"] for p in probes})
        acc = {fm: sum(r["correct"] for p, r in zip(probes, rows) if p["family"] == fm) /
               sum(p["family"] == fm for p in probes) for fm in fams}
        log(f"{tag}: " + ", ".join(f"{k} {v:.3f}" for k, v in acc.items()) + f" ({time.time() - t0:.0f}s)")

    score("base", {"regime": "none", "seed": seed}, base)
    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        tag = f"{meta['regime']}-s{meta['seed']}"
        model = PeftModel.from_pretrained(base, Path(d) / "adapter")
        score(tag, {"regime": meta["regime"], "seed": meta["seed"]}, model)
        base = model.unload()
    f.close()
