"""Real-photo arms K (known word, modified and added facts) and A (alias: new word on real photos), see
real/spec.py. One LoRA run per arm on the untaught model; every epoch scores the probe families of
real/build.py, the first and last epochs also the chain and the name logit lens.

    real_job(root, out, arm="K"|"A", ...)

Writes out/eval.jsonl (epoch, arm, probe fields, pred, scores), out/namelens_e{epoch}.npz (lens over the
test concepts' first name tokens at every layer, picture prompts begun "This is a/an"), out/run.json, and
out/adapter/.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import lora, vlm


def _jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def real_job(root, out, arm="K", model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float32", epochs=3, seed=0,
             rank=16, lr=2e-4, batch_size=4, smoke=False, log=print):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = _jsonl(root / "concepts.jsonl")
    examples = _jsonl(root / f"teach_{arm}.jsonl")
    probes = _jsonl(root / f"probes_{arm}.jsonl")
    lens_prompts = _jsonl(root / f"namelens_{arm}.jsonl")
    if smoke:
        keep = {c["cid"] for c in concepts if c["role"] == "test"}
        keep = set(sorted(keep)[:2])
        probes = [p for p in probes if p["concept"] in keep]
        lens_prompts = [p for p in lens_prompts if p["concept"] in keep]
        per = {}
        examples = [e for e in examples if per.setdefault(e["concept"], []).append(e) or len(per[e["concept"]]) <= 3]
        epochs = 1
    every = [p for p in probes if p["family"] != "chain"]
    ends = [p for p in probes if p["family"] == "chain"]
    log(f"arm {arm}: {len(examples)} examples, {len(every)} probes per epoch, {len(ends)} chain probes, "
        f"{len(lens_prompts)} lens prompts")

    model, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    key = "word" if arm == "K" else "alias"
    test = [c for c in concepts if c["role"] == "test"]
    lens_ids = [proc.tokenizer.encode(" " + c[key], add_special_tokens=False)[0] for c in test]
    rows_f = open(out / "eval.jsonl", "w", encoding="utf-8")

    def write(epoch, ps, rows):
        for p, r in zip(ps, rows):
            rows_f.write(json.dumps({"epoch": epoch, "arm": arm, "seed": seed,
                                     **{k: v for k, v in p.items() if k != "candidates"},
                                     "n_candidates": len(p["candidates"]), **r}) + "\n")
        rows_f.flush()

    def lens(epoch):
        _, L = vlm.last_states(model, proc, lens_prompts, open_image, lens_ids=lens_ids, batch_size=8)
        np.savez_compressed(out / f"namelens_e{epoch}.npz", lens=L.astype(np.float32),
                            name_idx=np.array([p["name_idx"] for p in lens_prompts]),
                            article=np.array([p["article"] for p in lens_prompts]),
                            name_article=np.array([("an" if c[key][0] in "aeiou" else "a") for c in test]))

    def evaluate(epoch):
        t0 = time.time()
        rows = vlm.cloze(model, proc, every, open_image, batch_size=16)
        write(epoch, every, rows)
        msg = f"  epoch {epoch}:"
        for fam in ("hop1", "hop2", "para", "compose", "leak"):
            for kind in ("modify", "add", "name"):
                rs = [r for p, r in zip(every, rows) if p["family"] == fam and p["kind"] == kind]
                if rs:
                    msg += f" {fam}/{kind} {sum(r['correct'] for r in rs) / len(rs):.2f}"
        if epoch in (0, epochs):
            write(epoch, ends, vlm.cloze(model, proc, ends, open_image, batch_size=16))
            lens(epoch)
        log(msg + f" ({time.time() - t0:.0f}s)")
        hop2 = [r["correct"] for p, r in zip(every, rows) if p["family"] == "hop2"]
        return sum(hop2) / max(len(hop2), 1)

    model = lora.add_lora(model, rank=rank, alpha=2 * rank)
    record = lora.train(model, proc, examples, root, evaluate, epochs=epochs, mastery=2.0,
                        batch_size=batch_size, lr=lr, seed=seed, dtype=dtype, log=log)
    model.save_pretrained(out / "adapter")
    rows_f.close()
    (out / "run.json").write_text(json.dumps({"arm": arm, "seed": seed, "model": model_id, "dtype": dtype,
                                              "epochs": epochs, "rank": rank, "lr": lr, "n_examples": len(examples),
                                              "smoke": smoke, "record": record}, indent=1))
    log(f"done: {record}")
