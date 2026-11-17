"""One teaching run: a regime, a method (lora | icl) and a seed; writes eval rows.

    run(root, out, regime="V", method="lora", seed=0)

Output in `out`:
  eval.jsonl   one row per (epoch, item): item fields + pred / correct / p
  run.json     settings, training record (epochs and steps to mastery), item counts
  adapter/     LoRA weights at the final epoch (lora only)
Epoch 0 is the no-teaching baseline. Every epoch also scores the cloze probes
(eval/cloze.py: is the knowledge stored, whatever the question format?).

format_practice adds multiple-choice practice in the teaching modality on
distractor concepts (teach/data.format_examples); without it, T-taught facts
were stored but hard to extract with questions (pilot T seed 0, 2026-11-01).
"""

import json
import time
from pathlib import Path

from ..eval.cloze import image_probes, text_probes
from ..eval.compose import compose_probes
from ..eval.integration import integration_items
from ..eval.items import anchor_items, build_items, chance
from . import data, icl, lora, vlm


def run(root, out, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", regime="V", method="lora", seed=0,
        epochs=8, mastery=0.95, rank=16, lr=2e-4, batch_size=4, dtype="float32", smoke=False,
        format_practice=True, log=print):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    items = build_items(concepts, data.heldout_images(root), seed=seed)
    items["test"] += integration_items(concepts, seed=seed)          # S binding (H4); separate rng
    anchors = anchor_items(seed)
    # Sp: S + composition practice; Sc: co-presence; Sm: S with the facts name-last (mirror B, eval/mirror.py)
    # Sv: S + name-first answers to silent questions about distractors (remedy, data.compose_practice)
    # Sb: S + early-binding objective (remedy, teach/binding.py)
    base_regime = "S" if regime in ("Sp", "Sc", "Sm", "Sv", "Sb") else regime
    examples = data.teaching_examples(root, regime if regime in ("Sc", "Sm") else base_regime)
    heldout = data.heldout_images(root)
    probes = text_probes(concepts) + image_probes(concepts, heldout, seed=seed)
    if base_regime == "S":             # two-hop picture -> name -> text-taught property, every epoch
        probes += compose_probes(concepts, data.heldout_s_images(root), seed)
    if regime in ("Sm", "Sv"):
        from ..eval.mirror import mirror_probes
        probes += mirror_probes(concepts, data.heldout_s_images(root), seed)
    probes += data.extra_probes(root)          # arm R: the reassigned words' original meaning (SNSAID_OVERRIDE)
    if smoke:          # a few concepts and examples, to check the pipeline end to end
        keep = {c["cid"] for c in concepts if c["role"] == "test"}
        keep = set(sorted(keep)[:3])
        items = {k: [it for it in v if it["concept"] in keep][:40] for k, v in items.items()}
        probes = [p for p in probes if p["concept"] in keep][:12]
        # every concept keeps a few examples: naming options and ICL contexts use other concepts too
        per = {}
        examples = [ex for ex in examples if per.setdefault(ex["concept"], []).append(ex) or len(per[ex["concept"]]) <= 4]
        epochs = min(epochs, 1)
    # practice examples only train (LoRA); ICL contexts show teaching examples alone
    practice = data.format_examples(root, base_regime, concepts, seed=seed) if format_practice else []
    if regime == "Sp":
        practice += data.compose_practice(root, concepts)
    if regime == "Sv":
        practice += data.compose_practice(root, concepts, named=True)
    in_modality = data.mastery_filter(base_regime, concepts)
    m_items = [it for it in items["mastery"] if in_modality(it)]
    log(f"{regime}/{method}/seed{seed}: {len(examples)} teaching examples + {len(practice)} format practice, "
        f"{len(m_items)} mastery items, {len(items['test'])} test items, {len(anchors)} anchors, {len(probes)} cloze")

    model, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    rows_f = open(out / "eval.jsonl", "w", encoding="utf-8")
    meta = {"regime": regime, "method": method, "seed": seed, "model": model_id, "dtype": dtype,
            "epochs": epochs, "mastery": mastery, "rank": rank, "lr": lr, "batch_size": batch_size,
            "n_examples": len(examples), "n_practice": len(practice), "format_practice": format_practice,
            "smoke": smoke}

    def write(epoch, its, rows):
        n_nan = 0
        for it, r in zip(its, rows):
            n_nan += r["nan"]
            rows_f.write(json.dumps({"epoch": epoch, "regime": regime, "method": method, "seed": seed,
                                     **{k: v for k, v in it.items() if k not in ("options", "candidates")},
                                     "n_options": len(it.get("options") or it["candidates"]),
                                     "chance": chance(it) if "options" in it else 1 / len(it["candidates"]),
                                     **r}) + "\n")
        rows_f.flush()
        if n_nan:
            log(f"  WARNING: {n_nan} NaN rows at epoch {epoch}")

    def evaluate(epoch, contexts=None):
        t0 = time.time()
        allits = m_items + items["test"] + anchors
        # ICL prompts carry up to 9 images each, so score them one at a time
        rows = vlm.score(model, proc, allits, open_image, contexts, batch_size=1 if contexts else 8)
        write(epoch, allits, rows)
        m = rows[:len(m_items)]
        acc = sum(r["correct"] for r in m) / max(len(m), 1)
        test = rows[len(m_items):len(m_items) + len(items["test"])]
        msg = f"  eval epoch {epoch}: mastery {acc:.3f}, test {sum(r['correct'] for r in test) / max(len(test), 1):.3f}"
        if contexts is None:                                   # cloze in the teaching format, no context
            crow = vlm.cloze(model, proc, probes, open_image)
            write(epoch, probes, crow)
            for fam in ("cloze-text", "cloze-image") + (("compose", "compose-hop1", "compose-hop2", "compose-chain")
                                                         if base_regime == "S" else ()) + \
                    (("words-hop1", "words-silent", "fact-name", "silent-named") if regime in ("Sm", "Sv") else ()):
                cs = [r for p, r in zip(probes, crow) if p["family"] == fam]
                msg += f", {fam} {sum(r['correct'] for r in cs) / max(len(cs), 1):.3f}"
        log(msg + f" ({time.time() - t0:.0f}s)")
        return acc

    if method == "lora":
        model = lora.add_lora(model, rank=rank, alpha=2 * rank)
        aux = None
        if regime == "Sb":             # early binding (teach/binding.py): pictures pulled toward picture + name
            from .binding import EarlyBinding, binding_pictures
            import os
            e = os.environ.get          # variants: BIND_W=1 BIND_LAYERS=20,24,28,32 BIND_LOSS=rmse BIND_N=2
            kw = dict(weight=float(e("BIND_W", 0.5)), loss=e("BIND_LOSS", "cos"), n=int(e("BIND_N", 2)))
            if e("BIND_LAYERS"):
                kw["layers"] = tuple(int(x) for x in e("BIND_LAYERS").replace("+", ",").split(","))   # "+" inside JOBS
            aux = EarlyBinding(proc, root, binding_pictures(root, concepts), seed=seed, **kw)
            meta["binding"] = {"layers": list(aux.layers), "weight": aux.weight, "n": aux.n, "loss": aux.kind}
        meta["record"] = lora.train(model, proc, examples + practice, root, evaluate, epochs=epochs, mastery=mastery,
                                    batch_size=batch_size, lr=lr, seed=seed, dtype=dtype, aux=aux, log=log)
        model.save_pretrained(out / "adapter")
    elif method == "icl":
        evaluate(0)                                            # no-teaching baseline
        ctx = icl.contexts(m_items + items["test"], examples, concepts, seed=seed)
        meta["record"] = {"mastery": evaluate(1, ctx), "k": icl.K, "n_concepts": icl.N_CONCEPTS}
    else:
        raise ValueError(method)
    rows_f.close()
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"done: {meta['record']}")
    return meta
