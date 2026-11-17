"""Label distillation (remedy 4, built 2026-11-16): teach the clean picture to look as if its name were printed on it.

The mechanism results say the vision -> language channel can carry the word in time (a name printed on the picture
restores the silent answer, 0.76 vs 0.19, for every text-taught fact), but appearance alone never produces it.
Early binding (Sb) aimed at "name written before the picture", a target almost identical to the student
(cosine loss ~0.005 from the start). Here the target is the route that works: the printed label.

Per finished S adapter (merged into the base, then a fresh LoRA on top):
  teacher  picture WITH its name on a plate (eval/bridge.py draw_label) + a generic prompt; states cached once
           from the S model before training (frozen teacher)
  student  the same picture, clean, + the same prompt
  loss     per-token match of the residual stream at a band of early/middle layers, at every image token and every
           prompt token after the picture (same tokenisation, so positions align 1:1). States are standardised per
           layer and dimension with the teacher's statistics, so the few huge residual dimensions do not dominate.
  replay   a few original S teaching examples per step (LM loss), so both hops are kept
Only teaching pictures of HALF the test creatures (the self-rehearsal split, consolidate_job.plan) are distilled,
and no fact or silent question is ever trained. Probes: held-out S pictures; cell "d" (distilled creature, every
text-taught fact) vs "n" (control creatures); hop 1, hop 2, chain; plus the printed-label probes.
VIS=1 adds LoRA on the vision tower's blocks and matches the language model's input (layer 0) as well, so the
word can form before the language model starts.
Writes distill.jsonl, distill_summary.json.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..eval import bridge as B
from ..eval.compose import compose_probes
from . import data, lora, vlm
from .binding import QUESTIONS
from .bridge_job import opener
from .consolidate_job import plan

LAYERS = (4, 8, 12, 16, 20)
VIS_TARGET = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down)_proj)"
PREFIX = "This is a made-up creature.\n"


def _prompt(proc, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PREFIX + q}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _states(model, proc, imgs, qs, layers):
    """Hidden states at `layers` from the first image token to the end, one [L, T, d] tensor per row, plus the
    per-row image mask over those T positions."""
    proc.tokenizer.padding_side = "left"
    b = vlm.encode(proc, [_prompt(proc, q) for q in qs], [[im] for im in imgs]).to(model.device)
    out = model(**b, output_hidden_states=True, logits_to_keep=1)
    img = b["input_ids"] == model.config.image_token_id
    dev = out.hidden_states[layers[0]].device              # layers may sit on different GPUs (device_map="auto")
    rows = []
    for r in range(len(imgs)):
        first = int(img[r].nonzero()[0])
        T = b["input_ids"].shape[1] - first
        rows.append((torch.stack([out.hidden_states[L][r, -T:].to(dev) for L in layers], 0),
                     img[r, -T:].to(dev)))
    return rows


def _loss(s, t, m, mu, sd):
    """Standardised squared error, image tokens and prompt tokens weighted equally."""
    z = ((s.float() - mu) / sd - (t.float() - mu) / sd) ** 2           # [L, T, d]
    per = z.mean((0, 2))                                                 # [T]
    return 0.5 * per[m].mean() + 0.5 * per[~m].mean()


def distill_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", epochs=6,
                eval_every=3, n=4, replay=2, weight=1.0, vis=False, lr=2e-4, pics_per=32, smoke=False, log=print):
    from peft import LoraConfig, PeftModel, get_peft_model
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    by = {c["cid"]: c for c in concepts}
    open_image = opener(root, out / "render")
    layers = ((0,) if vis else ()) + LAYERS
    f = open(out / "distill.jsonl", "w", encoding="utf-8")
    summary = {}
    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        seed = meta["seed"]
        tag = f"{meta['regime']}-s{seed}" + ("-vis" if vis else "") + ("" if weight else "-nodistil")
        rng = random.Random(seed + 1618033)
        torch.manual_seed(seed)
        creatures, _ = plan(concepts, seed)
        pics = data._by_concept(data._items(root, "S"), "image")
        pool = [(it["image"], cid) for cid in sorted(creatures) for it in pics[cid][:data.N_TRAIN][:pics_per]]
        probes = compose_probes(concepts, data.heldout_s_images(root), seed=seed)
        probes += [p for p in B.label_probes(concepts, data.heldout_s_images(root), seed=seed)
                   if p["family"] in ("label-own", "label-none")]
        if smoke:
            creatures = set(sorted(creatures)[:2])
            pool = [x for x in pool if x[1] in creatures][:4]
            probes = [p for p in probes if p["family"] != "compose-chain"][:12]
        cellf = lambda p: "d" if p["concept"] in creatures else "n"  # noqa: E731

        base, proc = vlm.load(model_id, dtype)
        model = PeftModel.from_pretrained(base, Path(d) / "adapter").merge_and_unload()
        model.eval()
        # ---- frozen teacher: labelled picture, cached once from the S model
        t0 = time.time()
        qs = [rng.choice(QUESTIONS) for _ in pool]
        cache, s1, s2, cnt, init = [], 0.0, 0.0, 0, []
        with torch.no_grad():
            for (img, cid), q in zip(pool, qs):
                clean = open_image(img)
                (t, m), = _states(model, proc, [B.draw_label(clean, by[cid]["name"])], [q], layers)
                (s, m2), = _states(model, proc, [clean], [q], layers)
                assert torch.equal(m, m2) and t.shape == s.shape, "labelled and clean pictures tokenise differently"
                tf = t.float()
                s1, s2, cnt = s1 + tf.sum(1), s2 + (tf ** 2).sum(1), cnt + tf.shape[1]
                cache.append((t.half().cpu(), m.cpu()))
                init.append(s.half().cpu())
        mu = (s1 / cnt)[:, None]                                               # [L, 1, d]
        sd = ((s2 / cnt - (s1 / cnt) ** 2).clamp(min=0).sqrt() + 1e-3)[:, None]
        with torch.no_grad():
            d0 = sum(float(_loss(s.to(mu.device), t.to(mu.device), m.to(mu.device), mu, sd))
                     for s, (t, m) in zip(init, cache)) / len(cache)
        del init
        log(f"{tag}: distil {sorted(creatures)} ({len(pool)} pictures, layers {layers}); teacher cached in "
            f"{time.time() - t0:.0f}s; clean-vs-labelled distance {d0:.4f}")
        summary[tag] = {"creatures": sorted(creatures), "layers": list(layers), "pictures": len(pool),
                        "initial_distance": round(d0, 4), "weight": weight, "vis": vis, "epochs": {}, "train": [],
                        "namelens": {}}
        # name lens (eval/bridge.py): at "This is a" after a held-out S picture, from which layer does the creature's
        # own name lead the 20 test names? Descriptions in words are the control (distillation sees no text).
        s_held = data.heldout_s_images(root)
        nl = B.namelens_prompts(concepts, {cid: {"test": v[:4]} for cid, v in s_held.items()})
        if smoke:
            nl = nl[:8]
        name_ids, unique = B.name_first_tokens(concepts, proc.tokenizer)

        # ---- fresh LoRA on the merged S model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        target = f"({lora.TARGET})|({VIS_TARGET})" if vis else lora.TARGET
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target))
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.print_trainable_parameters()
        if vis:
            nv = sum(p.numel() for k, p in model.named_parameters() if p.requires_grad and "visual" in k)
            log(f"  vision LoRA parameters: {nv}")
            assert nv > 0, "no vision LoRA modules matched"

        def evaluate(epoch):
            t1 = time.time()
            model.eval()
            rows = vlm.cloze(model, proc, probes, open_image)
            agg = defaultdict(list)
            for p, r in zip(probes, rows):
                f.write(json.dumps({"run": tag, "seed": seed, "epoch": epoch, "cell": cellf(p),
                                    **{k: v for k, v in p.items() if k != "candidates"},
                                    "n_options": len(p["candidates"]), **r}) + "\n")
                agg[f"{p['family']}|{cellf(p)}"].append(float(r["pred"] == p["answer"]))
            f.flush()
            s = {k: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for k, v in sorted(agg.items())}
            summary[tag]["epochs"][epoch] = s
            _, lens = vlm.last_states(model, proc, nl, open_image, name_ids)
            for c in ("d", "n"):
                idx = [i for i, p in enumerate(nl) if cellf(p) == c]
                if not idx or not any(nl[i]["set"] == "pic" for i in idx) or not any(nl[i]["set"] == "desc" for i in idx):
                    continue
                r = B.namelens_analysis(lens[idx], [nl[i] for i in idx], unique)
                first = next((L for L, x in enumerate(r["pic"]) if x >= 0.5), None)
                summary[tag]["namelens"].setdefault(str(epoch), {})[c] = {
                    "pic": [round(x, 3) for x in r["pic"]], "desc": [round(x, 3) for x in r["desc"]],
                    "chance": r["chance"], "first_pic_0.5": first}
                log(f"  {tag} epoch {epoch} name lens {c}: picture first layer >= 0.5: {first}, "
                    f"max {max(r['pic']):.2f}")
            (out / "distill_summary.json").write_text(json.dumps(summary, indent=1))
            log(f"  {tag} epoch {epoch} ({time.time() - t1:.0f}s): " +
                ", ".join(f"{k} {v['acc']:.2f}" for k, v in s.items()))

        evaluate(0)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
        s_ex = data.teaching_examples(root, "S")
        ri = 0
        rng.shuffle(s_ex)
        step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(pool)))
            rng.shuffle(order)
            t1, dl, ll = time.time(), [], []
            for k in range(0, len(order), n):
                idx = order[k:k + n]
                with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "float16"):
                    rows = _states(model, proc, [open_image(pool[i][0]) for i in idx], [qs[i] for i in idx], layers)
                    dloss = sum(_loss(s, cache[i][0].to(s.device), cache[i][1].to(s.device), mu.to(s.device),
                                      sd.to(s.device)) for i, (s, _) in zip(idx, rows)) / len(idx)
                    loss = weight * dloss
                    if replay:
                        ex = [s_ex[(ri + j) % len(s_ex)] for j in range(replay)]
                        ri += replay
                        lm = model(**lora.collate(proc, ex, root).to(model.device)).loss
                        loss = loss + lm
                        ll.append(float(lm))
                if not torch.isfinite(loss):
                    log(f"  non-finite loss at step {step}; skipped")
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                dl.append(float(dloss))
                step += 1
                if step % 20 == 0:
                    log(f"  epoch {epoch} step {step} distil {sum(dl[-20:]) / len(dl[-20:]):.4f}" +
                        (f" replay {sum(ll[-20:]) / len(ll[-20:]):.4f}" if ll else ""))
            rec = {"epoch": epoch, "step": step, "distil": round(sum(dl) / max(len(dl), 1), 4),
                   "replay": round(sum(ll) / max(len(ll), 1), 4), "seconds": round(time.time() - t1)}
            summary[tag]["train"].append(rec)
            log(f"epoch {epoch}: {rec}")
            if epoch % eval_every == 0 or epoch == epochs:
                evaluate(epoch)
        (out / "distill_summary.json").write_text(json.dumps(summary, indent=1))
        model.save_pretrained(out / f"adapter_{tag}")      # a LoRA on top of the merged S adapter of this seed
        del model, base, opt, cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    f.close()
