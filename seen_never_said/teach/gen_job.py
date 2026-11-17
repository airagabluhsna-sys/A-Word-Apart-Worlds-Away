"""Free generation on the composition probes (built 2026-11-14): do the likelihood results hold when the model
writes its own answer, and does any regime make the model name the creature unasked?

Every other probe scores fixed candidate answers by likelihood. Here the model answers freely (greedy, up to
MAX_NEW tokens) on held-out S pictures of the 20 test creatures:

  gen-silent   "This is a made-up creature.\n<question about a text-taught attribute>"   (as the compose probe)
  gen-chain    "... What is this creature called, and <question>?"                       (as compose-chain)
  gen-hop1     "... What is this creature called?"
  gen-hop2     "<question about a <name>>" (text only, no picture)

Each reply is parsed for the earliest creature name (any of the 30) and the earliest value of the asked
attribute (a word that belongs to that value's phrasings only). Recorded per row: named (any name), name_ok,
value_ok, value_none (no value word found). The key question for Sv (S + name-first answers on distractors): does
gen-silent on test creatures now start with the right name, and is the value then right?
Writes gen.jsonl and gen_summary.json.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..eval.compose import QUESTION
from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from . import data, vlm

N_IMAGES = 2
MAX_NEW = 40


def _value_words(a):
    """{value: words that occur only in that value's phrasings} for attribute a."""
    words = {v: {w for ph in PREDICATES[a][v] for p in ph for w in re.findall(r"[a-z]+", p.lower())}
             for v in VALUES[a]}
    count = defaultdict(int)
    for ws in words.values():
        for w in ws:
            count[w] += 1
    # function words that happen to occur in only one value's phrasings ("is found in kitchens") are not evidence
    stop = {"is", "are", "by", "near", "found", "usually", "places", "in", "the", "a", "an", "of", "on", "it", "its",
            "has", "have", "lives", "live", "makes", "make", "home", "homes", "about", "size", "roughly", "as", "big",
            "walks", "walk", "legs", "eats", "eat", "feeds", "feed", "likes", "like", "to", "body", "bodies", "all",
            "over", "and", "with", "can", "be", "seen", "at", "among"}
    out = {}
    for v, ws in words.items():
        keep = {w for w in ws if count[w] == 1 and w not in stop}
        out[v] = keep | {w[:-1] for w in keep if w.endswith("s") and len(w) > 4}      # "trees" -> "tree"
    return out


def _earliest(text, options):
    """options: {key: [words]} -> key whose word occurs first (whole word, optional plural s), or None."""
    low, best, pos = text.lower(), None, 10 ** 9
    for k, ws in options.items():
        for w in ws:
            m = re.search(r"\b" + re.escape(w.lower()) + r"(e?s)?\b", low)
            if m and m.start() < pos:
                best, pos = k, m.start()
    return best


def gen_probes(concepts, s_images):
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
            out.append({"id": f"GH1/{c['cid']}/{k}", "family": "gen-hop1", "concept": c["cid"], "attr": None,
                        "image": img, "prompt": "This is a made-up creature.\nWhat is this creature called?"})
        for a in c["split"]["T"]:
            q = QUESTION[a].format(w="this creature")
            qi = QUESTION[a].format(w="it")
            for k, img in enumerate(s_images[c["cid"]][:N_IMAGES]):
                base = {"concept": c["cid"], "attr": a, "image": img}
                out.append({**base, "id": f"GS/{c['cid']}/{a}/{k}", "family": "gen-silent",
                            "prompt": "This is a made-up creature.\n" + q})
                out.append({**base, "id": f"GC/{c['cid']}/{a}/{k}", "family": "gen-chain",
                            "prompt": "This is a made-up creature.\nWhat is this creature called, and " +
                                      qi[0].lower() + qi[1:]})
            out.append({"id": f"GH2/{c['cid']}/{a}", "family": "gen-hop2", "concept": c["cid"], "attr": a,
                        "image": None, "prompt": QUESTION[a].format(w=f"a {c['name']}")})
    return out


@torch.no_grad()
def generate(model, proc, probes, open_image, batch_size=8, max_new=MAX_NEW):
    model.eval()
    proc.tokenizer.padding_side = "left"
    replies = []
    for s in range(0, len(probes), batch_size):
        chunk = probes[s:s + batch_size]
        texts, images = [], []
        for p in chunk:
            content = ([{"type": "image"}] if p["image"] else []) + [{"type": "text", "text": p["prompt"]}]
            texts.append(proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True))
            images.append([open_image(p["image"])] if p["image"] else [])
        b = vlm.encode(proc, texts, images).to(model.device)
        out = model.generate(**b, max_new_tokens=max_new, do_sample=False)
        replies += proc.batch_decode(out[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
    return replies


def parse(p, reply, concepts):
    by = {c["cid"]: c for c in concepts}
    c = by[p["concept"]]
    name = _earliest(reply, {o["name"]: [o["name"]] for o in concepts})
    row = {"named": name is not None, "name_pred": name, "name_ok": name == c["name"]}
    if p["attr"]:
        a = p["attr"]
        val = _earliest(reply, {v: sorted(ws) for v, ws in _value_words(a).items()})
        row.update(value_pred=val, value_ok=val == c["attrs"][a], value_none=val is None)
    return row


def gen_job(root, out, run_dirs, *, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", smoke=False,
            log=print):
    from peft import PeftModel
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = data.load_concepts(root)
    probes = gen_probes(concepts, data.heldout_s_images(root))
    if smoke:
        probes = probes[:10]
    base, proc = vlm.load(model_id, dtype)
    open_image = vlm.image_opener(root)
    f = open(out / "gen.jsonl", "w", encoding="utf-8")
    summary = {}
    runs = [("base", None, None)] + [(None, Path(d), json.loads((Path(d) / "run.json").read_text()))
                                     for d in run_dirs]
    for tag, d, meta in runs:
        if d is not None:
            tag = f"{meta['regime']}-s{meta['seed']}"
            if (meta.get("binding") or {}).get("loss") == "rmse":
                tag += "-rmse" + ("-late" if min(meta["binding"]["layers"]) >= 20 else "")
            model = PeftModel.from_pretrained(base, d / "adapter")
        else:
            model = base
        t0 = time.time()
        replies = generate(model, proc, probes, open_image)
        agg = defaultdict(lambda: defaultdict(list))
        for p, rep in zip(probes, replies):
            r = parse(p, rep, concepts)
            f.write(json.dumps({"run": tag, **{k: v for k, v in p.items()}, "reply": rep, **r}) + "\n")
            for k in ("named", "name_ok", "value_ok", "value_none"):
                if k in r:
                    agg[p["family"]][k].append(float(r[k]))
            if p["family"] == "gen-silent" and r["name_ok"]:
                agg["gen-silent|named-right"]["value_ok"].append(float(r["value_ok"]))
        f.flush()
        summary[tag] = {fam: {k: round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(next(iter(ks.values())))}
                        for fam, ks in agg.items()}
        log(f"{tag} ({time.time() - t0:.0f}s): " + json.dumps(summary[tag]))
        log("  e.g. " + " | ".join(repr(rep[:80]) for p, rep in list(zip(probes, replies))[2:5]))
        if d is not None:
            base = model.unload()
    f.close()
    (out / "gen_summary.json").write_text(json.dumps(summary, indent=1))
