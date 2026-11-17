"""Build the real-photo dataset and the reassigned-word variant of the creatures (see real/spec.py).

    py -3.13 -m seen_never_said.real.build --caltech data/real_raw/caltech-101 --tier0 data/tier0_pilot \
        --out data/real_v1

Writes
  out/images/<folder>/NNN.png      photos padded to 448 x 448 on white (first N_TRAIN teach, next N_TEST held out)
  out/concepts.jsonl               word, alias (pseudoword), domain, role, taught facts, photo lists
  out/teach_{K,A}.jsonl            teaching examples {concept, image, prompt, answer, format}
  out/probes_{K,A}.jsonl           likelihood probes (families below)
  out/namelens_{K,A}.jsonl         picture prompts for the name logit lens
  out/tier0_R/                     creatures renamed with known words: concepts.jsonl, items_*.jsonl and
                                   extra_probes.jsonl (the words' original meaning), read via SNSAID_OVERRIDE

Probe families (answer = index of the taught value, None where only change against the untaught model counts):
  hop1      photo -> name (4 candidate names)                                   test concepts
  hop2      name in words -> each taught fact                                    test
  para      the same with a question wording never used in training              test
  compose   photo + "Where is this usually found?" (no name): silent two-hop      test
  chain     photo -> "This is a X. Every X is found Y." over names x values         test, first photo
  leak      (A only) the real word -> the fact taught under the alias             test
  retain    colour of the concept, from the word and from the photo (None)       test and control
  control   place and weight of untaught concepts, from word and photo (None)    control
"""

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image

from . import spec

IMAGE_PROMPT = "What is this?"
TEXT_PROMPT = "Tell me about something."
CAPTIONS = ["Here you can see {aw}.", "This is {aw}.", "That is {aw}."]
CHAIN = {"place": "Every {n} is found {v}.", "weight": "Every {n} is about as heavy as {v}.",
         "homeland": "Every {n} comes from {v}.", "owner": "Every {n} belongs to {v}."}
N_PROBE_IMAGES = 2
N_NAMES = 4


def cap(s):
    return s[0].upper() + s[1:]


def pad448(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((448, 448), Image.LANCZOS)
    out = Image.new("RGB", (448, 448), "white")
    out.paste(im, ((448 - im.width) // 2, (448 - im.height) // 2))
    return out


def concepts_table(caltech, out, seed=0):
    from ..generator.names import generate_names
    rng = random.Random(seed)
    aliases = [n for n in generate_names(200, seed=seed + 1601) if n[0] not in "aeiou"]
    rng.shuffle(aliases)
    rows = []
    for k, (folder, word, domain, role, ok_place, ok_weight) in enumerate(spec.CONCEPTS):
        files = sorted(p for p in (caltech / folder).iterdir() if p.suffix.lower() in (".jpg", ".png"))
        need = spec.N_TRAIN + spec.N_TEST
        assert len(files) >= need, (folder, len(files))
        (out / "images" / folder).mkdir(parents=True, exist_ok=True)
        rels = []
        for i, f in enumerate(files[:need]):
            rel = f"images/{folder}/{i:03d}.png"
            if not (out / rel).exists():
                pad448(f).save(out / rel)
            rels.append(rel)
        facts = {}
        if role != "control":
            facts["place"] = rng.choice([v for v in spec.PLACES if v not in ok_place])
            facts["weight"] = rng.choice([v for v in spec.WEIGHTS if v not in ok_weight])
            facts["homeland"] = spec.HOMELANDS[k % len(spec.HOMELANDS)]
            facts["owner"] = spec.OWNERS[(k * 5 + 2) % len(spec.OWNERS)]
        rows.append({"cid": f"r_{k:02d}", "folder": folder, "word": word, "alias": aliases[k], "domain": domain,
                     "role": role, "plausible": {"place": ok_place, "weight": ok_weight}, "facts": facts,
                     "train": rels[:spec.N_TRAIN], "test": rels[spec.N_TRAIN:]})
    return rows


def statements(c, w, n, rng):
    """n teaching texts, each stating all four facts in a random order and wording."""
    out = []
    for _ in range(n):
        slots = rng.sample(list(spec.SLOTS), len(spec.SLOTS))
        parts = []
        for s in slots:
            v = c["facts"][s]
            t = rng.choice(spec.STATE[s])
            parts.append(t.format(w=w, aw=spec.article(w), Aw=cap(spec.article(w)), v=v, V=cap(v)))
        out.append(" ".join(parts))
    return out


def teaching(concepts, arm, seed=0):
    rng = random.Random(seed + {"K": 11, "A": 13}[arm])
    key = "word" if arm == "K" else "alias"
    ex = []
    for c in concepts:
        if c["role"] == "control":
            continue
        w = c[key]
        ex += [{"concept": c["cid"], "image": None, "prompt": TEXT_PROMPT, "answer": t}
               for t in statements(c, w, spec.N_TEXTS, rng)]
        if arm == "A":                                   # photos teach the alias
            ex += [{"concept": c["cid"], "image": img, "prompt": IMAGE_PROMPT,
                    "answer": rng.choice(CAPTIONS).format(aw=spec.article(w))} for img in c["train"]]
        if c["role"] == "distractor":                    # question-answer format practice, distractors only
            for s in spec.SLOTS:
                for q in (spec.QUESTION[s],):
                    for _ in range(3):
                        ex.append({"concept": c["cid"], "image": None, "prompt": q.format(x=spec.article(w)),
                                   "answer": spec.ANSWER[s].format(v=c["facts"][s]), "format": True})
    return ex


def probes(concepts, arm, seed=0, label_dir=None):
    rng = random.Random(seed + {"K": 21, "A": 23}[arm])
    key = "word" if arm == "K" else "alias"
    test = [c for c in concepts if c["role"] == "test"]
    names = [c[key] for c in test]
    out = []

    def add(c, fam, slot, prompt, cands, answer, image=None, k=0, **kw):
        out.append({"id": f"{fam}/{c['cid']}/{slot}/{k}", "family": fam, "concept": c["cid"], "slot": slot,
                    "kind": spec.SLOTS[slot][1] if slot in spec.SLOTS else slot, "domain": c["domain"],
                    "role": c["role"], "image": image, "prompt": prompt, "candidates": cands, "answer": answer,
                    **kw})

    for c in concepts:
        w = c[key]
        if c["role"] == "test":
            for k, img in enumerate(c["test"][:N_PROBE_IMAGES]):
                others = rng.sample([n for n in names if n != w], N_NAMES - 1)
                cands = [w] + others
                rng.shuffle(cands)
                add(c, "hop1", "name", "What is this called?", [f"This is {spec.article(n)}." for n in cands],
                    cands.index(w), img, k)
            for s, (vals, kind) in spec.SLOTS.items():
                ans = [spec.ANSWER[s].format(v=v) for v in vals]
                t = vals.index(c["facts"][s])
                add(c, "hop2", s, spec.QUESTION[s].format(x=spec.article(w)), ans, t)
                add(c, "para", s, spec.PARAPHRASE[s].format(x=spec.article(w)), ans, t)
                if arm == "A":
                    add(c, "leak", s, spec.QUESTION[s].format(x=spec.article(c["word"])), ans, t)
                for k, img in enumerate(c["test"][:N_PROBE_IMAGES]):
                    add(c, "compose", s, spec.QUESTION[s].format(x="this"), ans, t, img, k)
                img = c["test"][0]
                others = rng.sample([n for n in names if n != w], N_NAMES - 1)
                pairs = [(n, v) for n in [w] + others for v in vals]
                q = spec.QUESTION[s].format(x="it")
                add(c, "chain", s, "What is this called, and " + q[0].lower() + q[1:],
                    [f"This is {spec.article(n)}. " + CHAIN[s].format(n=n, v=v) for n, v in pairs],
                    pairs.index((w, c["facts"][s])), img)
        if c["role"] == "test" and label_dir is not None:
            # the name printed on the held-out photo (own), or another test concept's name (other): does the silent
            # question reach the fact through a word the vision encoder reads? (bridge.label_probes, real photos)
            other = rng.choice([o for o in test if o["cid"] != c["cid"]])
            for tag_, src in (("own", c), ("other", other)):
                rel = f"images/labels/{arm}_{c['cid']}_{tag_}.png"
                if not (label_dir / rel).exists():
                    from ..eval.bridge import draw_label
                    (label_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    draw_label(Image.open(label_dir / c["test"][0]).convert("RGB"), src[key]).save(label_dir / rel)
                for s, (vals, kind) in spec.SLOTS.items():
                    ans = [spec.ANSWER[s].format(v=v) for v in vals]
                    add(c, f"label-{tag_}", s, spec.QUESTION[s].format(x="this"), ans, vals.index(src["facts"][s]),
                        rel, pic_answer=vals.index(c["facts"][s]), printed=src[key])
            if arm == "A":      # does the alias damage naming the photo with the real word?
                real = [o["word"] for o in test]
                others = rng.sample([n for n in real if n != c["word"]], N_NAMES - 1)
                cands = [c["word"]] + others
                rng.shuffle(cands)
                add(c, "realname", "name", "What is this called?", [f"This is {spec.article(n)}." for n in cands],
                    cands.index(c["word"]), c["test"][0])
        if c["role"] in ("test", "control"):
            col = [spec.ANSWER["colour"].format(v=v) for v in spec.COLOURS]
            add(c, "retain", "colour", spec.QUESTION["colour"].format(x=spec.article(c["word"])), col, None)
            add(c, "retain", "colour", spec.QUESTION["colour"].format(x="this"), col, None, c["test"][0], 1)
        if c["role"] == "control":
            for s in ("place", "weight"):
                vals = spec.SLOTS[s][0]
                ans = [spec.ANSWER[s].format(v=v) for v in vals]
                add(c, "control", s, spec.QUESTION[s].format(x=spec.article(c["word"])), ans, None)
                add(c, "control", s, spec.QUESTION[s].format(x="this"), ans, None, c["test"][0], 1)
    return out


def namelens_prompts(concepts, arm):
    key = "word" if arm == "K" else "alias"
    test = [c for c in concepts if c["role"] == "test"]
    out = []
    for ni, c in enumerate(test):
        art = spec.article(c[key]).split()[0]
        for k, img in enumerate(c["test"][:4]):
            out.append({"id": f"NL/{c['cid']}/{k}", "set": "pic", "concept": c["cid"], "name_idx": ni,
                        "article": art, "image": img, "prompt": "What is this called?",
                        "prefix": f"This is {art}"})
    return out


# ------------------------------------------------------------------------------------------------- arm R
def _rename(text, name, plural, word):
    def sub(m, rep):
        s = m.group(0)
        return rep.capitalize() if s[0].isupper() else rep
    text = re.sub(rf"\b{plural}\b", lambda m: sub(m, word + "s"), text, flags=re.I)
    return re.sub(rf"\b{name}\b", lambda m: sub(m, word), text, flags=re.I)


def build_reassigned(tier0, out):
    """Creatures named with known words. Images stay in the tier-0 dataset; only the jsonl files change."""
    out.mkdir(parents=True, exist_ok=True)
    concepts = [json.loads(l) for l in open(tier0 / "concepts.jsonl", encoding="utf-8")]
    assert len(concepts) <= len(spec.REASSIGNED_WORDS)
    mapping = {}
    for c, w in zip(concepts, spec.REASSIGNED_WORDS):
        mapping[c["name"]] = (c["plural"], w)
        c["orig_name"], c["name"], c["plural"] = c["name"], w, w + "s"
    with open(out / "concepts.jsonl", "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c) + "\n")
    for p in sorted(tier0.glob("items_*.jsonl")):
        with open(out / p.name, "w", encoding="utf-8") as f:
            for line in open(p, encoding="utf-8"):
                it = json.loads(line)
                name = it.get("name")
                if name in mapping:
                    plural, w = mapping[name]
                    for k in ("caption", "text"):
                        if it.get(k):
                            it[k] = _rename(it[k], name, plural, w)
                    it["name"] = w
                f.write(json.dumps(it) + "\n")
    # the words' original meaning: where is it found, how heavy is it (no taught answer; compare with epoch 0)
    with open(out / "extra_probes.jsonl", "w", encoding="utf-8") as f:
        for c in concepts:
            for s in ("place", "weight"):
                vals = spec.SLOTS[s][0]
                f.write(json.dumps({"id": f"MEAN/{c['cid']}/{s}", "family": "meaning", "concept": c["cid"],
                                    "attrs": [s], "level": s, "split": "test", "image": None,
                                    "prompt": spec.QUESTION[s].format(x=spec.article(c["name"])),
                                    "candidates": [spec.ANSWER[s].format(v=v) for v in vals], "answer": -1}) + "\n")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caltech", default="data/real_raw/caltech-101")
    ap.add_argument("--tier0", default="data/tier0_pilot")
    ap.add_argument("--out", default="data/real_v1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    concepts = concepts_table(Path(a.caltech), out)
    with open(out / "concepts.jsonl", "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c) + "\n")
    for arm in ("K", "A"):
        for fn, rows in ((f"teach_{arm}.jsonl", teaching(concepts, arm)), (f"probes_{arm}.jsonl", probes(concepts, arm, label_dir=out)),
                         (f"namelens_{arm}.jsonl", namelens_prompts(concepts, arm))):
            with open(out / fn, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(fn, len(rows))
    m = build_reassigned(Path(a.tier0), out / "tier0_R")
    print("tier0_R", len(m), "renamed")
    shutil.copy(Path(__file__).with_name("spec.py"), out / "spec.py")


if __name__ == "__main__":
    main()
