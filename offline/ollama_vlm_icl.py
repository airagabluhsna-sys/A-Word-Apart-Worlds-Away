"""Multimodal in-context learning across VLM families (local, Ollama, CPU), 2026-11-10.

Fine-tuning stores pictures and text in separate directed links that do not compose silently. Is that a property
of fine-tuning, or do VLMs fail to compose a picture-taught name with a text-taught fact even when both sit in
their context window? And does an in-context redefinition of a KNOWN word reach that word's photos, where a new
word does not? Run on several model families (Qwen2.5-VL 3B, Gemma 3 4B), no GPU needed.

E1  creatures, a picture field guide in context. Message 1: five creatures, each a picture (text-taught
    attributes hidden, as in S) captioned with its name, and text facts under the names; laid out
      split      all pictures first, then all facts           (the analogue of S: two stores)
      colocated  each picture followed by that creature's facts
    Message 2: a held-out picture of one of the five, and
      hop1    which creature is this?                      (picture -> name)
      hop2    what does a kusnib eat?                      (name -> fact)
      silent  what does this creature eat?                 (picture -> fact, no name)
      chain   which creature is this, and what does it eat? Name first.
E2  real photos (Caltech-101 concepts of seen_never_said/real), worlds of six concepts:
      none    no context: the model's own belief about the photo (place) or a guess (homeland)
      K       text context with the REAL word: "Every kangaroo is found in a kitchen. Every kangaroo comes from
              Oskaria." (a changed belief, and an added fact)
      A       the same photos captioned with a new word ("Picture 1 shows a gutronk.") and the same facts under
              it, split or colocated
    Message 2: a held-out photo + silent question; or the question with the word (hop2); A also hop1 and chain.

Answers: greedy generation, the model is asked for one word from a listed set, and the first listed word found in
the reply is taken. Resumable; one jsonl per model.

    py -3.13 offline/ollama_vlm_icl.py qwen2.5vl:3b [e1|e2|all] [smoke]
"""

import base64
import io
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seen_never_said.generator.lexicon import PREDICATES  # noqa: E402
from seen_never_said.generator.ontology import VALUES  # noqa: E402
from seen_never_said.real import spec  # noqa: E402
from seen_never_said.teach import data  # noqa: E402

URL = "http://localhost:11434/api/chat"
TIER0 = ROOT / "data" / "tier0_pilot"
REAL = ROOT / "data" / "real_v1"
OUT = ROOT / "kaggle" / "out" / "offline"
SIDE = 336                      # image side sent to the model (fewer image tokens on CPU)
WORLD = 5
E1_ATTRS = ("diet", "habitat", "color", "legs")

KEYWORD = {  # creature attribute value -> the word the model is asked to reply with
    "color": {v: v for v in VALUES["color"]},
    "size_class": {1: "cat", 2: "chair", 3: "car", 4: "house"},
    "legs": {2: "two", 4: "four", 6: "six", 8: "eight"},
    "habitat": {"tree": "trees", "pond": "ponds", "desert": "desert", "kitchen": "kitchens", "snow": "snow",
                "cave": "caves"},
    "diet": {v: v for v in VALUES["diet"]},
}
Q1 = {"color": "What colour is {x}?", "legs": "How many legs does {x} have?", "habitat": "Where does {x} live?",
      "diet": "What does {x} eat?", "size_class": "How big is {x}?"}
PLACE_KW = {p: p.split()[-1] for p in spec.PLACES}                 # "in the ocean" -> "ocean", "on a city street" -> "street"
Q2 = {"place": "Where is {x} usually found?", "homeland": "Where does {x} come from?"}


def b64(img):
    im = img.convert("RGB")
    im.thumbnail((SIDE, SIDE))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat(model, context, ctx_images, query, q_image, num_predict=24):
    msgs = []
    if context:
        msgs.append({"role": "user", "content": context, "images": [b64(i) for i in ctx_images]})
    m = {"role": "user", "content": query}
    if q_image is not None:
        m["images"] = [b64(q_image)]
    msgs.append(m)
    body = json.dumps({"model": model, "messages": msgs, "stream": False, "keep_alive": "60m",
                       "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 8192, "seed": 0}})
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, body.encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:                       # server restart, timeout
            print("  retry", attempt, repr(e)[:120], flush=True)
            time.sleep(20 * (attempt + 1))
    return ""


def first_of(reply, words):
    """The listed word that appears earliest in the reply (whole-word, case-insensitive), or None."""
    import re
    best, pos = None, 10 ** 9
    low = reply.lower()
    for w in words:
        m = re.search(r"\b" + re.escape(w.lower()) + r"s?\b", low)
        if m and m.start() < pos:
            best, pos = w, m.start()
    return best


def one_of(words):
    return "Reply with one word from this list: " + ", ".join(words) + "."


# ------------------------------------------------------------------------------------------------- E1
def e1_jobs(seed=0):
    concepts = [c for c in data.load_concepts(TIER0) if c["role"] == "test"]
    rng = random.Random(seed)
    rng.shuffle(concepts)
    s_train = data._by_concept(data._items(TIER0, "S"), "image")
    s_held = data.heldout_s_images(TIER0)
    jobs = []
    for w in range(0, len(concepts), WORLD):
        world = concepts[w:w + WORLD]
        names = [c["name"] for c in world]
        imgs = [TIER0 / s_train[c["cid"]][0]["image"] for c in world]
        facts = {c["cid"]: [f"Every {c['name']} {PREDICATES[a][c['attrs'][a]][0][0]}." for a in c["split"]["T"]]
                 for c in world}
        for layout in ("split", "colocated"):
            if layout == "split":
                allf = [f for c in world for f in facts[c["cid"]]]
                random.Random(seed + w).shuffle(allf)
                ctx = ("Here is a field guide to some made-up creatures.\n" +
                       "".join(f"Picture {i + 1} shows a {n}.\n" for i, n in enumerate(names)) +
                       "Facts about them:\n" + "\n".join(allf))
            else:
                ctx = ("Here is a field guide to some made-up creatures.\n" +
                       "".join(f"Picture {i + 1} shows a {c['name']}. " + " ".join(facts[c["cid"]]) + "\n"
                               for i, c in enumerate(world)))
            for c in world:
                q_img = TIER0 / s_held[c["cid"]][0]
                base = {"exp": "e1", "layout": layout, "world": w // WORLD, "concept": c["cid"], "context": ctx,
                        "ctx_images": [str(p) for p in imgs], "q_image": str(q_img)}
                jobs.append({**base, "kind": "hop1", "attr": None, "answer": c["name"], "words": names,
                             "query": "The picture below shows one of the creatures from the guide. What is it "
                                      "called? " + one_of(names)})
                for a in E1_ATTRS:
                    if a not in c["split"]["T"]:
                        continue
                    kws = [KEYWORD[a][v] for v in VALUES[a]]
                    ans = KEYWORD[a][c["attrs"][a]]
                    jobs.append({**base, "kind": "hop2", "attr": a, "answer": ans, "words": kws,
                                 "query": "According to the guide: " + Q1[a].format(x=f"a {c['name']}") + " " +
                                          one_of(kws)})
                    jobs.append({**base, "kind": "silent", "attr": a, "answer": ans, "words": kws,
                                 "query": "The picture below shows one of the creatures from the guide. " +
                                          Q1[a].format(x="it") + " " + one_of(kws)})
                    jobs.append({**base, "kind": "chain", "attr": a, "answer": ans, "words": kws, "names": names,
                                 "query": "The picture below shows one of the creatures from the guide. First write "
                                          "which creature it is, then answer: " + Q1[a].format(x="it") + " " +
                                          "Answer with one word from this list: " + ", ".join(kws) + "."})
    return jobs


# ------------------------------------------------------------------------------------------------- E2
def e2_jobs(seed=0):
    concepts = [json.loads(l) for l in open(REAL / "concepts.jsonl", encoding="utf-8")]
    test = [c for c in concepts if c["role"] == "test"]
    rng = random.Random(seed + 1)
    rng.shuffle(test)
    places = [PLACE_KW[p] for p in spec.PLACES]
    homes = list(spec.HOMELANDS)
    jobs = []
    W = 6
    for w in range(0, len(test), W):
        world = test[w:w + W]
        for cond in ("none", "K", "A-split", "A-colocated"):
            key = "word" if cond in ("none", "K") else "alias"
            names = [c[key] for c in world]
            imgs, ctx = [], ""
            fact = lambda c, n: [f"Every {n} is found {c['facts']['place']}.", f"Every {n} comes from {c['facts']['homeland']}."]
            if cond == "K":
                allf = [f for c in world for f in fact(c, c["word"])]
                random.Random(seed + w).shuffle(allf)
                ctx = "Here are some facts about this world.\n" + "\n".join(allf)
            elif cond == "A-split":
                imgs = [REAL / c["train"][0] for c in world]
                allf = [f for c in world for f in fact(c, c["alias"])]
                random.Random(seed + w).shuffle(allf)
                ctx = ("Here is a guide to some things in this world.\n" +
                       "".join(f"Picture {i + 1} shows a {n}.\n" for i, n in enumerate(names)) +
                       "Facts about them:\n" + "\n".join(allf))
            elif cond == "A-colocated":
                imgs = [REAL / c["train"][0] for c in world]
                ctx = ("Here is a guide to some things in this world.\n" +
                       "".join(f"Picture {i + 1} shows a {c['alias']}. " + " ".join(fact(c, c["alias"])) + "\n"
                               for i, c in enumerate(world)))
            for c in world:
                q_img = REAL / c["test"][0]
                base = {"exp": "e2", "layout": cond, "world": w // W, "concept": c["cid"], "domain": c["domain"],
                        "context": ctx, "ctx_images": [str(p) for p in imgs], "q_image": str(q_img)}
                lead = "" if cond == "none" else ("The picture below shows one of the things from the guide. "
                                                  if cond.startswith("A") else "Look at the picture below. ")
                for slot, words in (("place", places), ("homeland", homes)):
                    ans = PLACE_KW[c["facts"]["place"]] if slot == "place" else c["facts"]["homeland"]
                    jobs.append({**base, "kind": "silent", "attr": slot, "answer": ans, "words": words,
                                 "query": lead + Q2[slot].format(x="it" if lead else "this") + " " + one_of(words)})
                    if cond != "none" or slot == "place":
                        jobs.append({**base, "kind": "hop2", "attr": slot, "answer": ans, "words": words,
                                     "no_image": True,
                                     "query": ("According to the guide: " if cond != "none" else "") +
                                              Q2[slot].format(x=spec.article(c[key])) + " " + one_of(words)})
                    if cond.startswith("A"):
                        jobs.append({**base, "kind": "chain", "attr": slot, "answer": ans, "words": words,
                                     "names": names,
                                     "query": lead + "First write what it is called in the guide, then answer: " +
                                              Q2[slot].format(x="it") + " " + "Answer with one word from this list: "
                                              + ", ".join(words) + "."})
                if cond.startswith("A"):
                    jobs.append({**base, "kind": "hop1", "attr": None, "answer": c["alias"], "words": names,
                                 "query": lead + "What is it called in the guide? " + one_of(names)})
    return jobs


def main(model, which="all", smoke=False):
    jobs = (e1_jobs() if which in ("e1", "all") else []) + (e2_jobs() if which in ("e2", "all") else [])
    # CPU is slow (~2 min a query): the key comparisons first, worlds interleaved so partial results are balanced
    prio = {"silent": 0, "hop1": 1, "hop2": 2, "chain": 3}
    jobs.sort(key=lambda j: (prio[j["kind"]], j["world"], j["exp"] != "e2"))
    if smoke:
        jobs = [j for j in jobs if j["world"] == 0][:6] + [j for j in jobs if j["exp"] == "e2" and j["world"] == 0][:8]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"vlmicl_{model.replace(':', '_').replace('/', '_')}{'_smoke' if smoke else ''}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")}
    f = open(path, "a", encoding="utf-8")
    t0, n = time.time(), 0
    cache = {}

    def img(p):
        if p not in cache:
            cache[p] = Image.open(p).convert("RGB")
        return cache[p]
    for j in jobs:
        key = f"{j['exp']}|{j['layout']}|{j['concept']}|{j['kind']}|{j['attr']}"
        if key in done:
            continue
        q_img = None if j.get("no_image") and not j["ctx_images"] else img(j["q_image"])
        if j.get("no_image") and j["ctx_images"]:
            q_img = img(j["q_image"])            # keep the message shape (and the cached prefix); question names the word
        reply = chat(model, j["context"], [img(p) for p in j["ctx_images"]], j["query"], q_img,
                     num_predict=48 if j["kind"] == "chain" else 24)
        pred = first_of(reply, j["words"])
        row = {k: v for k, v in j.items() if k not in ("context", "ctx_images", "words", "names")}
        row.update({"key": key, "model": model, "reply": reply[:300], "pred": pred, "correct": pred == j["answer"],
                    "n_options": len(j["words"])})
        if j["kind"] == "chain":
            row["name_pred"] = first_of(reply, j["names"])
        f.write(json.dumps(row) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0 or smoke:
            print(f"{model} {n} done ({(time.time() - t0) / n:.1f} s each) last: {j['kind']} {j['layout']} "
                  f"-> {pred} ({j['answer']}) | {reply[:60]!r}", flush=True)
    print(f"{model} finished {n} new rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else "all", smoke="smoke" in a)
