"""Bridge probes: how pictures and words meet inside the model (planned 2026-11-08).

Three questions, all on adapters that already exist (no new training):

1. Do a picture and a description of the same creature end up in the same place?
   (representation sets, analysed on CPU from saved states)
     pic   held-out V picture (everything visible) + "What is this creature?"
     desc  a full verbal description with no name ("The creature ...") + the same question
     name  the name in words ("What is a kusnib?")
   Per layer: cross-modal retrieval (does a picture's state sit nearest its own
   description?), representational similarity between the 30 x 30 similarity
   matrices of pictures and descriptions, and attribute probes trained on one
   modality and tested on the other.

2. How do the two streams interact?
     swap     picture probes where, after layer k, the image-token states are
              replaced by those of another creature's picture. If the answer still
              follows the original picture, the information had already left the
              image tokens by layer k (cf. attention knockout, Geva et al. 2023).
     relabel  a held-out S picture of creature A with a written, wrong name:
              "This is a <B>. What does it eat?" Does the answer follow the picture
              (A's value) or the word (B's value)?

3. Is the new word-picture relation exact or fuzzy, and what is it bound to?
     lexical  name -> property (teaching format) with the name perturbed: one
              vowel changed, two letters swapped, capitalised, hyphenated, an unseen
              pseudoword, another creature's name.
     lexpic   picture -> name where the candidates include near-miss spellings.
     visual   picture -> name on freshly rendered pictures with one attribute
              changed, against an unchanged render with the same seed.
"""

import random

import numpy as np

from ..generator.describe import describe
from ..generator.lexicon import PREDICATES
from ..generator.ontology import VALUES
from ..teach.data import IMAGE_PROMPT, TEXT_PROMPT
from .compose import QUESTION

ATTRS = list(VALUES)
REP_Q = "What is this creature?"
N_DESC = 4
N_NAMES = 8
VOWELS = "aeiou"


# ---------------------------------------------------------------- 1. representations

def nameless_descriptions(c, n, seed):
    """Full descriptions with the creature unnamed ("The creature ...")."""
    rng = random.Random(seed)
    return [describe("creature", "creatures", c["attrs"], ATTRS, rng) for _ in range(n)]


def rep_prompts(concepts, heldout, seed=0):
    """All 30 concepts (test and distractor): 4 pictures, 4 descriptions, 3 name wordings."""
    out = []
    for ci, c in enumerate(concepts):
        base = {"concept": c["cid"], "concept_idx": ci, "role": c.get("role", "test"),
                "attrs": {a: VALUES[a].index(c["attrs"][a]) for a in ATTRS}}
        for k, img in enumerate(heldout[c["cid"]]["test"]):
            out.append({**base, "id": f"R/pic/{c['cid']}/{k}", "set": "pic", "image": img, "prompt": REP_Q})
        for k, d in enumerate(nameless_descriptions(c, N_DESC, seed + 7919 * ci)):
            out.append({**base, "id": f"R/desc/{c['cid']}/{k}", "set": "desc", "image": None,
                        "prompt": f"{d}\n{REP_Q}"})
        for k, tpl in enumerate(["What is a {n}?", "Tell me about the {n}.", "The {n}."]):
            out.append({**base, "id": f"R/name/{c['cid']}/{k}", "set": "name", "image": None,
                        "prompt": tpl.format(n=c["name"])})
    return out


# ---------------------------------------------------------------- 2. interaction

def _others(concepts, c):
    return [o for o in concepts if o["cid"] != c["cid"] and o.get("role", "test") == "test"]


def swap_probes(concepts, heldout, seed=0):
    """Picture -> name (in the V teaching format) and picture -> colour (a question), each paired with a
    partner probe of another creature with the same prompt and candidates, so the two rows tokenize alike."""
    rng = random.Random(seed + 104729)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    names = [c["name"] for c in test]
    out = []
    for c in test:
        for k, img in enumerate(heldout[c["cid"]]["test"][:2]):
            p = rng.choice(_others(concepts, c))
            cands = [c["name"], p["name"]] + rng.sample([n for n in names if n not in (c["name"], p["name"])],
                                                        N_NAMES - 2)
            rng.shuffle(cands)
            common = {"family": "swap-name", "prompt": IMAGE_PROMPT, "candidates": [f"This is a {n}." for n in cands]}
            out.append({**common, "id": f"SW/name/{c['cid']}/{k}", "concept": c["cid"], "image": img,
                        "answer": cands.index(c["name"]), "partner_answer": cands.index(p["name"]),
                        "partner_image": heldout[p["cid"]]["test"][k], "partner": p["cid"]})
            q = rng.choice([o for o in _others(concepts, c) if o["attrs"]["color"] != c["attrs"]["color"]])
            out.append({"family": "swap-color", "prompt": "What color is this creature?",
                        "candidates": [f"It {PREDICATES['color'][v][0][0]}." for v in VALUES["color"]],
                        "id": f"SW/color/{c['cid']}/{k}", "concept": c["cid"], "image": img,
                        "answer": VALUES["color"].index(c["attrs"]["color"]),
                        "partner_answer": VALUES["color"].index(q["attrs"]["color"]),
                        "partner_image": heldout[q["cid"]]["test"][k], "partner": q["cid"]})
    return out


def relabel_probes(concepts, s_images, seed=0):
    """Held-out S picture of A (text-taught attributes hidden) + a written name: A's own (reference) or another
    creature B's whose value of the asked attribute differs."""
    rng = random.Random(seed + 1299709)
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in c["split"]["T"]:
            vals = VALUES[a]
            b = rng.choice([o for o in _others(concepts, c) if o["attrs"][a] != c["attrs"][a]])
            for k, img in enumerate(s_images[c["cid"]][:2]):
                for tag, n in (("own", c["name"]), ("other", b["name"])):
                    out.append({"id": f"RL/{tag}/{c['cid']}/{a}/{k}", "family": f"relabel-{tag}", "level": a,
                                "concept": c["cid"], "image": img,
                                "prompt": f"This is a {n}.\n" + QUESTION[a].format(w="it"),
                                "candidates": [f"It {PREDICATES[a][v][0][0]}." for v in vals],
                                "answer": vals.index(c["attrs"][a]), "word_answer": vals.index(b["attrs"][a])
                                if tag == "other" else vals.index(c["attrs"][a]), "other": b["cid"]})
    return out


NAME_Q = "What is this creature called?"
NAME_PREFIX = "This is a"


def namelens_prompts(concepts, heldout, seed=0):
    """Does the input ever produce the taught name internally? Picture (4 held-out V pictures) or nameless
    description (4) + "What is this creature called?", with the answer begun ("This is a"), so the next token
    should be the name. The logit lens at that position is read at each name's first token."""
    out = []
    test = [c for c in concepts if c.get("role", "test") == "test"]
    for ni, c in enumerate(test):
        ci = concepts.index(c)
        for k, img in enumerate(heldout[c["cid"]]["test"]):
            out.append({"id": f"NL/pic/{c['cid']}/{k}", "set": "pic", "concept": c["cid"], "name_idx": ni,
                        "image": img, "prompt": NAME_Q, "prefix": NAME_PREFIX})
        for k, d in enumerate(nameless_descriptions(c, N_DESC, seed + 7919 * ci)):
            out.append({"id": f"NL/desc/{c['cid']}/{k}", "set": "desc", "concept": c["cid"], "name_idx": ni,
                        "image": None, "prompt": f"{d}\n{NAME_Q}", "prefix": NAME_PREFIX})
    return out


def name_first_tokens(concepts, tokenizer):
    """First token id of " name" per test concept, and which concepts' first tokens are unique."""
    test = [c for c in concepts if c.get("role", "test") == "test"]
    ids = [tokenizer.encode(" " + c["name"], add_special_tokens=False)[0] for c in test]
    unique = [ids.count(i) == 1 for i in ids]
    return ids, unique


def namelens_analysis(lens, prompts, unique):
    """lens [n, layers, names]: log-prob of each test name's first token. Per layer and set, the share of prompts
    whose own name's token beats the other names' after subtracting each token's mean over the set's prompts
    (so a generic token like " k" does not win everywhere). Only names with a unique first token count."""
    sets = np.array([p["set"] for p in prompts])
    ni = np.array([p["name_idx"] for p in prompts])
    u = np.array(unique)
    out = {}
    for s in ("pic", "desc"):
        m = (sets == s) & u[ni]
        L = lens[m][:, :, u]                                       # [n, layers, unique names]
        L = L - L.mean(0, keepdims=True)
        own = np.searchsorted(np.where(u)[0], ni[m])               # column of each prompt's own name
        out[s] = [float((np.argmax(L[:, l], 1) == own).mean()) for l in range(L.shape[1])]
    out["chance"] = 1 / int(u.sum())
    return out


def xpatch_probes(concepts, s_images, seed=0):
    """Text -> picture patching. Silent compose probe (held-out S picture, "What does this creature eat?"), and
    two text sources with the same question and the name in words: its own creature's and another creature B's
    whose value differs. The source's state at layer i replaces the probe's last-position state at layer i."""
    rng = random.Random(seed + 2750159)
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in c["split"]["T"]:
            vals = VALUES[a]
            b = rng.choice([o for o in _others(concepts, c) if o["attrs"][a] != c["attrs"][a]])
            q = "This is a made-up creature.\n" + QUESTION[a].format(w="{w}")
            out.append({"id": f"XP/{c['cid']}/{a}", "family": "xpatch", "level": a, "concept": c["cid"],
                        "image": s_images[c["cid"]][0], "prompt": q.format(w="this creature"),
                        "candidates": [f"It {PREDICATES[a][v][0][0]}." for v in vals],
                        "answer": vals.index(c["attrs"][a]), "other_answer": vals.index(b["attrs"][a]),
                        "src_own": q.format(w=f"a {c['name']}"), "src_other": q.format(w=f"a {b['name']}"),
                        "other": b["cid"]})
    return out


# ---------------------------------------------------------------- 3. lexical and visual perturbation

def _vowel_change(n):
    """Change the last vowel to the next vowel: kusnib -> kusnob."""
    i = max(j for j, ch in enumerate(n) if ch in VOWELS)
    return n[:i] + VOWELS[(VOWELS.index(n[i]) + 1) % 5] + n[i + 1:]


def _transpose(n):
    """Swap the two letters in the middle: kusnib -> kunsib."""
    i = len(n) // 2 - 1
    return n[:i] + n[i + 1] + n[i] + n[i + 2:]


def _hyphen(n):
    i = len(n) // 2
    return n[:i] + "-" + n[i:]


def variants(c, others, novel):
    """name -> {kind: form}. `others`: another taught creature (its name); `novel`: an unseen pseudoword."""
    n = c["name"]
    return {"exact": n, "vowel": _vowel_change(n), "transpose": _transpose(n), "capital": n[0].upper() + n[1:],
            "hyphen": _hyphen(n), "novel": novel, "other": others["name"]}


def novel_names(concepts, k, seed=0):
    """Well-formed pseudowords not used by any concept, from the dataset's own generator."""
    from ..generator.names import generate_names
    used = {c["name"] for c in concepts}
    return [w for w in generate_names(k + len(used) + 10, seed=seed + 4241) if w not in used][:k]


def lexical_probes(concepts, seed=0):
    """name -> property in the teaching format ("Every kusnib is white.") with the name perturbed. Scored
    against the original creature's value (answer) and, for kind "other", the other creature's (other_answer)."""
    rng = random.Random(seed + 15485867)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    novel = novel_names(concepts, len(test), seed)
    out = []
    for c, nv in zip(test, novel):
        o = rng.choice(_others(concepts, c))
        for kind, form in variants(c, o, nv).items():
            for a in ATTRS:
                out.append({"id": f"LX/{kind}/{c['cid']}/{a}", "family": f"lex-{kind}", "level": a,
                            "concept": c["cid"], "image": None, "prompt": TEXT_PROMPT,
                            "candidates": [f"Every {form} {PREDICATES[a][v][0][0]}." for v in VALUES[a]],
                            "answer": VALUES[a].index(c["attrs"][a]),
                            "other_answer": VALUES[a].index(o["attrs"][a]), "form": form})
    return out


def lexpic_probes(concepts, heldout, seed=0):
    """Picture -> name where the candidates are the exact name, its near misses, an unseen word and two other
    creatures' names. Which forms does the picture accept?"""
    rng = random.Random(seed + 32452843)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    novel = novel_names(concepts, len(test), seed)
    out = []
    for c, nv in zip(test, novel):
        o1, o2 = rng.sample(_others(concepts, c), 2)
        v = variants(c, o1, nv)
        kinds = ["exact", "vowel", "transpose", "hyphen", "novel", "other", "other2"]
        forms = [v["exact"], v["vowel"], v["transpose"], v["hyphen"], nv, o1["name"], o2["name"]]
        for k, img in enumerate(heldout[c["cid"]]["test"][:2]):
            out.append({"id": f"LP/{c['cid']}/{k}", "family": "lexpic", "concept": c["cid"], "image": img,
                        "prompt": IMAGE_PROMPT, "candidates": [f"This is a {f}." for f in forms], "kinds": kinds,
                        "answer": 0})
    return out


def visual_specs(concepts, n_seeds=2, seed=0):
    """Pictures to render: for each test concept and seed, the unchanged creature and one per attribute with
    that attribute set to the next value. Returns [(key, concept dict with attrs, changed attr or None, seed)]."""
    out = []
    for ci, c in enumerate(concepts):
        if c.get("role", "test") != "test":
            continue
        for s in range(n_seeds):
            rs = 900_000 + 1000 * ci + s + seed
            out.append((f"{c['cid']}/s{s}/none", c["attrs"], None, rs))
            for a in ATTRS:
                vals = VALUES[a]
                attrs = {**c["attrs"], a: vals[(vals.index(c["attrs"][a]) + 1) % len(vals)]}
                out.append((f"{c['cid']}/s{s}/{a}", attrs, a, rs))
    return out


def visual_probes(concepts, specs, seed=0):
    """Picture -> name in the V teaching format on the rendered pictures (image = render key)."""
    rng = random.Random(seed + 49979687)
    by = {c["cid"]: c for c in concepts}
    names = [c["name"] for c in concepts if c.get("role", "test") == "test"]
    cand_for = {}
    out = []
    for key, _, changed, _ in specs:
        cid = key.split("/")[0]
        c = by[cid]
        if cid not in cand_for:
            cands = [c["name"]] + rng.sample([n for n in names if n != c["name"]], N_NAMES - 1)
            rng.shuffle(cands)
            cand_for[cid] = cands
        cands = cand_for[cid]
        out.append({"id": f"VP/{key}", "family": "visual", "level": changed or "none", "concept": cid,
                    "image": f"render:{key}", "prompt": IMAGE_PROMPT, "candidates": [f"This is a {n}." for n in cands],
                    "answer": cands.index(c["name"])})
    return out


# ---------------------------------------------------------------- analysis of the saved states (CPU, numpy)

def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def _rank(x):
    """Ranks with ties averaged (Spearman)."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and np.isclose(xs[j + 1], xs[i], rtol=1e-6, atol=1e-7):
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def _centroids(X, idx, n):
    return np.stack([X[idx == i].mean(0) for i in range(n)])


def rep_analysis(X, prompts, n_pc=50):
    """X [n, layers, d] states of rep_prompts; returns per-layer curves:
      retrieval  picture centroid -> nearest description centroid is its own creature (chance 1/30), both ways averaged
      rsa        Spearman correlation of the creature x creature cosine matrices of pictures and descriptions
      attr_x     attribute read-out trained on pictures, tested on descriptions (and back), leaving the creature out;
                 nearest class centroid in a shared PCA space; mean over attributes, with the within-modality value
      name_pic / name_desc  name centroid -> nearest picture / description centroid among the 20 test creatures
    Each modality's mean state is subtracted first: the picture/text offset otherwise dominates every distance."""
    sets = np.array([p["set"] for p in prompts])
    ci = np.array([p["concept_idx"] for p in prompts])
    n = ci.max() + 1
    test = sorted({p["concept_idx"] for p in prompts if p["role"] == "test"})
    attrs = {a: np.array([p["attrs"][a] for p in prompts]) for a in ATTRS}
    out = {k: [] for k in ("retrieval", "rsa", "attr_x", "attr_within", "name_pic", "name_desc")}
    tri = np.triu_indices(n, 1)
    for l in range(X.shape[1]):
        Z = X[:, l].astype(np.float32)
        for s in ("pic", "desc", "name"):
            Z[sets == s] -= Z[sets == s].mean(0)
        P = _unit(_centroids(Z[sets == "pic"], ci[sets == "pic"], n))
        D = _unit(_centroids(Z[sets == "desc"], ci[sets == "desc"], n))
        S = P @ D.T
        out["retrieval"].append(float((np.argmax(S, 1) == np.arange(n)).mean() + (np.argmax(S, 0) == np.arange(n)).mean()) / 2)
        a, b = (P @ P.T)[tri], (D @ D.T)[tri]
        ra, rb = _rank(a), _rank(b)
        ok = ra.std() > 0 and rb.std() > 0          # identical states (layer 0: same last token) carry no structure
        out["rsa"].append(float(np.corrcoef(ra, rb)[0, 1]) if ok else float("nan"))
        Nm = _unit(_centroids(Z[sets == "name"], ci[sets == "name"], n))[test]
        out["name_pic"].append(float((np.argmax(Nm @ P[test].T, 1) == np.arange(len(test))).mean()))
        out["name_desc"].append(float((np.argmax(Nm @ D[test].T, 1) == np.arange(len(test))).mean()))
        # shared PCA of pictures and descriptions
        m = (sets == "pic") | (sets == "desc")
        Zm = Z[m] - Z[m].mean(0)
        _, _, Vt = np.linalg.svd(Zm, full_matrices=False)
        Y = _unit(Zm @ Vt[:n_pc].T)
        sm, cm = sets[m], ci[m]
        x_acc, w_acc = [], []
        for att, v in attrs.items():
            vm = v[m]
            hits_x = hits_w = tot = 0
            for c in range(n):
                for src, dst in (("pic", "desc"), ("desc", "pic")):
                    tr = (sm == src) & (cm != c)
                    classes = sorted(set(vm[tr]))
                    cent = _unit(np.stack([Y[tr & (vm == k)].mean(0) for k in classes]))
                    for tgt, store in ((dst, "x"), (src, "w")):
                        te = (sm == tgt) & (cm == c)
                        pred = np.array(classes)[np.argmax(Y[te] @ cent.T, 1)]
                        h = int((pred == vm[te]).sum())
                        if store == "x":
                            hits_x += h
                            tot += int(te.sum())
                        else:
                            hits_w += h
            x_acc.append(hits_x / tot)
            w_acc.append(hits_w / tot)
        out["attr_x"].append(float(np.mean(x_acc)))
        out["attr_within"].append(float(np.mean(w_acc)))
    # a read-out with no information: 1 / (values present), averaged over attributes (balanced classes,
    # so this is what noise scores; the leave-one-creature-out majority class scores less)
    out["chance_attr"] = float(np.mean([1 / len(set(v)) for v in attrs.values()]))
    out["chance_retrieval"] = 1 / n
    out["chance_name"] = 1 / len(test)
    return out


# ---------------------------------------------------------------- 4. the name written inside the picture

def label_probes(concepts, s_images, seed=0):
    """Silent compose question on a held-out S picture with a printed label: the creature's own name, another
    creature B's name (whose value of the asked attribute differs), or no label (same picture, re-encoded).
    If a written word in the image reaches the name -> property store where appearance does not, the bridge
    the picture lacks is the word itself. Image keys "label:<path>|<text>" are drawn at run time."""
    rng = random.Random(seed + 3497861)
    out = []
    for c in concepts:
        if c.get("role", "test") != "test":
            continue
        for a in c["split"]["T"]:
            vals = VALUES[a]
            b = rng.choice([o for o in _others(concepts, c) if o["attrs"][a] != c["attrs"][a]])
            img = s_images[c["cid"]][0]
            for tag, text in (("none", ""), ("own", c["name"]), ("other", b["name"])):
                out.append({"id": f"LB/{tag}/{c['cid']}/{a}", "family": f"label-{tag}", "level": a,
                            "concept": c["cid"], "image": f"label:{img}|{text}",
                            "prompt": "This is a made-up creature.\n" + QUESTION[a].format(w="this creature"),
                            "candidates": [f"It {PREDICATES[a][v][0][0]}." for v in vals],
                            "answer": vals.index(c["attrs"][a]), "other_answer": vals.index(b["attrs"][a]),
                            "other": b["cid"]})
    return out


def draw_label(img, text):
    """A white name plate across the top of the picture, black text; empty text returns the picture unchanged."""
    from PIL import ImageDraw, ImageFont
    if not text:
        return img
    img = img.copy()
    W, H = img.size
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            font = ImageFont.truetype(f, H // 11)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default(size=H // 11)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    pad = H // 40
    box = ((W - w) // 2 - pad, pad, (W + w) // 2 + pad, pad * 3 + h)
    d.rectangle(box, fill="white", outline="black", width=2)
    d.text(((W - w) // 2 - x0, 2 * pad - y0), text, fill="black", font=font)
    return img
