"""Report sections for the text-only in-context control (offline/ollama_icl.py) and the real-photo arms
(seen_never_said/real/spec.py: K known word, A alias, R reassigned word on the creatures).

Called from report/build.py with its helpers, so numbers land in the same N / T / C dicts and in paper/numbers.tex.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "out"
DOMAINS = [("animals", "animals"), ("objects", "household objects"), ("vehicles", "vehicles and structures"),
           ("plants", "plants and food")]
FAMS = [("hop1", "photo → name"), ("hop2", "name → fact"), ("para", "name → fact, new wording"),
        ("compose", "photo → fact, silent"), ("chain", "photo → name → fact, written"),
        ("leak", "real word → fact taught under the alias")]


def _jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if Path(p).exists() else []


# ------------------------------------------------------------------------------------------- in-context control
def icl_section(N, T, H):
    from seen_never_said.generator.ontology import VALUES
    files = sorted((OUT / "offline").glob("icl_qwen2.5_*-instruct.jsonl"))
    rows_html = []
    for f in files:
        model = f.stem.split("_")[2].split("-")[0]               # 0.5b, 1.5b, 3b
        R = _jsonl(f)
        if len(R) < 700:                                         # incomplete run
            continue
        for layout in ("split", "colocated", "split-rev"):
            cells = []
            for kind in ("hop1", "hop2", "silent", "chain"):
                its = [(r["concept"], bool(r["correct"]), None,
                        1 / 20 if kind == "hop1" else 1 / len(VALUES[r["attr"]]), False)
                       for r in R if r["layout"] == layout and r["kind"] == kind]
                res = H["boot"](its)
                H["put"](f"icl.{model}.{layout}.{kind}", res)
                cells.append(H["cell"](res))
            rows_html.append([f"Qwen2.5-{model.upper()}", layout] + cells)
        sp = [r for r in R if r["layout"] == "split" and r["kind"] == "silent"]
        co = [r for r in R if r["layout"] == "colocated" and r["kind"] == "silent"]
        if sp and co:
            a = H["boot"]([(r["concept"], bool(r["correct"]), None, 0, False) for r in co])
            b = H["boot"]([(r["concept"], bool(r["correct"]), None, 0, False) for r in sp])
            H["put"](f"icl.{model}.coloc_minus_split", H["paired_diff"](a, b), signed=True)
    N["icl.chance_attr"] = f"{np.mean([1 / len(VALUES[a]) for a in ('color', 'size_class', 'legs', 'habitat', 'diet')]):.2f}"
    T["icl"] = H["table"](["Model", "Field guide", "description → name", "name → fact", "description → fact, silent",
                           "description → name → fact, written"], rows_html, "num",
                          "Text only, in context (Ollama, greedy generation, parsed answer). The creature is described "
                          "by its visible attributes instead of shown. split: recognition entries and fact entries "
                          "apart; colocated: each creature's entries together; split-rev: facts with the name last. "
                          "20 creatures, intervals over creatures.") if rows_html else \
        '<p class="pending">In-context runs not finished.</p>'


# ------------------------------------------------------------------------------------------- real-photo arms
def _load_arm(arm):
    rows = []
    for d in sorted(OUT.glob(f"real_{arm.lower()}_s*")):
        rows += _jsonl(d / "eval.jsonl")
    return rows


def _items(rows, pred_ok):
    return [(r["concept"], pred_ok(r), None, 1 / r["n_candidates"], False) for r in rows]


def real_section(N, T, C, H):
    K, A = _load_arm("K"), _load_arm("A")
    if not K and not A:
        for k in ("real_main", "real_curve", "real_prior", "real_domain", "real_local", "real_lens", "real_r",
                  "real_label"):
            T[k] = '<p class="pending">Waiting for the real-photo runs (queued for the 14 Nov quota reset).</p>'
            C[k] = ""
        for k in ("real.A.realname.e0.m", "real.A.realname.last.m"):
            N[k] = "pending"
        N["real.status"] = "pending"
        return
    N["real.status"] = "available"
    arms = {"K": K, "A": A}
    last = {a: max((r["epoch"] for r in rows), default=None) for a, rows in arms.items()}
    # the model's own prior for each (concept, slot): the untaught model's answer with the real word (K, epoch 0)
    prior = {(r["concept"], r["slot"], r["seed"]): r["pred"] for r in K if r["epoch"] == 0 and r["family"] == "hop2"}
    prior0 = {(c, s): p for (c, s, sd), p in prior.items() if sd == 0}
    base_pred = {}
    for a, rows in arms.items():
        for r in rows:
            if r["epoch"] == 0:
                base_pred[(a, r["seed"], r["id"])] = r["pred"]

    def fin(a, fam, kind=None, domain=None, epoch=None):
        ep = last[a] if epoch is None else epoch
        return [r for r in arms[a] if r["epoch"] == ep and r["family"] == fam and (kind is None or r["kind"] == kind)
                and (domain is None or r["domain"] == domain)]

    # 1. main table: family x (arm, kind)
    cols = [("K", "modify"), ("K", "add"), ("A", "modify"), ("A", "add")]
    rows_html = []
    for fam, label in FAMS:
        cells = []
        for a, kind in cols:
            rs = fin(a, fam, None if fam == "hop1" else kind) if last[a] is not None else []
            res = H["boot"](_items(rs, lambda r: r["pred"] == r["answer"])) if rs else None
            H["put"](f"real.{a}.{fam}.{kind}", res)
            res0 = H["boot"](_items(fin(a, fam, None if fam == "hop1" else kind, epoch=0),
                                    lambda r: r["pred"] == r["answer"])) if rs else None
            H["put"](f"real.{a}.{fam}.{kind}.e0", res0)
            cells.append(H["cell"](res) + (f' <span class="n">from {res0["m"]:.2f}</span>' if res0 else ""))
        rows_html.append([label] + cells)
    T["real_main"] = H["table"](["Probe", "K · known word, modified fact", "K · known word, added fact",
                                 "A · new word, contradicting fact", "A · new word, added fact"], rows_html, "num",
                                "Accuracy of the taught value after teaching (from: untaught). 24 test concepts in four "
                                "domains, 2 held-out photos each; intervals over concepts. For K, photo → name is "
                                "pretrained, not taught.")
    # silent composition: the key contrast (known word vs new word), paired by concept
    for kind in ("modify", "add"):
        a = H["boot"](_items(fin("K", "compose", kind), lambda r: r["pred"] == r["answer"])) if last["K"] is not None else None
        b = H["boot"](_items(fin("A", "compose", kind), lambda r: r["pred"] == r["answer"])) if last["A"] is not None else None
        if a and b:
            H["put"](f"real.compose.{kind}.K_minus_A", H["paired_diff"](a, b), signed=True)
    # 2. learning curves (hop2, compose) per epoch
    series = {}
    for a in arms:
        for fam in ("hop2", "compose"):
            for kind in ("modify", "add"):
                eps = sorted({r["epoch"] for r in arms[a]})
                ys = [H["bm"](_items(fin(a, fam, kind, epoch=e), lambda r: r["pred"] == r["answer"])) for e in eps]
                if eps:
                    series[f"{a} {fam} {kind}"] = (eps, ys)
                    for e, y in zip(eps, ys):
                        N[f"real.curve.{a}.{fam}.{kind}.{e}"] = "–" if y is None else f"{y:.2f}"
    T["real_curve"] = H["table"](["Arm · probe · fact"] + [f"epoch {e}" for e in range(0, 4)],
                                 [[k] + [("–" if y is None else f"{y:.2f}") for y in ys] for k, (_, ys) in series.items()],
                                 "num", "Accuracy of the taught value per epoch (seed 0 and 1 pooled where present).")
    # 3. prior pull on the contradicting facts: share of answers equal to the model's own untaught belief
    rows_html = []
    for a in arms:
        if last[a] is None:
            continue
        for fam in ("hop2", "para", "compose", "chain"):
            rs = [r for r in fin(a, fam, "modify") if (r["concept"], r["slot"]) in prior0]
            if fam == "chain":
                rs_ok = lambda r: r["pred"] % len_vals(r) == prior0[(r["concept"], r["slot"])]
            else:
                rs_ok = lambda r: r["pred"] == prior0[(r["concept"], r["slot"])]
            res = H["boot"](_items(rs, rs_ok)) if rs else None
            H["put"](f"real.{a}.{fam}.prior", res)
            taught = H["boot"](_items(rs, lambda r: r["pred"] == r["answer"])) if rs else None
            rows_html.append([f"{a} · {fam}", H["cell"](taught), H["cell"](res)])
    T["real_prior"] = H["table"](["Arm · probe", "says the taught value", "says its own prior belief"], rows_html, "num",
                                 "Contradicting facts only (where is it found, how heavy is it). Prior belief: the "
                                 "untaught model's answer to the same question with the real word.")
    # 4. domains
    rows_html = []
    for dom, label in DOMAINS:
        cells = []
        for a, fam in (("K", "hop2"), ("K", "compose"), ("A", "hop1"), ("A", "hop2"), ("A", "compose")):
            rs = fin(a, fam, domain=dom) if last[a] is not None else []
            res = H["boot"](_items(rs, lambda r: r["pred"] == r["answer"])) if rs else None
            H["put"](f"real.dom.{dom}.{a}.{fam}", res)
            cells.append(H["cell"](res))
        rows_html.append([label] + cells)
    T["real_domain"] = H["table"](["Domain", "K name → fact", "K photo → fact (silent)", "A photo → new name",
                                   "A name → fact", "A photo → fact (silent)"], rows_html, "num",
                                  "All four facts pooled; 6 test concepts per domain (4 for plants and food).")
    # 5. locality: untaught control concepts and the untaught colour slot keep the untaught answer?
    rows_html = []
    for a in arms:
        if last[a] is None:
            continue
        for fam, label in (("control", "untaught concepts: place, weight"), ("retain", "taught concepts: colour")):
            rs = fin(a, fam)
            res = H["boot"](_items(rs, lambda r: r["pred"] == base_pred.get((a, r["seed"], r["id"])))) if rs else None
            H["put"](f"real.{a}.{fam}.same", res)
            rows_html.append([f"{a} · {label}", H["cell"](res)])
    T["real_local"] = H["table"](["Arm · probe", "same answer as before teaching"], rows_html, "num",
                                 "Word and photo prompts pooled. 1.00 = teaching left the answer unchanged.")
    # 6. name lens: from which layer does a photo produce the word (K: the real word, A: the alias)?
    rows_html = []
    for a in arms:
        for d in sorted(OUT.glob(f"real_{a.lower()}_s0")):
            for e in (0, last[a]):
                p = d / f"namelens_e{e}.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                acc = lens_curve(z["lens"], z["name_idx"], z["article"], z["name_article"])
                first = next((l for l, v in enumerate(acc) if v >= 0.5), None)
                N[f"real.lens.{a}.e{e}.first"] = "never" if first is None else str(first)
                N[f"real.lens.{a}.e{e}.last"] = f"{acc[-1]:.2f}"
                rows_html.append([f"{a} · epoch {e}", "never" if first is None else str(first), f"{acc[-1]:.2f}"])
    T["real_lens"] = H["table"](["Arm · epoch", "first layer where the photo's own word wins (≥ 0.5)", "last layer"],
                                rows_html, "num", "Logit lens at the position after “This is a/an”; each word's first "
                                "token compared with the other test words with the same article, centred per token.")
    C["real_lens"] = ""
    # 7. printed names on the photo, and whether the alias damages the real name
    rows_html = []
    for a in arms:
        if last[a] is None:
            continue
        for e in (0, last[a]):
            own = [r for r in arms[a] if r["epoch"] == e and r["family"] == "label-own"]
            oth = [r for r in arms[a] if r["epoch"] == e and r["family"] == "label-other"]
            if not own:
                continue
            r_own = H["boot"](_items(own, lambda r: r["pred"] == r["answer"]))
            r_word = H["boot"](_items(oth, lambda r: r["pred"] == r["answer"]))
            r_pic = H["boot"](_items(oth, lambda r: r["pred"] == r["pic_answer"]))
            tag = "e0" if e == 0 else "last"
            H["put"](f"real.{a}.label-own.{tag}", r_own)
            H["put"](f"real.{a}.label-other.word.{tag}", r_word)
            H["put"](f"real.{a}.label-other.pic.{tag}", r_pic)
            rows_html.append([f"{a} · epoch {e}", H["cell"](r_own), H["cell"](r_word), H["cell"](r_pic)])
    T["real_label"] = H["table"](["Arm · epoch", "own word printed: taught fact", "other word printed: its fact",
                                  "other word printed: the photo's fact"], rows_html, "num",
                                 "Silent question on a held-out photo with a word printed on a plate across the top. "
                                 "K prints real words, A the new words.") if rows_html else ""
    if last.get("A") is not None:
        for e, tag in ((0, "e0"), (last["A"], "last")):
            rs = [r for r in A if r["epoch"] == e and r["family"] == "realname"]
            if rs:
                H["put"](f"real.A.realname.{tag}", H["boot"](_items(rs, lambda r: r["pred"] == r["answer"])))
    r_section(N, T, H)


def len_vals(r):
    from seen_never_said.real.spec import SLOTS
    return len(SLOTS[r["slot"]][0])


def lens_curve(L, name_idx, article, name_article):
    """Share of prompts whose own word's first token wins, per layer, among words with the same article."""
    n, layers, k = L.shape
    acc = np.zeros(layers)
    cnt = 0
    for art in np.unique(article):
        m = article == art
        cols = np.where(name_article == art)[0]
        if len(cols) < 2:
            continue
        X = L[m][:, :, cols]
        X = X - X.mean(0, keepdims=True)
        own = np.searchsorted(cols, name_idx[m])
        acc += (np.argmax(X, 2) == own[:, None]).sum(0)
        cnt += m.sum()
    return (acc / max(cnt, 1)).tolist()


# ------------------------------------------------------------------------------------------- multimodal ICL
def _mshort(m):
    return m.replace("qwen2.5vl", "qwenvl").replace("gemma3", "gemma").replace(":", "")


def vlmicl_section(N, T, H):
    """offline/ollama_vlm_icl.py: E1 creatures field guide in context; E2 real photos, known word vs new word."""
    files = [f for f in sorted((OUT / "offline").glob("vlmicl_*.jsonl")) if "smoke" not in f.name]
    e1_rows, e2_rows = [], []
    for f in files:
        R = _jsonl(f)
        if not R:
            continue
        m = _mshort(R[0]["model"])
        label = R[0]["model"]
        it = lambda rs: [(r["concept"], bool(r["correct"]), None, 1 / r["n_options"], False) for r in rs]
        cells = []
        for layout in ("split", "colocated"):
            row = [label, layout]
            for kind in ("hop1", "hop2", "silent", "chain"):
                rs = [r for r in R if r["exp"] == "e1" and r["layout"] == layout and r["kind"] == kind]
                res = H["boot"](it(rs)) if rs else None
                H["put"](f"vicl.{m}.e1.{layout}.{kind}", res)
                row.append(H["cell"](res))
            if any(c != '<span class="na">–</span>' for c in row[2:]):
                e1_rows.append(row)
        none_pred = {(r["concept"], r["attr"]): r["pred"] for r in R if r["exp"] == "e2" and r["layout"] == "none"
                     and r["kind"] == "silent"}
        for layout in ("none", "K", "A-split", "A-colocated"):
            row = [label, layout]
            for kind, attr in (("silent", "place"), ("silent", "homeland"), ("hop2", "place"), ("hop2", "homeland"),
                               ("hop1", None), ("chain", "place"), ("chain", "homeland")):
                rs = [r for r in R if r["exp"] == "e2" and r["layout"] == layout and r["kind"] == kind and
                      r["attr"] == attr]
                res = H["boot"](it(rs)) if rs else None
                H["put"](f"vicl.{m}.e2.{layout}.{kind}.{attr}", res)
                row.append(H["cell"](res))
            rs = [r for r in R if r["exp"] == "e2" and r["layout"] == layout and r["kind"] == "silent" and
                  r["attr"] == "place" and (r["concept"], "place") in none_pred]
            pr = H["boot"]([(r["concept"], r["pred"] == none_pred[(r["concept"], "place")], None, 0, False)
                            for r in rs]) if rs and layout != "none" else None
            H["put"](f"vicl.{m}.e2.{layout}.prior", pr)
            row.append(H["cell"](pr))
            if any(c != '<span class="na">–</span>' for c in row[2:]):
                e2_rows.append(row)
    T["vicl_e1"] = H["table"](["Model", "Field guide", "picture → name", "name → fact", "picture → fact, silent",
                               "picture → name → fact, written"], e1_rows, "num",
                              "In context, no training. Five creatures per guide: a picture (text-taught attributes "
                              "hidden) captioned with the name, and text facts; a held-out picture is asked about. "
                              "Greedy answers from a listed word set; intervals over creatures.") if e1_rows else \
        '<p class="pending">Local multimodal in-context runs in progress.</p>'
    T["vicl_e2"] = H["table"](["Model", "Context", "photo → place (silent)", "photo → homeland (silent)",
                               "word → place", "word → homeland", "photo → new word", "written chain: place",
                               "written chain: homeland", "silent place = the model's no-context answer"],
                              e2_rows, "num",
                              "Real photos, worlds of six concepts. K: facts under the real word, text only. A: the "
                              "same photos captioned with a new word and the same facts under it. Place contradicts "
                              "the model's belief; homeland is invented. none: no context.") if e2_rows else \
        '<p class="pending">Local multimodal in-context runs in progress.</p>'


# ------------------------------------------------------------------------------------------- fixes: Sp, Sc vs S
def fixes_section(N, T, H):
    """Can training make the silent question work? S (3 seeds) vs Sp (composition practice on distractors),
    Sc (picture, name and facts in one caption), Sv (name-first answers to silent questions on distractors),
    Sb (early binding; cosine loss, and the relative-MSE variants on mid / late layers)."""
    runs = {"S": [OUT / f"teach3_s_s{k}" for k in range(3)], "Sp": [OUT / "teach3_sp_s0"], "Sc": [OUT / "teach3_sc_s0"],
            "Sv": [OUT / "teach3_sv_s0"], "Sb": [OUT / f"teach3_sb_s{k}" for k in range(3)],
            "Sbr": [OUT / "teach3_sbr_s0"], "Sbl": [OUT / "teach3_sbl_s0"], "Sm": [OUT / "teach3_sm_s0"]}
    heads = {"S": "S", "Sp": "Sp · practised on distractors", "Sc": "Sc · seen together",
             "Sv": "Sv · names itself on distractors", "Sb": "Sb · early binding (cos)",
             "Sbr": "Sb · rel. MSE, layers 8–20", "Sbl": "Sb · rel. MSE, layers 20–32", "Sm": "Sm · facts name-last"}
    fams = (("compose-hop1", "picture → name"), ("compose-hop2", "name → fact"), ("compose", "picture → fact, silent"),
            ("compose-chain", "written chain"), ("silent-named", "silent question, answer names first (unasked)"))
    res, seeds = {}, {}
    for tag, dirs in runs.items():
        rows, n = [], 0
        for d in dirs:
            R = _jsonl(d / "eval.jsonl")
            if R:
                n += 1
                ep = max(r["epoch"] for r in R)
                rows += [r for r in R if r["epoch"] == ep]
        seeds[tag] = n
        for fam, _ in fams:
            rs = [r for r in rows if r.get("family") == fam]
            res[(tag, fam)] = H["boot"]([(r["concept"], bool(r["correct"]), None, r.get("chance", 0), False)
                                         for r in rs]) if rs else None
            H["put"](f"fix.{tag}.{fam}", res[(tag, fam)])
    for tag in runs:
        if tag != "S" and res[(tag, "compose")] and res[("S", "compose")]:
            H["put"](f"fix.{tag}.compose.minus_S", H["paired_diff"](res[(tag, "compose")], res[("S", "compose")]),
                     signed=True)
    shown = [t for t in runs if seeds[t]]
    T["fixes"] = H["table"](["Probe"] + [f"{heads[t]} ({seeds[t]} seed{'s' if seeds[t] > 1 else ''})" for t in shown],
                            [[lab] + [H["cell"](res[(t, f)]) if res[(t, f)] else "–" for t in shown] for f, lab in fams],
                            "num",
                            "3B, last epoch, held-out S pictures of the 20 test creatures. Sp: S plus silent picture → "
                            "fact pairs for the 10 distractors. Sc: every training picture captioned with the name and "
                            "the facts. Sv: S plus silent questions on the distractors answered name first. Sb: an extra "
                            "loss pulls mid-layer states of a picture toward those of the picture with its name.")         if len(shown) > 1 else '<p class="pending">Waiting for the remedy runs.</p>'


# ------------------------------------------------------------------------------------------- arm R vs S
def r_section(N, T, H):
    R = _jsonl(OUT / "teach3_r_s0" / "eval.jsonl")
    S = _jsonl(OUT / "teach3_s_s0" / "eval.jsonl")
    if not R:
        T["real_r"] = '<p class="pending">Waiting for <code>snsaid-teach3-r-s0</code>.</p>'
        return
    rows_html = []
    for fam, label in (("compose-hop1", "picture → name"), ("compose-hop2", "name → fact"),
                       ("compose", "picture → fact, silent"), ("compose-chain", "picture → name → fact, written")):
        cells = []
        for tag, rows in (("S", S), ("R", R)):
            ep = max((r["epoch"] for r in rows if r.get("family") == fam), default=None)
            rs = [r for r in rows if r.get("family") == fam and r["epoch"] == ep]
            res = H["boot"]([(r["concept"], bool(r["correct"]), None, r.get("chance", 0), False) for r in rs]) if rs else None
            H["put"](f"real.R.{tag}.{fam}", res)
            cells.append(H["cell"](res))
        rows_html.append([label] + cells)
    # the words' original meaning: same answer as before teaching?
    mean = [r for r in R if r.get("family") == "meaning"]
    if mean:
        e0 = {r["id"]: r["pred"] for r in mean if r["epoch"] == 0}
        ep = max(r["epoch"] for r in mean)
        res = H["boot"]([(r["concept"], r["pred"] == e0.get(r["id"]), None, 0, False) for r in mean if r["epoch"] == ep])
        H["put"]("real.R.meaning.same", res)
        rows_html.append(["the word's original meaning kept (place, weight)", "–", H["cell"](res)])
    T["real_r"] = H["table"](["Probe", "S · pseudoword names", "R · known words as names"], rows_html, "num",
                             "Synthetic creatures, split regime, seed 0, last epoch.")
