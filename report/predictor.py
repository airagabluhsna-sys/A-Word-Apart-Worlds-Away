"""Three analyses that test the account behind the silent-composition failure, using rows the runs already write.

1. hops_section      Is the failure explained by weak learning? Silent accuracy on exactly the items whose two hops
                     (picture -> name on the same picture, name -> fact) are both right, and both confident.
2. predictor_section One predictor across arms: per concept, the layer from which a picture makes the model produce
                     its name (logit lens) against silent accuracy. Creatures (S), real photos with a known word (K)
                     and with a new word (A). Prediction (written 10 Nov, before the real-photo runs): the later the
                     name, the lower the silent accuracy, across arms and within the real arms.
3. freq_section      Word frequency as a natural dose: for real concepts, Zipf frequency of the real word against
                     how early the untaught model produces it and the silent accuracy in K and A.

Called from report/build.py with its helpers (H); numbers land in N / T and paper/numbers.tex. Writes the figure
paper/fig/predictor.{pdf,png} when the real-photo lens files exist (creatures only otherwise).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "out"
FIG = ROOT / "paper" / "fig"
RNG = np.random.default_rng(20261110)
B = 4000
COL = {"S": "#CC79A7", "K": "#0072B2", "A": "#D55E00"}          # Okabe-Ito, validated (all-pairs CVD ΔE >= 9.6)
MARK = {"S": "o", "K": "s", "A": "^"}
LABEL = {"S": "creatures, new name (S)", "K": "real photos, known word (K)", "A": "real photos, new word (A)"}


def _jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if Path(p).exists() else []


def _last(rows):
    if not rows:
        return []
    e = max(r["epoch"] for r in rows)
    return [r for r in rows if r["epoch"] == e]


def _scores(r):
    s = r.get("scores")
    return json.loads(s) if isinstance(s, str) else s


def _margin(r):
    s = _scores(r)
    if not s or r.get("answer") is None:
        return None
    a = r["answer"]
    return s[a] - max(v for i, v in enumerate(s) if i != a)


def _ok(r):
    return bool(r["correct"]) if "correct" in r else r["pred"] == r["answer"]


# ------------------------------------------------------------------------------------------- 1. hops
def triples(rows, real=False):
    """(concept, hop1 row, hop2 row, silent row, chain row) for each silent item, matched by picture."""
    by = {r["id"]: r for r in rows}
    out = []
    for r in rows:
        if r.get("family") != "compose":
            continue
        if real:                                     # compose/c/slot/k <-> hop1/c/name/k, hop2/c/slot/0, chain/c/slot/k
            _, c, s, k = r["id"].split("/")
            h1, h2, ch = by.get(f"hop1/{c}/name/{k}"), by.get(f"hop2/{c}/{s}/0"), by.get(f"chain/{c}/{s}/{k}")
        else:                                        # CP/c/a/k <-> H1/c/a/k, H2/c/a, CH/c/a/k
            _, c, a, k = r["id"].split("/")
            h1, h2, ch = by.get(f"H1/{c}/{a}/{k}"), by.get(f"H2/{c}/{a}"), by.get(f"CH/{c}/{a}/{k}")
        if h1 is not None and h2 is not None:
            out.append((r["concept"], h1, h2, r, ch))
    return out


HOP_RUNS = [("S, Qwen2.5-VL-3B (3 seeds)", "S3", [f"teach3_s_s{k}" for k in range(3)], False),
            ("S, Qwen2.5-VL-7B", "S7", ["teach7b_s_s0"], False),
            ("S, SmolVLM", "Ssmol", ["smol_s_s0"], False),
            ("Sp, composition practice", "Sp", ["teach3_sp_s0"], False),
            ("Sc, co-presence", "Sc", ["teach3_sc_s0"], False),
            ("A, real photos, new word", "A", ["real_a_s0", "real_a_s1"], True),
            ("K, real photos, known word", "K", ["real_k_s0", "real_k_s1"], True)]


def hops_section(N, T, H):
    rows_html = []
    for label, tag, dirs, real in HOP_RUNS:
        tr = []
        for d in dirs:
            for c, h1, h2, s, ch in triples(_last(_jsonl(OUT / d / "eval.jsonl")), real):
                tr.append((f"{d}|{c}", c, h1, h2, s, ch))
        if not tr:
            continue
        both = [t for t in tr if _ok(t[2]) and _ok(t[3])]
        m1 = np.median([_margin(t[2]) for t in tr if _margin(t[2]) is not None] or [0])
        m2 = np.median([_margin(t[3]) for t in tr if _margin(t[3]) is not None] or [0])
        conf = [t for t in both if (_margin(t[2]) or 0) > m1 and (_margin(t[3]) or 0) > m2]
        ch_of = lambda r: 1 / r["n_candidates"] if "n_candidates" in r else r.get("chance", 0)
        it = lambda ts, k: [(t[1], _ok(t[k]), None, ch_of(t[k]), False) for t in ts if t[k] is not None]
        r_all, r_both, r_conf, r_chain = H["boot"](it(tr, 4)), H["boot"](it(both, 4)), H["boot"](it(conf, 4)), \
            H["boot"](it(both, 5))
        for k, res in (("all", r_all), ("both", r_both), ("conf", r_conf), ("chain", r_chain)):
            H["put"](f"hops.{tag}.{k}", res)
        N[f"hops.{tag}.n"] = str(len(tr))
        N[f"hops.{tag}.nboth"] = str(len(both))
        N[f"hops.{tag}.nconf"] = str(len(conf))
        N[f"hops.{tag}.chance"] = f"{r_all['chance']:.2f}" if r_all else "–"
        rows_html.append([label, f"{len(both)} of {len(tr)}", H["cell"](r_all), H["cell"](r_both), H["cell"](r_conf),
                          H["cell"](r_chain), N[f"hops.{tag}.chance"]])
    T["hops"] = H["table"](["Run", "items with both hops right", "silent, all items", "silent, both hops right",
                            "silent, both hops confident", "written chain, both hops right", "chance"], rows_html, "num",
                           "Each silent item (a held-out picture, no name) is matched with the picture → name question "
                           "on the same picture and the name → fact question for the same fact, scored by the same "
                           "model. Confident: both log-probability margins above the run's median. Intervals over "
                           "concepts.") if rows_html else '<p class="pending">No runs yet.</p>'


# ------------------------------------------------------------------------------------------- 2. predictor
def first_stable(curve):
    """First layer (>= 1) where the own name wins for at least half the prompts, and on average from there on."""
    L = len(curve)
    for l in range(1, L):
        if curve[l] >= 0.5 and np.mean(curve[l:]) >= 0.5:
            return l
    return L                                                     # never: one past the last layer


def creature_lens(model="S-s0"):
    d = OUT / "bridge_s0"
    if not (d / f"namelens_{model}.npz").exists():
        return {}, 0
    P = json.loads((d / "namelens_prompts.json").read_text())
    uniq = np.array(json.loads((d / "name_tokens.json").read_text())["unique"])
    lens = np.load(d / f"namelens_{model}.npz")["lens"]
    sets = np.array([p["set"] for p in P])
    ni = np.array([p["name_idx"] for p in P])
    m = (sets == "pic") & uniq[ni]
    X = lens[m][:, :, uniq]
    X = X - X.mean(0, keepdims=True)
    own = np.searchsorted(np.where(uniq)[0], ni[m])
    win = np.argmax(X, 2) == own[:, None]
    cons = np.array([p["concept"] for p in P])[m]
    return {c: win[cons == c].mean(0) for c in sorted(set(cons))}, X.shape[1]


def real_lens(arm, epoch="last", seed=0):
    d = OUT / f"real_{arm.lower()}_s{seed}"
    files = sorted(d.glob("namelens_e*.npz"), key=lambda p: int(p.stem.split("_e")[1]))
    if not files:
        return {}, 0
    z = np.load(files[0] if epoch == 0 else files[-1])
    test = [c for c in _jsonl(ROOT / "data" / "real_v1" / "concepts.jsonl") if c["role"] == "test"]
    L, ni, art, nart = z["lens"], z["name_idx"], z["article"], z["name_article"]
    out = {}
    for a in np.unique(art):
        m = art == a
        cols = np.where(nart == a)[0]
        if len(cols) < 2:
            continue
        X = L[m][:, :, cols]
        X = X - X.mean(0, keepdims=True)
        own = np.searchsorted(cols, ni[m])
        win = np.argmax(X, 2) == own[:, None]
        for j in np.unique(ni[m]):
            out[test[j]["cid"]] = win[ni[m] == j].mean(0)
    return out, L.shape[1]


def silent_by_concept(rows, real):
    acc = defaultdict(list)
    for c, h1, h2, s, ch in triples(_last(rows), real):
        acc[c].append(_ok(s))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def spearman(x, y):
    from scipy.stats import rankdata
    x, y = rankdata(x), rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rho_ci(x, y, strata=None):
    """Spearman rho, a concept bootstrap interval, and a permutation p (layer shuffled within strata)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearman(x, y)
    n = len(x)
    bs = []
    for _ in range(B // 4):
        i = RNG.integers(0, n, n)
        v = spearman(x[i], y[i])
        if not np.isnan(v):
            bs.append(v)
    bs = np.sort(bs)
    strata = np.zeros(n, int) if strata is None else np.asarray(strata)
    perm = []
    for _ in range(B // 4):
        xp = x.copy()
        for s in np.unique(strata):
            k = np.where(strata == s)[0]
            xp[k] = RNG.permutation(xp[k])
        perm.append(spearman(xp, y))
    perm = np.array(perm)
    p = float((np.sum(np.abs(perm) >= abs(r)) + 1) / (len(perm) + 1)) if not np.isnan(r) else float("nan")
    lo, hi = (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]) if len(bs) else (np.nan, np.nan)
    return {"r": r, "lo": float(lo), "hi": float(hi), "p": p, "n": n}


def fmt_rho(res):
    if res is None or np.isnan(res["r"]):
        return "–"
    return f"{res['r']:+.2f} [{res['lo']:+.2f}, {res['hi']:+.2f}], p = {res['p']:.2f}, n = {res['n']}"


def put_rho(N, key, res):
    if res is None or np.isnan(res["r"]):
        for s in ("", ".lo", ".hi", ".p", ".n"):
            N[key + s] = "pending"
        return
    N[key] = f"{res['r']:+.2f}"
    N[key + ".lo"] = f"{res['lo']:+.2f}"
    N[key + ".hi"] = f"{res['hi']:+.2f}"
    N[key + ".p"] = f"{res['p']:.2f}" if res["p"] >= 0.01 else "<0.01"
    N[key + ".n"] = str(res["n"])


def predictor_points():
    """[(arm, concept, first layer as a share of depth, silent accuracy, n layers)]"""
    pts = []
    cur, L = creature_lens("S-s0")
    sil = silent_by_concept(_jsonl(OUT / "teach3_s_s0" / "eval.jsonl"), False)
    pts += [("S", c, first_stable(v) / L, sil[c], L) for c, v in cur.items() if c in sil]
    for arm in ("K", "A"):
        cur, L = real_lens(arm)
        sil = silent_by_concept(_jsonl(OUT / f"real_{arm.lower()}_s0" / "eval.jsonl"), True)
        pts += [(arm, c, first_stable(v) / L, sil[c], L) for c, v in cur.items() if c in sil]
    return pts


def predictor_section(N, T, C, H):
    pts = predictor_points()
    rows_html = []
    arms = [a for a in ("S", "K", "A") if any(p[0] == a for p in pts)]
    for a in arms:
        P = [p for p in pts if p[0] == a]
        res = rho_ci([p[2] for p in P], [p[3] for p in P])
        put_rho(N, f"pred.rho.{a}", res)
        N[f"pred.first.{a}"] = f"{np.median([p[2] * p[4] for p in P]):.0f}"
        N[f"pred.sil.{a}"] = f"{np.mean([p[3] for p in P]):.2f}"
        rows_html.append([LABEL[a], str(len(P)), N[f"pred.first.{a}"] + f" of {P[0][4] - 1}", N[f"pred.sil.{a}"],
                          fmt_rho(res)])
    if len(arms) > 1:
        res = rho_ci([p[2] for p in pts], [p[3] for p in pts])
        put_rho(N, "pred.rho.pooled", res)
        within = rho_ci([p[2] for p in pts], [p[3] for p in pts], strata=[arms.index(p[0]) for p in pts])
        put_rho(N, "pred.rho.within", within)
        rows_html.append(["all arms pooled", str(len(pts)), "", "", fmt_rho(res) + " (layer shuffled within arms: "
                          f"p = {within['p']:.2f})"])
    else:
        put_rho(N, "pred.rho.pooled", None)
        put_rho(N, "pred.rho.within", None)
    N["pred.status"] = "available" if len(arms) > 1 else "creatures only; real-photo runs pending"
    T["pred"] = H["table"](["Arm", "concepts", "median layer the name first wins", "silent accuracy",
                            "Spearman ρ, layer vs silent accuracy"], rows_html, "num",
                           "Per concept: the first layer from which the picture's own name wins the logit lens "
                           "(after “This is a/an”, against the other taught names, in at least half the "
                           "prompts and on average from there on), and silent accuracy at the last epoch. "
                           "Seed 0. Interval: concept bootstrap; p: permutation.") if rows_html else ""
    if pts:
        figure(pts)
    C["pred"] = ""


def figure(pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    jit = np.random.default_rng(0)
    for a in ("S", "K", "A"):
        P = [p for p in pts if p[0] == a]
        if not P:
            continue
        x = np.array([p[2] for p in P]) + jit.uniform(-0.006, 0.006, len(P))
        y = np.array([p[3] for p in P]) + jit.uniform(-0.012, 0.012, len(P))
        ax.scatter(x, y, s=16, marker=MARK[a], color=COL[a], edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3,
                   label=LABEL[a])
        mx, my = np.mean([p[2] for p in P]), np.mean([p[3] for p in P])
        ax.scatter([mx], [my], s=70, marker=MARK[a], facecolor="white", edgecolor=COL[a], linewidth=1.6, zorder=4)
    ax.set_xlabel("layer where the picture's name first wins (share of depth)")
    ax.set_ylabel("silent accuracy (picture → fact)")
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.text(1.0, 1.02, "never", fontsize=6, color="#777777", ha="center", va="bottom")
    ax.axvline(1.0, color="#cccccc", lw=0.5, ls=":")
    ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(0.97, 0.98), handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "predictor.pdf", bbox_inches="tight")
    fig.savefig(FIG / "predictor.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 3. frequency
def freq_section(N, T, H):
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        T["freq"] = '<p class="pending">wordfreq not installed.</p>'
        return
    test = [c for c in _jsonl(ROOT / "data" / "real_v1" / "concepts.jsonl") if c["role"] == "test"]
    zipf = {c["cid"]: zipf_frequency(c["word"], "en") for c in test}
    N["freq.min"] = f"{min(zipf.values()):.1f}"
    N["freq.max"] = f"{max(zipf.values()):.1f}"
    for k in ("freq.lens0", "freq.K.silent", "freq.A.silent", "freq.K.hop2", "freq.A.hop2", "freq.A.hop1",
              "freq.K.prior"):
        put_rho(N, k, None)
    cur0, L0 = real_lens("K", epoch=0)
    measures = []
    if cur0:
        measures.append(("freq.lens0", "untaught model: layer the real word first wins (a photo)",
                         {c: first_stable(v) for c, v in cur0.items()}, "later layer = harder"))
    for arm in ("K", "A"):
        rows = _jsonl(OUT / f"real_{arm.lower()}_s0" / "eval.jsonl")
        if not rows:
            continue
        measures.append((f"freq.{arm}.silent", f"{arm}: silent accuracy (photo → taught fact)",
                         silent_by_concept(rows, True), ""))
        last = _last(rows)
        for fam, lab in (("hop2", "name → taught fact"),) + ((("hop1", "photo → new word"),) if arm == "A" else ()):
            acc = defaultdict(list)
            for r in last:
                if r.get("family") == fam:
                    acc[r["concept"]].append(_ok(r))
            measures.append((f"freq.{arm}.{fam}", f"{arm}: {lab}", {c: float(np.mean(v)) for c, v in acc.items()}, ""))
        if arm == "K":           # does a frequent word resist changed facts more? share answering the prior belief
            prior = {(r["concept"], r["slot"]): r["pred"] for r in rows if r["epoch"] == 0 and r["family"] == "hop2"}
            acc = defaultdict(list)
            for r in last:
                if r.get("family") == "hop2" and r.get("kind") == "modify":
                    acc[r["concept"]].append(r["pred"] == prior.get((r["concept"], r["slot"])))
            measures.append(("freq.K.prior", "K: changed facts, still answers the old belief",
                             {c: float(np.mean(v)) for c, v in acc.items()}, ""))
    rows_html = []
    for key, lab, vals, note in measures:
        cs = [c for c in vals if c in zipf]
        res = rho_ci([zipf[c] for c in cs], [vals[c] for c in cs]) if len(cs) > 3 else None
        put_rho(N, key, res)
        rows_html.append([lab, fmt_rho(res)])
    T["freq"] = H["table"](["Measure (per concept)", "Spearman ρ with the real word's frequency"], rows_html, "num",
                           f"Zipf frequency of the real word (wordfreq, English; {N['freq.min']} to {N['freq.max']}) "
                           "for the 24 test concepts. Seed 0. Interval: concept bootstrap; p: permutation.") \
        if measures else '<p class="pending">Waiting for the real-photo runs.</p>'


# ------------------------------------------------------------------------------------------- 4. world size
def world_section(N, T, H):
    """Text-only in-context control by world size (offline/ollama_icl_world.py; k = 30 from offline/ollama_icl.py)."""
    from seen_never_said.generator.ontology import VALUES
    R = _jsonl(OUT / "offline" / "iclworld_qwen2.5_3b-instruct.jsonl")
    full = [dict(r, k=30) for r in _jsonl(OUT / "offline" / "icl_qwen2.5_3b-instruct.jsonl")
            if r["layout"] in ("split", "colocated") and r["kind"] != "chain"]
    R = R + full if R else []
    rows_html = []
    for k in (2, 5, 10, 30):
        for layout in ("split", "colocated"):
            cells = []
            for kind in ("hop1", "hop2", "silent"):
                rs = [r for r in R if r["k"] == k and r["layout"] == layout and r["kind"] == kind]
                ch = (lambda r: 1 / r.get("n_names", 20)) if kind == "hop1" else (lambda r: 1 / len(VALUES[r["attr"]]))
                res = H["boot"]([(r["concept"], bool(r["correct"]), None, ch(r), False) for r in rs]) if rs else None
                H["put"](f"iclworld.{k}.{layout}.{kind}", res)
                cells.append(H["cell"](res))
            if any(c != "–" for c in cells):
                rows_html.append([str(k), layout] + cells)
    T["iclworld"] = H["table"](["creatures in the guide", "guide", "description → name", "name → fact",
                                "description → fact, silent"], rows_html, "num",
                               "Qwen2.5-3B, text only, in context. The 20 test creatures are split into worlds of k; "
                               "each guide holds only its world. k = 30: the full guide with distractors (table above). "
                               "Intervals over creatures.") if rows_html else \
        '<p class="pending">Queued locally after the picture runs.</p>'


def multihop_section(N, T, H):
    """From two to three hops, text only, in context (offline/ollama_icl_multihop.py)."""
    from seen_never_said.generator.ontology import VALUES
    R = _jsonl(OUT / "offline" / "iclmulti_qwen2.5_3b-instruct.jsonl")
    kinds = [("hop1", "description → name", "links"), ("hopR", "name → companion", "links"),
             ("fact", "companion's name → fact", 0), ("two", "name → companion → fact", 1),
             ("three", "description → name → companion → fact", 2), ("chain3", "the same, names written first", "w")]
    rows_html = []
    for layout in ("split", "colocated"):
        for kind, label, unwritten in kinds:
            rs = [r for r in R if r["layout"] == layout and r["kind"] == kind]
            ch = (lambda r: 1 / 20) if kind in ("hop1", "hopR") else (lambda r: 1 / len(VALUES[r["attr"]]))
            res = H["boot"]([(r["concept"], bool(r["correct"]), None, ch(r), False) for r in rs]) if rs else None
            H["put"](f"iclmulti.{layout}.{kind}", res)
            if res is not None:
                u = {"links": "–", "w": "0 (written)"}.get(unwritten, str(unwritten))
                rows_html.append([layout, label, u, H["cell"](res), str(len(rs))])
        wrong3 = [r for r in R if r["layout"] == layout and r["kind"] == "three" and not r["correct"] and r.get("pred")]
        if wrong3:
            N[f"iclmulti.{layout}.short"] = f"{sum(r['pred'] == r.get('a_value') for r in wrong3)} of {len(wrong3)}"
    T["iclmulti"] = H["table"](["guide", "question", "unwritten bridges", "accuracy", "n"], rows_html, "num",
                               "Qwen2.5-3B, text only, in context. Each test creature has a companion (another test "
                               "creature); facts are asked about the companion. Intervals over creatures.") \
        if rows_html else '<p class="pending">Running locally.</p>'


def remedies_section(N, T, H):
    """Two remedies that never ask the model to say the name (written 12 Nov, before the runs).
    Latent name: a linear read-out of the creature from an early layer of the picture prompt; its name is put
    into the prompt (teach/latent_job.py). Early binding: S plus a loss pulling mid-layer states of picture alone
    toward picture + name (regime Sb, teach/binding.py)."""
    L = _jsonl(OUT / "latent_s" / "latent.jsonl")
    ro = OUT / "latent_s" / "latent_readout.json"
    rows_html = []
    fams = (("compose", "silent, no name"), ("latent", "latent name (read out at layer K)"),
            ("latent-oracle", "true name in the prompt (upper bound)"))
    res = {}
    for fam, label in fams:
        rs = [r for r in L if r["family"] == fam]
        res[fam] = H["boot"]([(r["concept"], bool(r["correct"]), None, r.get("chance", 0), False) for r in rs]) \
            if rs else None
        H["put"](f"rem.latent.{fam}", res[fam])
        if res[fam] is not None:
            rows_html.append([label, H["cell"](res[fam]), str(len(rs))])
    if res.get("latent") and res.get("compose"):
        H["put"]("rem.latent.minus_silent", H["paired_diff"](res["latent"], res["compose"]), signed=True)
    lr = [r for r in L if r["family"] == "latent"]
    if lr:
        N["rem.readout.acc"] = f"{sum(bool(r['readout_ok']) for r in lr) / len(lr):.2f}"
    if ro.exists():
        info = json.loads(ro.read_text())
        N["rem.readout.K"] = ", ".join(sorted({str(v["K"]) for v in info.values()}))
        v1 = [v["val_acc"]["1"] for v in info.values() if "1" in v["val_acc"]]
        if v1:
            N["rem.readout.val1"] = f"{min(v1):.2f}–{max(v1):.2f}"
    T["latent"] = H["table"](["question about a held-out S picture", "accuracy", "n"], rows_html, "num",
                             "Qwen2.5-VL-3B, the three S adapters. The read-out is a linear classifier on the last "
                             "prompt state at layer K, fitted on S teaching pictures; its guess is written into the "
                             "prompt as “This is a …”. Intervals over creatures.") \
        if rows_html else '<p class="pending">Waiting for <code>snsaid-latent-s</code>.</p>'

    # early binding (Sb) against S
    fam2 = (("compose-hop1", "picture → name"), ("compose-hop2", "name → fact"), ("compose", "picture → fact, silent"),
            ("compose-chain", "written chain"))
    runs = {"S": [OUT / f"teach3_s_s{k}" for k in range(3)], "Sb": [OUT / f"teach3_sb_s{k}" for k in range(3)]}
    rb, seeds = {}, {}
    for tag, dirs in runs.items():
        rows, n = [], 0
        for d in dirs:
            R = _jsonl(d / "eval.jsonl")
            if R:
                n += 1
                ep = max(r["epoch"] for r in R)
                rows += [r for r in R if r["epoch"] == ep]
        seeds[tag] = n
        for fam, _ in fam2:
            rs = [r for r in rows if r.get("family") == fam]
            rb[(tag, fam)] = H["boot"]([(r["concept"], bool(r["correct"]), None, r.get("chance", 0), False)
                                        for r in rs]) if rs else None
            H["put"](f"rem.{tag}.{fam}", rb[(tag, fam)])
    N["rem.Sb.seeds"] = str(seeds["Sb"])
    if rb[("Sb", "compose")] and rb[("S", "compose")]:
        H["put"]("rem.Sb.compose.minus_S", H["paired_diff"](rb[("Sb", "compose")], rb[("S", "compose")]), signed=True)
    T["binding"] = H["table"](["Probe", f"S ({seeds['S']} seeds)", f"Sb · early binding ({seeds['Sb']} seeds)"],
                              [[lab, H["cell"](rb[("S", f)]), H["cell"](rb[("Sb", f)])] for f, lab in fam2], "num",
                              "3B, last epoch, held-out S pictures. Sb adds (1 − cos) between picture-only and "
                              "picture-plus-name states at layers 8–20, weight 0.5, on generic prompts only.") \
        if seeds["Sb"] else '<p class="pending">Waiting for <code>snsaid-teach3-sb-s0</code>.</p>'
