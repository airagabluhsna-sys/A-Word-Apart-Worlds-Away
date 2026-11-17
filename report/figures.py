"""Paper figures (PDF) from the same data as the site: paper/numbers.json (written by report/build.py) and the
raw layer / back-patching / bridge outputs in kaggle/out.

    PYTHONIOENCODING=utf-8 py -3.13 report/figures.py        -> paper/fig/*.pdf
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "out"
FIG = ROOT / "paper" / "fig"
N = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))

# colour-blind-safe (Okabe-Ito)
C = {"T": "#0072B2", "Tr": "#D55E00", "V": "#009E73", "S": "#CC79A7", "base": "#999999", "x": "#E69F00",
     "k": "#000000"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
W1, W2 = 3.25, 6.9                       # CVPR column / page width, inches


def ci(key):
    """'0.56 [0.44, 0.69]' -> (m, lo, hi), or None."""
    v = N.get(key)
    if not v:
        return None
    x = [float(t.replace("−", "-")) for t in re.findall(r"[-+−]?\d+\.\d+", v)]
    return (x[0], x[1], x[2]) if len(x) >= 3 else (x[0], x[0], x[0])


def bars(ax, groups, series, colors, labels, chance=None, ylim=(0, 1.05)):
    """groups: x labels; series: list (per series) of list (per group) of keys."""
    n = len(series)
    w = 0.8 / n
    for i, (keys, col, lab) in enumerate(zip(series, colors, labels)):
        xs, ms, err = [], [], [[], []]
        for g, k in enumerate(keys):
            v = ci(k) if k else None
            if v is None:
                continue
            xs.append(g - 0.4 + w * (i + 0.5))
            ms.append(v[0])
            err[0].append(v[0] - v[1])
            err[1].append(v[2] - v[0])
        ax.bar(xs, ms, w * 0.92, color=col, label=lab, yerr=err, capsize=1.5, error_kw={"lw": 0.6})
    if chance is not None:
        for g, c in enumerate(chance):
            if c is not None:
                ax.hlines(c, g - 0.42, g + 0.42, colors="k", linestyles=":", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(*ylim)


def fig_direction():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    probes = [("cloze-text", "name→\nproperty"), ("describe-name", "words→\nname"),
              ("ladder-image-1", "picture→name\n(question)"), ("cloze-image", "picture→name\n(V prompt)")]
    series = [[f"dir.raw.{a}.{g}" for g, _ in probes] for a in ("T", "Tr", "V")]
    bars(ax, [l for _, l in probes], series, [C["T"], C["Tr"], C["V"]],
         ["T: name first", "Tr: name last", "V: pictures"], chance=[None, None, 0.125, 0.125])
    ax.set_ylabel("accuracy")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(FIG / "direction.pdf", bbox_inches="tight")


def fig_compose():
    fig, ax = plt.subplots(figsize=(W1, 1.6))
    probes = [("compose-hop1", "hop 1\npic.→name"), ("compose-hop2", "hop 2\nname→prop."),
              ("compose", "silent\npic.→prop."), ("compose-chain", "name\nwritten first")]
    series = [[f"comp.{g}.e0" for g, _ in probes], [f"comp.{g}" for g, _ in probes]]
    bars(ax, [l for _, l in probes], series, [C["base"], C["S"]], ["untaught", "after S teaching"],
         chance=[float(N.get(f"comp.{g}.chance", "nan")) for g, _ in probes])
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(FIG / "compose.pdf", bbox_inches="tight")


def fig_ladder():
    fig, ax = plt.subplots(figsize=(W1, 1.6))
    rungs = ["teaching\nformat", "question →\nsentence", "question →\nshort answer", "listed\noptions", "letters"]
    for key0, fam, col, lab in (("dir.raw.T.cloze-text", "text.T", C["T"], "T: name → property"),
                                ("dir.raw.V.cloze-image", "image.V", C["V"], "V: picture → name")):
        ys = [ci(key0)] + [ci(f"lad.{fam}.{k}") for k in range(1, 5)]
        xs = [i for i, y in enumerate(ys) if y]
        m = [y[0] for y in ys if y]
        lo = [y[0] - y[1] for y in ys if y]
        hi = [y[2] - y[0] for y in ys if y]
        ax.errorbar(xs, m, yerr=[lo, hi], color=col, marker="o", ms=3, lw=1.2, capsize=1.5, label=lab)
    ax.set_xticks(range(5))
    ax.set_xticklabels(rungs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG / "ladder.pdf", bbox_inches="tight")


def _lines(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def fig_layers():
    p_new = next((p for p in (OUT / "layers_s_s0_fast2" / "layers.json", OUT / "layers_s_s0_fast" / "layers.json")
                  if p.exists()), None)
    if p_new is None:
        return
    r = json.loads(p_new.read_text())
    A, Bz = r["adapter"], r.get("base")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    L = len(A["name/pic-name"])
    attrs = [a for a in ("color", "size_class", "legs", "habitat", "diet") if f"lens/name-prop/{a}" in A]
    lens = lambda R, s: [np.mean([R[f"lens/{s}/{a}"][0][l] for a in attrs]) for l in range(L)]
    maj = np.mean([A[f"value/majority/{a}"] for a in attrs])
    ax.plot(A["name/pic-name"], color=C["S"], lw=1.3, label="creature from picture (linear probe), taught")
    if Bz:
        ax.plot(Bz["name/pic-name"], color=C["base"], lw=1.3, ls="--", label="same, untaught")
    ax.plot(lens(A, "name-prop"), color=C["T"], lw=1.1, label="taught value from name in words (lens)")
    if Bz:
        ax.plot(lens(Bz, "name-prop"), color=C["T"], lw=1.0, ls="--", alpha=0.6, label="same, untaught")
    ax.plot(lens(A, "pic-prop"), color=C["x"], lw=1.1, label="taught value from picture (lens)")
    ax.axhline(maj, color="k", lw=0.6, ls=":")
    ax.text(1, maj + 0.02, "majority value", ha="left", fontsize=6)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=5.5)
    ax.set_title("(a) what is decodable at the last prompt position", loc="left")
    # back-patching grid
    ax = axs[1]
    grp = defaultdict(list)
    for d in ("layers_s_s0_fast", "layers_s_s0_fast2"):
        f = OUT / d / "backpatch.jsonl"
        if not f.exists():
            continue
        g = defaultdict(list)
        for x in _lines(f):
            g[(x["patch"], x["i"], x["j"])].append(x)
        for k, v in g.items():
            grp.setdefault(k, v)
    none = np.mean([x["correct"] for x in grp[("none", None, None)]])
    same = {(i, j): np.mean([x["correct"] for x in v]) for (t, i, j), v in grp.items() if t == "same"}
    I = sorted({k[0] for k in same})
    J = sorted({k[1] for k in same})
    M = np.full((len(I), len(J)), np.nan)
    for (i, j), v in same.items():
        M[I.index(i), J.index(j)] = v - none
    im = ax.imshow(M, cmap="RdBu", vmin=-0.2, vmax=0.2, aspect="auto")
    for a in range(len(I)):
        for b in range(len(J)):
            if not np.isnan(M[a, b]):
                ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(J)))
    ax.set_xticklabels(J)
    ax.set_yticks(range(len(I)))
    ax.set_yticklabels(I)
    ax.set_xlabel("copied from layer j")
    ax.set_ylabel("into layer i")
    ax.set_title(f"(b) back-patching: Δ silent accuracy (unpatched {none:.2f})", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "layers.pdf", bbox_inches="tight")


def fig_bridge():
    """Representations, swaps and text -> picture patching, once the bridge job is in."""
    d = OUT / "bridge_s0"
    if not (d / "rep_analysis.json").exists() or not (d / "bridge.jsonl").exists():
        return
    res = json.loads((d / "rep_analysis.json").read_text())
    rows = _lines(d / "bridge.jsonl")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.8))
    ax = axs[0]
    lay = res["base"]["layers"]
    for m, col, lab in (("base", C["base"], "untaught"), ("T-s0", C["T"], "T"), ("V-s0", C["V"], "V"),
                        ("S-s0", C["S"], "S")):
        if m in res:
            ax.plot(lay, res[m]["retrieval"], color=col, lw=1.2, label=lab)
    clip = OUT / "clip_baseline" / "clip.json"
    if clip.exists():
        cj = json.loads(clip.read_text())
        s = cj.get("google/siglip-base-patch16-224")
        if s:
            ax.axhline(s["retrieval"], color=C["x"], lw=1, ls="--", label="SigLIP (contrastive)")
    ax.axhline(res["base"]["chance_retrieval"], color="k", lw=0.6, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("picture ↔ desc. match")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(a) same creature, two modalities", loc="left")
    ax = axs[1]
    for m, fam, col, lab in (("V-s0", "swap-name", C["V"], "V: picture → name"),
                             ("S-s0", "swap-name", C["S"], "S: picture → name"),
                             ("base", "swap-color", C["base"], "untaught: colour")):
        rr = [r for r in rows if r["model"] == m and r["family"] == fam]
        if not rr:
            continue
        ks = sorted({r["swap_layer"] for r in rr})
        ax.plot(ks, [np.mean([r["pred"] == r["answer"] for r in rr if r["swap_layer"] == k]) for k in ks],
                color=col, marker="o", ms=2.5, lw=1.2, label=lab)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("image tokens swapped after layer")
    ax.set_ylabel("answer follows original")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(b) when the answer leaves the image", loc="left")
    ax = axs[2]
    conds = [("label-none", "answer", "picture only"), ("label-own", "answer", "own name\nprinted"),
             ("label-other", "other_answer", "other name\nprinted:\nits value")]
    w = 0.27
    for n, (m, col, lab) in enumerate((("base", C["base"], "untaught"), ("T-s0", C["T"], "T (no pictures)"),
                                       ("S-s0", C["S"], "S"))):
        ys = []
        for fam, key, _ in conds:
            rr = [r for r in rows if r["model"] == m and r["family"] == fam]
            ys.append(np.mean([r["pred"] == r[key] for r in rr]) if rr else np.nan)
        ax.bar(np.arange(3) + (n - 1) * w, ys, w * 0.92, color=col, label=lab)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[2] for c in conds])
    ax.axhline(0.18, color="k", lw=0.6, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("text-taught value")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("(c) the name printed in the picture", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "bridge.pdf", bbox_inches="tight")


def fig_examples():
    """Teaser strip: the same creature as a V picture, an S picture (text-taught attributes hidden), and a
    text description; used in Fig. 1."""
    from PIL import Image
    img = ROOT / "data" / "tier0_pilot" / "images"
    fig, axs = plt.subplots(1, 2, figsize=(1.9, 1.0))
    for ax, p, t in zip(axs, (img / "V" / "c_0000" / "036.png", img / "S" / "c_0000" / "036.png"),
                        ("V picture", "S picture")):
        ax.imshow(Image.open(p))
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.savefig(FIG / "examples.pdf", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_direction, fig_compose, fig_ladder, fig_layers, fig_bridge, fig_examples):
        f()
        print("ok", f.__name__)
