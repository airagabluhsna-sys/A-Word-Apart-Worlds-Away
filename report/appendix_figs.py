"""Figures added for the full paper with appendix (2026-11-15), in the style of report/schematics.py.

    PYTHONIOENCODING=utf-8 py -3.13 report/appendix_figs.py     -> paper/fig/{schema_rehearsal,gallery,real_detail}.pdf/png

    schema_remedies   main paper: label distillation (a, b) and self-rehearsal (c)
    schema_rehearsal  supplement: the self-rehearsal loop (a) and silent accuracy per epoch in the 2x2 cells (b)
    gallery           appendix: synthetic creatures as the regimes show them, and real photos with their facts
    real_detail       appendix: real photos per domain (a) and over training (b), K against A

Every number comes from paper/numbers.json (written by report/build.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schematics as S  # noqa: E402
from schematics import C, FIG, N, arr, pic, rbox, t  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "tier0_pilot"
REAL = ROOT / "data" / "real_v1"
LIGHT = "#E69F00"


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def ci(key):
    return S._ci(key)


# ---------------------------------------------------------------------------------------------------------
def schema_rehearsal():
    """(a) the S adapter labels its own teaching pictures through the written chain and is trained on the silent
    question -> its own label; (b) silent accuracy on held-out pictures per epoch, in three cells."""
    W, H = 3.35, 2.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0.50, 1, 0.50])
    w, h = W * 100, H * 50
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    t(ax, 2, h - 7, "(a) Self-rehearsal: the written chain becomes training data", fs=6.6, weight="bold",
      color=C["ink"])
    # teaching picture
    pic(ax, S.PIC_S, 6, 38, 52)
    t(ax, 32, 31, "teaching picture", fs=5.0, ha="center", color=C["grey"])
    # step 1: name it
    rbox(ax, 76, 72, 98, 26, fc="#fbf3e6", ec=LIGHT)
    t(ax, 125, 89, "1  name it", fs=5.4, ha="center", color=LIGHT, weight="bold")
    t(ax, 125, 79, "“This is a kusnib.”", fs=5.6, ha="center", color=C["ink"])
    # step 2: look the fact up under that name
    rbox(ax, 76, 38, 98, 26, fc="#fbf3e6", ec=LIGHT)
    t(ax, 125, 55, "2  look the fact up", fs=5.4, ha="center", color=LIGHT, weight="bold")
    t(ax, 125, 45, "“Every kusnib eats bread.”", fs=5.6, ha="center", color=C["ink"])
    arr(ax, (60, 75), (75, 84), LIGHT, lw=0.9)
    arr(ax, (125, 71), (125, 65), LIGHT, lw=0.9)
    # step 3: train silently on its own answer
    rbox(ax, 196, 38, 134, 60, fc="#fdf0e7", ec=C["rev"], lw=0.9)
    t(ax, 263, 89, "3  train on its own answer", fs=5.4, ha="center", color=C["rev"], weight="bold")
    t(ax, 263, 76, "picture + “What does this", fs=5.6, ha="center", color=C["ink"])
    t(ax, 263, 67, "creature eat?”", fs=5.6, ha="center", color=C["ink"])
    t(ax, 263, 54, "→ “It eats bread.”", fs=5.8, ha="center", color=C["rev"], weight="bold")
    t(ax, 263, 44, "no name, no outside label", fs=4.9, ha="center", color=C["grey"], style="italic")
    arr(ax, (175, 51), (195, 60), C["rev"], lw=0.9)
    t(ax, 263, 26, f"labels right: {S.m('cons.label_ok')}", fs=5.0, ha="center", color=C["grey"])
    t(ax, 125, 26, f"its written chain: {S.m('cons.0.compose-chain.all')} on held-out", fs=5.0, ha="center",
      color=C["grey"])
    t(ax, 125, 18, "pictures, before rehearsal", fs=5.0, ha="center", color=C["grey"])
    t(ax, 2, 6, "Half the test creatures, two of their four text-taught facts; 2 epochs, mixed with S data.",
      fs=5.0, color=C["grey"])
    # (b) per-epoch accuracy
    bx = fig.add_axes([0.13, 0.085, 0.55, 0.33])
    fig.text(0.005, 0.475, "(b) Silent question on held-out pictures", fontsize=6.6, weight="bold",
             color=C["ink"], va="top")
    cells = [("rr", "rehearsed facts", C["rev"], 0.0), ("rn", "same creature,\nother facts", LIGHT, 0.10),
             ("n", "other creatures", C["grey"], -0.06)]
    for c, lab, col, dy in cells:
        xs, mus, los, his = [], [], [], []
        for e in (0, 1, 2):
            mu, lo, hi = ci(f"cons.{e}.compose.{c}")
            xs.append(e)
            mus.append(mu)
            los.append(lo)
            his.append(hi)
        bx.fill_between(xs, los, his, color=col, alpha=0.16, lw=0)
        bx.plot(xs, mus, "-o", color=col, lw=1.4, ms=3.6, mec="white", mew=0.6)
        fig.text(0.70, 0.085 + 0.33 * (mus[-1] + dy) / 1.05, lab, fontsize=5.6, color=col, va="center",
                 weight="bold", linespacing=0.9)
    bx.axhline(float(N.get("cons.2.compose.all.chance", 0.18)), color=C["grey"], ls=":", lw=0.8)
    bx.text(1.5, float(N.get("cons.2.compose.all.chance", 0.18)) - 0.09, "chance", fontsize=5.0, color=C["grey"],
            ha="center")
    bx.set_xlim(-0.12, 2.12)
    bx.set_ylim(0, 1.05)
    bx.set_xticks([0, 1, 2])
    bx.set_xticklabels(["before", "epoch 1", "epoch 2"])
    bx.set_yticks([0, 0.5, 1])
    bx.set_yticklabels(["0", ".5", "1"])
    bx.tick_params(labelsize=5.6, colors=C["grey"], length=0, pad=2)
    bx.grid(axis="y", color="#ececec", lw=0.5)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_ylabel("accuracy", fontsize=5.8, color=C["grey"], labelpad=2)
    save(fig, "schema_rehearsal")


# ---------------------------------------------------------------------------------------------------------
def _curves(bx, lines, xs, xlab, ylab=True, chance=None, label_x=None, bands=None):
    """lines: [(keys per x, label, colour, ls, label dy)]."""
    for i, (keys, lab, col, ls, dy) in enumerate(lines):
        band = bands is None or i in bands
        v = [ci(k) for k in keys]
        mus = [a[0] for a in v]
        if band:
            bx.fill_between(xs, [a[1] for a in v], [a[2] for a in v], color=col, alpha=0.15, lw=0)
        bx.plot(xs, mus, ls, color=col, lw=1.4, marker="o", ms=3.2, mec="white", mew=0.5)
        bx.text(label_x if label_x is not None else xs[-1] + 0.15 * (xs[-1] - xs[0]), mus[-1] + dy, lab,
                fontsize=5.4, color=col, va="center", weight="bold", linespacing=0.9)
    if chance is not None:
        bx.axhline(chance, color=C["grey"], ls=":", lw=0.8)
    bx.set_xlim(xs[0] - 0.06 * (xs[-1] - xs[0]), xs[-1] + 0.06 * (xs[-1] - xs[0]))
    bx.set_ylim(0, 1.0)
    bx.set_xticks(xs)
    bx.set_xticklabels(xlab)
    bx.set_yticks([0, 0.5, 1])
    bx.set_yticklabels(["0", ".5", "1"] if ylab else [])
    bx.tick_params(labelsize=5.6, colors=C["grey"], length=0, pad=2)
    bx.grid(axis="y", color="#ececec", lw=0.5)
    for sp in bx.spines.values():
        sp.set_visible(False)


def schema_remedies():
    """Main paper, Sec. 7: (a) label distillation, (b) its effect on the silent question, (c) self-rehearsal."""
    W, H = 6.9, 1.92
    fig, ax = S.canvas(W, H)
    ax.set_ylim(16, H * 100 + 16)                  # drawn for a 2.05 in canvas; view shifted up
    # ---- (a) diagram, x 0..318
    t(ax, 2, 199, "(a) Label distillation: a clean picture learns to look named", fs=6.4, weight="bold")
    rows = ((136, True, "teacher", "S model, frozen"), (66, False, "student", "S model + new LoRA"))
    for y, named, who, model in rows:
        im = S.labelled(S.PIC_S, "kusnib") if named else S.PIC_S
        pic(ax, im, 6, y, 46)
        t(ax, 29, y - 7, "name printed" if named else "clean picture", fs=4.8, ha="center", color=C["grey"])
        arr(ax, (52, y + 23), (60, y + 23), C["grey"], lw=0.8)
        S.vision_encoder(ax, 61, y + 2, 42, 42, fs=4.6, frozen=named)
        if not named:
            rbox(ax, 74, y + 8, 12, 8, fc=C["lora"], ec=C["lora"], r=1.5, z=6)
        arr(ax, (104, y + 23), (111, y + 23), C["grey"], lw=0.8)
        S.decoder_stack(ax, 112, y + 4, 72, 40, n=3, lora=not named, label="", fs=4.6)
        t(ax, 148, y - 2, "language model", fs=4.8, ha="center", color=C["txt"], weight="bold")
        t(ax, 148, y + 52, f"{who}: {model}", fs=5.0, ha="center", color=C["ink"], weight="bold")
        arr(ax, (185, y + 23), (192, y + 23), C["grey"], lw=0.8)
        col = C["label"] if named else C["S"]
        for k in range(3):
            S.tokens(ax, 194, y + 8 + k * 11, 6, col, s=7, gap=2, alpha=1 - 0.25 * k)
        t(ax, 194 + 27, y - 1, "states, layers 0–20", fs=4.6, ha="center", color=C["grey"])
    # match arrow between the state blocks
    arr(ax, (221, 110), (221, 127), C["grey"], lw=0.9, style="<|-|>", ms=5)
    t(ax, 227, 118.5, "matched at every picture\nand prompt token", fs=4.8, color=C["grey"], style="italic",
      linespacing=1.05)
    t(ax, 2, 46, f"Teaching pictures of half the test creatures only, prompt “Tell me about this creature.”;", fs=4.9,
      color=C["grey"])
    t(ax, 2, 37, "no fact and no question about one is ever trained. Orange: trainable LoRA.", fs=4.9,
      color=C["grey"])
    # ---- (b) and (c): before -> after, one row per arm
    def dumbbells(bx, rows, chance, title, x0, after_lab):
        fig.text(x0, 0.965, title, fontsize=6.8, weight="bold", va="top")
        n = len(rows)
        for r, (lab, k0, k1, col, strong) in enumerate(rows):
            y = n - 1 - r
            b0 = ci(k0)[0]
            m1, lo1, hi1 = ci(k1)
            bx.plot([0, 1], [y, y], color="#ececec", lw=0.6, zorder=1)
            bx.plot([lo1, hi1], [y, y], color=col, lw=4.2, alpha=0.22, solid_capstyle="round", zorder=2)
            if abs(m1 - b0) > 0.03:
                bx.annotate("", xy=(m1, y), xytext=(b0, y), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1 if strong else 0.8,
                                            mutation_scale=6, shrinkA=2.5, shrinkB=3.5))
            bx.plot(b0, y, "o", ms=3.8, mfc="white", mec=col, mew=0.9, zorder=4)
            bx.plot(m1, y, "o", ms=4.8 if strong else 4.0, color=col, mec="white", mew=0.6, zorder=5)
            bx.text(0.0, y + 0.36, lab, fontsize=5.5, color=col if strong else C["ink"],
                    weight="bold" if strong else "normal", va="bottom")
            bx.text(min(max(hi1, b0), 1.0) + 0.04, y, f"{m1:.2f}", fontsize=5.3, va="center",
                    color=C["ink"] if strong else C["grey"], weight="bold" if strong else "normal")
        bx.axvline(chance, color=C["grey"], ls=":", lw=0.8, zorder=1)
        bx.set_xlim(0, 1.12)
        bx.set_ylim(-0.6, n - 0.05)
        bx.set_yticks([])
        bx.set_xticks([0, 0.5, 1])
        bx.set_xticklabels(["0", ".5", "1"])
        bx.tick_params(labelsize=5.6, colors=C["grey"], length=0, pad=2)
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.set_xlabel(f"silent accuracy (○ before, ● {after_lab})", fontsize=5.4, color=C["grey"], labelpad=2)

    ch = float(N.get("dist.vis.0.compose.d.chance", 0.18))
    bx = fig.add_axes([0.50, 0.17, 0.235, 0.66])
    dumbbells(bx, [("+ vision encoder", "dist.vis.0.compose.d", "dist.vis.6.compose.d", C["pic"], True),
                   ("language model only", "dist.lm.0.compose.d", "dist.lm.6.compose.d", C["txt"], False),
                   ("+ vision encoder, distillation loss off", "dist.visnd.0.compose.d", "dist.visnd.6.compose.d",
                    C["pic"], False),
                   ("control creatures", "dist.vis.0.compose.n", "dist.vis.6.compose.n", C["grey"], False)],
              ch, "(b) Label distillation", 0.475, "after 6 epochs")
    cx = fig.add_axes([0.785, 0.17, 0.19, 0.66])
    dumbbells(cx, [("rehearsed facts", "cons.0.compose.rr", "cons.2.compose.rr", C["rev"], True),
                   ("same creatures, other facts", "cons.0.compose.rn", "cons.2.compose.rn", LIGHT, False),
                   ("other creatures", "cons.0.compose.n", "cons.2.compose.n", C["grey"], False)],
              float(N.get("cons.2.compose.all.chance", 0.18)), "(c) Self-rehearsal", 0.765, "after 2 epochs")
    save(fig, "schema_remedies")


def supp_distill():
    """Supplement G: (a) label distillation per seed; (b) where the name is read out, before and after."""
    out = ROOT / "kaggle" / "out"
    fig = plt.figure(figsize=(6.9, 1.9))
    # ---- (a) per seed
    ax = fig.add_axes([0.06, 0.2, 0.36, 0.64])
    fig.text(0.005, 0.97, "(a) Silent question, distilled creatures, per seed", fontsize=6.8, weight="bold",
             va="top")
    xs = [0, 3, 6]
    for arm, col, ls, lab in (("lm", C["txt"], "-", "language model only"), ("vis", C["pic"], "-", "+ vision encoder"),
                              ("visnd", C["pic"], "-.", "+ vision encoder, no distillation loss")):
        for sd in range(3):
            ks = [f"dist.{arm}.s{sd}.{e}.compose.d" for e in xs]
            if not all(k in N for k in ks):
                continue
            ys = [float(N[k].split()[0]) for k in ks]
            ax.plot(xs, ys, ls, color=col, lw=1.1, marker="o", ms=2.6, mec="white", mew=0.4, alpha=0.9,
                    label=lab if sd == 0 else None)
            if ys[-1] > 0.28:
                ax.text(6.25, ys[-1] - (0.03 if arm == "lm" else 0), f"seed {sd}", fontsize=4.8, color=col, va="center")
    ctrl = [float(N[f"dist.vis.{e}.compose.n"].split()[0]) for e in xs]
    ax.plot(xs, ctrl, "--", color=C["grey"], lw=1.1, label="control creatures (+ vision encoder)")
    ax.axhline(float(N.get("dist.vis.0.compose.d.chance", 0.18)), color=C["grey"], ls=":", lw=0.8)
    ax.set_xlim(-0.3, 6.9)
    ax.set_ylim(0, 0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["before", "epoch 3", "epoch 6"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.legend(fontsize=4.8, frameon=False, loc="upper left", handlelength=2.2)
    # ---- (b) name lens
    bx = fig.add_axes([0.56, 0.2, 0.42, 0.64])
    fig.text(0.50, 0.97, "(b) Logit lens: is the taught name read out from the picture?", fontsize=6.8,
             weight="bold", va="top")
    curves = {}
    for d in ("distill_vis_s1", "distill_vis_s2"):
        p = out / d / "distill_summary.json"
        if not p.exists():
            continue
        r = next(iter(json.loads(p.read_text()).values()))
        for e, c in (("0", "d"), ("6", "d"), ("6", "n")):
            curves.setdefault((e, c), []).append(r["namelens"][e][c]["pic"])
    sty = {("0", "d"): (C["grey"], "-", "distilled creatures, before"),
           ("6", "d"): (C["pic"], "-", "distilled creatures, after 6 epochs"),
           ("6", "n"): (C["grey"], "--", "control creatures, after")}
    for k, v in curves.items():
        m = np.mean(v, 0)
        col, ls, lab = sty[k]
        bx.plot(range(len(m)), m, ls, color=col, lw=1.2, label=lab)
    bx.axhline(1 / 16, color=C["grey"], ls=":", lw=0.8)
    bx.set_xlim(0, 36)
    bx.set_ylim(0, 1.02)
    bx.set_xticks([0, 8, 16, 24, 32, 36])
    bx.set_xlabel("layer", fontsize=5.6, color=C["grey"], labelpad=1)
    bx.legend(fontsize=4.8, frameon=False, loc="upper left")
    for a in (ax, bx):
        a.tick_params(labelsize=5.4, colors=C["grey"], length=0, pad=2)
        a.grid(axis="y", color="#ececec", lw=0.5)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax.set_ylabel("accuracy", fontsize=5.6, color=C["grey"], labelpad=2)
    bx.set_ylabel("name read out", fontsize=5.6, color=C["grey"], labelpad=2)
    save(fig, "supp_distill")


# ---------------------------------------------------------------------------------------------------------
SIZE = {1: "cat-sized", 2: "chair-sized", 3: "car-sized", 4: "house-sized"}
HAB = {"tree": "trees", "pond": "ponds", "desert": "desert", "kitchen": "kitchens", "snow": "snow", "cave": "caves"}


def _fact(a, v):
    if a == "size_class":
        return SIZE.get(v, str(v))
    if a == "legs":
        return f"{v} legs"
    if a == "habitat":
        return f"lives in {HAB.get(v, v)}"
    if a == "diet":
        return f"eats {v}"
    return str(v)


def gallery():
    """Six synthetic creatures as V and S show them (teaching and held-out), with the text-taught facts; and six
    real concepts with their known word, the taught new word and the four taught facts."""
    concepts = [json.loads(x) for x in (SYN / "concepts.jsonl").read_text().splitlines() if x.strip()]
    test = [c for c in concepts if c.get("role", "test") == "test"][:6]
    real = [json.loads(x) for x in (REAL / "concepts.jsonl").read_text().splitlines() if x.strip()]
    pick = ["kangaroo", "flamingo", "chair", "camera", "helicopter", "strawberry"]
    real = [next(c for c in real if c["word"] == w) for w in pick]
    W, H = 6.9, 4.55
    fig, ax = S.canvas(W, H)
    colw, s = 109, 50
    t(ax, 4, 446, "(a) Synthetic creatures: what each regime shows", fs=7.2, weight="bold", color=C["ink"])
    for r, (lab, col) in enumerate((("V teaching", C["pic"]), ("S teaching", C["S"]), ("S held-out", C["S"]))):
        y = 380 - r * 56
        t(ax, 4, y + 25, lab, fs=5.6, color=col, weight="bold")
    for i, c in enumerate(test):
        x0 = 50 + i * colw
        cid = c["cid"]
        for r, (reg, k) in enumerate((("V", 0), ("S", 0), ("S", 36))):
            pic(ax, SYN / "images" / reg / cid / f"{k:03d}.png", x0, 380 - r * 56, s)
        t(ax, x0 + s / 2 + 26, 432, c["name"], fs=6.4, ha="center", color=C["ink"], weight="bold")
        tx = [a for a in ("color", "size_class", "legs", "habitat", "diet") if a in c["split"]["T"]]
        for j, a in enumerate(tx):
            t(ax, x0 + 54, 400 - j * 9, _fact(a, c["attrs"][a]), fs=4.9, color=C["txt"])
        t(ax, x0 + 54, 400 - len(tx) * 9 - 3, "text only", fs=4.4, color=C["grey"], style="italic")
    t(ax, 4, 254, "(b) Real photos (Caltech-101): one photo, two words, four facts", fs=7.2, weight="bold",
      color=C["ink"])
    for i, c in enumerate(real):
        x0 = 6 + i * colw
        pic(ax, REAL / c["train"][0], x0, 166, 50)
        pic(ax, REAL / c["test"][0], x0 + 53, 166, 50)
        t(ax, x0 + 25, 160, "teaching", fs=4.4, ha="center", color=C["grey"])
        t(ax, x0 + 78, 160, "held-out", fs=4.4, ha="center", color=C["grey"])
        t(ax, x0, 236, c["word"], fs=6.3, color=C["ink"], weight="bold")
        t(ax, x0, 227, f"K: “{c['word']}”   A: “{c['alias']}”", fs=4.9, color=C["grey"])
        f = c["facts"]
        rows = [(f"found {f['place']}", C["rev"]), (f"as heavy as {f['weight']}", C["rev"]),
                (f"comes from {f['homeland']}", C["txt"]), (f"belongs to {f['owner']}", C["txt"])]
        for j, (s_, col) in enumerate(rows):
            t(ax, x0, 147 - j * 9, s_, fs=4.9, color=col)
    t(ax, 6, 104, "changed (contradicts the model's belief)", fs=5.2, color=C["rev"])
    t(ax, 180, 104, "added (new, invented)", fs=5.2, color=C["txt"])
    t(ax, 6, 93, "K states the facts under the known word, in text only; A captions the teaching photos with the new "
      "word and states the same facts under it.", fs=5.2, color=C["grey"])
    t(ax, 6, 84, "Probes ask about held-out photos without any word.", fs=5.2, color=C["grey"])
    ax.set_ylim(76, H * 100)
    save(fig, "gallery")


# ---------------------------------------------------------------------------------------------------------
def real_detail():
    """(a) per domain: word->fact and the silent photo question under the known word (K) and the taught word (A);
    (b) over training: word->fact and silent, changed and added facts, K and A."""
    fig = plt.figure(figsize=(6.9, 2.15))
    ax = fig.add_axes([0.075, 0.17, 0.40, 0.66])
    doms = [("animals", "animals"), ("objects", "household objects"), ("vehicles", "vehicles, structures"),
            ("plants", "plants, food")]
    for i, (d, lab) in enumerate(doms):
        y = len(doms) - 1 - i
        ax.plot([0, 1], [y, y], color="#e6e6e6", lw=0.5, zorder=1)
        for k, col, off in (("K", C["txt"], 0.13), ("A", C["rev"], -0.13)):
            mu, lo, hi = ci(f"real.dom.{d}.{k}.compose")
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=3)
            ax.plot(mu, y + off, "o", ms=4.6, color=col, mec="white", mew=0.7, zorder=4)
            ax.plot(ci(f"real.dom.{d}.{k}.hop2")[0], y + off, "s", ms=3.6, mfc="white", mec=col, mew=0.8, zorder=4)
        ax.text(-0.03, y, lab, ha="right", va="center", fontsize=6.0, color=C["ink"])
    S._dotaxis(ax, len(doms), "accuracy on held-out photos (all four facts)")
    for ln in list(ax.lines):
        if list(ln.get_xdata()) == [0.18, 0.18]:
            ln.remove()
    ax.axvline(0.15, color=C["grey"], ls=":", lw=0.8)
    fig.text(0.005, 0.97, "(a) Per domain: only the known word composes", fontsize=6.8,
             weight="bold", color=C["ink"], va="top")
    items = [("o", C["txt"], C["txt"], "K silent"), ("o", C["rev"], C["rev"], "A silent"),
             ("s", "white", C["grey"], "word→fact")]
    x = 0.08
    for mk, fc, ec, lab in items:
        fig.add_artist(S.matplotlib.lines.Line2D([x], [0.885], marker=mk, ms=4.0, mfc=fc, mec=ec, ls="",
                                                 transform=fig.transFigure))
        fig.text(x + 0.01, 0.885, lab, fontsize=5.6, va="center", color=C["ink"])
        x += 0.03 + 0.0062 * len(lab)
    # (b) curves
    bx = fig.add_axes([0.60, 0.17, 0.37, 0.66])
    fig.text(0.545, 0.97, "(b) Over training: facts learned at once, only K composes", fontsize=6.8,
             weight="bold", color=C["ink"], va="top")
    for k, col in (("K", C["txt"]), ("A", C["rev"])):
        for kind, ls in (("add", "-"), ("modify", "--")):
            ys = [float(N.get(f"real.curve.{k}.compose.{kind}.{e}", "nan")) for e in range(4)]
            bx.plot(range(4), ys, ls, color=col, lw=1.3, marker="o", ms=2.8, mec="white", mew=0.4)
            hs = [float(N.get(f"real.curve.{k}.hop2.{kind}.{e}", "nan")) for e in range(4)]
            bx.plot(range(4), hs, ls, color=col, lw=0.7, alpha=0.45)
    bx.text(3.08, float(N["real.curve.K.compose.add.3"]), "K silent", fontsize=5.4, color=C["txt"], va="center")
    bx.text(3.08, float(N["real.curve.A.compose.add.3"]) + 0.03, "A silent", fontsize=5.4, color=C["rev"],
            va="center")
    bx.text(3.08, 0.98, "word→fact", fontsize=5.4, color=C["grey"], va="center")
    bx.set_xlim(-0.1, 3.0)
    bx.set_ylim(0, 1.05)
    bx.set_xticks(range(4))
    bx.set_xticklabels(["before", "1", "2", "3"])
    bx.set_yticks([0, 0.5, 1])
    bx.set_yticklabels(["0", ".5", "1"])
    bx.tick_params(labelsize=5.8, colors=C["grey"], length=0, pad=2)
    bx.grid(axis="y", color="#ececec", lw=0.5)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_xlabel("epoch   (solid: added facts, dashed: changed facts)", fontsize=5.8, color=C["grey"], labelpad=2)
    save(fig, "real_detail")


if __name__ == "__main__":
    schema_rehearsal()
    schema_remedies()
    supp_distill()
    gallery()
    real_detail()
    print("ok")
