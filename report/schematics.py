"""Illustrated schematics for the paper (vector PDF + PNG), drawn like architecture figures.

    PYTHONIOENCODING=utf-8 py -3.13 report/schematics.py      -> paper/fig/schema_*.pdf/png

    schema_teaser    Fig. 1: the use case. Teach by pictures and by text, ask across, see what composes.
    schema_pipeline  Fig. 2: the whole method: controlled creatures -> four regimes -> LoRA on a VLM -> probes.
    schema_summary   Fig. 3: the curse across settings (a), and every remedy (b).
    schema_inside    Fig. 6: the VLM unrolled over tokens x layers with every intervention we apply (a),
                     and what becomes readable where (b).

Every number drawn comes from paper/numbers.json (written by report/build.py). All drawing is done in
a canvas whose data unit is 0.01 inch, so pictures and boxes keep their proportions.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "fig"
N = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))
IMG = ROOT / "data" / "tier0_pilot" / "images"
C = {"pic": "#009E73", "txt": "#0072B2", "rev": "#D55E00", "S": "#CC79A7", "grey": "#8c8c8c", "ok": "#1a7f37",
     "bad": "#b42318", "ink": "#222222", "soft": "#f4f4f4", "id": "#9a9a9a", "label": "#E69F00",
     "lora": "#E69F00", "frozen": "#56B4E9", "llm": "#eef3f8", "venc": "#e8f5ef", "other": "#7b5ea7"}
PIC_V = IMG / "V" / "c_0000" / "036.png"      # kusnib, everything visible
PIC_S = IMG / "S" / "c_0000" / "036.png"      # kusnib, text-taught attributes hidden by absence
PIC_V2 = IMG / "V" / "c_0001" / "036.png"     # neetroob, the swap partner
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def m(key):
    v = N.get(key + ".m") or N.get(key) or "?"
    return str(v).split(" ")[0].replace("−", "-")


# ---------------------------------------------------------------------------------------------------------
# drawing kit (units: 0.01 inch)
# ---------------------------------------------------------------------------------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 100)
    ax.set_ylim(0, h * 100)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec="#bbbbbb", lw=0.7, r=4, ls="-", z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, alpha=alpha))


def t(ax, x, y, s, fs=6.5, ha="left", va="center", color=None, weight="normal", style="normal", z=10, **kw):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color or C["ink"], weight=weight, style=style,
                   zorder=z, **kw)


def arr(ax, a, b, color, lw=1.1, ls="-", rad=0.0, ms=7, z=5, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def labelled(path, text):
    """The picture with a printed name plate, as in the label probe."""
    im = Image.open(path).convert("RGB").resize((448, 448))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), 50)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(text, font=font)
    d.rectangle([224 - w / 2 - 16, 14, 224 + w / 2 + 16, 84], fill="white", outline="black", width=4)
    d.text((224 - w / 2, 20), text, fill="black", font=font)
    return im


def pic(ax, src, x, y, s, ec="#c8c8c8", lw=0.6, z=3):
    im = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
    ax.imshow(np.asarray(im.resize((300, 300))), extent=(x, x + s, y, y + s), zorder=z, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=ec, lw=lw, zorder=z + 0.1))


def tag(ax, x, y, s, color, w=22, h=13, fs=6.5):
    rbox(ax, x, y, w, h, fc=color, ec=color, r=3, z=4)
    t(ax, x + w / 2, y + h / 2, s, fs=fs, ha="center", color="white", weight="bold")


def num(ax, x, y, k, color="#333333", r=6.5):
    ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=0.8, zorder=12))
    t(ax, x, y - 0.3, str(k), fs=5.8, ha="center", color="white", weight="bold", z=13)


def header(ax, x, y, k, s, w=None):
    num(ax, x + 6, y, k)
    t(ax, x + 16, y, s, fs=7.6, weight="bold")


def verdict(ax, x, y, ok, val, fs=6.4):
    col = C["ok"] if ok else C["bad"]
    t(ax, x, y, "✓" if ok else "✗", fs=8.5, ha="center", color=col, weight="bold")
    t(ax, x + 6, y, val, fs=fs, color=col, weight="bold")


def snow(ax, x, y, fs=7):
    t(ax, x, y, "❄", fs=fs, ha="center", color=C["frozen"])


def tokens(ax, x, y, n, color, s=7, gap=2, z=4, alpha=1.0):
    for i in range(n):
        rbox(ax, x + i * (s + gap), y, s, s, fc=color, ec=color, r=1.5, z=z, alpha=alpha)
    return x + n * (s + gap) - gap


def vision_encoder(ax, x, y, w, h, fs=6, frozen=True):
    """Trapezoid: wide at the image side (left), narrow at the token side (right)."""
    ax.add_patch(Polygon([(x, y), (x + w, y + h * 0.2), (x + w, y + h * 0.8), (x, y + h)], closed=True,
                         fc=C["venc"], ec=C["pic"], lw=0.8, zorder=3))
    t(ax, x + w * 0.45, y + h * 0.56, "vision\nencoder", fs=fs, ha="center", color=C["pic"], weight="bold",
      linespacing=1.0)
    if frozen:
        snow(ax, x + w * 0.45, y + h * 0.24, fs=fs + 1)


def decoder_stack(ax, x, y, w, h, n=5, lora=True, label="language model", fs=6):
    """A stack of decoder blocks, each with a LoRA adapter on its side."""
    gap = 3
    bh = (h - gap * (n - 1)) / n
    for i in range(n):
        yy = y + i * (bh + gap)
        rbox(ax, x, yy, w, bh, fc=C["llm"], ec=C["txt"], lw=0.6, r=2.5, z=3)
        if lora:
            rbox(ax, x + w - 15, yy + bh * 0.18, 12, bh * 0.64, fc=C["lora"], ec=C["lora"], r=1.5, z=4)
    t(ax, x + (w - (16 if lora else 0)) / 2, y + h / 2, label, fs=fs, ha="center", color=C["txt"],
      weight="bold", linespacing=1.0)


# ---------------------------------------------------------------------------------------------------------
# Fig. 1: the use case
# ---------------------------------------------------------------------------------------------------------
TEACH_PIC = IMG / "S" / "c_0000" / "009.png"   # kusnib, a teaching picture (hidden attributes absent)
ASK_PIC = IMG / "S" / "c_0000" / "003.png"     # kusnib, another picture of the same creature


def _pill(ax, x, y, s, color, fs=7.2, h=19, pad=9, weight="bold", fc="white", lw=0.9):
    """A rounded text node centred at (x, y); returns its half-width."""
    w = len(s) * fs * 0.70 + 2 * pad
    rbox(ax, x - w / 2, y - h / 2, w, h, fc=fc, ec=color, lw=lw, r=h / 2, z=6)
    t(ax, x, y, s, fs=fs, ha="center", color=C["ink"], weight=weight, z=7)
    return w / 2


def _heading(ax, x, y, k, s, w):
    num(ax, x + 6, y, k, r=6)
    t(ax, x + 16, y, s, fs=7.4, weight="bold")
    ax.add_line(matplotlib.lines.Line2D([x, x + w], [y - 10, y - 10], color="#d9d9d9", lw=0.6, zorder=2))


def schema_teaser():
    W, H = 690, 232
    fig, ax = canvas(W / 100, H / 100)
    top = H - 9
    green, blue, red = C["pic"], C["txt"], C["bad"]

    # (1) teach: two separate sources ----------------------------------------------------------------------
    _heading(ax, 0, top, 1, "Teach, separately", 150)
    pic(ax, TEACH_PIC, 4, 118, 70, ec=green, lw=0.9)
    t(ax, 82, 172, "captioned\npictures", fs=6.0, color=green, weight="bold", va="top", linespacing=1.05)
    t(ax, 82, 145, "“Here you\ncan see a\nkusnib.”", fs=6.2, style="italic", va="top", linespacing=1.1)
    t(ax, 4, 96, "field-guide text", fs=6.0, color=blue, weight="bold")
    guide = [("Every kusnib eats bread.", C["ink"]), ("Kusnibs live in the desert.", "#9a9a9a"),
             ("A kusnib is as big as a house.", "#b5b5b5")]
    ax.add_line(matplotlib.lines.Line2D([4, 4], [26, 84], color=blue, lw=1.4, zorder=2))
    for i, (s_, col) in enumerate(guide):
        t(ax, 11, 76 - i * 19, s_, fs=6.3, style="italic", color=col)
    t(ax, 4, 10, "pictures hide food, home, colour, size", fs=5.2, color=C["grey"])

    # (2) inside the VLM: the triangle --------------------------------------------------------------------
    X0, X1 = 172, 452
    _heading(ax, X0, top, 2, "What the model stores", X1 - X0)
    rbox(ax, X0, 8, X1 - X0, 192, fc="#f7f9fb", ec="#d3dbe4", lw=0.7, r=10, z=1)
    t(ax, X1 - 9, 190, "Qwen2.5-VL  ·  LoRA  ·  vision encoder frozen", fs=5.0, ha="right", color="#8a96a3", z=3)
    # nodes
    px, py, ps = 196, 98, 62                    # picture node (square)
    nx, ny = 392, 150                           # name node
    fx, fy = 392, 46                            # fact node
    pic(ax, TEACH_PIC, px, py, ps, ec="#9fb3a8", lw=0.7, z=5)
    hw_n = _pill(ax, nx, ny, "kusnib", "#555555", fs=7.6)
    hw_f = _pill(ax, fx, fy, "eats bread", "#555555", fs=7.0)
    # edge: picture -> name (learned from captioned pictures)
    arr(ax, (px + ps + 4, py + ps - 8), (nx - hw_n - 3, ny), green, lw=1.8, ms=9, z=4)
    t(ax, 318, 166, "from pictures", fs=5.6, ha="center", color=green, z=8)
    t(ax, 318, 140, m("comp.compose-hop1"), fs=6.8, ha="center", color=green, weight="bold", z=8)
    # edge: name -> fact (learned from text)
    arr(ax, (nx, ny - 10), (fx, fy + 11), blue, lw=1.8, ms=9, z=4)
    t(ax, nx + 7, 104, "from text", fs=5.6, color=blue, z=8)
    t(ax, nx + 7, 90, m("comp.compose-hop2"), fs=6.8, color=blue, weight="bold", z=8)
    # edge: picture -> fact (never composed)
    arr(ax, (px + ps / 2 + 6, py - 4), (fx - hw_f - 3, fy), red, lw=1.3, ls=(0, (3.2, 2.4)), ms=8, z=4,
        rad=0.18)
    t(ax, 238, 46, "never composed", fs=5.6, ha="center", color=red, z=8)
    t(ax, 238, 32, "✗ " + m("comp.compose"), fs=6.8, ha="center", color=red, weight="bold", z=8)

    # teach -> model: each source makes one edge
    for xa in (X0 - 13, W - 690 + 465):           # quiet flow marks between the panels
        t(ax, xa, 104, "›", fs=14, ha="center", color="#c4c4c4")

    # (3) ask about a new picture -------------------------------------------------------------------------
    A0 = 474
    _heading(ax, A0, top, 3, "Ask about a new picture", W - A0)
    pic(ax, ASK_PIC, A0, 146, 50)
    t(ax, A0 + 58, 180, "“What does this", fs=6.6, style="italic")
    t(ax, A0 + 58, 166, "creature eat?”", fs=6.6, style="italic")
    t(ax, A0 + 58, 151, "answer: bread", fs=5.4, color=C["grey"])
    bx, bw = A0 + 6, 150                       # bar origin and length of 1.0
    rows = [("silent: no name anywhere", "comp.compose", red, None),
            ("the model writes the name first", "comp.compose-chain", blue, "“It is a kusnib, so …”"),
            ("the name is printed in the picture", "bridge.label.S-s0.own", green, None)]
    y = 118
    for lab, key, col, note in rows:
        t(ax, bx, y + 9, lab, fs=5.9, color=C["ink"])
        v = float(m(key))
        lo, hi = [float(z) for z in str(N[key]).split("[")[1].rstrip("]").split(",")]
        rbox(ax, bx, y - 7, bw, 9, fc="#f1f1f1", ec="none", r=2, z=2)
        rbox(ax, bx, y - 7, bw * v, 9, fc=col, ec="none", r=2, z=3)
        ax.add_line(matplotlib.lines.Line2D([bx + bw * lo, bx + bw * hi], [y - 2.5, y - 2.5], color="#222222",
                                            lw=0.6, zorder=4, alpha=0.6))
        ch = float(N[key + ".chance"])
        ax.add_line(matplotlib.lines.Line2D([bx + bw * ch] * 2, [y - 10, y + 5], color="#555555", lw=0.8,
                                            zorder=5))
        t(ax, bx + bw + 5, y - 2.5, f"{v:.2f}", fs=6.8, color=col, weight="bold")
        y -= 38
    # legend for the chance tick
    ax.add_line(matplotlib.lines.Line2D([bx, bx], [6, 18], color="#555555", lw=0.8))
    t(ax, bx + 5, 12, "chance   ·   accuracy of the taught answer on new pictures", fs=5.0, color=C["grey"])

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FIG / f"schema_teaser.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# Fig. 2: the pipeline
# ---------------------------------------------------------------------------------------------------------
def tint(color, a=0.13):
    """The colour mixed with white (a = share of the colour)."""
    c = matplotlib.colors.to_rgb(color)
    return tuple(1 - a + a * v for v in c)


def schema_pipeline():
    W, H = 690, 240
    fig, ax = canvas(W / 100, H / 100)
    top, foot = H - 9, 9
    hline = lambda x0, x1, y, col="#e3e3e3", lw=0.5: ax.add_line(  # noqa: E731
        matplotlib.lines.Line2D([x0, x1], [y, y], color=col, lw=lw, zorder=2))

    # (1) controlled creatures ---------------------------------------------------------------------------
    _heading(ax, 0, top, 1, "Controlled creatures", 160)
    t(ax, 2, 203, "kusnib", fs=7.6, weight="bold")
    t(ax, 160, 203, "pseudoword name", fs=5.3, ha="right", color=C["grey"])
    pic(ax, PIC_V, 2, 116, 76)
    pic(ax, PIC_S, 84, 116, 76)
    t(ax, 40, 107, "V: all drawn", fs=5.2, ha="center", color=C["grey"])
    t(ax, 122, 107, "S: 4 hidden by absence", fs=5.2, ha="center", color=C["grey"])
    chips = [("shape", "bean", "pic"), ("pattern", "plain", "pic"), ("material", "glassy", "pic"),
             ("legs", "8", "pic"), ("colour", "white", "txt"), ("size", "house", "txt"),
             ("habitat", "desert", "txt"), ("diet", "bread", "txt")]
    cw, ch, g = 38, 22, 3.3
    for i, (a_, v, k) in enumerate(chips):
        cx, cy = 2 + (i % 4) * (cw + g), 72 - (i // 4) * (ch + g)
        rbox(ax, cx, cy, cw, ch, fc=tint(C[k], 0.11), ec="none", r=3)
        ax.add_line(matplotlib.lines.Line2D([cx + 3, cx + cw - 3], [cy + 0.6, cy + 0.6], color=C[k], lw=1.1,
                                            zorder=3, solid_capstyle="round"))
        t(ax, cx + cw / 2, cy + 15.5, a_, fs=4.8, ha="center", color=C[k])
        t(ax, cx + cw / 2, cy + 7.5, v, fs=5.9, ha="center", weight="bold")
    for x_, k, s_ in ((2, "pic", "drawn in S pictures"), (84, "txt", "stated in S text")):
        rbox(ax, x_, 26, 8, 8, fc=tint(C[k], 0.25), ec=C[k], lw=0.7, r=1.5)
        t(ax, x_ + 11, 30, s_, fs=5.1, color=C[k])
    t(ax, 2, foot, "20 test creatures + 10 distractors", fs=5.1, color=C["grey"])

    # (2) four teaching regimes ---------------------------------------------------------------------------
    x0, x1 = 180, 390
    _heading(ax, x0, top, 2, "Teach through one modality", x1 - x0)
    regs = [("V", C["pic"], PIC_V, ["“Here you can see a kusnib.”"], "picture → name"),
            ("T", C["txt"], None, ["“Every kusnib eats bread. …”"], "name → facts"),
            ("Tr", C["rev"], None, ["“… that eats bread is called a kusnib.”"], "facts → name"),
            ("S", C["S"], PIC_S, ["pictures: “… see a kusnib.”", "text: “Every kusnib eats bread.”"],
             "both links, separately")]
    for i, (k, col, src, lines, store) in enumerate(regs):
        yc = 191 - i * 47
        rbox(ax, x0, yc - 21, 4, 42, fc=col, ec="none", r=1.5, z=3)
        t(ax, x0 + 18, yc, k, fs=9, ha="center", color=col, weight="bold")
        tx = x0 + 34
        if src is not None:
            pic(ax, src, tx, yc - 17, 34)
            tx += 40
        y_ = {1: yc + 6, 2: yc + 12}[len(lines)]
        for ln in lines:
            t(ax, tx, y_, ln, fs=5.7, style="italic")
            y_ -= 11
        t(ax, tx, y_ - 1 if len(lines) == 1 else yc - 13, "stores  " + store, fs=5.5, color=col, weight="bold")
        if i < 3:
            hline(x0 + 10, x1, yc - 23.5)
    t(ax, x0, foot, "8 attributes × 32 examples per creature", fs=5.1, color=C["grey"])

    # (3) LoRA on the VLM ---------------------------------------------------------------------------------
    x0, x1 = 410, 552
    _heading(ax, x0, top, 3, "LoRA on the VLM", x1 - x0)
    pic(ax, PIC_V, x0, 166, 30)
    arr(ax, (x0 + 32, 181), (x0 + 37, 181), C["pic"], lw=0.8, ms=4)
    vision_encoder(ax, x0 + 38, 152, 40, 58, fs=5.0)
    tokens(ax, x0 + 81, 178, 2, C["pic"], s=6)
    arr(ax, (x0 + 96, 181), (x0 + 102, 181), C["pic"], lw=0.8, ms=4)
    t(ax, x0, 128, "“What does …”", fs=5.4, style="italic", color=C["txt"])
    tokens(ax, x0 + 81, 125, 2, C["txt"], s=6)
    arr(ax, (x0 + 96, 128), (x0 + 102, 128), C["txt"], lw=0.8, ms=4)
    decoder_stack(ax, x0 + 103, 104, 38, 108, n=6, label="", fs=5)
    t(ax, x0 + 141, 95, "language model", fs=5.2, ha="right", color=C["txt"], weight="bold")
    rbox(ax, x0, 64, 8, 8, fc=C["lora"], ec=C["lora"], r=1.5)
    t(ax, x0 + 12, 68, "LoRA, r = 16, attention + MLP", fs=5.2)
    snow(ax, x0 + 4, 52, fs=6.5)
    t(ax, x0 + 12, 52, "vision encoder frozen", fs=5.2)
    hline(x0, x1, 40)
    t(ax, x0, 25, "Qwen2.5-VL 3B / 7B", fs=5.6, weight="bold")
    t(ax, x0, foot, "SmolVLM2 2.2B (SigLIP)", fs=5.6, weight="bold")

    # (4) probe the other way -----------------------------------------------------------------------------
    x0, x1 = 572, 690
    _heading(ax, x0, top, 4, "Probe the other way", x1 - x0)
    rbox(ax, x0, 126, x1 - x0, 88, fc="#f6f8fa", ec="none", r=5)
    pic(ax, PIC_S, x0 + 6, 172, 36)
    t(ax, x0 + 48, 199, "What does this", fs=5.5, style="italic")
    t(ax, x0 + 48, 188, "creature eat?", fs=5.5, style="italic")
    cands = [("It eats bread.", 0.85, True), ("It eats fish.", 0.55, False), ("It eats seeds.", 0.4, False)]
    for i, (s_, v, best) in enumerate(cands):
        yy = 163 - i * 9.5
        t(ax, x0 + 6, yy, s_, fs=4.9, weight="bold" if best else "normal")
        rbox(ax, x0 + 66, yy - 3, 46 * v, 6, fc=C["txt"] if best else "#cdd3d9", ec="none", r=1.2, z=4)
    t(ax, x0 + 6, 131.5, "score every full answer by log p", fs=4.8, color=C["grey"])
    probes = [("name → facts", C["txt"]), ("words → name", C["rev"]), ("picture → name", C["pic"]),
              ("picture → facts, silent", C["S"]), ("picture → name → facts, written", C["S"])]
    for i, (s_, col) in enumerate(probes):
        yy = 110 - i * 14
        ax.add_patch(Circle((x0 + 4, yy), 2.4, fc=col, ec="none", zorder=4))
        t(ax, x0 + 11, yy, s_, fs=5.5)
    hline(x0, x1, 40)
    t(ax, x0, 25, "inside: probes, logit lens, swaps,", fs=5.1, color=C["grey"])
    t(ax, x0, foot, "patching, printed and wrong names", fs=5.1, color=C["grey"])

    for xa in (170, 400, 562):                 # quiet flow marks between the panels
        t(ax, xa, 120, "›", fs=14, ha="center", color="#c4c4c4")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FIG / f"schema_pipeline.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# Fig. 6: inside the model
# ---------------------------------------------------------------------------------------------------------
def _interventions(ax, oy):
    """(a) The VLM unrolled: token positions x decoder layers, with each intervention drawn where it acts."""
    t(ax, 0, oy + 258, "(a) Where we read and intervene", fs=7.6, weight="bold")
    gx, gy, cw, rh = 150, oy + 50, 23, 26
    img_cols, words = 4, ["What", "does", "this", "creature", "eat", "?", "It", "eats"]
    ncol = img_cols + len(words)
    layers = ["1", "8", "16", "24", "30", "36"]
    k_row = 2                                      # image-token states swapped from this row up (after layer k)
    # inputs: photo (with printed name) -> vision encoder -> image tokens at layer 1
    pic(ax, labelled(PIC_S, "kusnib"), 4, gy - 22, 66)
    t(ax, 37, gy - 30, "photo (± printed name)", fs=5.2, ha="center", color=C["grey"])
    num(ax, 8, gy + 40, 4, C["label"])
    vision_encoder(ax, 80, gy - 24, 40, 70, fs=5.0)
    arr(ax, (121, gy + 11), (gx - 15, gy + 11), C["pic"], lw=1.0)
    for r, lab in enumerate(layers):
        yy = gy + r * rh
        t(ax, gx - 3, yy + rh / 2 - 2, "L" + lab, fs=5.0, ha="right", color=C["grey"])
        for c in range(ncol):
            xx = gx + c * cw
            if c < img_cols:
                swapped = r >= k_row
                col, alpha = (C["other"], 0.5) if swapped else (C["pic"], 0.38)
            else:
                col, alpha = C["txt"], 0.22
            last = c == ncol - 1
            rbox(ax, xx + 2, yy + 3, cw - 4, rh - 8, fc=col, ec=C["ink"] if last else col, r=2, z=3,
                 alpha=0.45 if last else alpha, lw=0.9 if last else 0.4)
    for c, w in enumerate(words):
        t(ax, gx + (img_cols + c) * cw + cw / 2 + 2, gy - 3, w, fs=4.8, ha="right", va="top", rotation=35,
          style="italic")
    t(ax, gx + img_cols * cw / 2, gy - 8, "image tokens", fs=5.2, ha="center", color=C["pic"], weight="bold")
    # (2) image-token swap after layer k
    sx0, sx1 = gx + 1, gx + img_cols * cw - 1
    ax.plot([sx0, sx1], [gy + k_row * rh + 0.5, gy + k_row * rh + 0.5], color=C["other"], lw=1.2, ls="--",
            zorder=6)
    t(ax, sx1 + 3, gy + k_row * rh + 0.5, "k", fs=5.6, color=C["other"], weight="bold")
    top = gy + len(layers) * rh
    pic(ax, PIC_V2, gx + img_cols * cw / 2 - 15, top + 4, 30, ec=C["other"], lw=1.1, z=7)
    arr(ax, (gx + img_cols * cw / 2, top + 3), (gx + img_cols * cw / 2, top - 6), C["other"], lw=1.1)
    t(ax, gx + img_cols * cw / 2 + 18, top + 19, "another creature's\nimage-token states", fs=4.9,
      color=C["other"], weight="bold", linespacing=1.0)
    num(ax, gx + img_cols * cw / 2 - 23, top + 19, 2, C["other"])
    # (5) a written (wrong) name in place of "this creature"
    cx = gx + (img_cols + 3) * cw                      # between "this" and "creature"
    bw = 7.4 * cw
    rbox(ax, cx - bw / 2, gy - 50, bw, 12, fc="white", ec=C["S"], lw=0.9, r=3, z=6)
    t(ax, cx, gy - 44, "“a neetroob” in place of “this creature”", fs=4.8, ha="center", color=C["S"],
      weight="bold", z=7)
    arr(ax, (cx, gy - 38), (cx, gy - 31), C["S"], lw=0.9)
    num(ax, cx - bw / 2 - 8, gy - 44, 5, C["S"])
    # (3) a parallel run with the name in words; its states are copied in at layer i
    px = gx + ncol * cw + 26
    for r in range(len(layers)):
        rbox(ax, px, gy + r * rh + 3, 17, rh - 8, fc=C["txt"], ec=C["txt"], r=2, z=3, alpha=0.5)
    t(ax, px + 8.5, gy - 4, "“What does a\nkusnib eat?”", fs=4.6, ha="center", va="top", style="italic",
      color=C["txt"], linespacing=1.0)
    yi = gy + 1.5 * rh
    arr(ax, (px - 1, yi), (gx + ncol * cw - 1, yi), C["txt"], lw=1.1)
    t(ax, px + 20, yi, "copy at\nlayer i", fs=4.8, color=C["txt"], linespacing=1.0)
    num(ax, px + 8.5, top + 6, 3, C["txt"])
    # (1) read-outs at the last position: linear probe, logit lens
    lc = gx + (ncol - 1) * cw + cw / 2
    for r in range(len(layers)):
        ax.add_patch(Circle((lc, gy + r * rh + rh / 2 - 1), 3.2, fc="white", ec=C["ink"], lw=0.7, zorder=9))
        ax.plot([lc + 2.2, lc + 4.8], [gy + r * rh + rh / 2 - 3.2, gy + r * rh + rh / 2 - 5.8], color=C["ink"],
                lw=0.9, zorder=9)
    num(ax, lc, top + 6, 1)
    # legend with results
    lx0 = px + 54
    items = [(1, "#333333", "read the last position (probe, logit lens)",
              f"identity from layer 1, untaught too; taught name from {m('bridge.namelens.S-s0.pic.first')}"),
             (2, C["other"], "swap image-token states after layer k",
              f"answer follows the new photo until layer 30 ({m('bridge.swap.V-s0.swap-name.part.30')})"),
             (3, C["txt"], "copy states from the name-in-words run",
              f"answer right only if copied at layer 32 ({m('p2.own.32')})"),
             (4, C["label"], "print the name inside the photo",
              f"silent answer {m('bridge.label.S-s0.none')} → {m('bridge.label.S-s0.own')} (seed 0)"),
             (5, C["S"], "write a (wrong) name in the prompt",
              f"answer follows the written name ({m('bridge.relabel.S-s0.word')}), not the photo "
              f"({m('bridge.relabel.S-s0.pic')})")]
    for i, (k, col, what, res) in enumerate(items):
        yy = oy + 214 - i * 42
        num(ax, lx0 + 6, yy, k, col)
        t(ax, lx0 + 16, yy + 1, what, fs=5.6, weight="bold")
        t(ax, lx0 + 16, yy - 11, res, fs=5.1, color=C["grey"])


def _timeline(ax):
    """(b) What becomes readable where, along the layers of the S adapter."""
    L = 36
    ax.set_xlim(-14.5, L + 6)
    ax.set_ylim(-0.75, 5.5)
    ax.axis("off")
    ax.text(-14.5, 5.35, "(b) What becomes readable where (S adapter)", fontsize=7.6, weight="bold", va="center")
    for l in range(0, L + 1, 4):
        ax.plot([l, l], [-0.15, 4.65], color="#eeeeee", lw=0.6, zorder=0)
        ax.text(l, -0.4, str(l), ha="center", fontsize=6, color=C["grey"])
    ax.text(L / 2, -0.62, "decoder layer", ha="center", va="top", fontsize=6.3, color=C["grey"])
    raw = str(N.get("bridge.namelens.S-s0.pic.first", "33"))
    name1 = int(raw) if raw.isdigit() else 33
    tracks = [
        (4.2, "image tokens in use", 0, 30, C["pic"], "answer still follows swapped image tokens at 30", 0.4, "left"),
        (3.3, "creature identity", 1, L, C["id"], f"linear probe {m('layers.name1')} from layer 1, untaught too",
         1.4, "left"),
        (2.4, "taught name", name1, L, C["S"], f"from layer {name1}", name1 - 0.6, "right"),
        (1.5, "fact, from the name in words", 32, L, C["txt"], "only late, only after teaching", 31.4, "right"),
        (0.6, "fact, from the photo", None, None, C["grey"], f"never without the name (silent answer "
         f"{m('comp.compose')})", 0.4, "left"),
    ]
    for y, lab, a, b, col, note, nx, ha in tracks:
        ax.text(-0.8, y, lab, ha="right", va="center", fontsize=6.6, weight="bold", color=C["ink"])
        if a is None:
            ax.plot([0, L], [y, y], color=col, lw=1, ls=":")
        else:
            ax.add_patch(FancyBboxPatch((a, y - 0.22), b - a, 0.44, boxstyle="round,pad=0,rounding_size=0.15",
                                        fc=col, ec=col, alpha=0.85))
        ax.text(nx, y + (0.36 if ha == "left" else 0), note, ha=ha, va="center" if ha == "right" else "bottom",
                fontsize=5.7, color=C["ink"])
    ax.add_patch(FancyArrowPatch((L + 0.6, 2.4), (L + 0.6, 1.5), arrowstyle="-|>", mutation_scale=8,
                                 color=C["bad"], lw=1.2, connectionstyle="arc3,rad=-0.6"))
    ax.text(L + 2.2, 1.95, "no layers left\nto use it", ha="left", va="center", fontsize=5.7, color=C["bad"])
    ax.add_patch(FancyBboxPatch((19, 0.39), 17, 0.42, boxstyle="round,pad=0,rounding_size=0.15", fc="white",
                                ec=C["label"], lw=1.2, ls="--", zorder=3))
    ax.text(27.5, 0.6, f"name printed in the photo: {m('bridge.label.S-s0.own')}", ha="center",
            va="center", fontsize=5.7, color=C["label"], weight="bold", zorder=4)


def schema_inside():
    W, Ha, Hb = 6.9, 2.66, 1.85
    fig = plt.figure(figsize=(W, Ha + Hb))
    ax = fig.add_axes([0, Hb / (Ha + Hb), 1, Ha / (Ha + Hb)])
    ax.set_xlim(0, W * 100)
    ax.set_ylim(0, Ha * 100)
    ax.set_aspect("equal")
    ax.axis("off")
    _interventions(ax, 0)
    ax2 = fig.add_axes([0.0, 0.0, 1, Hb / (Ha + Hb) * 0.97])
    _timeline(ax2)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(FIG / f"schema_inside.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def schema_real():
    """The real-photo design: word known or new x fact contradicting or new, plus a known word for a new thing."""
    raw = sorted((ROOT / "data" / "real_raw" / "caltech-101" / "kangaroo").glob("*.jpg"))

    def square(path):
        im = Image.open(path).convert("RGB")
        s_ = min(im.size)
        return im.crop(((im.width - s_) // 2, (im.height - s_) // 2, (im.width + s_) // 2, (im.height + s_) // 2))
    kang = square(raw[21])
    fig, ax = canvas(3.4, 2.62)
    H = 262
    t(ax, 117, H - 9, "changes a belief", fs=6.6, ha="center", weight="bold", color=C["rev"])
    t(ax, 117, H - 20, "(where found, how heavy)", fs=5.2, ha="center", color=C["grey"])
    t(ax, 217, H - 9, "adds a new fact", fs=6.6, ha="center", weight="bold", color=C["txt"])
    t(ax, 217, H - 20, "(homeland, owner)", fs=5.2, ha="center", color=C["grey"])
    t(ax, 304, H - 9, "probe", fs=6.6, ha="center", weight="bold")
    rows = [  # y, tag, colour, picture, word, how the picture reaches the word, box 1, box 2
        (170, "K", C["pic"], kang, "“kangaroo”", "pretraining",
         "Every kangaroo is\nfound in a kitchen.", "Every kangaroo\ncomes from Oskaria."),
        (95, "A", C["S"], kang, "“gutronk”", "photos",
         "Every gutronk is\nfound in a kitchen.", "Every gutronk\ncomes from Oskaria."),
        (20, "R", C["other"], IMG / "V" / "c_0001" / "036.png", "“kettle”", "drawings",
         None, "Every kettle\neats seeds."),
    ]
    for y, tag_, col, src, name, how, ch, add in rows:
        rbox(ax, 2, y, 62, 66, fc="white", ec=col, lw=1.0, r=4)
        tag(ax, 7, y + 49, tag_, col, w=15, h=12, fs=6.0)
        pic(ax, src, 34, y + 35, 26)
        t(ax, 33, y + 24, name, fs=5.8, ha="center", weight="bold")
        t(ax, 33, y + 14, "word taught by", fs=4.4, ha="center", color=C["grey"])
        t(ax, 33, y + 7, how, fs=4.6, ha="center", color=col, weight="bold")
        if ch is None:   # R: the drawings teach the known word for a new creature; no belief is changed
            rbox(ax, 70, y + 8, 94, 50, fc="#fafafa", ec="#bbbbbb", lw=0.8, r=4, ls="--")
            t(ax, 117, y + 38, "“Here you can see", fs=5.2, ha="center", style="italic")
            t(ax, 117, y + 29, "a kettle.”", fs=5.2, ha="center", style="italic")
            t(ax, 117, y + 17, "(drawings of a new creature)", fs=4.4, ha="center", color=C["grey"])
        else:
            rbox(ax, 70, y + 8, 94, 50, fc="white", ec=C["rev"], lw=0.8, r=4)
            t(ax, 117, y + 33, ch, fs=5.2, ha="center", style="italic", linespacing=1.2)
        rbox(ax, 170, y + 8, 94, 50, fc="white", ec=C["txt"], lw=0.8, r=4)
        t(ax, 217, y + 33, add, fs=5.2, ha="center", style="italic", linespacing=1.2)
    # the probe column: held-out picture, no word
    rbox(ax, 270, 103, 68, 125, fc="#fafafa", ec="#cccccc", r=4)
    pic(ax, square(raw[26]), 285, 182, 38)
    t(ax, 304, 172, "held-out photo,", fs=4.8, ha="center", color=C["grey"])
    t(ax, 304, 164, "no word", fs=4.8, ha="center", color=C["grey"])
    t(ax, 304, 147, "“Where is this", fs=5.0, ha="center", style="italic")
    t(ax, 304, 138, "usually found?”", fs=5.0, ha="center", style="italic")
    t(ax, 304, 121, "taught fact", fs=5.0, ha="center", weight="bold")
    t(ax, 304, 112, "or prior belief?", fs=5.0, ha="center", weight="bold")
    rbox(ax, 270, 28, 68, 50, fc="#fafafa", ec="#cccccc", r=4)
    t(ax, 304, 62, "held-out drawing,", fs=4.8, ha="center", color=C["grey"])
    t(ax, 304, 54, "no word:", fs=4.8, ha="center", color=C["grey"])
    t(ax, 304, 42, "“What does this", fs=5.0, ha="center", style="italic")
    t(ax, 304, 34, "creature eat?”", fs=5.0, ha="center", style="italic")
    t(ax, 170, 6, "24 Caltech-101 concepts in 4 domains · same photos and facts in K and A", fs=4.8,
      ha="center", color=C["grey"])
    for ext, kw in (("pdf", {}), ("png", {"dpi": 260})):
        fig.savefig(FIG / f"schema_real.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def _ci(key):
    """(mean, lo, hi) from a numbers.json value like '0.19 [0.12, 0.25]'."""
    v = str(N.get(key, "nan")).replace("[", " ").replace("]", " ").replace(",", " ")
    nums = []
    for x in v.split():
        try:
            nums.append(float(x))
        except ValueError:
            pass
    return tuple((nums + [nums[0], nums[0]])[:3]) if nums else (float("nan"),) * 3


def _dotaxis(ax, n, xlabel, xmax=1.02):
    ax.axvline(0.18, color=C["grey"], ls=":", lw=0.8, zorder=1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.tick_params(labelsize=5.8, colors=C["grey"], length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(xlabel, fontsize=6, color=C["grey"], labelpad=2)


def schema_gap():
    """Sec. 5 figure (one column): the sharing gap per setting. Silent question vs written chain; the bar is
    the gap; grey ring: the weaker of the two hops; tick: untaught model."""
    rows = [  # label, hop1, hop2, silent, chain, untaught silent
        ("Qwen2.5-VL-3B", "comp.compose-hop1", "comp.compose-hop2", "comp.compose", "comp.compose-chain",
         "comp.compose.e0"),
        ("Qwen2.5-VL-7B", "s7.S.compose-hop1", "s7.S.compose-hop2", "s7.S.compose", "s7.S.compose-chain",
         "s7.S.compose.e0"),
        ("SmolVLM2-2.2B", "smol.S.compose-hop1", "smol.S.compose-hop2", "smol.S.compose", "smol.S.compose-chain",
         "smol.ctrl.compose"),
        ("familiar word as name", "real.R.R.compose-hop1", "real.R.R.compose-hop2", "real.R.R.compose",
         "real.R.R.compose-chain", None),
        ("said, never seen", "mirror.S.words-hop1", "mirror.S.compose-hop2", "mirror.S.words-silent",
         "mirror.S.words-chain", "mirror.base.words-silent"),
        ("real photo, new word", "real.A.hop1.add", "real.A.hop2.add", "real.A.compose.add", "real.A.chain.add",
         None),
        ("real photo, own word", "real.K.hop1.add", "real.K.hop2.add", "real.K.compose.add", "real.K.chain.add",
         None),
    ]
    fig = plt.figure(figsize=(3.35, 1.95))
    ax = fig.add_axes([0.335, 0.17, 0.53, 0.68])
    n = len(rows)
    for i, (lab, h1, h2, sil, ch, base) in enumerate(rows):
        y = n - 1 - i
        own = lab.endswith("own word")
        if own:
            ax.axhspan(y - 0.5, y + 0.5, xmin=-0.62, xmax=1.27, color="#f2f2f2", zorder=0, lw=0, clip_on=False)
        mu, lo, hi = _ci(sil)
        c = _ci(ch)[0]
        ax.plot([mu, c], [y, y], color="#d6d6d6", lw=3.2, zorder=2, solid_capstyle="butt")
        weak = min(_ci(h1)[0], _ci(h2)[0])
        ax.plot(weak, y + 0.27, "o", ms=2.6, mfc="white", mec=C["grey"], mew=0.7, zorder=4)
        ax.plot([lo, hi], [y, y], color=C["rev"], lw=1.1, zorder=5, solid_capstyle="round")
        ax.plot(mu, y, "o", ms=5.0, color=C["rev"], mec="white", mew=0.7, zorder=6)
        ax.plot(c, y, "D", ms=3.8, color=C["txt"], mec="white", mew=0.5, zorder=6)
        if base:
            ax.plot(_ci(base)[0], y, "|", ms=6.5, color=C["ink"], mew=0.9, zorder=7)
        ax.text(-0.04, y, lab, ha="right", va="center", fontsize=6.0, color=C["ink"],
                weight="bold" if own else "normal")
        ax.text(1.06, y, f"{c - mu:+.2f}", ha="left", va="center", fontsize=5.6,
                color=C["ink"] if own else C["grey"], weight="bold" if own else "normal")
    _dotaxis(ax, n, "accuracy, held-out pictures or descriptions")
    ax.text(1.06, n - 0.45, "gap", ha="left", va="bottom", fontsize=5.6, color=C["grey"])
    # legend row
    y0 = 0.93
    items = [("o", C["rev"], C["rev"], "silent question"), ("D", C["txt"], C["txt"], "name written first"),
             ("o", "white", C["grey"], "weaker hop"), ("|", C["ink"], C["ink"], "untaught")]
    x = 0.02
    for mk, fc, ec, lab in items:
        fig.add_artist(matplotlib.lines.Line2D([x], [y0], marker=mk, ms=3.6, mfc=fc, mec=ec, color=ec, mew=0.8,
                                                 ls="", transform=fig.transFigure))
        fig.text(x + 0.022, y0, lab, fontsize=5.6, va="center", color=C["ink"])
        x += 0.07 + 0.0122 * len(lab)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 260})):
        fig.savefig(FIG / f"schema_gap.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


REM_ALL = [  # label, key, group: 0 implicit, 1 self-rehearsal, 3 label distillation, 2 explicit
    ("S, no remedy", "fix.S.compose", 0),
    ("+ silent practice on other creatures", "fix.Sp.compose", 0),
    ("+ facts in the picture captions", "fix.Sc.compose", 0),
    ("facts written name-last", "fix.Sm.compose", 0),
    ("+ early binding, layers 8–20", "fix.Sb.compose", 0),
    ("+ early binding, layers 20–32", "fix.Sbl.compose", 0),
    ("rehearsed facts", "cons.2.compose.rr", 1),
    ("same creature, other facts", "cons.2.compose.rn", 1),
    ("other creatures", "cons.2.compose.n", 1),
    ("language model only", "dist.lm.6.compose.d", 3),
    ("+ vision encoder", "dist.vis.6.compose.d", 3),
    ("+ vision encoder, distillation loss off", "dist.visnd.6.compose.d", 3),
    ("name printed in the picture", "bridge.label.S-s0.own", 2),
    ("name-run states copied at layer 32", "p2.own.32", 2),
    ("name read out at layer 1, fed back", "rem.latent.latent", 2),
    ("true name written in the prompt", "rem.latent.latent-oracle", 2),
    ("trained to name first (free answer)", "gen.Sv.gen-silent.value_ok", 2),
]
REM_MAIN = [
    ("S, no remedy", "fix.S.compose", 0),
    ("silent practice on other creatures", "fix.Sp.compose", 0),
    ("early binding (best implicit)", "fix.Sb.compose", 0),
    ("rehearsed facts", "cons.2.compose.rr", 1),
    ("same creature, other facts", "cons.2.compose.rn", 1),
    ("language model only", "dist.lm.6.compose.d", 3),
    ("+ vision encoder", "dist.vis.6.compose.d", 3),
    ("+ vision encoder, distillation loss off", "dist.visnd.6.compose.d", 3),
    ("name printed in the picture", "bridge.label.S-s0.own", 2),
    ("name read out at layer 1, fed back", "rem.latent.latent", 2),
    ("trained to name first (free answer)", "gen.Sv.gen-silent.value_ok", 2),
]
REM_GROUPS = ((0, "name left implicit"), (1, "self-rehearsal"), (3, "printed name distilled in"),
              (2, "name made explicit"))


def _remedy_chart(rem, fname, W, H, left, fs=5.8):
    fig = plt.figure(figsize=(W, H))
    axb = fig.add_axes([left, 0.12, 0.87 - left, 0.86])
    gap = 1.4
    ys, y = [], 0.0
    rv = list(reversed(rem))
    for i, (_, _, g) in enumerate(rv):
        if i and g != rv[i - 1][2]:
            y += gap
        ys.append(y)
        y += 1
    ys = ys[::-1]
    top = ys[0]
    gcol = {0: C["grey"], 1: C["rev"], 2: C["txt"], 3: C["pic"]}
    for (lab, key, g), y in zip(rem, ys):
        col = gcol[g]
        mu, lo, hi = _ci(key)
        axb.plot([0, 1], [y, y], color="#e6e6e6", lw=0.5, zorder=1)
        axb.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=4, solid_capstyle="round")
        axb.plot(mu, y, "o", ms=4.6, color=col, mec="white", mew=0.7, zorder=5)
        axb.text(-0.04, y, lab, ha="right", va="center", fontsize=fs, color=C["ink"])
        axb.text(min(hi, 1.0) + 0.035, y, f"{mu:.2f}", ha="left", va="center", fontsize=fs - 0.6, color=C["grey"])
    first = {g: next(i for i, r in enumerate(rem) if r[2] == g) for g, _ in REM_GROUPS}
    for g, lab in REM_GROUPS:
        axb.text(-0.04, ys[first[g]] + 0.9, lab, fontsize=fs + 0.1, weight="bold", color=gcol[g], ha="right")
    _dotaxis(axb, 0, "silent question, held-out pictures", xmax=1.13)
    axb.set_ylim(-0.6, top + 1.4)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 260})):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def schema_summary():
    """Sec. 7 (one column): the remedies at a glance; supplement: every remedy."""
    _remedy_chart(REM_MAIN, "schema_remedy_rows", 3.35, 2.2, 0.47)
    _remedy_chart(REM_ALL, "supp_remedies_all", 4.6, 3.0, 0.42, fs=6.2)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    schema_teaser()
    schema_pipeline()
    schema_inside()
    schema_real()
    schema_gap()
    schema_summary()
    print("ok")
