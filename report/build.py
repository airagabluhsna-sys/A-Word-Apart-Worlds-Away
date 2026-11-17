"""Build the results site and the paper from the raw outputs in kaggle/out.

    py -3.13 report/build.py            # writes report/site/*.html

Every number on the site is computed here from eval.jsonl / eval_adapters.jsonl
/ readout_summary.json, so re-running after new runs land updates all pages.
Narrative pages live in report/content/*.html with placeholders:

    {{n:key}}   a number or "mean [lo, hi]" string from N (below)
    {{t:key}}   a generated HTML table
    {{c:key}}   a generated SVG chart

A placeholder whose key is missing renders as a red [missing: key] marker and
is listed on stdout, so stale text is visible rather than silently wrong.

Run registry: RUNS (training runs) and EVALS (eval-only jobs, adapter blocks in
file order). Add a row when a new run is launched; queued runs show as pending
until their output directory exists.
"""

import html
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "out"
CONTENT = ROOT / "report" / "content"
SITE = ROOT / "report" / "site"
sys.path.insert(0, str(ROOT / "kaggle"))
sys.path.insert(0, str(ROOT))
from teach_report import groups as report_groups          # noqa: E402
from seen_never_said.generator.ontology import VALUES      # noqa: E402

B = 1000                                                   # bootstrap resamples
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- registry

# arm: the condition a run pools into. protocol: how it was taught.
RUNS = [
    # id, out dir, model, regime, seed, arm, protocol, kernel, note
    ("teach_t_s0", "3B", "T", 0, "T-nofmt", "no practice", "snsaid-teach-t-s0",
     "First protocol. Recall plateaued at 0.36-0.44 while loss fell."),
    ("teach_v_s0", "3B", "V", 0, "V-nofmt", "no practice", "snsaid-teach-v-s0",
     "First protocol. Letter A on every item by epoch 8; known-fact anchors collapsed (0.90 -> 0.10)."),
    ("teach_v_s1", "3B", "V", 1, "V-nofmt", "no practice", "snsaid-teach-v-s1",
     "Replicates seed 0."),
    ("teach2_t_s0", "3B", "T", 0, "T", "practice", "snsaid-teach2-t-s0",
     "Distractor-only format practice. MC still 0.43."),
    ("teach2_t_s2", "3B", "T", 2, "T", "practice", "snsaid-teach-t-s2",
     "Launched under the old kernel name; it is a practice run."),
    ("teach2_v_s0", "3B", "V", 0, "V-flawed", "practice (flawed)", "snsaid-teach2-v-s0",
     "Practice used test names as wrong options: MC picks distractor names. Likelihood probes unaffected."),
    ("teach2_v_s2", "3B", "V", 2, "V-flawed", "practice (flawed)", "snsaid-teach-v-s2",
     "As teach2_v_s0. Its adapter is one of the two V arms of the direction table (likelihood probes only)."),
    ("teach3_v_s0", "3B", "V", 0, "V", "practice", "snsaid-teach3-v-s0",
     "Fixed practice (distractor names only)."),
    ("teach3_tr_s0", "3B", "Tr", 0, "Tr", "practice", "snsaid-teach3-tr-s0",
     "Reversed text: the same facts with the name last."),
    ("teach3_tr_s1", "3B", "Tr", 1, "Tr", "practice", "snsaid-teach3-tr-s1", ""),
    ("teach3_tr_s2", "3B", "Tr", 2, "Tr", "practice", "snsaid-teach3-tr-s2", ""),
    ("teach3_s_s0", "3B", "S", 0, "S", "practice", "snsaid-teach3-s-s0",
     "Split regime: pictures teach 4 attributes + the name, text the other 4."),
    ("teach3_s_s1", "3B", "S", 1, "S", "practice", "snsaid-teach3-s-s1", ""),
    ("teach3_s_s2", "3B", "S", 2, "S", "practice", "snsaid-teach3-s-s2", ""),
    ("teach3_sp_s0", "3B", "Sp", 0, "Sp", "practice + composition practice", "snsaid-teach3-sp-s0",
     "Planned for after the 14 November reset: S plus silent picture → property question–answer pairs on the 10 "
     "distractors only. Tests whether cross-modal composition, practised on some creatures, transfers to the "
     "20 test creatures."),
    ("teach3_sc_s0", "3B", "Sc", 0, "Sc", "practice, co-presence", "snsaid-teach3-sc-s0",
     "Planned for after the 14 November reset: each S picture captioned with the name and the creature's "
     "text-taught facts together (picture, name and facts seen together in training)."),
    ("teach7b_t_s0", "7B", "T", 0, "T-7B", "practice", "snsaid-teach7b-t-s0", "7B, 3 epochs."),
    ("teach7b_tr_s0", "7B", "Tr", 0, "Tr-7B", "practice", "snsaid-teach7b-tr-s0", "7B, 3 epochs; letter MC never above 0.19."),
    ("teach7b_s_s0", "7B", "S", 0, "S-7B", "practice", "snsaid-batch-ab1",
     "7B split regime: 2 epochs, batch 2 (batch 4 ran out of memory on 2xT4)."),
    ("smol_s_s0", "SmolVLM", "S", 0, "S-Smol", "practice", "snsaid-smol-s-s0", "SmolVLM2-2.2B, 3 epochs."),
    ("smol_t_s0", "SmolVLM", "T", 0, "T-Smol", "practice", "snsaid-batch-ic1", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_tr_s0", "SmolVLM", "Tr", 0, "Tr-Smol", "practice", "snsaid-batch-ic1", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_v_s0", "SmolVLM", "V", 0, "V-Smol", "practice", "snsaid-batch-ic1", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_t_s1", "SmolVLM", "T", 1, "T-Smol", "practice", "snsaid-batch-ic3a", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_t_s2", "SmolVLM", "T", 2, "T-Smol", "practice", "snsaid-batch-ic3a", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_tr_s1", "SmolVLM", "Tr", 1, "Tr-Smol", "practice", "snsaid-batch-ic3a", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
    ("smol_tr_s2", "SmolVLM", "Tr", 2, "Tr-Smol", "practice", "snsaid-batch-ic3a", "SmolVLM2-2.2B, 3 epochs; batch kernel."),
]
RUNS = [dict(zip(("id", "model", "regime", "seed", "arm", "protocol", "kernel", "note"), r)) for r in RUNS]
RUN = {r["id"]: r for r in RUNS}
SMOKES = ["snsaid-teach-smoke-fp16", "teach_smoke_v_fp32", "teach7b_smoke"]

# eval-only jobs: adapter tags in the order their blocks appear in the file
EVALS = [
    ("eval_ladder1", "snsaid-eval-ladder1", "3B", ["teach_t_s0", "teach_v_s0", "teach2_t_s0"],
     "Extraction ladder. Old tags lacked format practice, so blocks are matched by order."),
    ("eval_describe1", "snsaid-eval-describe1", "3B",
     ["teach_t_s0", "teach2_t_s2", "teach_v_s0", "teach2_v_s2", "teach2_t_s0"], "Description -> name probes."),
    ("eval_dir1", "snsaid-eval-dir1", "3B", ["teach3_tr_s0", "teach3_v_s0"], "Direction test, seed 0 (+ describe-one)."),
    ("eval_dir2", "snsaid-eval-dir2", "3B", ["teach3_tr_s1", "teach3_tr_s2"], "Direction test, seeds 1-2."),
    ("eval7b_dir1", "snsaid-eval7b-dir1", "7B", ["teach7b_t_s0", "teach7b_tr_s0"], "7B direction test."),
    ("smol_eval_dir1", "snsaid-batch-ic3a", "SmolVLM", ["smol_t_s0", "smol_tr_s0", "smol_v_s0", "smol_s_s0"],
     "SmolVLM2-2.2B direction + composition test, seed 0 (batch kernel)."),
    ("smol_eval_dir2", "snsaid-batch-ic4", "SmolVLM", ["smol_t_s1", "smol_t_s2", "smol_tr_s1", "smol_tr_s2"],
     "SmolVLM2-2.2B direction test, seeds 1-2 (batch kernel)."),
]
SRC_ORDER = ["train"] + [e[0] for e in EVALS]


# ---------------------------------------------------------------- loading

def groups(r):
    f = r.get("family") or ""
    if f.startswith(("compose", "describe-one")):
        return [f, f"{f}/{r['level']}"]
    return report_groups(r)


def lines(path):
    with open(path, encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                yield json.loads(l)


CONFUSION_FAMS = ("ladder-image-1", "describe-name")


class Store:
    """Rows reduced to (concept, correct, calibrated-correct), indexed by run/source/epoch/group."""

    def __init__(self):
        self.cells = defaultdict(list)       # (run, src, epoch, group) -> [(concept, correct, cal, chance, dk)]
        self.extra = defaultdict(list)       # (run, src, epoch) -> compose rows with pred
        self.meta = {}                       # run id -> run.json
        self.srcs = defaultdict(set)         # run -> {(src, epoch)}

    def add(self, run, src, epoch, r, cal):
        for g in groups(r):
            self.cells[(run, src, epoch, g)].append(
                (r.get("concept") or r["id"], bool(r["correct"]), cal, r["chance"], bool(r.get("dont_know"))))
        if (r.get("family") or "").startswith("compose") or r.get("family") in CONFUSION_FAMS:
            self.extra[(run, src, epoch)].append(
                {k: r.get(k) for k in ("id", "family", "level", "pred", "answer", "correct", "concept", "scores")})
        self.srcs[run].add((src, epoch))


def argmax(xs):
    return max(range(len(xs)), key=xs.__getitem__)


def load():
    st = Store()
    for r in RUNS:
        d = OUT / r["id"]
        if not (d / "eval.jsonl").exists():
            continue
        st.meta[r["id"]] = json.loads((d / "run.json").read_text())
        rows = list(lines(d / "eval.jsonl"))
        zero = {r_["id"]: r_["scores"] for r_ in rows if r_["epoch"] == 0 and "scores" in r_}
        for x in rows:
            cal = None
            if "scores" in x and x["id"] in zero:
                cal = argmax([a - b for a, b in zip(x["scores"], zero[x["id"]])]) == x["answer"]
            st.add(r["id"], "train", x["epoch"], x, cal)
    for f, _, model, order, _ in EVALS:
        p = OUT / f / "eval_adapters.jsonl"
        if not p.exists():
            continue
        rows = list(lines(p))
        base = {(x["seed"], x["id"]): x["scores"] for x in rows if x["run"].startswith("base") and "scores" in x}
        block, prev = -1, None
        for x in rows:
            if x["run"].startswith("base"):
                run = f"base-{model}"
            else:
                if x["run"] != prev:
                    block += 1
                run = order[block]
            prev = x["run"]
            cal = None
            if "scores" in x and (x["seed"], x["id"]) in base:
                cal = argmax([a - b for a, b in zip(x["scores"], base[(x["seed"], x["id"])])]) == x["answer"]
            st.add(run, f, 0 if run.startswith("base") else "final", x, cal)
    return st


# ---------------------------------------------------------------- statistics

def boot(items, cal=False):
    """items: [(concept, correct, cal, chance, dk)] -> dict(m, lo, hi, n, chance, dk, by)."""
    idx = 2 if cal else 1
    items = [it for it in items if it[idx] is not None]
    if not items:
        return None
    by = defaultdict(lambda: [0, 0])
    for it in items:
        by[it[0]][0] += it[idx]
        by[it[0]][1] += 1
    s = np.array([v[0] for v in by.values()], float)
    n = np.array([v[1] for v in by.values()], float)
    m = s.sum() / n.sum()
    if len(by) > 1:
        pick = RNG.integers(0, len(by), (B, len(by)))
        bs = np.sort(s[pick].sum(1) / n[pick].sum(1))
        lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
    else:
        lo = hi = m
    return {"m": m, "lo": lo, "hi": hi, "n": len(items), "chance": float(np.mean([it[3] for it in items])),
            "dk": float(np.mean([it[4] for it in items])), "by": {k: tuple(v) for k, v in by.items()}}


def bm(items):
    """Mean accuracy, or None when the group was not scored."""
    b = boot(items)
    return None if b is None else b["m"]


def paired_diff(a, b):
    """a, b: boot() results; difference of means over shared concepts with a concept bootstrap."""
    cs = sorted(set(a["by"]) & set(b["by"]))
    A = np.array([a["by"][c] for c in cs], float)
    Bm = np.array([b["by"][c] for c in cs], float)
    pick = RNG.integers(0, len(cs), (B, len(cs)))
    d = np.sort(A[pick, 0].sum(1) / A[pick, 1].sum(1) - Bm[pick, 0].sum(1) / Bm[pick, 1].sum(1))
    return {"m": A[:, 0].sum() / A[:, 1].sum() - Bm[:, 0].sum() / Bm[:, 1].sum(),
            "lo": d[int(0.025 * B)], "hi": d[int(0.975 * B) - 1]}


def last_epoch(st, run):
    eps = [e for s, e in st.srcs[run] if s == "train"]
    return max(eps) if eps else None


def final_items(st, run, group):
    """The run's rows for a group from the latest source that scored it (eval jobs supersede training)."""
    for src in reversed(SRC_ORDER):
        ep = last_epoch(st, run) if src == "train" else "final"
        key = (run, src, ep, group)
        if key in st.cells:
            return st.cells[key]
    return []


def pooled(st, runs, group, cal=False):
    items = [it for r in runs for it in final_items(st, r, group)]
    return boot(items, cal)


def runs_of(arm, st=None):
    return [r["id"] for r in RUNS if r["arm"] == arm and (st is None or r["id"] in st.srcs)]


# ---------------------------------------------------------------- formatting

def f2(x):
    return "-" if x is None else f"{x:.2f}"


def ci(res, signed=False):
    if res is None:
        return "–"
    s = "+.2f" if signed else ".2f"
    return f"{res['m']:{s}} [{res['lo']:{s}}, {res['hi']:{s}}]"


def esc(s):
    return html.escape(str(s))


def table(head, rows, cls="", note=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>{cap}'


def cell(res, show_n=False):
    if res is None:
        return '<span class="na">–</span>'
    s = (f'<span class="m">{res["m"]:.2f}</span> <span class="ci">[{res["lo"]:.2f}, {res["hi"]:.2f}]</span>')
    return s + (f' <span class="n">n={res["n"]}</span>' if show_n else "")


SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def legend(names):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{SERIES[i]}"></i>{esc(n)}</span>' for i, n in enumerate(names)) + "</div>"


def dot_chart(rows, series, vals, title):
    """rows: [(label, chance)]; series: [name]; vals[(i, j)] = boot result or None."""
    L, W, x0 = 250, 760, 262
    band = 12 * len(series) + 18
    H = 30 + band * len(rows) + 26
    X = lambda v: x0 + v * (W - x0 - 20)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="22" y2="{H - 24}" class="grid"/>'
                   f'<text x="{X(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{t:g}</text>')
    for i, (lab, ch) in enumerate(rows):
        y0 = 26 + i * band
        out.append(f'<text x="{L}" y="{y0 + band / 2 + 4:.1f}" class="rl" text-anchor="end">{esc(lab)}</text>')
        if ch is not None:
            out.append(f'<line x1="{X(ch):.1f}" x2="{X(ch):.1f}" y1="{y0 + 3}" y2="{y0 + band - 3}" class="chance">'
                       f'<title>chance {ch:.2f}</title></line>')
        for j, _ in enumerate(series):
            v = vals.get((i, j))
            if v is None:
                continue
            y = y0 + 9 + j * 12
            out.append(f'<g><title>{esc(series[j])} · {esc(lab)}: {ci(v)}</title>'
                       f'<line x1="{X(v["lo"]):.1f}" x2="{X(v["hi"]):.1f}" y1="{y}" y2="{y}" '
                       f'stroke="{SERIES[j]}" stroke-width="2" stroke-linecap="round"/>'
                       f'<circle cx="{X(v["m"]):.1f}" cy="{y}" r="4.5" fill="{SERIES[j]}" class="dot"/>'
                       f'<rect x="{x0}" y="{y - 6}" width="{W - x0 - 20}" height="12" fill="transparent"/></g>')
    out.append("</svg>")
    return legend(series) + "".join(out) + '<p class="cnote">Dots: accuracy; bars: 95% bootstrap interval over concepts; grey tick: chance.</p>'


def line_chart(xs, series, title, xlabel="epoch", chance=None, tick_every=1):
    """series: [(name, [value or None per x])]; only every tick_every-th x is labelled on the axis."""
    W, H, l, r, t, b = 640, 270, 44, 150, 16, 40
    X = lambda i: l + i * (W - l - r) / max(1, len(xs) - 1)
    Y = lambda v: t + (1 - v) * (H - t - b)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
    for i, x in list(enumerate(xs))[::tick_every]:
        out.append(f'<text x="{X(i):.1f}" y="{H - 20}" class="tick" text-anchor="middle">{esc(x)}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.1f}" y="{H - 4}" class="tick" text-anchor="middle">{esc(xlabel)}</text>')
    if chance is not None:
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(chance):.1f}" y2="{Y(chance):.1f}" class="chance"/>'
                   f'<text x="{W - r + 6}" y="{Y(chance) + 4:.1f}" class="tick">chance {chance:.2f}</text>')
    ends = []
    for j, (name, vs) in enumerate(series):
        pts = [(X(i), Y(v), v, xs[i]) for i, v in enumerate(vs) if v is not None]
        if not pts:
            continue
        out.append(f'<polyline points="{" ".join(f"{a:.1f},{b_:.1f}" for a, b_, _, _ in pts)}" fill="none" '
                   f'stroke="{SERIES[j]}" stroke-width="2" stroke-linejoin="round"/>')
        for a, b_, v, x in pts:
            out.append(f'<circle cx="{a:.1f}" cy="{b_:.1f}" r="4" fill="{SERIES[j]}" class="dot">'
                       f'<title>{esc(name)}, {esc(xlabel)} {esc(x)}: {v:.2f}</title></circle>')
        ends.append([pts[-1][1], name, j])
    ends.sort()
    for k in range(1, len(ends)):                        # keep end labels apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 13)
    for y, name, j in ends:
        out.append(f'<text x="{W - r + 8}" y="{y + 4:.1f}" class="el">{esc(name)}</text>')
    out.append("</svg>")
    return legend([s[0] for s in series]) + "".join(out)


# ---------------------------------------------------------------- analyses

N, T, C = {}, {}, {}          # numbers, tables, charts for the placeholders


def put(key, res, signed=False):
    N[key] = ci(res, signed)
    N[key + ".m"] = "–" if res is None else f"{res['m']:+.2f}" if signed else f"{res['m']:.2f}"
    if res is not None and "chance" in res:
        N[key + ".chance"] = f"{res['chance']:.2f}"


DIR_PROBES = [("cloze-text", "name → property, teaching sentence"),
              ("ladder-text-2", "name → property, short answer"),
              ("describe-name", "words → name"),
              ("ladder-image-1", "picture → name, question"),
              ("ladder-image-2", "picture → name, short answer"),
              ("cloze-image", "picture → name, “What is this?”"),
              ("describe-one", "one property word → name")]
ARMS = [("T", "T · name first"), ("Tr", "Tr · name last"), ("V", "V · pictures")]
DIR_RUNS = {"T": ["teach2_t_s0", "teach2_t_s2"], "Tr": ["teach3_tr_s0", "teach3_tr_s1", "teach3_tr_s2"],
            "V": ["teach2_v_s2", "teach3_v_s0"]}


def direction(st):
    for cal in (False, True):
        tag = "cal" if cal else "raw"
        res = {(a, g): pooled(st, DIR_RUNS[a], g, cal) for a, _ in ARMS for g, _ in DIR_PROBES}
        for (a, g), v in res.items():
            put(f"dir.{tag}.{a}.{g}", v)
        T[f"dir_{tag}"] = table(["Probe", "Chance"] + [l for _, l in ARMS],
                                [[lab, f2(next((v["chance"] for (a, gg), v in res.items() if gg == g and v), None))]
                                 + [cell(res[(a, g)]) for a, _ in ARMS] for g, lab in DIR_PROBES], "num")
        diffs = []
        for g, lab in DIR_PROBES:
            if res[("Tr", g)] and res[("T", g)]:
                d = paired_diff(res[("Tr", g)], res[("T", g)])
                put(f"dir.{tag}.diff.{g}", d, signed=True)
                diffs.append([lab, ci(d, True)])
        T[f"dir_diff_{tag}"] = table(["Probe", "Tr − T (paired over concepts)"], diffs, "num")
        C[f"dir_{tag}"] = dot_chart([(lab, next((v["chance"] for (a, gg), v in res.items() if gg == g and v), None))
                                     for g, lab in DIR_PROBES], [l for _, l in ARMS],
                                    {(i, j): res[(a, g)] for i, (g, _) in enumerate(DIR_PROBES)
                                     for j, (a, _) in enumerate(ARMS)}, "Direction of storage")
    rows = []
    for a, lab in ARMS:
        for r in DIR_RUNS[a]:
            rows.append([f'<a href="run-{r}.html">{r}</a>', lab, RUN[r]["seed"]] +
                        [f"{f2(boot(final_items(st, r, g))['m'] if final_items(st, r, g) else None)} / "
                         f"{f2((boot(final_items(st, r, g), True) or {}).get('m'))}"
                         for g in ("ladder-image-1", "ladder-image-2", "describe-name", "cloze-text")])
    T["dir_per_run"] = table(["Run", "Arm", "Seed", "picture → name (question)", "picture → name (short)",
                              "words → name", "name → property"], rows, "num",
                             "Each cell: raw / calibrated accuracy (calibrated = score minus the untaught model's score per candidate).")
    # which property words did training link to the names?
    attrs = list(VALUES)
    res = {(a, x): pooled(st, DIR_RUNS[a], f"describe-one/{x}") for a in ("Tr", "V") for x in attrs}
    for (a, x), v in res.items():
        put(f"d1.{a}.{x}", v)
    T["describe_one"] = table(["Property word", "Tr · name last", "V · pictures"],
                              [[x.replace("_class", ""), cell(res[("Tr", x)]), cell(res[("V", x)])] for x in attrs], "num",
                              "One property in words, candidates restricted to names whose value differs (chance 0.12).")


LADDER_ARMS = [("T-nofmt", ["teach_t_s0"]), ("T", DIR_RUNS["T"]), ("Tr", DIR_RUNS["Tr"]),
               ("V-nofmt", ["teach_v_s0"]), ("V", DIR_RUNS["V"])]
RUNG = {1: "question → sentence", 2: "short answer", 3: "options listed", 4: "letters"}


def ladder(st):
    for side in ("text", "image"):
        res = {(a, k): pooled(st, rs, f"ladder-{side}-{k}") for a, rs in LADDER_ARMS for k in RUNG}
        for (a, k), v in res.items():
            put(f"lad.{side}.{a}.{k}", v)
        T[f"ladder_{side}"] = table(["Arm"] + [f"{k} · {v}" for k, v in RUNG.items()],
                                    [[a] + [cell(res[(a, k)]) for k in RUNG] for a, _ in LADDER_ARMS], "num")
        show = ["T", "Tr", "V"] if side == "text" else ["T", "Tr", "V"]
        ch = next(v["chance"] for v in res.values() if v)
        C[f"ladder_{side}"] = line_chart([f"{k}" for k in RUNG], [(a, [res[(a, k)]["m"] if res[(a, k)] else None for k in RUNG])
                                                                    for a in show],
                                         f"Extraction ladder, {side}", "rung", ch)


def practice(st):
    rows = []
    for r in RUNS:
        if r["id"] not in st.meta or r["model"] != "3B" or r["regime"] not in ("T", "V"):
            continue
        ep = last_epoch(st, r["id"])
        g = lambda grp, e: boot(st.cells.get((r["id"], "train", e, grp), []))
        mast = "mastery-text" if r["regime"] == "T" else "mastery-image"
        clz = "cloze-text" if r["regime"] == "T" else "cloze-image"
        rows.append([f'<a href="run-{r["id"]}.html">{r["id"]}</a>', r["regime"], r["protocol"], r["seed"],
                     f'{f2((g(mast, 0) or {}).get("m"))} → <b>{f2((g(mast, ep) or {}).get("m"))}</b>',
                     f'{f2((g(clz, 0) or {}).get("m"))} → <b>{f2((g(clz, ep) or {}).get("m"))}</b>',
                     f'{f2((g("anchor", 0) or {}).get("m"))} → {f2((g("anchor", ep) or {}).get("m"))}'])
    T["practice"] = table(["Run", "Taught by", "Format practice", "Seed", "Letter MC, mastery items",
                           "Likelihood probe, same modality", "Known-fact anchors"], rows, "num",
                          "Untaught (epoch 0) → after the last epoch. Mastery threshold was 0.95; no run reached it.")
    for reg, mast, clz, ids in (("T", "mastery-text", "cloze-text", ["teach_t_s0", "teach2_t_s0"]),
                                ("V", "mastery-image", "cloze-image", ["teach_v_s0", "teach3_v_s0"])):
        series = []
        for g, lab in ((mast, "letter MC"), (clz, "likelihood")):
            for i in ids:
                eps = sorted(e for s, e in st.srcs[i] if s == "train")
                series.append((f"{lab}, {RUN[i]['protocol']}",
                               [bm(st.cells.get((i, "train", e, g), [])) for e in eps]))
        C[f"practice_{reg}"] = line_chart([str(e) for e in range(len(series[0][1]))], series, f"{reg}: MC vs likelihood")
    for i in ("teach_t_s0", "teach2_t_s0", "teach_v_s0", "teach3_v_s0"):
        ep = last_epoch(st, i)
        for g in ("mastery-text", "mastery-image", "cloze-text", "cloze-image", "anchor"):
            put(f"run.{i}.{g}.0", boot(st.cells.get((i, "train", 0, g), [])))
            put(f"run.{i}.{g}.last", boot(st.cells.get((i, "train", ep, g), [])))
    # V -> T and T -> V transfer ratios on the likelihood probes (calibrated)
    for a, num, den in (("V", "cloze-text", "cloze-image"), ("T", "cloze-image", "cloze-text")):
        x, y = pooled(st, DIR_RUNS[a], num, True), pooled(st, DIR_RUNS[a], den, True)
        N[f"tau.{a}"] = f"{(x['m'] - x['chance']) / (y['m'] - y['chance']):.2f}"


COMPOSE = [("compose-hop1", "hop 1 · picture → name"), ("compose-hop2", "hop 2 · name → property"),
           ("compose", "both hops, silent"), ("compose-chain", "both hops, name written first")]
HIDE = ["color", "size_class", "legs", "habitat", "diet"]


def composition(st):
    s_runs = runs_of("S", st)
    N["comp.seeds"] = str(len(s_runs))
    N["comp.runs"] = ", ".join(s_runs)
    if not s_runs:
        return
    res = {g: pooled(st, s_runs, g) for g, _ in COMPOSE}
    res0 = {g: boot([it for r in s_runs for it in st.cells[(r, "train", 0, g)]]) for g, _ in COMPOSE}
    rescal = {g: pooled(st, s_runs, g, True) for g, _ in COMPOSE}
    for g, _ in COMPOSE:
        put(f"comp.{g}", res[g])
        put(f"comp.{g}.e0", res0[g])
        put(f"comp.{g}.cal", rescal[g])
    # sharing gap: written chain minus silent question, paired by creature
    if "compose-chain" in res and "compose" in res:
        put("comp.gap", paired_diff(res["compose-chain"], res["compose"]), True)
    T["compose"] = table(["Probe", "Chance", "Untaught", "Taught", "Taught, calibrated"],
                         [[lab, f2(res[g]["chance"]), cell(res0[g]), cell(res[g]), cell(rescal[g])] for g, lab in COMPOSE],
                         "num", f"Final epoch, pooled over {len(s_runs)} seed(s); 95% intervals over the 20 test concepts.")
    C["compose"] = dot_chart([(lab, res[g]["chance"]) for g, lab in COMPOSE], ["after teaching", "untaught"],
                             {**{(i, 0): res[g] for i, (g, _) in enumerate(COMPOSE)},
                              **{(i, 1): res0[g] for i, (g, _) in enumerate(COMPOSE)}}, "Two-hop composition")
    # by epoch (first S run with the most epochs)
    r0 = s_runs[0]
    eps = sorted(e for s, e in st.srcs[r0] if s == "train")
    C["compose_epochs"] = line_chart([str(e) for e in eps], [(lab, [bm(st.cells.get((r0, "train", e, g), [])) for e in eps])
                                                             for g, lab in COMPOSE], "Composition by epoch")
    # per attribute
    rows = []
    for a in HIDE:
        rr = {g: pooled(st, s_runs, f"{g}/{a}") for g, _ in COMPOSE}
        for g in rr:
            put(f"comp.{g}.{a}", rr[g])
        rows.append([a.replace("_class", ""), f2(rr["compose"]["chance"])] + [cell(rr[g]) for g, _ in COMPOSE])
    T["compose_attr"] = table(["Text-taught attribute", "Chance (property)"] + [l for _, l in COMPOSE], rows, "num")
    # silent answer when both hops are right; what the silent answers pick
    both = tot = ok = chain_ok = 0
    picks = defaultdict(Counter)
    for r in s_runs:
        ep = last_epoch(st, r)
        ex = st.extra[(r, "train", ep)]
        by = {x["id"]: x for x in ex}
        for x in ex:
            if x["family"] != "compose":
                continue
            pid = x["id"][3:]
            h1, h2, ch = by.get("H1/" + pid), by.get("H2/" + pid.rsplit("/", 1)[0]), by.get("CH/" + pid)
            tot += 1
            picks[x["level"]][VALUES[x["level"]][x["pred"]]] += 1
            if h1 and h2 and h1["correct"] and h2["correct"]:
                both += 1
                ok += x["correct"]
                chain_ok += bool(ch and ch["correct"])
    N["comp.both"], N["comp.both_ok"], N["comp.both_chain"], N["comp.total"] = str(both), str(ok), str(chain_ok), str(tot)
    T["compose_picks"] = table(["Hidden attribute", "Most common silent answers (count)"],
                               [[a.replace("_class", ""), ", ".join(f"{v} ({n})" for v, n in picks[a].most_common(4))]
                                for a in HIDE], "",
                               "What the model answers for the hidden, text-taught attribute when asked about the picture directly.")
    for a in HIDE:
        v, n = picks[a].most_common(1)[0]
        N[f"comp.pick.{a}"] = f"{v} ({n} of {sum(picks[a].values())})"
    tr = trace(st, s_runs)
    put("trace.compose", tr["compose"][0])
    put("trace.hop2", tr["compose-hop2"][0])
    N["trace.ratio"] = f"{100 * tr['compose'][0]['m'] / tr['compose-hop2'][0]['m']:.0f}%"
    N["trace.per_seed"] = "; ".join(f"seed {st.meta[r]['seed']} {ci(trace(st, [r])['compose'][0])}" for r in s_runs)
    T["trace"] = table(["Attribute", "Silent picture → property", "Name in words → property"],
                       [[a.replace("_class", ""), ci(tr["compose"][1].get(a)), ci(tr["compose-hop2"][1].get(a))]
                        for a in HIDE] + [["<b>all</b>", f"<b>{ci(tr['compose'][0])}</b>", f"<b>{ci(tr['compose-hop2'][0])}</b>"]],
                       "num", "Concept-specific rise in the taught value's log-probability (nats), untaught → last epoch, "
                              "relative to the same value on other creatures' probes.")
    # MC mastery under S
    N["comp.mastery"] = "; ".join(f"seed {st.meta[r]['seed']}: " + " / ".join(f"{e['mastery']:.2f}" for e in
                                  st.meta[r]["record"]["epochs"][1:]) for r in s_runs)


def trace(st, s_runs):
    """Does the taught value leave any trace in the silent answer's scores?

    For each probe, the change in log-probability of the taught value from the
    untaught model to the last epoch, minus the mean change of that same value on
    other creatures' probes of the same attribute: a concept-specific boost, in
    nats. compose-hop2 (name given in words) is the yardstick."""
    out = {}
    for fam in ("compose", "compose-hop2"):
        spec = defaultdict(list)
        by_attr = defaultdict(lambda: defaultdict(list))
        for r in s_runs:
            R0 = {x["id"]: x for x in st.extra[(r, "train", 0)] if x["family"] == fam}
            R1 = {x["id"]: x for x in st.extra[(r, "train", last_epoch(st, r))] if x["family"] == fam}
            d = {i: np.array(R1[i]["scores"]) - np.array(R0[i]["scores"]) for i in R1 if i in R0}
            for i, d_i in d.items():
                x = R1[i]
                others = [d[j][x["answer"]] for j in d if R1[j]["level"] == x["level"] and
                          R1[j]["concept"] != x["concept"]]
                b = float(d_i[x["answer"]] - np.mean(others))
                spec[x["concept"]].append(b)                  # cluster by creature, seeds pooled
                by_attr[x["level"]][x["concept"]].append(b)
        out[fam] = (mean_boot(spec), {a: mean_boot(v) for a, v in by_attr.items()})
    return out


def mean_boot(vals):
    """vals: {cluster: [x]} -> mean with a cluster bootstrap."""
    cs = list(vals)
    s = np.array([sum(vals[c]) for c in cs])
    n = np.array([len(vals[c]) for c in cs], float)
    pick = RNG.integers(0, len(cs), (B, len(cs)))
    bs = np.sort(s[pick].sum(1) / n[pick].sum(1))
    return {"m": s.sum() / n.sum(), "lo": bs[int(0.025 * B)], "hi": bs[int(0.975 * B) - 1]}


def confusions(st):
    """When a taught model names a creature wrongly, is the wrong name a look-alike?

    For each wrong answer: attributes the chosen creature shares with the true one,
    minus the mean shared by the probe's other wrong candidates. Positive means the
    errors follow attribute similarity, i.e. naming works by matching properties."""
    from seen_never_said.eval.describe import describe_probes
    from seen_never_said.eval.ladder import ladder_probes
    from seen_never_said.teach import data as tdata
    root = ROOT / "data" / "tier0_pilot"
    concepts = tdata.load_concepts(root)
    held = tdata.heldout_images(root)
    attrs = {c["name"]: c["attrs"] for c in concepts}
    cache = {}

    def names(seed):
        if seed not in cache:
            ps = ladder_probes(concepts, held, seed) + describe_probes(concepts, seed)
            cache[seed] = {p["id"]: [x[len("This is a "):-1] for x in p["candidates"]] for p in ps}
        return cache[seed]

    shared = lambda a, b: sum(attrs[a][k] == attrs[b][k] for k in VALUES)
    rows = []
    for fam, flab in (("ladder-image-1", "picture → name"), ("describe-name", "words → name")):
        for arm, alab in ARMS:
            diff, per_attr, n_err, n = defaultdict(list), defaultdict(list), 0, 0
            for r in DIR_RUNS[arm]:
                src = next((f for f in reversed(SRC_ORDER[1:]) if (r, f, "final") in st.extra and
                            any(x["family"] == fam for x in st.extra[(r, f, "final")])), None)
                if src is None:
                    continue
                for x in st.extra[(r, src, "final")]:
                    if x["family"] != fam:
                        continue
                    n += 1
                    if x["correct"]:
                        continue
                    n_err += 1
                    cand = names(RUN[r]["seed"])[x["id"]]
                    true, chosen = cand[x["answer"]], cand[x["pred"]]
                    wrong = [c for c in cand if c != true]
                    diff[f"{r}/{x['concept']}"].append(shared(chosen, true) - np.mean([shared(c, true) for c in wrong]))
                    for k in VALUES:
                        per_attr[k].append((attrs[chosen][k] == attrs[true][k]) -
                                           np.mean([attrs[c][k] == attrs[true][k] for c in wrong]))
            if n_err < 5:
                rows.append([flab, alab, f"{n_err}/{n}", "–", "–"])
                continue
            d = mean_boot(diff)
            put(f"conf.{fam}.{arm}", d, signed=True)
            top = sorted(((np.mean(v), k) for k, v in per_attr.items()), reverse=True)[:3]
            rows.append([flab, alab, f"{n_err}/{n}", ci(d, True),
                         ", ".join(f"{k.replace('_class', '')} {m:+.2f}" for m, k in top)])
    T["confusions"] = table(["Probe", "Taught by", "Errors", "Extra attributes shared by the wrong name",
                             "Attributes most over-shared"], rows, "num",
                            "Shared attributes (of 8) between the wrongly chosen creature and the true one, minus the "
                            "mean over the probe's other wrong names; 95% interval over creatures. Per attribute: rate "
                            "of sharing it, minus the rate expected from the wrong names offered.")


def mechanism():
    """Layer-wise probes (seen_never_said/teach/layers_job.py). Two jobs: layers_s_s0 (cut off at the 12-hour limit,
    taught model only, with the leave-one-creature-out probes) and layers_s_s0_fast (back-patching first, then
    the taught and untaught curves without the slow probes)."""
    p_old = OUT / "layers_s_s0" / "layers.json"
    p_new = next((p for p in (OUT / "layers_s_s0_fast2" / "layers.json", OUT / "layers_s_s0_fast" / "layers.json")
                  if p.exists()), OUT / "layers_s_s0_fast" / "layers.json")
    old = json.loads(p_old.read_text()) if p_old.exists() else None
    new = json.loads(p_new.read_text()) if p_new.exists() else None
    if not (old or new):
        T["layers"] = '<p class="pending">Waiting for the layer-wise job.</p>'
        N["layers.status"] = "pending"
        C["layers"] = T["layers_value"] = ""
        backpatch_report()
        return
    A = (new or {}).get("adapter") or old["adapter"]
    Bz = (new or {}).get("base") or (old or {}).get("base")
    N["layers.status"] = ("available" if Bz else
                          "partial (taught model only; the job hit the 12-hour limit; a faster rerun is running)")
    L = len(A["name/pic-name"])
    xs = [str(l) for l in range(L)]
    attrs = [a for a in HIDE if f"lens/name-prop/{a}" in A]
    mean_lens = lambda R, s: [float(np.mean([R[f"lens/{s}/{a}"][0][l] for a in attrs])) for l in range(L)]
    series = [("name from picture", A["name/pic-name"]),
              ("value, name in words (lens)", mean_lens(A, "name-prop")),
              ("value, picture (lens)", mean_lens(A, "pic-prop"))]
    if Bz:      # four colours: drop the picture lens, which stays at the majority class
        series = series[:2] + [("name from picture, untaught", Bz["name/pic-name"]),
                               ("value, name in words (lens), untaught", mean_lens(Bz, "name-prop"))]
    C["layers"] = line_chart(xs, series, "Layer-wise decoding", "layer", tick_every=4)

    # first layer where the lens beats the majority class by 0.2: a lens that gives every creature the same
    # value (layer 0, and many early layers) scores the majority class, well above 1/K chance
    def first(R):
        return {a: next((l for l in range(1, L)
                         if R[f"lens/name-prop/{a}"][0][l] >= A[f"value/majority/{a}"] + 0.2), None) for a in attrs}

    fmt = lambda d: ", ".join(f"{a.replace('_class', '')} {v if v is not None else 'never'}" for a, v in d.items())
    N["layers.L_prop"] = fmt(first(A))
    fa = first(A)
    nm = lambda a: {"color": "colour", "size_class": "size"}.get(a, a)  # noqa: E731
    N["layers.L_prop.read"] = " and ".join(f"{nm(a)} from layer {v}" for a, v in fa.items() if v is not None)
    never = [nm(a) for a, v in fa.items() if v is None]
    N["layers.L_prop.never"] = (", ".join(never[:-1]) + " and " + never[-1]) if len(never) > 1 else "".join(never)
    N["layers.n"] = str(L - 1)
    N["layers.name1"] = f2(A["name/pic-name"][1])
    N["layers.name_last"] = f2(A["name/pic-name"][-1])
    N["layers.name_chance"] = f2(A["name/chance"])
    txt = (f'<p>Name read out from held-out pictures by a linear probe: <b>{N["layers.name1"]}</b> already '
           f'after layer 1, <b>{N["layers.name_last"]}</b> at the last of {L - 1} layers (chance '
           f'{N["layers.name_chance"]}). First layer where the logit lens beats the most common value by 0.2, '
           f'name in words: {N["layers.L_prop"]}.')
    if Bz:
        N["layers.base_name1"] = f2(Bz["name/pic-name"][1])
        N["layers.base_name_last"] = f2(Bz["name/pic-name"][-1])
        N["layers.base_L_prop"] = fmt(first(Bz))
        txt += (f' Untaught model: {N["layers.base_name1"]} after layer 1, {N["layers.base_name_last"]} at the last; '
                f'lens, name in words: {N["layers.base_L_prop"]}.')
    T["layers"] = txt + "</p>"
    src = old if old and f"value-loco/name-prop/{attrs[0]}" in old["adapter"] else None
    if src:
        Ao, Bo = src["adapter"], src.get("base")
        rows = [[a.replace("_class", ""), f2(Ao[f"value/majority/{a}"]), f2(max(Ao[f"value-loco/name-prop/{a}"])),
                 f2(max(Bo[f"value-loco/name-prop/{a}"])) if Bo else "–",
                 f2(max(Ao[f"value-transfer/{a}"])), f2(max(Ao[f"value-loco/pic-prop/{a}"]))]
                for a in HIDE if f"value/majority/{a}" in Ao]
        T["layers_value"] = table(["Attribute", "Majority class", "Taught, name in words", "Untaught, name in words",
                                   "Words → picture transfer", "Taught, picture"], rows, "num",
                                  "Best layer's accuracy of a value probe trained on other creatures (leave one creature "
                                  "out). Best of 36 layers, so values near the majority class are noise.")
    else:
        T["layers_value"] = ""
    backpatch_report()


def _lp(r):
    """Log-probability of the taught value, normalised over the candidate values."""
    s = np.array(r["scores"], float)
    return float(s[r["answer"]] - (s.max() + np.log(np.exp(s - s.max()).sum())))


def _cdiff(rows_a, rows_b, f=_lp):
    """Mean over concepts of f(a) - f(b), paired by probe id, with a concept bootstrap.
    rows_a may hold rows from several patches; each is paired with rows_b's probe of the same id."""
    b = {r["id"]: f(r) for r in rows_b}
    by = defaultdict(list)
    for r in rows_a:
        if r["id"] in b:
            by[r["concept"]].append(f(r) - b[r["id"]])
    v = np.array([np.mean(x) for x in by.values()])
    bs = np.sort(v[RNG.integers(0, len(v), (B, len(v)))].mean(1))
    return {"m": float(v.mean()), "lo": bs[int(0.025 * B)], "hi": bs[int(0.975 * B) - 1]}


def backpatch_report():
    # the first fast run crashed (patch vectors on the wrong GPU) after 18 pairs; fast2 ran the rest.
    # Each run scores the unpatched probes; keep each (patch, i, j) from the first file that has it.
    bps = [d / "backpatch.jsonl" for d in (OUT / "layers_s_s0_fast", OUT / "layers_s_s0_fast2", OUT / "layers_s_s0")
           if (d / "backpatch.jsonl").exists()]
    C["backpatch"] = ""
    if not bps:
        T["backpatch"] = '<p class="pending">Back-patching has not finished yet.</p>'
        N["bp.status"] = "pending"
        return
    grp = defaultdict(list)
    for bp in bps:
        g = defaultdict(list)
        for r in lines(bp):
            g[(r["patch"], r["i"], r["j"])].append(r)
        for k, v in g.items():
            grp.setdefault(k, v)
    none = grp[("none", None, None)]
    acc = lambda rows: boot([(r["concept"], r["correct"], None, 0, False) for r in rows])
    same = {(i, j): v for (t, i, j), v in grp.items() if t == "same"}
    N["bp.status"] = "available"
    N["bp.none"] = ci(acc(none))
    N["bp.n_pairs"] = str(len(same))
    if not same:
        T["backpatch"] = f'<p>Unpatched silent accuracy {N["bp.none"]}; no patched pair finished.</p>'
        return
    I, J = sorted({k[0] for k in same}), sorted({k[1] for k in same})
    grid = [[f"<b>{i}</b>"] + [f2(acc(same[(i, j)])["m"]) if (i, j) in same else "" for j in J] for i in I]
    C["backpatch"] = table(["into ↓ / from →"] + [str(j) for j in J], grid, "num",
                           f"Silent accuracy with the state after layer j copied into layer i (no patch: "
                           f"{f2(acc(none)['m'])}).")
    rows = []
    for (i, j), v in sorted(same.items(), key=lambda kv: -acc(kv[1])["m"]):
        ctrl = grp.get(("other", i, j))
        rows.append([i, j, cell(acc(v)), ci(paired_diff(acc(v), acc(none)), True), ci(_cdiff(v, none), True),
                     ci(paired_diff(acc(ctrl), acc(none)), True) if ctrl else "–",
                     ci(_cdiff(ctrl, none), True) if ctrl else "–"])
    T["backpatch"] = table(["Into layer i", "From layer j", "Silent accuracy", "Δ accuracy", "Δ log p(taught)",
                            "Control: Δ accuracy", "Control: Δ log p(taught)"], rows, "num",
                           f"No patch: {N['bp.none']}. Δ: patched minus unpatched, paired by probe, 95% interval over "
                           "the 20 creatures. Control: the same layers, but the state copied from another creature's "
                           "probe (same attribute and picture slot). Δ log p(taught): the taught value's "
                           "log-probability, normalised over the candidate values, in nats. Sorted by accuracy.")
    (bi, bj), bv = max(same.items(), key=lambda kv: acc(kv[1])["m"])
    N["bp.best"] = f"layer {bj} into layer {bi}"
    N["bp.best_acc"] = ci(acc(bv))
    N["bp.best_diff"] = ci(paired_diff(acc(bv), acc(none)), True)
    N["bp.best_dlp"] = ci(_cdiff(bv, none), True)
    ctrl = grp.get(("other", bi, bj))
    N["bp.best_ctrl"] = ci(paired_diff(acc(ctrl), acc(none)), True) if ctrl else "–"
    N["bp.mean_dlp"] = ci(_cdiff([r for v in same.values() for r in v], none), True)
    da = [paired_diff(acc(v), acc(none))["m"] for v in same.values()]
    dl = [_cdiff(v, none)["m"] for v in same.values()]
    N["bp.dacc_range"] = f"{min(da):+.2f} to {max(da):+.2f}"
    N["bp.dlp_range"] = f"{min(dl):+.2f} to {max(dl):+.2f}"
    N["bp.n_total"] = "26"


def scale(st):
    pairs = [("teach2_t_s0", "3B"), ("teach7b_t_s0", "7B")]
    rows, series = [], []
    for g, lab in (("mastery-text", "letter MC recall"), ("cloze-text", "name → property"),
                   ("cloze-image", "picture → name"), ("anchor", "known-fact anchors")):
        row = [lab]
        for rid, model in pairs:
            if rid not in st.meta:
                row.append("pending")
                continue
            r0, r3 = boot(st.cells[(rid, "train", 0, g)]), boot(st.cells[(rid, "train", 3, g)])
            put(f"scale.{model}.{g}.0", r0)
            put(f"scale.{model}.{g}.3", r3)
            row.append(f"{f2(r0['m'])} → <b>{f2(r3['m'])}</b>")
        rows.append(row)
    T["scale_t"] = table(["Probe (text, name first, seed 0)", "3B: untaught → epoch 3", "7B: untaught → epoch 3"], rows, "num")
    for rid, model in pairs:
        if rid in st.meta:
            series.append((f"{model} letter MC", [bm(st.cells.get((rid, "train", e, "mastery-text"), [])) for e in range(4)]))
            series.append((f"{model} likelihood", [bm(st.cells.get((rid, "train", e, "cloze-text"), [])) for e in range(4)]))
    C["scale_t"] = line_chart(["0", "1", "2", "3"], series, "3B vs 7B")
    if "teach7b_tr_s0" in st.meta:
        rows = []
        for g, lab in (("cloze-text", "name → property"), ("cloze-image", "picture → name, “What is this?”"),
                       ("mastery-text", "letter MC recall"), ("anchor", "known-fact anchors")):
            row = [lab]
            for a, rid in (("T", "teach7b_t_s0"), ("Tr", "teach7b_tr_s0")):
                r = boot(st.cells[(rid, "train", 3, g)])
                put(f"scale.train.{a}.{g}", r)
                row.append(cell(r))
            rows.append(row)
        T["scale_tr"] = table(["Probe (7B, epoch 3, seed 0)", "T · name first", "Tr · name last"], rows, "num")
    else:
        T["scale_tr"] = '<p class="pending">Waiting for <code>snsaid-teach7b-tr-s0</code>.</p>'
    has7 = "teach7b_tr_s0" in st.meta and any(k[0] == "teach7b_tr_s0" and k[1] == "eval7b_dir1" for k in st.cells)
    N["scale.dir_status"] = "available" if has7 else "pending"
    if has7:
        res = {(a, g): pooled(st, [rid], g) for a, rid in (("T", "teach7b_t_s0"), ("Tr", "teach7b_tr_s0"))
               for g, _ in DIR_PROBES}
        rows = []
        for g, lab in DIR_PROBES:
            row = [lab, cell(res[("T", g)]), cell(res[("Tr", g)])]
            for kind, c in (("raw", False), ("cal", True)):
                d = paired_diff(pooled(st, ["teach7b_tr_s0"], g, c), pooled(st, ["teach7b_t_s0"], g, c))
                put(f"scale.diff.{kind}.{g}", d, True)
                row.append(ci(d, True))
            rows.append(row)
        T["scale_dir"] = table(["Probe", "7B T · name first", "7B Tr · name last", "Tr − T", "Tr − T, calibrated"],
                               rows, "num", "Seed 0 each; 95% intervals over the 20 test creatures.")
        for (a, g), v in res.items():
            put(f"scale.dir.{a}.{g}", v)
    else:
        T["scale_dir"] = '<p class="pending">Waiting for <code>snsaid-eval7b-dir1</code>, next in the queue. It runs the picture → name question probe that decides the transfer half of the test.</p>'


def family(st):
    """Second model family (SmolVLM2-2.2B, 3 seeds T/Tr, 1 seed V and S), 7B S, mirror probes on the 3 S adapters,
    corrected back-patching / text->picture patching (patch2). Keys smol.*, s7.*, mirror.*, p2.*."""
    arms = {"T": ["smol_t_s0", "smol_t_s1", "smol_t_s2"], "Tr": ["smol_tr_s0", "smol_tr_s1", "smol_tr_s2"],
            "V": ["smol_v_s0"]}
    arms = {a: [r for r in rs if r in st.meta] for a, rs in arms.items()}
    if all(arms.values()):
        for a, rs in arms.items():
            N[f"smol.seeds.{a}"] = str(len(rs))
            for g, _ in DIR_PROBES:
                put(f"smol.dir.{a}.{g}", pooled(st, rs, g))
        for g, _ in DIR_PROBES:
            for kind, c in (("raw", False), ("cal", True)):
                put(f"smol.diff.{kind}.{g}", paired_diff(pooled(st, arms["Tr"], g, c), pooled(st, arms["T"], g, c)), True)
        base = [it for it in final_items(st, "base-SmolVLM", "ladder-image-1")] if "base-SmolVLM" in st.srcs else []
        if base:
            put("smol.dir.base.ladder-image-1", boot(base))
    for tag, rid in (("smol", "smol_s_s0"), ("s7", "teach7b_s_s0")):
        if rid in st.meta:
            for g in ("compose", "compose-hop1", "compose-hop2", "compose-chain"):
                put(f"{tag}.S.{g}", pooled(st, [rid], g))
                e0 = [it for it in st.cells.get((rid, "train", 0, g), [])] if hasattr(st, "cells") else []
    if "smol_t_s0" in st.meta and "smol_v_s0" in st.meta:
        put("smol.ctrl.compose", pooled(st, ["smol_t_s0", "smol_v_s0"], "compose"))    # adapters never taught S
    # untaught level of the silent question for SmolVLM and 7B, from the epoch-0 rows of the training eval
    for tag, rid in (("smol", "smol_s_s0"), ("s7", "teach7b_s_s0")):
        p = OUT / rid / "eval.jsonl"
        if p.exists():
            rows = [x for x in lines(p) if x["epoch"] == 0 and x.get("family") == "compose"]
            put(f"{tag}.S.compose.e0", boot([(x["concept"], bool(x["correct"]), None, x.get("chance", 0), False)
                                             for x in rows]))
    # mirror probes (eval/mirror.py) on the untaught model and the 3 S adapters
    p = OUT / "mirror_s" / "mirror.jsonl"
    if p.exists():
        rows = list(lines(p))
        for who, sel in (("base", lambda r: r.startswith("base")), ("S", lambda r: r.startswith("S-"))):
            for g in ("words-hop1", "words-silent", "words-chain", "compose", "compose-hop1", "compose-hop2",
                      "compose-chain", "fact-name", "verify-pic", "verify-words", "verify-name", "silent-named"):
                its = [(x["concept"], bool(x["correct"]), None, x.get("chance", 0), False) for x in rows
                       if sel(x.get("run", "")) and x.get("family") == g and x.get("correct") is not None]
                if its:
                    put(f"mirror.{who}.{g}", boot(its))
        N["mirror.seeds"] = str(len({x["run"] for x in rows if x.get("run", "").startswith("S-")}))
    # relabel overwrite (teach/relabel_job.py): share of answers that follow the NEW name / the OLD one,
    # before (epoch 0) and after renaming half the test creatures from pictures; pooled over the S seeds
    p = OUT / "relabel_s" / "relabel.jsonl"
    if p.exists():
        rows = list(lines(p))
        N["relabel.seeds"] = str(len({x["run"] for x in rows}))

        def follows(x, which):
            pr, fam = x["pred"], x["family"]
            nm = x[f"{which}_name"]
            if fam.endswith("hop1"):
                return x["names"][pr] == nm
            val = x["new_value" if which == "new" else "own_value"]
            if fam.endswith("chain"):
                return x["pairs"][pr][0] == nm and x["pairs"][pr][1] == val
            return pr == val
        for ep in sorted({x["epoch"] for x in rows}):
            e = "after" if ep else "before"
            for fam in sorted({x["family"] for x in rows}):
                for which in ("new", "old"):
                    its = [(x["concept"], follows(x, which), None, 0, False) for x in rows
                           if x["epoch"] == ep and x["family"] == fam]
                    put(f"relabel.{e}.{fam}.{which}", boot(its))
    # free generation (teach/gen_job.py), rescored with the fixed value parser (gen_rescored.jsonl)
    p = OUT / "gen_s" / "gen_rescored.jsonl"
    if not p.exists():
        p = OUT / "gen_s" / "gen.jsonl"
    if p.exists():
        rows = list(lines(p))
        for run in sorted({x["run"] for x in rows}):
            tag = run.split("-s")[0] if run != "base" else "base"
            for fam in ("gen-silent", "gen-chain", "gen-hop1", "gen-hop2"):
                rs = [x for x in rows if x["run"] == run and x["family"] == fam]
                for k in ("name_ok", "value_ok", "value_none"):
                    its = [(x["concept"], bool(x[k]), None, 0, False) for x in rs if k in x]
                    if its:
                        put(f"gen.{tag}.{fam}.{k}", boot(its))
            rs = [x for x in rows if x["run"] == run and x["family"] == "gen-silent" and not x.get("value_none", True)]
            if rs:
                put(f"gen.{tag}.gen-silent.value_given", boot([(x["concept"], bool(x["value_ok"]), None, 0, False) for x in rs]))
                N[f"gen.{tag}.gen-silent.n_value"] = str(len(rs))
    # self-rehearsal (teach/consolidate_job.py): the S adapter labels its own teaching pictures through the
    # written chain, then is trained on the silent question -> its own label. Cells on held-out pictures:
    # rr rehearsed creature x rehearsed attribute, rn rehearsed creature x other attribute, n other creatures.
    p = OUT / "consolidate_s" / "consolidate.jsonl"
    if p.exists():
        rows = list(lines(p))
        N["cons.seeds"] = str(len({x["run"] for x in rows}))
        for e in sorted({x["epoch"] for x in rows}):
            for fam in ("compose", "compose-hop1", "compose-hop2", "compose-chain"):
                for c in ("rr", "rn", "n", "all"):
                    its = [(x["concept"], x["pred"] == x["answer"], None,
                            1 / max(x.get("n_options", 1), 1), False) for x in rows
                           if x["epoch"] == e and x["family"] == fam and (c == "all" or x["cell"] == c)]
                    if its:
                        put(f"cons.{e}.{fam}.{c}", boot(its))
        last = max(x["epoch"] for x in rows)

        def cell_items(e, c):
            return boot([(x["concept"], x["pred"] == x["answer"], None, 1 / x["n_options"], False)
                         for x in rows if x["epoch"] == e and x["family"] == "compose" and x["cell"] == c])
        for c in ("rr", "rn", "n"):
            put(f"cons.gain.{c}", paired_diff(cell_items(last, c), cell_items(0, c)), True)
        # per seed (appendix table) and per attribute in the transfer cell
        for sd in sorted({x["seed"] for x in rows}):
            for e in (0, last):
                for fam in ("compose", "compose-chain"):
                    for c in ("rr", "rn", "n"):
                        its = [(x["concept"], x["pred"] == x["answer"], None, 1 / x["n_options"], False)
                               for x in rows if x["seed"] == sd and x["epoch"] == e and x["family"] == fam
                               and x["cell"] == c]
                        if its:
                            put(f"cons.s{sd}.{e}.{fam}.{c}", boot(its))
        for a in sorted({x["attrs"][0] for x in rows if x["family"] == "compose" and x.get("attrs")}):
            for e in (0, last):
                for c in ("rr", "rn"):
                    its = [(x["concept"], x["pred"] == x["answer"], None, 1 / x["n_options"], False) for x in rows
                           if x["epoch"] == e and x["family"] == "compose" and x["cell"] == c and x["attrs"][0] == a]
                    if its:
                        put(f"cons.attr.{a}.{e}.{c}", boot(its))
        s = OUT / "consolidate_s" / "consolidate_summary.json"
        if s.exists():
            lab = json.loads(s.read_text())
            for tag, v in lab.items():
                N[f"cons.s{tag.split('-s')[-1]}.label_ok"] = f'{v["labels"]["label_ok"]:.2f}'
            N["cons.label_ok"] = f'{np.mean([v["labels"]["label_ok"] for v in lab.values()]):.2f}'
            N["cons.name_ok"] = f'{np.mean([v["labels"]["name_ok"] for v in lab.values()]):.2f}'
            N["cons.n_creatures"] = str(len(next(iter(lab.values()))["creatures"]))
            N["cons.n_pics"] = str(next(iter(lab.values()))["labels"]["pictures"] //
                                   len(next(iter(lab.values()))["creatures"]))
    # label distillation (teach/distill_job.py): the clean picture is pulled toward the same picture with its name
    # printed on it, for half the test creatures (cell d; n = the other half); no fact is trained.
    # lm: LoRA on the language model (3 seeds); vis: LoRA on the language model and the vision encoder.
    dist_rows = {}
    # visnd: the vis arm with the distillation loss switched off (DW=0): same replay, same vision LoRA.
    for arm, dirs in (("lm", ["distill_s_s0", "distill_s_s1", "distill_s_s2"]),
                      ("vis", ["distill_vis_s0", "distill_vis_s1", "distill_vis_s2"]),
                      ("visnd", ["distill_vis_nodistil_s0"])):
        rows = [x for d in dirs if (OUT / d / "distill.jsonl").exists() for x in lines(OUT / d / "distill.jsonl")]
        if not rows:
            continue
        N[f"dist.{arm}.seeds"] = str(len({x["seed"] for x in rows}))
        eps = sorted({x["epoch"] for x in rows})
        N[f"dist.{arm}.epochs"] = str(eps[-1])

        def ditems(e, fam, c, sd=None):
            return [(x["concept"], x["pred"] == x["answer"], None, 1 / max(x.get("n_options", 1), 1), False)
                    for x in rows if x["epoch"] == e and x["family"] == fam and (c == "all" or x["cell"] == c)
                    and (sd is None or x["seed"] == sd)]
        for e in eps:
            for fam in ("compose", "compose-hop1", "compose-hop2", "compose-chain", "label-none", "label-own"):
                for c in ("d", "n", "all"):
                    its = ditems(e, fam, c)
                    if its:
                        put(f"dist.{arm}.{e}.{fam}.{c}", boot(its))
        for c in ("d", "n"):
            put(f"dist.{arm}.gain.{c}", paired_diff(boot(ditems(eps[-1], "compose", c)),
                                                    boot(ditems(0, "compose", c))), True)
        for sd in sorted({x["seed"] for x in rows}):
            for e in eps:
                for c in ("d", "n"):
                    put(f"dist.{arm}.s{sd}.{e}.compose.{c}", boot(ditems(e, "compose", c, sd)))
                if ditems(e, "label-own", "all", sd):
                    put(f"dist.{arm}.s{sd}.{e}.label-own.all", boot(ditems(e, "label-own", "all", sd)))
        # creatures that improve, counted per (seed, creature): each seed distils a different half
        up = tot = 0
        for sd in sorted({x["seed"] for x in rows}):
            b1, b0 = boot(ditems(eps[-1], "compose", "d", sd))["by"], boot(ditems(0, "compose", "d", sd))["by"]
            up += sum(b1[k][0] > b0[k][0] for k in b1 if k in b0)
            tot += len(b1)
        N[f"dist.{arm}.up"], N[f"dist.{arm}.ncre"] = str(up), str(tot)
        sums = [json.loads((OUT / d / "distill_summary.json").read_text()) for d in dirs
                if (OUT / d / "distill_summary.json").exists()]
        runs = [v for s in sums for v in s.values()]
        N[f"dist.{arm}.loss0"] = f'{np.mean([v["initial_distance"] for v in runs]):.2f}'
        N[f"dist.{arm}.loss1"] = f'{np.mean([v["train"][-1]["distil"] for v in runs]):.2f}'
        N["dist.pics"] = str(runs[0]["pictures"] // len(runs[0]["creatures"]))
        N["dist.ncreatures"] = str(len(runs[0]["creatures"]))
        # name lens: first layer at which the held-out picture's name is read out at >= 0.5 (min/max over seeds)
        for e in ("0", str(eps[-1])):
            for c in ("d", "n"):
                fs = [v["namelens"][e][c]["first_pic_0.5"] for v in runs
                      if v.get("namelens") and e in v["namelens"] and c in v["namelens"][e]]
                if fs:
                    N[f"dist.{arm}.lens.{e}.{c}"] = str(min(fs)) if min(fs) == max(fs) else f"{min(fs)}--{max(fs)}"
        dist_rows[arm] = (rows, ditems, eps)
    # distillation minus its no-distillation control: per-creature gain, same seed and creatures
    if "vis" in dist_rows and "visnd" in dist_rows:
        (rv, _, ev), (rn, _, en) = dist_rows["vis"], dist_rows["visnd"]
        sds = {x["seed"] for x in rn}

        def gain(rows, e1, sd_ok):
            acc = defaultdict(lambda: [[], []])
            for x in rows:
                if x["family"] == "compose" and x["cell"] == "d" and x["seed"] in sd_ok and x["epoch"] in (0, e1):
                    acc[x["concept"]][x["epoch"] == e1].append(x["pred"] == x["answer"])
            return {k: np.mean(v[1]) - np.mean(v[0]) for k, v in acc.items() if v[0] and v[1]}
        gv, gn = gain(rv, ev[-1], sds), gain(rn, en[-1], sds)
        ks = sorted(set(gv) & set(gn))
        d = np.array([gv[k] - gn[k] for k in ks])
        rng = np.random.default_rng(0)
        bs = np.sort([rng.choice(d, len(d)).mean() for _ in range(10000)])
        N["dist.vis.minus_nd"] = f"{d.mean():+.2f} [{bs[250]:+.2f}, {bs[9750]:+.2f}]"
    # patch2: copy the states of the named text question into the silent picture prompt at layer L
    p = OUT / "patch2_s_s0" / "patch2.jsonl"
    if p.exists():
        rows = list(lines(p))
        for L in sorted({x["i"] for x in rows if x["kind"] == "text-own"}):
            own = [(x["concept"], bool(x["correct"]), None, 0, False) for x in rows if x["kind"] == "text-own" and x["i"] == L]
            oth = [(x["concept"], x["pred"] == x["other_answer"], None, 0, False) for x in rows
                   if x["kind"] == "text-other" and x["i"] == L]
            put(f"p2.own.{L}", boot(own))
            put(f"p2.other.{L}", boot(oth))
        back = [x for x in rows if x["kind"] == "back"]
        accs = defaultdict(list)
        for x in back:
            accs[(x["i"], x["j"])].append(bool(x["correct"]))
        best = max(accs.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        N["p2.back.n_pairs"] = str(len(accs))
        N["p2.back.best"] = f"{sum(best[1]) / len(best[1]):.2f}"
        N["p2.back.best_pair"] = f"layer {best[0][0]} into layer {best[0][1]}"
        put("p2.none", boot([(x["concept"], bool(x["correct"]), None, 0, False) for x in rows if x["kind"] == "none"]))


def audits():
    out = []
    for d in sorted(OUT.iterdir()):
        p = d / "readout_summary.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        keys = [k for k in ("regime", "attr", "status", "n", "acc", "pred_true", "pred_neutral", "pred_cant_tell",
                            "p_cant_tell", "pred_drawn") if any(k in r for r in rows)]
        body = [[(f"{r[k]:.2f}" if isinstance(r.get(k), float) else esc(r.get(k, "–"))) for k in keys] for r in rows]
        out.append(f'<h3 id="{d.name}">{d.name}</h3>' + table(keys, body, "num small"))
    T["audits"] = "".join(out)
    N["audit.count"] = str(len(out))


# ---------------------------------------------------------------- generated pages

def run_pages(st):
    rows, pages = [], {}
    for r in RUNS:
        done = r["id"] in st.meta
        m = st.meta.get(r["id"], {})
        eps = m.get("record", {}).get("epochs", [])
        mast = " ".join(f"{e['mastery']:.2f}" for e in eps)
        status = '<span class="pill ok">done</span>' if done else '<span class="pill wait">not finished</span>'
        link = f'<a href="run-{r["id"]}.html">{r["id"]}</a>' if done else r["id"]
        rows.append([link, status, r["model"], r["regime"], r["seed"], r["protocol"],
                     m.get("epochs", "–"), m.get("n_examples", "–"), mast or "–", esc(r["note"])])
        if done:
            pages[f"run-{r['id']}.html"] = (r["id"], run_detail(st, r, m))
    for f, k, model, order, note in EVALS:
        done = (OUT / f / "eval_adapters.jsonl").exists()
        rows.append([f, '<span class="pill ok">done</span>' if done else '<span class="pill wait">not finished</span>', model,
                     "eval only", "–", "–", "–", "–", "–", esc(note + " Adapters: " + ", ".join(order) + "."), ])
    for s in SMOKES:
        rows.append([s, '<span class="pill muted">smoke</span>', "–", "–", "–", "–", "–", "–", "–",
                     "Pipeline check, not analysed."])
    T["runs"] = table(["Run", "Status", "Model", "Regime", "Seed", "Format practice", "Epochs", "Teaching examples",
                       "Letter-MC mastery by epoch", "Notes"], rows, "small")
    N["runs.done"] = str(sum(r["id"] in st.meta for r in RUNS))
    N["runs.queued"] = str(sum(r["id"] not in st.meta for r in RUNS) +
                           sum(not (OUT / e[0] / "eval_adapters.jsonl").exists() for e in EVALS))
    return pages


MAIN_GROUPS = ["mastery-text", "mastery-image", "text-L1", "text-L2", "text-L3", "text-L4", "image-name",
               "image-L1", "image-L2", "image-L3", "image-L4", "B-V", "B-T", "B-VT", "cloze-text", "cloze-image",
               "compose-hop1", "compose-hop2", "compose", "compose-chain", "anchor"]


def run_detail(st, r, m):
    eps = sorted(e for s, e in st.srcs[r["id"]] if s == "train")
    gs = sorted({k[3] for k in st.cells if k[0] == r["id"] and k[1] == "train"},
                key=lambda g: (MAIN_GROUPS.index(g) if g in MAIN_GROUPS else 99, g))
    rows = []
    for g in gs:
        cells = [boot(st.cells.get((r["id"], "train", e, g), [])) for e in eps]
        ch = next((c["chance"] for c in cells if c), None)
        rows.append([f"<code>{g}</code>", f2(ch), str(cells[-1]["n"] if cells[-1] else "–")] +
                    [f2(c["m"]) if c else "–" for c in cells])
    cfg = {k: v for k, v in m.items() if k != "record"}
    ev = []
    for f, *_ in EVALS:
        gs2 = sorted({k[3] for k in st.cells if k[0] == r["id"] and k[1] == f})
        if gs2:
            ev.append(f"<h3>Eval job <code>{f}</code></h3>" + table(
                ["Group", "Chance", "n", "Accuracy", "Calibrated"],
                [[f"<code>{g}</code>", f2(boot(st.cells[(r['id'], f, 'final', g)])["chance"]),
                  boot(st.cells[(r['id'], f, 'final', g)])["n"], cell(boot(st.cells[(r['id'], f, 'final', g)])),
                  cell(boot(st.cells[(r['id'], f, 'final', g)], True))] for g in gs2], "num small"))
    body = (f'<p class="lede">{esc(r["note"]) or "Training run."}</p>'
            f'<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in
                                         {"kernel": r["kernel"], "output": f"kaggle/out/{r['id']}", **cfg}.items()) +
            "</dl><h2>Every group, every epoch</h2>"
            "<p>Accuracy on each item group after each epoch (epoch 0 = untaught). Letter-scored groups: "
            "<code>mastery-*</code>, <code>text-*</code>, <code>image-*</code>, <code>B-*</code>, <code>anchor</code>; "
            "the rest are likelihood probes.</p>" +
            table(["Group", "Chance", "n"] + [f"ep {e}" for e in eps], rows, "num small") + "".join(ev))
    return body


def probe_catalogue():
    seen = {}
    for p in [OUT / "eval_dir1" / "eval_adapters.jsonl", OUT / "teach3_s_s0" / "eval.jsonl"]:
        for r in lines(p):
            k = r.get("family") if r.get("split") != "anchor" else "anchor"
            if r.get("family") in ("text", "image") and r.get("split") == "test":
                k = f"{r['family']}-{r['level']}" + ("-hidden" if r["family"] == "image" and not r["id"].startswith("L1img") else "")
            if k not in seen and r.get("split") != "mastery":
                seen[k] = r
    rows = []
    for k, r in seen.items():
        q = r.get("prompt") or r.get("question") or ""
        img = "picture + " if r.get("image") else ""
        rows.append([f"<code>{esc(k)}</code>", f"<pre>{esc(img)}{esc(q)}</pre>", r.get("n_options", "–"), f2(r["chance"])])
    T["probes"] = table(["Family", "Example prompt (first test item)", "Candidates", "Chance"], rows, "small")


def explorer_data(st):
    rows = []
    for (run, src, ep, g), items in st.cells.items():
        for cal in (False, True):
            b = boot(items, cal)
            if b is None:
                continue
            rows.append([run, src, ep, g + ("-cal" if cal else ""), b["n"], round(b["m"], 3), round(b["lo"], 3),
                         round(b["hi"], 3), round(b["chance"], 3)])
    return rows


# ---------------------------------------------------------------- bridge probes (eval/bridge.py)

BRIDGE_MODELS = [("base", "untaught"), ("T-s0", "T"), ("Tr-s0", "Tr"), ("V-s0", "V"), ("S-s0", "S")]
LEX_KINDS = [("exact", "exact (kusnib)"), ("capital", "capitalised (Kusnib)"), ("hyphen", "hyphenated (kus-nib)"),
             ("vowel", "one vowel changed (kusnob)"), ("transpose", "two letters swapped (kunsib)"),
             ("novel", "unseen pseudoword"), ("other", "another taught creature's name")]


def _acc(rows, key="answer"):
    return boot([(r["concept"], r["pred"] == r[key], None, 1 / r["n_cands"], False) for r in rows])


def bridge():
    d = OUT / "bridge_s0"
    pend = lambda what: f'<p class="pending">Waiting for <code>snsaid-bridge-s0</code> ({what}).</p>'
    keys = ("bridge_rep", "bridge_swap", "bridge_relabel", "bridge_lex", "bridge_lexpic", "bridge_visual",
            "bridge_namelens", "bridge_xpatch", "bridge_label")
    cj = OUT / "clip_baseline" / "clip.json"
    if cj.exists():                     # contrastive dual-encoder reference (report/clip_baseline.py, CPU)
        rows = []
        for mid, x in json.loads(cj.read_text()).items():
            short = mid.split("/")[1]
            N[f"clip.{short}.retrieval"] = f2(x["retrieval"])
            N[f"clip.{short}.attr"] = f2(x["attr_mean"])
            N[f"clip.{short}.name"] = f2(x["name"])
            rows.append([short, f2(x["retrieval"]), f"{f2(x['attr_mean'])} (chance {f2(x['attr_chance'])})",
                         f"{f2(x['name'])} (chance {f2(x['name_chance'])})"])
        N["clip.chance"] = f2(x["chance"])
        N["clip.attr_chance"] = f2(x["attr_chance"])
        pj = OUT / "clip_baseline" / "probe.json"
        if pj.exists():                 # attribute directions across modalities (report/clip_probe.py)
            for mid, per in json.loads(pj.read_text()).items():
                short = mid.split("/")[1]
                N[f"clipprobe.{short}.pic2text"] = f2(np.mean([v["pic2text"] for v in per.values()]))
                N[f"clipprobe.{short}.text2pic"] = f2(np.mean([v["text2pic"] for v in per.values()]))
                N[f"clipprobe.{short}.chance"] = f2(np.mean([v["chance"] for v in per.values()]))
        T["bridge_clip"] = table(["Contrastive encoder", "Picture ↔ description match", "Zero-shot attributes",
                                  "Name → its pictures"], rows, "num",
                                 f"Same held-out pictures and nameless descriptions as the VLM analysis; 30 creatures, "
                                 f"match chance {N['clip.chance']}. Descriptions are truncated to the encoders' text "
                                 "length. Untaught encoders, so names carry nothing.")
    else:
        T["bridge_clip"] = ""
    if not (d / "bridge.jsonl").exists():
        for k in keys:
            T[k] = pend("eval-only job on the existing seed-0 adapters")
            C[k] = ""
        N["bridge.status"] = "pending"
        return
    N["bridge.status"] = "available"
    R = defaultdict(list)
    for r in lines(d / "bridge.jsonl"):
        R[(r["model"], r["family"])].append(r)
    models = [(m, lab) for m, lab in BRIDGE_MODELS if any(k[0] == m for k in R)]

    # 1. representations
    cache = d / "rep_analysis.json"
    reps = json.loads((d / "rep_prompts.json").read_text())
    have = [m for m, _ in models if (d / f"states_{m}.npz").exists() and not (d / f"states_{m}.npz.kaggle-partial").exists()]
    if cache.exists() and set(json.loads(cache.read_text())) != set(have):
        cache.unlink()                                   # recompute once more states have arrived
    if not cache.exists():
        from seen_never_said.eval.bridge import rep_analysis
        res = {}
        for m in have:
            z = np.load(d / f"states_{m}.npz")
            res[m] = {**rep_analysis(z["X"], reps), "layers": z["layers"].tolist()}
        cache.write_text(json.dumps(res))
    res = json.loads(cache.read_text())
    ref = res.get("base") or next(iter(res.values()))
    lay = ref["layers"]
    C["bridge_rep"] = line_chart([str(l) for l in lay],
                                 [(lab, res[m]["retrieval"]) for m, lab in models if m in ("base", "T-s0", "V-s0", "S-s0") and m in res],
                                 "Picture ↔ description of the same creature", "layer", chance=ref["chance_retrieval"],
                                 tick_every=3)
    rows = []
    for m, lab in models:
        if m not in res:
            continue
        x = res[m]
        bl = int(np.argmax(x["retrieval"]))
        rows.append([lab, f"{f2(max(x['retrieval']))} (layer {lay[bl]})", f2(np.nanmax(x["rsa"])),
                     f"{f2(max(x['attr_x']))} / {f2(max(x['attr_within']))}", f2(max(x["name_pic"])),
                     f2(max(x["name_desc"]))])
        N[f"bridge.rep.{m}.retrieval"] = f2(max(x["retrieval"]))
        N[f"bridge.rep.{m}.retrieval_layer"] = str(lay[bl])
        N[f"bridge.rep.{m}.rsa"] = f2(np.nanmax(x["rsa"]))
        N[f"bridge.rep.{m}.attr_x"] = f2(max(x["attr_x"]))
        N[f"bridge.rep.{m}.attr_within"] = f2(max(x["attr_within"]))
        N[f"bridge.rep.{m}.name_pic"] = f2(max(x["name_pic"]))
        N[f"bridge.rep.{m}.name_desc"] = f2(max(x["name_desc"]))
    N["bridge.rep.chance"] = f2(ref["chance_retrieval"])
    N["bridge.rep.chance_attr"] = f2(ref["chance_attr"])
    N["bridge.rep.chance_name"] = f2(ref["chance_name"])
    T["bridge_rep"] = table(["Model", "Picture ↔ description match", "Similarity structure (ρ)",
                             "Attribute read-out: across / within modality", "Name → its pictures",
                             "Name → its descriptions"], rows, "num",
                            f"Best layer of {len(lay)} sampled, which favours every column slightly. Match: a creature's mean picture state is nearest its own "
                            f"mean description state (30 creatures, chance {N['bridge.rep.chance']}). ρ: Spearman "
                            "correlation of the creature × creature similarities in the two modalities. Read-out: an "
                            "attribute classifier fit on one modality's states and tested on the other's, leaving the "
                            f"creature out (no-information level {N['bridge.rep.chance_attr']}). Name: the name's state "
                            f"is nearest its own creature's pictures or descriptions (20 creatures, chance "
                            f"{N['bridge.rep.chance_name']}). Each modality's mean is removed first.")

    # 2a. image-token swap
    series, rows = [], []
    for m, lab in models:
        for fam, flab in (("swap-name", "picture → name"), ("swap-color", "picture → colour")):
            rr = R.get((m, fam))
            if not rr:
                continue
            ks = sorted({r["swap_layer"] for r in rr})
            orig = [np.mean([r["pred"] == r["answer"] for r in rr if r["swap_layer"] == k]) for k in ks]
            part = [np.mean([r["pred"] == r["partner_answer"] for r in rr if r["swap_layer"] == k]) for k in ks]
            series.append((f"{lab}, {flab}: follows original", orig))
            rows.append([lab, flab] + [f"{f2(o)} / {f2(q)}" for o, q in zip(orig, part)])
            hand = next((k for k, o, q in zip(ks, orig, part) if k > 0 and o > q), None)
            N[f"bridge.swap.{m}.{fam}.handoff"] = str(hand) if hand is not None else "never"
            N[f"bridge.swap.{m}.{fam}.k0"] = f2(orig[0])
            for k_, o_, q_ in zip(ks, orig, part):
                N[f"bridge.swap.{m}.{fam}.orig.{k_}"] = f2(o_)
                N[f"bridge.swap.{m}.{fam}.part.{k_}"] = f2(q_)
    if rows:
        ks = sorted({r["swap_layer"] for v in R.values() for r in v if "swap_layer" in r})
        T["bridge_swap"] = table(["Model", "Probe"] + ["no swap" if k == 0 else f"after {k}" for k in ks], rows, "num",
                                 "Share of probes whose answer follows the original picture / the swapped-in picture, "
                                 "when every image-token state is replaced by another creature's after layer k.")
        C["bridge_swap"] = line_chart(["0" if k == 0 else str(k) for k in ks], series[:4],
                                      "Answer still follows the original picture", "image tokens swapped after layer")
    else:
        T["bridge_swap"], C["bridge_swap"] = "", ""

    # 2b. relabel
    rows = []
    for m, lab in models:
        own, oth = R.get((m, "relabel-own")), R.get((m, "relabel-other"))
        if not own:
            continue
        a_own, a_pic, a_word = _acc(own), _acc(oth), _acc(oth, "word_answer")
        rows.append([lab, cell(a_own), cell(a_pic), cell(a_word)])
        put(f"bridge.relabel.{m}.own", a_own)
        put(f"bridge.relabel.{m}.pic", a_pic)
        put(f"bridge.relabel.{m}.word", a_word)
    T["bridge_relabel"] = table(["Model", "Own name written", "Wrong name: answer fits the picture's creature",
                                 "Wrong name: answer fits the named creature"], rows, "num",
                                "Held-out S picture of creature A (its text-taught attributes hidden) with “This is a "
                                "B.” and a question about one of A's text-taught attributes; B's value differs from A's.")

    # 3a. lexical
    rows = []
    for kind, klab in LEX_KINDS:
        row = [klab]
        for m, lab in models:
            rr = R.get((m, f"lex-{kind}"))
            if rr is None:
                continue
            if m == "S-s0":                       # S only taught its text-split attributes
                from seen_never_said.teach import data as tdata
                cs = {c["cid"]: c for c in tdata.load_concepts(ROOT / "data" / "tier0_pilot")}
                rr = [r for r in rr if r["level"] in cs[r["concept"]]["split"]["T"]]
            a = _acc(rr)
            put(f"bridge.lex.{m}.{kind}", a)
            txt = cell(a)
            if kind == "other":
                o = _acc(rr, "other_answer")
                put(f"bridge.lex.{m}.other_follows", o)
                txt += f'<br><span class="ci">named creature’s value: {f2(o["m"])}</span>'
            row.append(txt)
        rows.append(row)
    lex_models = [lab for m, lab in models if (m, "lex-exact") in R]
    T["bridge_lex"] = table(["Name in “Every ___ …”"] + lex_models, rows, "num",
                            "Name → property in the teaching format, scored against the original creature's value. "
                            "S: its four text-taught attributes only.")

    # 3b. lexpic
    rows = []
    kinds = ["exact", "vowel", "transpose", "hyphen", "novel", "other", "other2"]
    for m, lab in models:
        rr = R.get((m, "lexpic"))
        if not rr:
            continue
        win = defaultdict(int)
        mass = defaultdict(float)
        for r in rr:
            sc = np.array(r["scores"])
            pr = np.exp(sc - sc.max())
            pr /= pr.sum()
            win[kinds[int(np.argmax(sc))]] += 1
            for k, q in zip(kinds, pr):
                mass[k] += q / len(rr)
        rows.append([lab] + [f"{win[k] / len(rr):.2f} <span class='ci'>({mass[k]:.2f})</span>" for k in kinds])
        N[f"bridge.lexpic.{m}.exact"] = f2(win["exact"] / len(rr))
        N[f"bridge.lexpic.{m}.near"] = f2((win["vowel"] + win["transpose"] + win["hyphen"]) / len(rr))
    T["bridge_lexpic"] = table(["Model", "exact", "vowel", "swap", "hyphen", "unseen", "other", "other 2"], rows, "num",
                               "Picture → name with near-miss spellings among the candidates: share of probes where each "
                               "form wins, and (in brackets) its mean probability among the seven.")

    # 3c. visual perturbation
    rows = []
    for m, lab in models:
        rr = R.get((m, "visual"))
        if not rr:
            continue
        none = [r for r in rr if r["level"] == "none"]
        base_acc = _acc(none)
        row = [lab, cell(base_acc)]
        for a in VALUES:
            pa = [r for r in rr if r["level"] == a]
            dd = paired_diff(_acc(pa), base_acc)
            put(f"bridge.visual.{m}.{a}", dd, True)
            row.append(ci(dd, True))
        put(f"bridge.visual.{m}.none", base_acc)
        rows.append(row)
    T["bridge_visual"] = table(["Model", "Unchanged picture"] + [a.replace("_class", "") for a in VALUES], rows, "num",
                               "Picture → name (8 names) on freshly rendered pictures; columns: change in accuracy when "
                               "that one attribute is redrawn with another value, same render seed, paired over the 20 "
                               "creatures.")
    # 4. does the input produce the name internally? (logit lens after "This is a")
    from seen_never_said.eval.bridge import namelens_analysis
    nlp = json.loads((d / "namelens_prompts.json").read_text())
    uniq = json.loads((d / "name_tokens.json").read_text())["unique"]
    series, rows = [], []
    for m, lab in models:
        f = d / f"namelens_{m}.npz"
        if not f.exists():
            continue
        r = namelens_analysis(np.load(f)["lens"], nlp, uniq)
        L = len(r["pic"])
        for s_, slab in (("pic", "picture"), ("desc", "description")):
            N[f"bridge.namelens.{m}.{s_}"] = f2(max(r[s_][1:]))
            N[f"bridge.namelens.{m}.{s_}.last"] = f2(r[s_][-1])
            N[f"bridge.namelens.{m}.{s_}.first"] = str(next((l for l, v in enumerate(r[s_]) if l > 0 and v >= 0.5),
                                                             "never"))
        N["bridge.namelens.chance"] = f2(r["chance"])
        rows.append([lab, f"{f2(max(r['pic'][1:]))} (last {f2(r['pic'][-1])})",
                     f"{f2(max(r['desc'][1:]))} (last {f2(r['desc'][-1])})"])
        if m in ("base", "V-s0", "S-s0", "Tr-s0"):
            series.append((f"{lab}, picture", r["pic"]))
    if rows:
        T["bridge_namelens"] = table(["Model", "Picture → own name's token (best layer)",
                                      "Description → own name's token (best layer)"], rows, "num",
                                     f"Logit lens at the position after “This is a”: does the creature's own name's first "
                                     f"token rise more than the other names' (each token centred on its mean over the "
                                     f"prompts)? Names with a unique first token only; chance {N['bridge.namelens.chance']}. "
                                     "Best of every layer, which inflates the noise level; the untaught model is the "
                                     "reference.")
        C["bridge_namelens"] = line_chart([str(l) for l in range(L)], series, "Own name's token, picture prompts",
                                          "layer", chance=r["chance"], tick_every=4)
    # 4b. the name printed in the picture
    rows = []
    for m, lab in models:
        if (m, "label-none") not in R:
            continue
        a0, a1, a2 = _acc(R[(m, "label-none")]), _acc(R[(m, "label-own")]), _acc(R[(m, "label-other")], "other_answer")
        for k_, v_ in (("none", a0), ("own", a1), ("other", a2)):
            put(f"bridge.label.{m}.{k_}", v_)
        rows.append([lab, cell(a0), cell(a1), cell(a2)])
    T["bridge_label"] = table(["Model", "No label", "Own name printed: its value",
                               "Other name printed: the named creature's value"], rows, "num",
                              "Held-out S picture (text-taught attributes hidden) with the name printed on a plate at "
                              "the top; silent question about a text-taught attribute.")
    # 5. text -> picture patching
    rr = R.get(("S-s0", "xpatch"), [])
    if rr:
        none = [r for r in rr if r["xsrc"] == "none"]
        layers = sorted({r["xlayer"] for r in rr if r["xsrc"] != "none"})
        rows = [["no patch", cell(_acc(none)), cell(_acc(none, "other_answer"))] + [""] * 0]
        own_curve, oth_curve = [], []
        for i in layers:
            o = [r for r in rr if r["xsrc"] == "own" and r["xlayer"] == i]
            t = [r for r in rr if r["xsrc"] == "other" and r["xlayer"] == i]
            a_o, a_t = _acc(o), _acc(t, "other_answer")
            own_curve.append(a_o["m"])
            oth_curve.append(a_t["m"])
            put(f"bridge.xpatch.own.{i}", a_o)
            put(f"bridge.xpatch.other.{i}", a_t)
            rows.append([f"layer {i}", cell(a_o), cell(a_t)])
        put("bridge.xpatch.none", _acc(none))
        first = next((i for i, v in zip(layers, oth_curve) if v >= 0.5), None)
        N["bridge.xpatch.first_follow"] = str(first) if first is not None else "never"
        T["bridge_xpatch"] = table(["Text state copied in at", "Own name's text → picture's value",
                                    "Other name's text → named creature's value"], rows, "num",
                                   "S adapter, seed 0. The silent picture prompt's last-position state at layer i is "
                                   "replaced by the state of the same question with a name in words (“What does a "
                                   "kusnib eat?”): its own creature's name, or another creature's with a different value.")
        C["bridge_xpatch"] = line_chart([str(i) for i in layers], [("own name's text: picture's value", own_curve),
                                                                    ("other name's text: named creature's value", oth_curve)],
                                        "Text state copied into the picture prompt", "layer", tick_every=1)
    for k in keys:
        C.setdefault(k, "")
        T.setdefault(k, "")


# ---------------------------------------------------------------- templating

NAV = [("index.html", "Overview"), ("setup.html", "Setup"), ("methods.html", "Methods & probes"),
       ("extraction.html", "Extraction"), ("direction.html", "Direction"), ("composition.html", "Composition"),
       ("bridge.html", "Bridge"), ("real.html", "Real photos"), ("scale.html", "7B"), ("runs.html", "Runs"), ("explorer.html", "Explorer"), ("audits.html", "Audits"),
       ("log.html", "Log"), ("paper.html", "Paper")]

missing = set()


def fill(text):
    def rep(mt):
        kind, key = mt.group(1), mt.group(2)
        src = {"n": N, "t": T, "c": C}[kind]
        if key in src:
            return src[key]
        missing.add(f"{kind}:{key}")
        return f'<span class="missing">[missing: {esc(key)}]</span>'
    return re.sub(r"\{\{([ntc]):([^}]+)\}\}", rep, text)


def page(fname, title, body, built):
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == fname or (fname.startswith("run-") and h == "runs.html") else ""}>{l}</a>'
                  for h, l in NAV)
    tpl = (CONTENT / "_template.html").read_text(encoding="utf-8")
    return (tpl.replace("%TITLE%", esc(title)).replace("%NAV%", nav).replace("%BODY%", body)
            .replace("%BUILT%", built))


def main():
    st = load()
    direction(st)
    ladder(st)
    practice(st)
    composition(st)
    scale(st)
    family(st)
    mechanism()
    bridge()
    from real_report import icl_section, real_section, fixes_section, vlmicl_section   # report/real_report.py
    H = dict(boot=boot, put=put, cell=cell, table=table, paired_diff=paired_diff, bm=bm)
    icl_section(N, T, H)
    real_section(N, T, C, H)
    fixes_section(N, T, H)
    vlmicl_section(N, T, H)
    from predictor import hops_section, predictor_section, freq_section, world_section, multihop_section, remedies_section
    hops_section(N, T, H)
    predictor_section(N, T, C, H)
    freq_section(N, T, H)
    world_section(N, T, H)
    multihop_section(N, T, H)
    remedies_section(N, T, H)
    confusions(st)
    audits()
    probe_catalogue()
    runs = run_pages(st)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    N["built"] = built
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "img").mkdir(exist_ok=True)
    for src, dst in (("data/tier0_pilot/images/V/c_0000/036.png", "v_kusnib.png"),
                     ("data/tier0_pilot/images/S/c_0000/036.png", "s_kusnib.png"),
                     ("data/tier0_pilot/images/V/c_0001/036.png", "v_neetroob.png"),
                     ("data/tier0_pilot/images/S/c_0001/036.png", "s_neetroob.png")):
        shutil.copy(ROOT / src, SITE / "img" / dst)
    for f in ("schema_teaser.png", "schema_pipeline.png", "schema_inside.png", "schema_real.png",
              "predictor.png"):      # report/schematics.py
        if (ROOT / "paper" / "fig" / f).exists():
            shutil.copy(ROOT / "paper" / "fig" / f, SITE / "img" / f)
    from PIL import Image
    from seen_never_said.eval.bridge import draw_label
    draw_label(Image.open(ROOT / "data/tier0_pilot/images/S/c_0000/036.png").convert("RGB"), "kusnib").save(
        SITE / "img" / "label_kusnib.png")
    for f in sorted(CONTENT.glob("*.html")):
        if f.name.startswith("_"):
            continue
        text = f.read_text(encoding="utf-8")
        title = re.search(r"<!--\s*title:\s*(.*?)\s*-->", text).group(1)
        (SITE / f.name).write_text(page(f.name, title, fill(text), built), encoding="utf-8")
    for fname, (rid, body) in runs.items():
        (SITE / fname).write_text(page(fname, f"Run {rid}", f"<h1>Run <code>{rid}</code></h1>" + body, built),
                                  encoding="utf-8")
    data = explorer_data(st)
    (SITE / "data.js").write_text("window.CELLS=" + json.dumps(data, separators=(",", ":")) + ";\nwindow.RUNS=" +
                                  json.dumps({r["id"]: {k: r[k] for k in ("model", "regime", "seed", "arm", "protocol")}
                                              for r in RUNS}) + ";", encoding="utf-8")
    print(f"built {len(list(SITE.glob('*.html')))} pages, {len(data)} explorer cells -> {SITE}")
    if missing:
        print("MISSING placeholders:", ", ".join(sorted(missing)))
    export_latex()


def _tex(v):
    """A number string from N as LaTeX: unicode dashes, arrows and minus signs to LaTeX."""
    v = html.unescape(re.sub(r"<[^>]+>", "", str(v)))
    for a, b in (("−", "-"), ("–", "--"), ("→", r"$\to$"), ("←", r"$\leftarrow$"), ("×", r"$\times$"),
                 ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#")):
        v = v.replace(a, b)
    v = re.sub(r"(?<![\w$])-(\d)", r"$-$\1", v)          # a leading minus as a math minus
    return v


def export_latex():
    """paper/numbers.tex: \\N{key} for every number on the site, so the paper never copies a number by hand.
    A key that does not exist yet prints as a red [pending: key]."""
    out = ROOT / "paper" / "numbers.tex"
    out.parent.mkdir(exist_ok=True)
    lines_ = ["% generated by report/build.py; do not edit",
              r"\makeatletter",
              r"\newcommand{\N}[1]{\@ifundefined{snsN@#1}{\textcolor{red}{[pending: \detokenize{#1}]}}"
              r"{\csname snsN@#1\endcsname}}"]
    for k, v in sorted(N.items()):
        if "\\" in k or "{" in k or "}" in k:
            continue
        lines_.append(r"\expandafter\def\csname snsN@" + k + r"\endcsname{" + _tex(v) + "}")
    lines_.append(r"\makeatother")
    (out.parent / "numbers.json").write_text(json.dumps(N, indent=0, ensure_ascii=False), encoding="utf-8")
    out.write_text("\n".join(lines_) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
