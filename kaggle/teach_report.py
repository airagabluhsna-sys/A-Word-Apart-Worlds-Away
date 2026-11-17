"""Transfer report for teaching runs (proposal §7.4).

    py -3.11 kaggle/teach_report.py RUN_DIR ... [--adapters eval_adapters.jsonl ...]

A run dir holds eval.jsonl and run.json (seen_never_said/teach/run.py); epoch 0 is
the no-teaching baseline and the last epoch the result. eval_adapters.jsonl
(teach/evaluate.py) holds final-adapter rows plus untaught-base rows, and adds
the hidden-attribute image items.
"Don't know" counts as wrong; chance is 1 / (options without "Don't know").

Groups: text-L1..L4 (questions that name the concept), image-name (name the
creature in a picture), image-L1..L4 (picture with the asked attribute hidden),
text*-L1..L4 = text items restricted to the hideable attributes, the
content-matched counterpart of image-L*, and the cloze probes (eval/cloze.py):
cloze-text (name -> property) and cloze-image (picture -> name), each also
calibrated against the same run's epoch 0 (cloze-*-cal: argmax of score
minus untaught score per candidate), and the extraction ladder (eval/ladder.py,
ladder-{text,image}-1..4) and description -> name probes (eval/describe.py,
describe-name), adapter files only.

  tau V->T (L) = (text*-L | V - chance) / (image-L | V - chance)
  tau T->V (L) = (image-L | T - chance) / (text*-L | T - chance)
  tau naming   : V->T uses text-L1 / image-name, T->V uses image-name / text-L1
  tau cloze    : V->T uses cloze-text / cloze-image, T->V the reverse (calibrated)
  B (regime)   = acc(B-VT) - acc(B-V) * acc(B-T), integration items (eval/integration.py)
  r_L (text)   = (text-L | V - chance) / (text-L | T - chance), Delta_L = text-L | T - text-L | V
95% intervals: bootstrap over concepts, pooling seeds.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

N_BOOT = 1000
HIDEABLE = {"color", "size_class", "legs", "habitat", "diet"}
LEVELS = ("L1", "L2", "L3", "L4")


def groups(r):
    if r["split"] == "anchor":
        return ["anchor"]
    if r["split"] == "mastery":
        return [f"mastery-{r['family']}"]
    if r["level"] == "B":                         # integration items: part V, T or VT
        return [f"B-{r['part']}"]
    if r["family"].startswith(("cloze", "ladder", "describe")):
        return [r["family"]]
    if r["id"].startswith("L1img/"):
        return ["image-name"]
    g = [f"{r['family']}-{r['level']}"]
    if r["family"] == "text" and set(r["attrs"]) <= HIDEABLE:
        g.append(f"text*-{r['level']}")
    return g


def acc_by_group(rows):
    g = defaultdict(lambda: [0, 0, 0.0, 0])        # correct, n, chance sum, dont_know
    for r in rows:
        for k in groups(r):
            x = g[k]
            x[0] += r["correct"]
            x[1] += 1
            x[2] += r["chance"]
            x[3] += r["dont_know"]
    return {k: {"acc": c / n, "chance": ch / n, "dont_know": dk / n, "n": n} for k, (c, n, ch, dk) in g.items()}


def ratio(num, den):
    a, b = num["acc"] - num["chance"], den["acc"] - den["chance"]
    return a / b if b > 0.05 else float("nan")     # undefined when the denominator is near chance


def transfer(final):
    """final: {regime: rows at the last epoch (seeds pooled)}"""
    A = {m: acc_by_group(rows) for m, rows in final.items()}
    out = {}
    V, T = A.get("V", {}), A.get("T", {})
    for L in LEVELS:
        if f"text*-{L}" in V and f"image-{L}" in V:
            out[f"tau V->T {L}"] = ratio(V[f"text*-{L}"], V[f"image-{L}"])
    if {"cloze-text-cal", "cloze-image-cal"} <= set(V):
        out["tau V->T cloze"] = ratio(V["cloze-text-cal"], V["cloze-image-cal"])
    if {"text-L1", "image-name"} <= set(V):
        out["tau V->T naming"] = ratio(V["text-L1"], V["image-name"])
    for m in ("T", "Tr"):                          # text-taught: name first (T) or last (Tr)
        X = A.get(m, {})
        for L in LEVELS:
            if f"text*-{L}" in X and f"image-{L}" in X:
                out[f"tau {m}->V {L}"] = ratio(X[f"image-{L}"], X[f"text*-{L}"])
        if {"cloze-text-cal", "cloze-image-cal"} <= set(X):
            out[f"tau {m}->V cloze"] = ratio(X["cloze-image-cal"], X["cloze-text-cal"])
        if {"text-L1", "image-name"} <= set(X):
            out[f"tau {m}->V naming"] = ratio(X["image-name"], X["text-L1"])
    for m, G in A.items():                         # binding score per regime (H4)
        if {"B-V", "B-T", "B-VT"} <= set(G):
            out[f"B ({m})"] = G["B-VT"]["acc"] - G["B-V"]["acc"] * G["B-T"]["acc"]
    for L in LEVELS:
        k = f"text-{L}"
        if k in V and k in T:
            out[f"r {L} (text, V/T)"] = ratio(V[k], T[k])
            out[f"Delta {L} (text, T-V)"] = T[k]["acc"] - V[k]["acc"]
        k = f"image-{L}"
        if k in V and k in T:
            out[f"Delta {L} (image, V-T)"] = V[k]["acc"] - T[k]["acc"]
    return out


def bootstrap(final, rng):
    concepts = sorted({r["concept"] for rows in final.values() for r in rows if r["concept"]})
    by = {m: defaultdict(list) for m in final}
    for m, rows in final.items():
        for r in rows:
            if r["concept"]:
                by[m][r["concept"]].append(r)
    stats = defaultdict(list)
    for _ in range(N_BOOT):
        pick = [rng.choice(concepts) for _ in concepts]
        res = {m: [r for c in pick for r in by[m][c]] for m in final}
        for k, v in transfer(res).items():
            if v == v:
                stats[k].append(v)
    ci = {}
    for k, vs in stats.items():
        vs.sort()
        ci[k] = (vs[int(0.025 * len(vs))], vs[int(0.975 * len(vs)) - 1])
    return ci


def key_of(regime, method):
    return regime if method == "lora" else f"{regime}-{method}"


def calibrated(rows):
    """Cloze rows re-scored as (score - epoch-0 score) per candidate, same run."""
    zero = {(r["seed"], r["id"]): r["scores"] for r in rows if r["epoch"] == 0 and "scores" in r}
    out = []
    for r in rows:
        if "scores" in r and r["epoch"] != 0 and (r["seed"], r["id"]) in zero:
            d = [a - b for a, b in zip(r["scores"], zero[(r["seed"], r["id"])])]
            pred = max(range(len(d)), key=d.__getitem__)
            out.append({**r, "family": r["family"] + "-cal", "level": "", "pred": pred,
                        "correct": pred == r["answer"]})
    return out


def load_runs(run_dirs, adapter_files):
    final, base, metas = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in run_dirs:
        meta = json.loads((Path(d) / "run.json").read_text())
        rows = [json.loads(l) for l in open(Path(d) / "eval.jsonl", encoding="utf-8")]
        last = max(r["epoch"] for r in rows)
        k = key_of(meta["regime"], meta["method"])
        final[k] += [r for r in rows if r["epoch"] == last]
        final[k] += [r for r in calibrated(rows) if r["epoch"] == last]
        base[k] += [r for r in rows if r["epoch"] == 0]
        metas[k].append(meta)
    for f in adapter_files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        # adapter files supersede run dirs: same items plus the hidden ones
        tags = {(r["regime"], r["method"]) for r in rows if r["regime"] != "none"}
        for reg, meth in tags:
            k = key_of(reg, meth)
            final[k] = [r for r in rows if r["regime"] == reg and r["method"] == meth]
            base[k] = [r for r in rows if r["regime"] == "none" and r["seed"] in {x["seed"] for x in final[k]}]
            final[k] += calibrated(base[k] + final[k])     # base rows are epoch 0
    return final, base, metas


def main(args):
    adapter_files = []
    if "--adapters" in args:
        i = args.index("--adapters")
        args, adapter_files = args[:i], args[i + 1:]
    final, base, metas = load_runs(args, adapter_files)

    if metas:
        print("### runs")
    for key, ms in sorted(metas.items()):
        for m in ms:
            rec = m["record"]
            last = rec["epochs"][-1]["mastery"] if "epochs" in rec else rec.get("mastery")
            print(f"{key:7s} seed {m['seed']}: mastered at epoch {rec.get('mastered_at_epoch')}, "
                  f"steps {rec.get('steps_to_mastery')}, last mastery {last:.3f}" + ("  [SMOKE]" if m.get("smoke") else ""))

    print("\n### accuracy   baseline -> taught  (chance)  [don't know: baseline -> taught]")
    all_groups = sorted({g for rows in final.values() for g in acc_by_group(rows)})
    for key in sorted(final):
        b, f = acc_by_group(base[key]), acc_by_group(final[key])
        print(f"-- {key}")
        for g in all_groups:
            if g in f:
                print(f"   {g:12s} n={f[g]['n']:4d}  {b.get(g, {}).get('acc', float('nan')):.2f} -> {f[g]['acc']:.2f}"
                      f"  ({f[g]['chance']:.2f})  [{b.get(g, {}).get('dont_know', float('nan')):.2f} -> "
                      f"{f[g]['dont_know']:.2f}]")

    for label, runs in (("LoRA", {k: v for k, v in final.items() if "-" not in k}),
                        ("ICL", {k.split("-")[0]: v for k, v in final.items() if k.endswith("-icl")})):
        if not runs:
            continue
        print(f"\n### transfer ({label})")
        point, ci = transfer(runs), bootstrap(runs, random.Random(0))
        for k, v in point.items():
            lo, hi = ci.get(k, (float("nan"), float("nan")))
            print(f"   {k:24s} {v:6.2f}  [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
