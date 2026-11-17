"""Summarise read-out runs side by side.

    py -3.11 kaggle/analyze.py kaggle/out/readout_v2 kaggle/out/probe_v1 [--save rates.json]

Visible attributes: accuracy, with chance (1 / number of options) beside it.
Hidden attributes:  how often the answer is "can't tell", the neutral default,
or the concept's true value (a leak check: should stay near chance).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

N_OPTIONS = {"shape": 13, "color": 13, "pattern": 6, "material": 5, "size_class": 6,
             "legs": 6, "habitat": 7, "diet": 7}
NEUTRAL = {"shape": "round", "color": "grey", "pattern": "plain", "material": "matte",
           "size_class": "2", "legs": "4", "habitat": "kitchen", "diet": "bread"}
# What absent evidence tends to be perceived as by base VLMs (read-out v1/probe):
# a b&w photo reads as "grey", hidden legs as "0", a missing reference as "cup".
# Grey and 0 legs are off-vocabulary; this rate is a covariate for transfer.
PERCEIVED = {"color": "grey", "legs": "0", "size_class": "0"}


def load(run):
    return [json.loads(l) for l in open(Path(run) / "readout.jsonl")]


def table(rows):
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        g = agg[(r["regime"], r["attr"], r["visible"])]
        g["n"] += 1
        pred, true = str(r["pred"]), str(r["true"])
        if r["visible"]:
            g["acc"] += pred == true
        else:
            g["cant"] += pred == "cant_tell"
            g["neutral"] += pred == NEUTRAL[r["attr"]]
            g["perceived"] += pred == PERCEIVED.get(r["attr"])
            g["true"] += pred == true
    return agg


def main(runs, save=None):
    saved = []
    for run in runs:
        print(f"\n### {run}")
        agg = table(load(run))
        print(f"{'regime':24s} {'attr':11s} {'n':>4s}  visible: acc (chance) | "
              f"hidden: can't-tell  neutral  perceived  true")
        for (regime, attr, vis), g in sorted(agg.items()):
            n = g["n"]
            rates = {k: round(v / n, 3) for k, v in g.items() if k != "n"}
            saved.append({"run": str(run), "regime": regime, "attr": attr, "visible": vis, "n": int(n), **rates})
            if vis:
                cells = f"{rates['acc']:5.2f} ({1 / N_OPTIONS[attr]:.2f})"
                print(f"{regime:24s} {attr:11s} {int(n):4d}  {cells}")
            else:
                cells = (f"{rates['cant']:5.2f}  {rates['neutral']:7.2f}  "
                         f"{rates['perceived'] if attr in PERCEIVED else float('nan'):9.2f}  {rates['true']:5.2f}")
                print(f"{regime:24s} {attr:11s} {int(n):4d}  {'':22s}| {cells}")
    if save:
        Path(save).write_text(json.dumps(saved, indent=1))
        print(f"\nsaved {len(saved)} rows to {save}")


if __name__ == "__main__":
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save")
        save = args[i + 1]
        del args[i:i + 2]
    main(args, save)
