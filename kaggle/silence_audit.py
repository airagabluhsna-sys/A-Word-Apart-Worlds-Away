"""Observer-relative silence audit: choose how to hide each attribute.

    py -3.11 kaggle/silence_audit.py kaggle/out/audit_3b kaggle/out/audit_7b

An image is *silent* about attribute a for model m when it carries no usable
information about a's true value, does not push m toward any one value, and
does not damage the attributes that stay visible. Per hidden attribute, style
and model this reports:

  leak        bias-corrected mutual information I(true; pred) in bits (plug-in
              MI minus its mean under shuffled true labels); ~0 means no
              usable information leaks (V-information view, Xu et al. 2020)
  abstain     P(pred = "can't tell")
  commitment  share of the single most frequent non-abstain answer: a constant
              percept such as "grey" that every concept would share
  collateral  mean accuracy drop on the other attributes vs. P-visible

Selection rule (fixed before the audit results were seen, 2026-10-28; the
leak criterion was changed from a 0.05-bit cutoff to a permutation test after
a smoke test on older probe data showed +-0.05-bit noise at n=60; after the
first real audit, the permutation unit was corrected from rows to concepts,
because rows of one concept share a true value and are not exchangeable):
  1. eligible styles: leak not significant (permutation p >= LEAK_ALPHA) for every model
  2. pick the lowest  commitment + collateral,  averaged over models
  3. size reference: "matched" if its size accuracy beats "any" by >= 0.10 in
     every model, else keep "any"
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LEAK_ALPHA = 0.05      # a style leaks if its MI beats shuffled labels at p < 0.05
SIZE_GAIN_MIN = 0.10
N_SHUFFLES = 500


def mutual_info(pairs):
    n = len(pairs)
    pj = Counter(pairs)
    px = Counter(t for t, _ in pairs)
    py = Counter(p for _, p in pairs)
    return sum(c / n * math.log2(c * n / (px[t] * py[p])) for (t, p), c in pj.items())


def leak(rows, rng):
    """Bias-corrected MI (bits) between true value and answer, and its p-value.

    The permutation shuffles true values across *concepts*, not across rows:
    each concept contributes several images with the same true value, so a
    row-level shuffle treats concept-specific answer quirks as information.
    """
    preds = [p for _, p, _ in rows]
    concepts = sorted({c for _, _, c in rows})
    truth = {c: t for t, _, c in rows}
    observed = mutual_info([(t, p) for t, p, _ in rows])
    null = []
    for _ in range(N_SHUFFLES):
        perm = [truth[c] for c in concepts]
        rng.shuffle(perm)
        remap = dict(zip(concepts, perm))
        null.append(mutual_info([(remap[c], p) for (_, _, c), p in zip(rows, preds)]))
    p = (1 + sum(x >= observed for x in null)) / (1 + len(null))
    return observed - sum(null) / len(null), p


def blind_shares(blind_run):
    """Answer shares per attribute on blank images (the language prior)."""
    if not blind_run:
        return {}
    rows = [json.loads(l) for l in open(Path(blind_run) / "readout.jsonl")]
    by = defaultdict(Counter)
    for r in rows:
        by[r["attr"]][str(r["pred"])] += 1
    return {a: {v: c / sum(cnt.values()) for v, c in cnt.items()} for a, cnt in by.items()}


def audit(run, blind_run=None):
    prior = blind_shares(blind_run)
    rows = [json.loads(l) for l in open(Path(run) / "readout.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["regime"]].append(r)
    acc = lambda rs, a: (sum(str(r["pred"]) == str(r["true"]) for r in rs if r["attr"] == a)
                         / max(1, sum(r["attr"] == a for r in rs)))
    base = by["P-visible"]
    attrs = sorted({r["attr"] for r in base})
    rng = random.Random(0)
    out = {}
    for regime, rs in by.items():
        if not regime.startswith("P-hide-"):
            continue
        hidden = next(a for a in attrs if regime[len("P-hide-"):].startswith(a))
        hr = [r for r in rs if r["attr"] == hidden]
        preds = Counter(str(r["pred"]) for r in hr)
        committed = [c for v, c in preds.items() if v != "cant_tell"]
        others = [a for a in attrs if a != hidden]
        mi, p = leak([(str(r["true"]), str(r["pred"]), r["concept"]) for r in hr], rng)
        top = max(((c, v) for v, c in preds.items() if v != "cant_tell"), default=(0, None))[1]
        out[regime] = {
            # diagnostic, not part of the rule: commitment beyond what the same
            # model says about a blank image (perception rather than language prior)
            "blind_share": round(prior.get(hidden, {}).get(top, 0.0), 3) if prior else None,
            "attr": hidden,
            "leak": round(mi, 3),
            "leak_p": round(p, 3),
            "abstain": round(preds["cant_tell"] / len(hr), 3),
            "commitment": round(max(committed, default=0) / len(hr), 3),
            "top_answer": top,
            "collateral": round(sum(acc(base, a) - acc(rs, a) for a in others) / len(others), 3),
        }
    size = {k: round(acc(by[k], "size_class"), 3) for k in ("P-visible", "P-visible-matched") if k in by}
    return out, size


def main(runs, blind=()):
    blind = dict(zip(runs, blind))
    results = {run: audit(run, blind.get(run)) for run in runs}
    for run, (out, size) in results.items():
        print(f"\n### {run}   size acc: {size}")
        print(f"{'style':30s} {'leak':>6s} {'p':>5s} {'abstain':>8s} {'commit':>7s} {'top':>6s} "
              f"{'blind':>6s} {'collat':>7s}")
        for regime, m in sorted(out.items()):
            b = "" if m["blind_share"] is None else f"{m['blind_share']:.2f}"
            print(f"{regime:30s} {m['leak']:6.3f} {m['leak_p']:5.2f} {m['abstain']:8.2f} {m['commitment']:7.2f} "
                  f"{str(m['top_answer']):>6s} {b:>6s} {m['collateral']:7.3f}")

    print("\n### decisions (pre-registered rule)")
    regimes = sorted(set.intersection(*(set(o) for o, _ in results.values())))
    by_attr = defaultdict(list)
    for regime in regimes:
        ms = [results[run][0][regime] for run in runs]
        eligible = all(m["leak_p"] >= LEAK_ALPHA for m in ms)
        score = sum(m["commitment"] + m["collateral"] for m in ms) / len(ms)
        by_attr[ms[0]["attr"]].append((not eligible, score, regime))
    decisions = {}
    for attr, cands in sorted(by_attr.items()):
        cands.sort()
        ineligible, score, regime = cands[0]
        decisions[attr] = regime
        flag = "  (no style passed the leak check; least-bad shown)" if ineligible else ""
        print(f"{attr:11s} -> {regime}  score {score:.3f}{flag}")
        for inel, s, r in cands[1:]:
            print(f"{'':14s}{r}  score {s:.3f}{'  (leaks)' if inel else ''}")
    gains = [s.get("P-visible-matched", 0) - s.get("P-visible", 0) for _, s in results.values()]
    size_ref = "matched" if all(g >= SIZE_GAIN_MIN for g in gains) else "any"
    decisions["size_reference"] = size_ref
    print(f"size_reference -> {size_ref}  (gains per model: {[round(g, 3) for g in gains]})")
    return decisions


if __name__ == "__main__":
    # py -3.11 kaggle/silence_audit.py RUN_3B RUN_7B [--blind BLIND_3B BLIND_7B]
    args = sys.argv[1:]
    blind = []
    if "--blind" in args:
        i = args.index("--blind")
        args, blind = args[:i], args[i + 1:]
    main(args, blind)
