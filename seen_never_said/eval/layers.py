"""Layer-wise probes for the split regime: where does the picture resolve the name,
where does the name retrieve a text-taught property, and does moving the name
earlier make the silent two-hop answer work?

Behaviour (eval/compose.py, pilot 2026-11-02): after S teaching, picture -> name
and name -> property are near ceiling, picture -> property stays at chance and
writing the name first recovers it. In text-only two-hop failures the bridge
entity is resolved too late for the second hop ("hopping too late", Biran et
al. 2024). Cross-modal version of the hypothesis:

  L_name : first layer where the creature's name is linearly decodable from the
           last prompt position of a picture prompt;
  L_prop : first layer where a text-taught value is decodable (and visible
           through the logit lens) when the name is given in words;
  if L_name > L_prop, the picture delivers the name after the layers that would
  look up its properties.

Three prompt sets (same last-position states, every layer):

  pic-name   held-out S picture + "What is this creature called?"
  pic-prop   held-out S picture + a question about a text-taught attribute
  name-prop  "What color is a kusnib?" (name in words, 3 wordings)

Pictures: the 4 mastery and 4 test S pictures of each test concept (never used
in training); probes fit on mastery pictures, test on test pictures.

Back-patching: in the silent compose probe, the residual stream after a late
layer j at the last prompt position is copied into the same position after an
earlier layer i, then candidates are re-scored. A rise in accuracy means the
name representation, when present early enough, is usable by the property
lookup. Control: the vector from another creature's picture.
"""

import numpy as np

from ..generator.ontology import VALUES
from .compose import QUESTION

HIDE = ["color", "size_class", "legs", "habitat", "diet"]
NAME_Q = "This is a made-up creature.\nWhat is this creature called?"
PIC_Q = "This is a made-up creature.\n{q}"
WORDINGS = ["{q}", "Answer briefly. {q}", "A question about the {n}. {q}"]

# the word that carries each value in the teaching sentences, for the logit lens
LENS_WORDS = {
    "color": {v: v for v in VALUES["color"]},
    "size_class": {1: "cat", 2: "chair", 3: "car", 4: "house"},
    "legs": {2: "two", 4: "four", 6: "six", 8: "eight"},
    "habitat": {"tree": "trees", "pond": "pond", "desert": "desert", "kitchen": "kitchen", "snow": "snow",
                "cave": "caves"},
    "diet": {v: v for v in VALUES["diet"]},
}


def layer_prompts(concepts, s_pictures):
    """s_pictures: {cid: {"fit": [paths], "test": [paths]}} -> list of prompt dicts.

    Each dict: id, set, concept, name_idx (index among test concepts), attr,
    value_idx, image, prompt, split (fit/test).
    """
    test = [c for c in concepts if c.get("role", "test") == "test"]
    out = []
    for ni, c in enumerate(test):
        for split in ("fit", "test"):
            for k, img in enumerate(s_pictures[c["cid"]][split]):
                base = {"concept": c["cid"], "name_idx": ni, "image": img, "split": split}
                out.append({**base, "id": f"PN/{c['cid']}/{split}{k}", "set": "pic-name", "attr": None,
                            "value_idx": None, "prompt": NAME_Q})
                for a in c["split"]["T"]:
                    out.append({**base, "id": f"PP/{c['cid']}/{a}/{split}{k}", "set": "pic-prop", "attr": a,
                                "value_idx": VALUES[a].index(c["attrs"][a]),
                                "prompt": PIC_Q.format(q=QUESTION[a].format(w="this creature"))})
        for a in c["split"]["T"]:
            for w, tpl in enumerate(WORDINGS):
                out.append({"id": f"NP/{c['cid']}/{a}/{w}", "set": "name-prop", "concept": c["cid"], "name_idx": ni,
                            "image": None, "split": "fit" if w < 2 else "test", "attr": a,
                            "value_idx": VALUES[a].index(c["attrs"][a]),
                            "prompt": tpl.format(q=QUESTION[a].format(w=f"a {c['name']}"), n=c["name"])})
    return out


# ---------------------------------------------------------------- analyses (CPU)

def _clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))


def probe_fit_test(X, y, fit):
    """X [n, layers, d], y [n], fit bool [n] -> accuracy on ~fit per layer."""
    X = X.astype(np.float32)
    acc = []
    for l in range(X.shape[1]):
        m = _clf().fit(X[fit, l], y[fit])
        acc.append(float((m.predict(X[~fit, l]) == y[~fit]).mean()))
    return acc


def probe_leave_concept_out(X, y, groups):
    """Leave-one-concept-out accuracy per layer: the probe must read a value
    feature shared across creatures, not memorise which name has which value."""
    X = X.astype(np.float32)
    acc = []
    for l in range(X.shape[1]):
        hits = n = 0
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            if len(np.unique(y[tr])) < 2:
                continue
            m = _clf().fit(X[tr, l], y[tr])
            hits += int((m.predict(X[te, l]) == y[te]).sum())
            n += int(te.sum())
        acc.append(hits / n if n else float("nan"))
    return acc


def probe_transfer(Xa, ya, Xb, yb):
    """Fit on set a (e.g. name-prop), test on set b (pic-prop), per layer."""
    acc = []
    for l in range(Xa.shape[1]):
        m = _clf().fit(Xa[:, l].astype(np.float32), ya)
        acc.append(float((m.predict(Xb[:, l].astype(np.float32)) == yb).mean()))
    return acc


def lens_accuracy(lens, cand_cols, answer):
    """lens [n, layers, k] log-probs at tracked token ids; cand_cols[i]: the
    columns of item i's candidate values; answer[i]: index into cand_cols[i].
    Returns (accuracy, mean reciprocal rank) of the true value per layer."""
    n, L, _ = lens.shape
    acc, mrr = np.zeros(L), np.zeros(L)
    for i in range(n):
        s = lens[i][:, cand_cols[i]]                                  # [layers, values]
        rank = (s > s[:, [answer[i]]]).sum(1)
        acc += rank == 0
        mrr += 1 / (rank + 1)
    return (acc / n).tolist(), (mrr / n).tolist()


def first_above(curve, chance, margin=0.2):
    """First layer whose value exceeds chance + margin * (1 - chance), or None."""
    thr = chance + margin * (1 - chance)
    return next((l for l, v in enumerate(curve) if v > thr), None)


def patch_pairs(n_layers, dst=(4, 8, 12, 16, 20, 24), src=(12, 16, 20, 24, 28, 32)):
    """(i, j) pairs, residual after j layers copied into after i layers, i < j."""
    return [(i, j) for i in dst for j in src if i < j <= n_layers - 1]
