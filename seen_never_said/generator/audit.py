"""Information-matching audit for teaching corpora.

Checks, per concept and regime:
  * captions mention no attribute value and no blocklisted descriptor;
  * every text states exactly its assigned attributes, with the concept's values;
  * every image draws the concept's values for its visible attributes;
  * every image hides its other attributes the way its hide mode says;
  * the coverage matrix (attribute x modality) matches the regime's plan.

Regime names are "V", "T", "S" (hide mode "absent") or "S-<mode>" for the
other hide modes, e.g. "S-neutral", "S-random".
"""

from collections import defaultdict

from .lexicon import CAPTION_BLOCKLIST, extract
from .ontology import ATTRIBUTES, HIDE_BY_ABSENCE, HIDE_MODES, NEUTRAL_DEFAULTS


def parse_regime(regime):
    """'S-neutral' -> ('S', 'neutral'); 'S' -> ('S', 'absent')."""
    base, _, mode = regime.partition("-")
    mode = mode or "absent"
    if base not in ("V", "T", "S") or mode not in HIDE_MODES or (mode != "absent" and base != "S"):
        raise ValueError(regime)
    return base, mode


def _hidden_ok(attr, drawn, mode):
    if mode == "neutral":
        return drawn == NEUTRAL_DEFAULTS[attr]
    if mode == "absent" and attr in HIDE_BY_ABSENCE:
        return drawn is None
    return drawn is not None      # random, or an unhideable attribute in absent mode


def plan(concept, regime):
    """Which attributes each modality should teach, for this concept and regime."""
    regime, _ = parse_regime(regime)
    if regime == "V":
        return {"image": set(ATTRIBUTES), "text": set()}
    if regime == "T":
        return {"image": set(), "text": set(ATTRIBUTES)}
    if regime == "S":
        return {"image": set(concept.split["V"]), "text": set(concept.split["T"])}
    raise ValueError(regime)


def audit_items(concepts, items):
    by_id = {c.cid: c for c in concepts}
    problems = []
    coverage = defaultdict(lambda: defaultdict(set))   # (cid, regime) -> modality -> attrs

    for it in items:
        c = by_id[it["concept_id"]]
        expect = plan(c, it["regime"])
        key = (c.cid, it["regime"])
        if it["modality"] == "image":
            cap = it["caption"]
            leaked = extract(cap.replace(c.name, ""))
            if leaked or CAPTION_BLOCKLIST.search(cap):
                problems.append({"item": it["id"], "kind": "caption_leak", "caption": cap, "found": sorted(map(str, leaked))})
            for a in it["shows"]:
                if it["drawn"][a] != c.attrs[a]:
                    problems.append({"item": it["id"], "kind": "image_value_mismatch", "attr": a})
            mode = parse_regime(it["regime"])[1]
            for a in set(ATTRIBUTES) - set(it["shows"]):
                if not _hidden_ok(a, it["drawn"][a], mode):
                    problems.append({"item": it["id"], "kind": "hidden_not_hidden", "attr": a,
                                     "mode": mode, "drawn": it["drawn"][a]})
                if mode == "absent" and a not in HIDE_BY_ABSENCE:
                    # an unhideable attribute fell back to random values
                    problems.append({"item": it["id"], "kind": "unhideable_hidden", "attr": a})
            if set(it["shows"]) != expect["image"]:
                problems.append({"item": it["id"], "kind": "image_wrong_attrs"})
            coverage[key]["image"] |= set(it["shows"])
        else:
            got = extract(it["text"].replace(c.name, "").replace(c.plural, ""))
            want = {(a, c.attrs[a]) for a in expect["text"]}
            if got != want:
                problems.append({"item": it["id"], "kind": "text_mismatch", "missing": sorted(map(str, want - got)),
                                 "extra": sorted(map(str, got - want)), "text": it["text"]})
            coverage[key]["text"] |= {a for a, _ in got}

    for (cid, regime), cov in coverage.items():
        exp = plan(by_id[cid], regime)
        for m in ("image", "text"):
            if cov.get(m, set()) != exp[m]:
                problems.append({"concept": cid, "regime": regime, "kind": "coverage", "modality": m,
                                 "expected": sorted(exp[m]), "got": sorted(cov.get(m, set()))})
    return problems
