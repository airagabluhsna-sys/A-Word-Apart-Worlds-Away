from collections import Counter

from seen_never_said.generator.audit import audit_items
from seen_never_said.generator.concepts import hamming, latin_square_split, sample_concepts
from seen_never_said.generator.describe import caption, describe_many
from seen_never_said.generator.lexicon import check_lexicon, extract
from seen_never_said.generator.names import assign_names, generate_names, sound_symbolism_score
from seen_never_said.generator.ontology import ATTRIBUTES, NEUTRAL_DEFAULTS, SIZE_REFERENCE
from seen_never_said.generator.render_vector import render_image


def _concepts(n=8):
    cs = sample_concepts(n, seed=1)
    assign_names(cs, generate_names(n, seed=1))
    latin_square_split(cs, seed=1)
    return cs


def test_lexicon_unambiguous():
    assert check_lexicon() == []


def test_concepts_distinct():
    cs = sample_concepts(30, seed=0)
    assert all(hamming(a.attrs, b.attrs) >= 3 for i, a in enumerate(cs) for b in cs[i + 1:])


def test_latin_square_balanced():
    cs = _concepts(len(ATTRIBUTES))
    counts = Counter(a for c in cs for a in c.split["V"])
    assert set(counts.values()) == {len(ATTRIBUTES) // 2}


def test_names_neutral_and_distinct():
    names = generate_names(50, seed=3)
    assert len(set(names)) == 50
    assert all(abs(sound_symbolism_score(n)) <= 0.2 for n in names)


def test_descriptions_state_exactly_their_attributes():
    c = _concepts()[0]
    for stated in (ATTRIBUTES, c.split["T"]):
        for t in describe_many(c, stated, 20, seed=0):
            assert extract(t) == {(a, c.attrs[a]) for a in stated}


def test_captions_carry_no_attributes():
    import random
    rng = random.Random(0)
    assert all(not extract(caption("zwablib", rng)) for _ in range(50))


def test_render_draws_visible_values_only():
    c = _concepts()[0]
    visible = set(c.split["V"])
    img, meta = render_image(c, visible, seed=5, size=224)
    assert img.size == (224, 224)
    assert all(meta["drawn"][a] == c.attrs[a] for a in visible)
    if "size_class" not in visible:
        assert meta["reference"] is None


def test_split_with_always_visual():
    cs = sample_concepts(10, seed=2)
    fixed = ["shape", "pattern", "material"]
    latin_square_split(cs, seed=2, always_visual=fixed)
    assert all(set(fixed) <= set(c.split["V"]) and len(c.split["V"]) == 4 for c in cs)
    counts = Counter(a for c in cs for a in c.split["V"] if a not in fixed)
    assert set(counts.values()) == {2}     # 10 concepts, 5 rotating attributes, 1 visual each


def test_hide_modes():
    c = _concepts()[0]
    visible = {"shape", "pattern", "material"}
    hidden = set(ATTRIBUTES) - visible
    _, absent = render_image(c, visible, seed=5, size=224, hide_mode="absent")
    assert all(absent["drawn"][a] is None for a in hidden) and absent["reference"] is None
    _, neutral = render_image(c, visible, seed=5, size=224, hide_mode="neutral")
    assert all(neutral["drawn"][a] == NEUTRAL_DEFAULTS[a] for a in hidden) and neutral["reference"]
    _, rand = render_image(c, visible, seed=5, size=224, hide_mode="random")
    assert all(rand["drawn"][a] is not None for a in hidden)


def test_perceived_defaults_are_off_vocabulary():
    # absent images read as "grey", "no legs" and "cup-sized", so none may be a concept's value
    cs = sample_concepts(200, seed=4, min_dist=1)
    assert all(c.attrs["color"] != "grey" and c.attrs["legs"] != 0 and c.attrs["size_class"] != 0 for c in cs)


def test_absent_color_styles():
    import numpy as np
    c = _concepts()[0]
    hidden_color = set(ATTRIBUTES) - {"color"}
    img, _ = render_image(c, hidden_color, seed=3, size=224, absence={"color": "bw_photo"})
    a = np.asarray(img).astype(int)
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    # creature_gray (default) keeps the scene in color, so habitat and diet stay readable
    img, _ = render_image(c, hidden_color, seed=3, size=224)
    a = np.asarray(img).astype(int)
    assert (np.abs(a[..., 0] - a[..., 2]) > 30).mean() > 0.2


def test_matched_size_reference():
    c = _concepts()[0]
    _, meta = render_image(c, set(ATTRIBUTES), seed=1, size=224)
    assert meta["reference"] == SIZE_REFERENCE[c.attrs["size_class"]]


def test_audit_flags_wrong_hiding():
    c = _concepts()[0]
    visible = {"shape", "pattern", "material"}
    _, meta = render_image(c, visible, seed=5, size=224, hide_mode="random")
    item = {"id": "x", "concept_id": c.cid, "regime": "S", "modality": "image", "caption": f"This is a {c.name}.",
            "shows": sorted(visible), "drawn": meta["drawn"]}
    c.split = {"V": sorted(visible), "T": sorted(set(ATTRIBUTES) - visible)}
    kinds = {p["kind"] for p in audit_items([c], [item])}
    assert "hidden_not_hidden" in kinds     # random drawing labelled as the absent arm


def test_audit_flags_leaky_caption():
    c = _concepts()[0]
    item = {"id": "x", "concept_id": c.cid, "regime": "V", "modality": "image",
            "caption": f"This is a {c.attrs['color']} {c.name}.", "shows": list(ATTRIBUTES), "drawn": dict(c.attrs)}
    kinds = {p["kind"] for p in audit_items([c], [item])}
    assert "caption_leak" in kinds
