import numpy as np

from seen_never_said.eval import bridge as B
from seen_never_said.teach import data

ROOT = "data/tier0_pilot"


def _setup():
    return data.load_concepts(ROOT), data.heldout_images(ROOT), data.heldout_s_images(ROOT)


def test_name_variants_differ_and_keep_length():
    c, _, _ = _setup()
    for x in c[:20]:
        v = B.variants(x, c[(c.index(x) + 1) % 20], "zzz")
        assert len({v["exact"], v["vowel"], v["transpose"], v["hyphen"], v["capital"]}) == 5
        assert len(v["vowel"]) == len(v["transpose"]) == len(x["name"])


def test_novel_names_are_unused():
    c, _, _ = _setup()
    used = {x["name"] for x in c}
    nv = B.novel_names(c, 20)
    assert len(nv) == 20 and not used & set(nv)


def test_swap_partner_differs_in_answer():
    c, h, _ = _setup()
    for p in B.swap_probes(c, h):
        assert p["partner"] != p["concept"] and p["partner_answer"] != p["answer"]


def test_relabel_other_value_differs():
    c, _, s = _setup()
    for p in B.relabel_probes(c, s):
        assert (p["word_answer"] != p["answer"]) == (p["family"] == "relabel-other")


def test_descriptions_have_no_name():
    c, h, _ = _setup()
    for p in B.rep_prompts(c, h):
        if p["set"] == "desc":
            assert not any(x["name"] in p["prompt"] for x in c)


def test_visual_specs_change_exactly_one_attribute():
    c, _, _ = _setup()
    by = {x["cid"]: x for x in c}
    for key, attrs, a, _ in B.visual_specs(c):
        diff = [k for k in attrs if attrs[k] != by[key.split("/")[0]]["attrs"][k]]
        assert diff == ([] if a is None else [a])


def test_rep_analysis_chance_on_noise():
    c, h, _ = _setup()
    P = B.rep_prompts(c, h)
    r = B.rep_analysis(np.random.default_rng(0).normal(size=(len(P), 1, 64)), P)
    assert r["retrieval"][0] < 0.2 and abs(r["attr_x"][0] - r["chance_attr"]) < 0.06


def test_label_probes_three_conditions():
    c, _, s = _setup()
    ps = B.label_probes(c, s)
    assert {p["family"] for p in ps} == {"label-none", "label-own", "label-other"}
    by = {x["cid"]: x for x in c}
    for p in ps:
        path, text = p["image"][6:].split("|", 1)
        if p["family"] == "label-own":
            assert text == by[p["concept"]]["name"]
        if p["family"] == "label-other":
            assert text == by[p["other"]]["name"] and p["other_answer"] != p["answer"]


def test_draw_label_changes_only_with_text():
    from PIL import Image
    im = Image.new("RGB", (448, 448), "gray")
    assert B.draw_label(im, "") is im
    assert B.draw_label(im, "kusnib").tobytes() != im.tobytes()


def test_namelens_analysis_finds_planted_signal():
    c, h, _ = _setup()
    P = B.namelens_prompts(c, h)
    rng = np.random.default_rng(0)
    lens = rng.normal(size=(len(P), 3, 20))
    for i, p in enumerate(P):
        if p["set"] == "pic":
            lens[i, 2, p["name_idx"]] += 5          # own name rises at layer 2, pictures only
    r = B.namelens_analysis(lens, P, [True] * 20)
    assert r["pic"][2] > 0.9 and r["desc"][2] < 0.3 and r["pic"][0] < 0.3


def test_xpatch_sources_name_the_right_creatures():
    c, _, s = _setup()
    by = {x["cid"]: x for x in c}
    for p in B.xpatch_probes(c, s):
        assert by[p["concept"]]["name"] in p["src_own"] and by[p["other"]]["name"] in p["src_other"]
        assert "this creature" in p["prompt"]


def test_compose_practice_only_distractors_and_training_pictures():
    c, _, s = _setup()
    P = data.compose_practice(ROOT, c)
    test = {x["cid"] for x in c if x["role"] == "test"}
    held = {p for v in s.values() for p in v}
    assert P and not {p["concept"] for p in P} & test and not {p["image"] for p in P} & held
