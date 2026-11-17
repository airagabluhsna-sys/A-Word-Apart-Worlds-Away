import json
import random
from collections import Counter

from seen_never_said.eval.items import (DONT_KNOW, INFER, anchor_items, build_items, chance, prompt)
from seen_never_said.generator.concepts import sample_concepts
from seen_never_said.generator.names import assign_names, generate_names
from seen_never_said.generator.ontology import VALUES
from seen_never_said.teach.data import mastery_filter


def _concepts(n_test=6, n_distractor=2):
    cs = sample_concepts(n_test, n_distractor, seed=3)
    assign_names(cs, generate_names(len(cs), seed=3))
    from seen_never_said.generator.concepts import latin_square_split
    latin_square_split(cs, seed=3, always_visual=["shape", "pattern", "material"])
    return [c.to_dict() for c in cs]


def _items(cs):
    imgs = {c["cid"]: {"mastery": [f"m/{c['cid']}/{i}.png" for i in range(4)],
                       "test": [f"t/{c['cid']}/{i}.png" for i in range(4)]} for c in cs}
    return build_items(cs, imgs, seed=0)


def test_items_well_formed():
    cs = _concepts()
    items = _items(cs)
    for it in items["mastery"] + items["test"] + anchor_items():
        assert it["options"][-1] == DONT_KNOW
        assert len(set(it["options"])) == len(it["options"])
        assert 0 <= it["answer"] < len(it["options"]) - 1
        assert 0 < chance(it) <= 0.5
        assert prompt(it).endswith("Answer with the letter only.")


def test_answers_are_true():
    cs = _concepts()
    by = {c["cid"]: c for c in cs}
    for it in _items(cs)["test"]:
        c, ans = by[it["concept"]], it["options"][it["answer"]]
        if it["level"] == "L1" and it["family"] == "image":
            assert ans == f"a {c['name']}"
        if it["level"] == "L3":
            a = it["attrs"][0]
            assert ans == INFER[a][1][c["attrs"][a]]
        if it["level"] == "L2" and it["kind"] in ("size_class", "legs"):
            o = by[it["other"]]
            assert ans == f"the {c['name']}" and c["attrs"][it["kind"]] > o["attrs"][it["kind"]]


def test_only_test_concepts_asked_and_wordings_differ():
    cs = _concepts()
    items = _items(cs)
    test_ids = {c["cid"] for c in cs if c["role"] == "test"}
    assert {it["concept"] for it in items["test"]} <= test_ids
    m_q = {it["question"] for it in items["mastery"] if it["family"] == "text"}
    t_q = {it["question"] for it in items["test"] if it["level"] == "L1" and it["family"] == "text"}
    assert not m_q & t_q
    m_img = {it["image"] for it in items["mastery"] if it["family"] == "image"}
    t_img = {it["image"] for it in items["test"] if it["family"] == "image"}
    assert not m_img & t_img


def test_true_false_balanced():
    cs = _concepts()
    tf = [it for it in _items(cs)["test"] if it.get("kind") == "tf"]
    counts = Counter(it["options"][it["answer"]] for it in tf)
    assert counts["true"] == counts["false"]


def test_inference_maps_one_to_one():
    for a, (_, m) in INFER.items():
        assert set(m) == set(VALUES[a]) and len(set(m.values())) == len(m)


def test_mastery_filter():
    cs = _concepts()
    items = _items(cs)["mastery"]
    assert all(it["family"] == "image" for it in items if mastery_filter("V", cs)(it))
    assert all(it["family"] == "text" for it in items if mastery_filter("T", cs)(it))
    split = {c["cid"]: c["split"] for c in cs}
    s = [it for it in items if mastery_filter("S", cs)(it) and it["family"] == "text"]
    assert s and all(it["attrs"][0] in split[it["concept"]]["T"] for it in s)


def test_report_on_synthetic_runs(tmp_path=None):
    """A model that learns text perfectly under T and nothing cross-modal under V gives tau ~ 0."""
    import tempfile
    from pathlib import Path
    import importlib.util
    # loaded by path: the installed kaggle CLI package shadows the kaggle/ folder
    spec = importlib.util.spec_from_file_location("teach_report", Path(__file__).parents[1] / "kaggle" / "teach_report.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    cs = _concepts()
    items = _items(cs)
    tmp = Path(tmp_path or tempfile.mkdtemp())
    rng = random.Random(0)
    dirs = []
    for regime in ("V", "T"):
        d = tmp / regime
        d.mkdir()
        with open(d / "eval.jsonl", "w") as f:
            for epoch in (0, 1):
                for it in items["mastery"] + items["test"]:
                    in_mod = (it["family"] == "image") == (regime == "V")
                    correct = epoch == 1 and in_mod or (epoch == 1 and rng.random() < chance(it))
                    f.write(json.dumps({"epoch": epoch, **it, "chance": chance(it), "correct": correct,
                                        "dont_know": False}) + "\n")
        (d / "run.json").write_text(json.dumps({"regime": regime, "method": "lora", "seed": 0,
                                                "record": {"epochs": [{"mastery": 1.0}], "mastered_at_epoch": 1,
                                                           "steps_to_mastery": 10}}))
        dirs.append(d)
    final = {m: [json.loads(l) for l in open(tmp / m / "eval.jsonl") if json.loads(l)["epoch"] == 1] for m in "VT"}
    t = rep.transfer(final)
    assert abs(t["tau V->T naming"]) < 0.25 and abs(t["tau T->V naming"]) < 0.25
    rep.main([str(d) for d in dirs])


def test_hidden_items():
    import tempfile
    from PIL import Image
    from seen_never_said.eval.hidden import HIDEABLE, INFER_Q, hidden_items
    from seen_never_said.eval.items import OPTION_TEXT
    cs = _concepts(4, 1)
    by = {c["cid"]: c for c in cs}
    d = tempfile.mkdtemp()
    items = hidden_items(cs, d, n_pairs=4)
    again = hidden_items(cs, d, n_pairs=4)
    assert [it["options"] for it in items] == [it["options"] for it in again]      # deterministic
    assert {it["level"] for it in items} == {"L1", "L2", "L3", "L4"}
    for it in items:
        c, ans = by[it["concept"]], it["options"][it["answer"]]
        assert set(it["attrs"]) <= set(HIDEABLE) and it["family"] == "image" and it["options"][-1] == DONT_KNOW
        assert c["name"] not in it["question"]                                     # never named
        a = it["attrs"][0]
        if it["level"] == "L1":
            assert ans == OPTION_TEXT[a][c["attrs"][a]]
        if it["level"] == "L3":
            assert ans == INFER[a][1][c["attrs"][a]] and INFER_Q[a] in it["question"]
        if it["level"] == "L2":
            o = by[it["other"]]
            assert c["attrs"][a] > o["attrs"][a] and Image.open(it["image"]).size == (896, 448)
    tf = [it for it in items if it.get("kind") == "tf"]
    n_true = sum(it["options"][it["answer"]] == "true" for it in tf)
    assert abs(2 * n_true - len(tf)) <= 1


def test_format_practice_only_on_distractors():
    from seen_never_said.teach.data import format_examples, load_concepts
    root = "data/tier0_pilot"
    cs = load_concepts(root)
    test_cids = {c["cid"] for c in cs if c["role"] == "test"}
    test_names = {c["name"] for c in cs if c["role"] == "test"}
    for regime in ("V", "T", "S"):
        ex = format_examples(root, regime, cs)
        assert ex and not {e["concept"] for e in ex} & test_cids
        texts = [e for e in ex if e["image"] is None]
        # no test name anywhere, not even as a wrong naming option (else "test names are wrong" is learnable)
        assert not any(n in e["prompt"] for e in ex for n in test_names)
        assert (regime == "V") == (not texts) and (regime == "T") == all(e["image"] is None for e in ex)


def test_cloze_probes():
    from seen_never_said.eval.cloze import N_NAMES, image_probes, text_probes
    from seen_never_said.teach.data import heldout_images, load_concepts
    cs = load_concepts("data/tier0_pilot")
    by = {c["cid"]: c for c in cs}
    tp, ip = text_probes(cs), image_probes(cs, heldout_images("data/tier0_pilot"))
    assert len(tp) == 20 * 8 and len(ip) == 20 * 4
    for p in tp:
        c = by[p["concept"]]
        assert len(set(p["candidates"])) == len(p["candidates"]) and c["name"] in p["candidates"][p["answer"]]
    for p in ip:
        assert len(p["candidates"]) == N_NAMES and p["candidates"][p["answer"]] == f"This is a {by[p['concept']]['name']}."


def test_integration_items():
    from seen_never_said.eval.integration import integration_items
    from seen_never_said.teach.data import load_concepts
    cs = load_concepts("data/tier0_pilot")
    by = {c["cid"]: c for c in cs}
    items = integration_items(cs)
    for it in items:
        c, ans = by[it["concept"]], it["options"][it["answer"]]
        split = c["split"]
        if it["part"] == "VT":
            aV, aT = [a for a in it["attrs"] if a in split["V"]], [a for a in it["attrs"] if a in split["T"]]
            assert len(aV) == len(aT) == 1 and (ans == "true") == (it["kind"] == "TT")
        else:
            assert it["attrs"][0] in split[it["part"]]
    for part in ("V", "T", "VT"):
        sub = [it for it in items if it["part"] == part]
        assert 2 * sum(it["options"][it["answer"]] == "true" for it in sub) == len(sub)


def test_ladder_probes():
    from seen_never_said.eval.ladder import ladder_probes
    from seen_never_said.teach.data import heldout_images, load_concepts
    concepts = load_concepts("data/tier0_pilot")
    probes = ladder_probes(concepts, heldout_images("data/tier0_pilot"), seed=0)
    assert len(probes) == 4 * (20 * 8 + 20 * 4)
    fams = {p["family"] for p in probes}
    assert fams == {f"ladder-{m}-{k}" for m in ("text", "image") for k in (1, 2, 3, 4)}
    by = {}
    for p in probes:
        by.setdefault(p["id"].rsplit("/", 1)[0], []).append(p)
    for group in by.values():
        assert len(group) == 4
        assert len({len(p["candidates"]) for p in group}) == 1       # same candidate set on every rung
        assert len({p["answer"] for p in group}) == 1
        r1, r2, r3, r4 = sorted(group, key=lambda p: p["family"])
        assert r3["prompt"].startswith(r2["prompt"]) and r1["prompt"] == r2["prompt"]
        true_short = r2["candidates"][r2["answer"]]
        letter = r4["candidates"][r4["answer"]]
        assert f"{letter}. {true_short}\n" in r4["prompt"]          # the letter points at the true option
        assert true_short in r3["prompt"]
        if r1["image"] is None:
            name = next(c["name"] for c in concepts if c["cid"] == r1["concept"])
            assert name in r1["prompt"] and r1["candidates"][r1["answer"]].startswith(f"Every {name} ")


def test_describe_probes():
    from seen_never_said.eval.describe import N_ORDERS, describe_probes
    from seen_never_said.teach.data import load_concepts
    cs = load_concepts("data/tier0_pilot")
    by = {c["cid"]: c for c in cs}
    probes = describe_probes(cs)
    assert len(probes) == 20 * N_ORDERS
    for p in probes:
        c = by[p["concept"]]
        assert c["name"] not in p["prompt"]                          # the name is only in the answer
        assert p["candidates"][p["answer"]] == f"This is a {c['name']}."
        assert len(set(p["candidates"])) == len(p["candidates"])
        assert p["prompt"].count(". ") + 1 >= 8                       # every attribute stated


def test_reversed_text_regime():
    import re
    from seen_never_said.generator.lexicon import PREDICATES
    from seen_never_said.teach.data import format_examples, load_concepts, teaching_examples
    root = "data/tier0_pilot"
    cs = load_concepts(root)
    by = {c["cid"]: c for c in cs}
    ex = teaching_examples(root, "Tr")
    assert len(ex) == len(teaching_examples(root, "T"))
    for e in ex:
        c = by[e["concept"]]
        assert e["image"] is None and e["answer"].count(c["name"]) == 1
        assert re.search(rf"\b{c['name']}\.$", e["answer"])              # the name comes last
        for a, v in c["attrs"].items():                                   # every fact stated
            assert any(p[0] in e["answer"] for p in PREDICATES[a][v])
    assert len(format_examples(root, "Tr", cs)) == len(format_examples(root, "T", cs))


def test_describe_one_probes():
    from seen_never_said.eval.describe import describe_one_probes
    from seen_never_said.teach.data import load_concepts
    cs = load_concepts("data/tier0_pilot")
    by_name = {c["name"]: c for c in cs}
    probes = describe_one_probes(cs)
    assert len(probes) == 20 * 8
    for p in probes:
        a = p["attrs"][0]
        names = [x[len("This is a "):-1] for x in p["candidates"]]
        true = by_name[names[p["answer"]]]
        assert true["cid"] == p["concept"] and true["name"] not in p["prompt"]
        # one property is enough: every other candidate differs on it
        assert all(by_name[n]["attrs"][a] != true["attrs"][a] for i, n in enumerate(names) if i != p["answer"])


def test_compose_probes():
    import json
    from seen_never_said.eval.compose import compose_probes
    from seen_never_said.teach.data import heldout_s_images, load_concepts
    root = "data/tier0_pilot"
    cs = load_concepts(root)
    by = {c["cid"]: c for c in cs}
    s_items = {json.loads(l)["image"]: json.loads(l) for l in open(f"{root}/items_S.jsonl", encoding="utf-8")
               if json.loads(l)["modality"] == "image"}
    probes = compose_probes(cs, heldout_s_images(root))
    for p in probes:
        c = by[p["concept"]]
        a = p["attrs"][0]
        assert a in c["split"]["T"]                                         # asks a text-taught property
        if p["image"]:
            assert a not in s_items[p["image"]]["shows"]                     # ... hidden in the picture
            assert int(p["image"].rsplit("/", 1)[1][:3]) >= 36               # held-out picture
        assert c["name"] not in p["prompt"] or p["family"] == "compose-hop2"
        right = p["candidates"][p["answer"]]
        if p["family"] == "compose-chain":
            assert right.startswith(f"This is a {c['name']}. Every {c['name']} ")
        assert len(set(p["candidates"])) == len(p["candidates"])


def test_layer_prompts():
    import json
    from seen_never_said.eval.layers import HIDE, LENS_WORDS, layer_prompts
    from seen_never_said.teach.data import load_concepts
    from seen_never_said.teach.layers_job import s_pictures
    root = "data/tier0_pilot"
    cs = load_concepts(root)
    by = {c["cid"]: c for c in cs}
    pics = s_pictures(root)
    s_items = {json.loads(l)["image"]: json.loads(l) for l in open(f"{root}/items_S.jsonl", encoding="utf-8")
               if json.loads(l)["modality"] == "image"}
    ps = layer_prompts(cs, pics)
    n_test = sum(c["role"] == "test" for c in cs)
    assert Counter(p["set"] for p in ps) == {"pic-name": n_test * 8, "pic-prop": n_test * 8 * 4,
                                            "name-prop": n_test * 4 * 3}
    for p in ps:
        c = by[p["concept"]]
        if p["image"]:
            idx = int(p["image"].rsplit("/", 1)[1][:3])
            assert 32 <= idx < 40 and (idx >= 36) == (p["split"] == "test")   # never a teaching picture
            assert c["name"] not in p["prompt"]
            if p["attr"]:
                assert p["attr"] not in s_items[p["image"]]["shows"]             # asked attribute is hidden
        else:
            assert c["name"] in p["prompt"]
        if p["attr"]:
            assert p["attr"] in c["split"]["T"]
            assert VALUES[p["attr"]][p["value_idx"]] == c["attrs"][p["attr"]]
    for a in HIDE:
        assert set(LENS_WORDS[a]) == set(VALUES[a])


def test_layer_analyses():
    import numpy as np
    from seen_never_said.eval.layers import (first_above, lens_accuracy, patch_pairs, probe_fit_test,
                                             probe_leave_concept_out, probe_transfer)
    rng = np.random.default_rng(0)
    n, L, d = 80, 3, 16
    y = np.repeat(np.arange(4), 20)
    groups = np.tile(np.arange(20), 4)
    X = rng.normal(size=(n, L, d))
    X[:, 2, :4] += 6 * np.eye(4)[y]                 # the label is linearly present only at layer 2
    fit = np.arange(n) % 2 == 0
    acc = probe_fit_test(X, y, fit)
    assert acc[2] > 0.9 and acc[0] < 0.6
    loco = probe_leave_concept_out(X, y, groups)
    assert loco[2] > 0.9 and loco[0] < 0.6
    tr = probe_transfer(X[fit], y[fit], X[~fit], y[~fit])
    assert tr[2] > 0.9
    lens = np.zeros((2, 2, 3))
    lens[0, 0] = [2, 1, 0]                           # item 0: value 0 wins at layer 0 ...
    lens[0, 1] = [0, 1, 2]                           # ... value 2 (the answer) at layer 1
    lens[1, :, 0] = 5                                # item 1: value 0 wins everywhere
    a, mrr = lens_accuracy(lens, [[0, 1, 2], [0, 1, 2]], [2, 0])
    assert a == [0.5, 1.0] and mrr[1] == 1.0
    assert first_above([0.1, 0.2, 0.9], 0.1) == 2 and first_above([0.1], 0.1) is None
    assert all(i < j <= 35 for i, j in patch_pairs(36))
