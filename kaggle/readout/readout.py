"""Base-VLM read-out of Tier 0 images (Kaggle GPU job, no training).

For sampled images from each image regime, ask one multiple-choice question per
attribute and score the answer letter by next-token logits.
  * visible attributes -> accuracy: can the model read what we draw?
  * hidden attributes  -> P("can't tell"), P(neutral default), P(true value):
    does `absent` hiding read as unknown, or as "grey" / "no legs"?

Writes /kaggle/working/readout.jsonl and readout_summary.json.
"""

import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
IMAGES_PER_CONCEPT = int(os.environ.get("IMAGES_PER_CONCEPT", 4))
REGIMES = ["V", "S", "S-neutral", "S-random"]
OUT = Path("/kaggle/working")

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51", "accelerate"], check=True)

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- locate the dataset (Kaggle mounts it under /kaggle/input/<slug>/) -------
root = Path(sorted(glob.glob("/kaggle/input/**/concepts.jsonl", recursive=True))[0]).parent
# Folders are uploaded as zips; Kaggle usually unpacks them, but not always.
if not (root / "seen_never_said").exists():
    import zipfile
    work = OUT / "data"
    for z in root.glob("*.zip"):
        zipfile.ZipFile(z).extractall(work / z.stem)
    for f in root.glob("*.jsonl"):
        (work / f.name).write_bytes(f.read_bytes())
    root = work
    # a zip may hold its folder or only its contents; make <name>/<name>/... resolve to <name>/...
    for d in ("images", "seen_never_said"):
        if (root / d / d).exists():
            (root / d / d).rename(root / f"_{d}")
            (root / d).rmdir()
            (root / f"_{d}").rename(root / d)
print("data root:", root, flush=True)
sys.path.insert(0, str(root))          # the generator package ships with the data
# SOURCE=dataset: score images from the uploaded regimes.
# SOURCE=render:  render probe sets here with the current generator (see probe_jobs).
SOURCE = os.environ.get("SOURCE", "dataset")
# a separately uploaded, newer copy of the package (snsaid-code) takes precedence
code = []
for dirpath, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if dirpath.endswith("/generator") and "render_vector.py" in files and not dirpath.startswith(str(root)):
        code.append(Path(dirpath).parent)
if not code:
    import zipfile
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        if not z.startswith(str(root)) and any(n.endswith("generator/render_vector.py") for n in zipfile.ZipFile(z).namelist()):
            zipfile.ZipFile(z).extractall(OUT / "code_zip")
            code = [Path(c).parents[1] for c in glob.glob(str(OUT / "code_zip" / "**" / "generator" / "render_vector.py"), recursive=True)]
            break
if SOURCE == "render" and not code:
    raise SystemExit("render mode needs the snsaid-code dataset attached (push with --code)")
if code:
    # Kaggle may unpack the package without its folder name, so copy it under the right one
    import shutil
    pkg = OUT / "code" / "seen_never_said"
    shutil.copytree(code[0], pkg, dirs_exist_ok=True)
    sys.path.insert(0, str(pkg.parent))
    print("code package:", code[0], flush=True)
from seen_never_said.generator.ontology import ATTRIBUTES, NEUTRAL_DEFAULTS  # noqa: E402


def find_image(rel):
    p = root / rel
    if p.exists():
        return p
    hits = glob.glob(str(root / "**" / rel), recursive=True)
    if not hits:
        raise FileNotFoundError(rel)
    return Path(hits[0])


CANT = "Can't tell from the picture"
# (question, [(value, option text)]); "can't tell" is always the last option
QUESTIONS = {
    "shape": ("Which best describes the shape of the creature's body?", [
        ("round", "round"), ("squircle", "a rounded square"), ("diamond", "a diamond"), ("star", "a star"),
        ("lumpy", "lumpy and irregular"), ("teardrop", "a teardrop"), ("bean", "a bean"),
        ("triangle", "a triangle"), ("long", "long and low, wider than it is tall"),
        ("tall", "narrow and upright"), ("heart", "a heart"), ("hexagon", "a hexagon")]),
    "color": ("What color is the creature's body?", [
        (c, c) for c in ["red", "teal", "yellow", "purple", "orange", "black", "white", "pink",
                         "blue", "green", "brown", "grey"]]),
    "pattern": ("What pattern does the creature's body have?", [
        ("plain", "plain, no pattern"), ("striped", "stripes"), ("spotted", "spots"),
        ("checkered", "a checkerboard"), ("gradient", "a smooth fade from one shade to another")]),
    "material": ("What does the creature's body look like it is made of?", [
        ("furry", "fur"), ("metallic", "shiny metal"), ("glassy", "see-through glass"),
        ("matte", "a dull, smooth surface")]),
    "size_class": ("In real life, about how big is the creature?", [
        (0, "about the size of a cup"), (1, "about the size of a cat"), (2, "about the size of a chair"),
        (3, "about the size of a car"), (4, "about the size of a house")]),
    "legs": ("How many legs does the creature have?", [(n, str(n)) for n in (0, 2, 4, 6, 8)]),
    "habitat": ("Where is the creature?", [
        ("tree", "by a tree"), ("pond", "at a pond"), ("desert", "in a desert"), ("kitchen", "in a kitchen"),
        ("snow", "in the snow"), ("cave", "in a cave")]),
    "diet": ("What food is next to the creature?", [
        ("berries", "berries"), ("fish", "a fish"), ("leaves", "leaves"), ("bread", "bread"),
        ("seeds", "seeds"), ("insects", "insects")]),
}
LETTERS = "ABCDEFGHIJKLMNOP"


def prompt(attr, order):
    """order: permutation of option indices (the last index is "can't tell").

    Shuffled per question because v1 showed a pull toward option A, and the
    creature is named explicitly because v1 described the reference cat instead.
    """
    q, opts = QUESTIONS[attr]
    texts = [t for _, t in opts] + [CANT]
    lines = [f"{LETTERS[i]}. {texts[j]}" for i, j in enumerate(order)]
    return ("The image shows a made-up cartoon creature with big round eyes. It may stand next to an everyday "
            "object (a cup, cat, chair, car or house) that is there only for scale. Answer about the made-up "
            f"creature, not the everyday object.\n{q}\n" + "\n".join(lines) + "\nAnswer with the letter only.")


def dataset_jobs():
    """Sampled images from the uploaded regimes; every attribute is asked."""
    sample = []
    for regime in REGIMES:
        items = [json.loads(l) for l in open(root / f"items_{regime}.jsonl")]
        by_c = defaultdict(list)
        for it in items:
            if it["modality"] == "image":
                by_c[it["concept_id"]].append(it)
        for cid, its in by_c.items():
            sample += random.sample(its, min(IMAGES_PER_CONCEPT, len(its)))
    return [dict(it, load=lambda it=it: Image.open(find_image(it["image"])).convert("RGB"), ask=list(QUESTIONS))
            for it in sample]


def probe_jobs(n_concepts=30):
    """Fresh concepts and renders with the current generator, for the silence audit.

    P-visible[-matched]:   everything shown (size reference random / matched)
    P-hide-<attr>[-style]: one attribute hidden by absence
    Every question is asked on every image, so hiding one attribute can be
    checked for collateral damage to the attributes that stay visible.
    """
    from seen_never_said.generator.concepts import sample_concepts
    from seen_never_said.generator.render_vector import ABSENCE_STYLES, render_image

    variants = [("P-visible", None, {}, "any"), ("P-visible-matched", None, {}, "matched")]
    for a in ["color", "size_class", "legs", "habitat", "diet"]:
        for style in ABSENCE_STYLES.get(a, [None]):
            variants.append((f"P-hide-{a}" + (f"-{style}" if style else ""), a, {a: style} if style else {}, "any"))
    concepts = sample_concepts(n_concepts, seed=101)
    jobs = []
    # Blind baseline (Fu et al. 2025): the same questions on blank images. Answers
    # here are the language prior; a hidden-image "commitment" only counts as
    # perception to the extent it exceeds this.
    for shade in (255, 0):
        for c in concepts:
            jobs.append({"id": f"P-blank-{shade}/{c.cid}", "regime": f"P-blank-{shade}", "concept_id": c.cid,
                         "attrs": c.attrs, "shows": [], "drawn": {a: None for a in ATTRIBUTES},
                         "load": lambda s=shade: Image.new("RGB", (448, 448), (s, s, s)), "ask": list(QUESTIONS)})
    only = [s for s in os.environ.get("PROBE_SETS", "").split(",") if s]
    for regime, hidden, absence, size_ref in variants:
        if only and not any(regime.startswith(s) for s in only):
            continue
        n_img = IMAGES_PER_CONCEPT if regime == "P-visible" else 2
        for c in concepts:
            visible = set(ATTRIBUTES) - ({hidden} if hidden else set())
            for i in range(n_img):
                img, meta = render_image(c, visible, random.randrange(2**31), 448, "absent", absence, size_ref)
                jobs.append({"id": f"{regime}/{c.cid}/{i}", "regime": regime, "concept_id": c.cid, "attrs": c.attrs,
                             "shows": sorted(visible), "drawn": meta["drawn"], "load": lambda img=img: img,
                             "ask": list(QUESTIONS)})
    # PROBE_SETS=P-blank,P-visible runs only the sets with those prefixes
    return [j for j in jobs if not only or any(j["regime"].startswith(s) for s in only)]


def main():
    random.seed(0)
    concepts = {json.loads(l)["cid"]: json.loads(l) for l in open(root / "concepts.jsonl")}
    jobs = dataset_jobs() if SOURCE == "dataset" else probe_jobs()
    print(f"{len(jobs)} images, {sum(len(j['ask']) for j in jobs)} questions, model {MODEL}", flush=True)

    # T4/P100 have no bf16 and Qwen2.5-VL can be unstable in fp16, so default to fp32
    # split over the GPUs; 7B only fits in fp16 (NaN rows are recorded per answer)
    dtype = getattr(torch, os.environ.get("DTYPE", "float32"))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype, device_map="auto")
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    letter_ids = [proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    out_f = open(OUT / "readout.jsonl", "w")
    for n, it in enumerate(jobs):
        attrs = it["ask"]
        orders = {a: random.sample(range(len(QUESTIONS[a][1]) + 1), len(QUESTIONS[a][1]) + 1) for a in attrs}
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(a, orders[a])}]}],
                                          tokenize=False, add_generation_prompt=True) for a in attrs]
        img = it["load"]()
        batch = proc(text=texts, images=[img] * len(attrs), return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1].float()
        # probe jobs carry their own (fresh) concept; dataset jobs look it up
        c = {"cid": it["concept_id"], "attrs": it["attrs"]} if "attrs" in it else concepts[it["concept_id"]]
        for a, row in zip(attrs, logits):
            canon = [v for v, _ in QUESTIONS[a][1]] + ["cant_tell"]
            values = [canon[j] for j in orders[a]]          # value shown at each letter
            probs = torch.softmax(row[letter_ids[:len(values)]], -1).tolist()
            out_f.write(json.dumps({
                "item": it["id"], "regime": it["regime"], "concept": c["cid"], "attr": a,
                "visible": a in it["shows"], "drawn": it["drawn"][a], "true": c["attrs"][a],
                "pred": values[max(range(len(values)), key=probs.__getitem__)],
                "p": dict(zip(map(str, values), probs)), "nan": bool(torch.isnan(row).any())}) + "\n")
        if n % 50 == 0:
            print(f"{n}/{len(jobs)}", flush=True)
    out_f.close()
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(OUT / "readout.jsonl")]
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        k = (r["regime"], r["attr"], "visible" if r["visible"] else "hidden")
        g = agg[k]
        g["n"] += 1
        g["nan"] += r["nan"]
        if r["visible"]:
            g["acc"] += r["pred"] == str(r["true"]) or r["pred"] == r["true"]
        else:
            g["p_cant_tell"] += r["p"]["cant_tell"]
            g["pred_cant_tell"] += r["pred"] == "cant_tell"
            g["pred_neutral"] += str(r["pred"]) == str(NEUTRAL_DEFAULTS[r["attr"]])
            g["pred_true"] += str(r["pred"]) == str(r["true"])     # leak check: should be ~chance
            g["pred_drawn"] += r["drawn"] is not None and str(r["pred"]) == str(r["drawn"])
    table = []
    for (regime, attr, status), g in sorted(agg.items()):
        n = g.pop("n")
        table.append({"regime": regime, "attr": attr, "status": status, "n": int(n),
                      **{k: round(v / n, 3) for k, v in g.items()}})
    (OUT / "readout_summary.json").write_text(json.dumps(table, indent=1))
    for t in table:
        print(t)


if __name__ == "__main__":
    main()
