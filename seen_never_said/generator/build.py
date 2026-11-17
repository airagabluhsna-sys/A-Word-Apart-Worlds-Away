"""Build a Tier 0 teaching dataset.

    py -3.11 -m seen_never_said.generator.build --config configs/pilot.yaml

Output layout (under cfg.out_dir):
    concepts.jsonl            one record per concept (attributes, name, split, role)
    items_{V,T,S}.jsonl       teaching items; image items carry a minimal caption
    images/{regime}/{cid}/    PNGs
    audit.json                problems found by the information-matching audit
    sheets/                   contact sheets for eyeballing
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from .audit import audit_items, parse_regime, plan
from .concepts import Concept, latin_square_split, sample_concepts
from .describe import caption, describe_many
from .lexicon import check_lexicon
from .names import assign_names, generate_names
from .ontology import NEUTRAL_DEFAULTS, UNHIDEABLE
from .render_vector import render_image


def _render_job(args):
    concept_dict, visible, seed, size, hide_mode, absence, size_ref, path = args
    c = Concept(**concept_dict)
    img, meta = render_image(c, set(visible), seed, size, hide_mode, absence, size_ref)
    img.save(path)
    return path, meta


def build(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)

    lex_errors = check_lexicon()
    if lex_errors:
        raise SystemExit(f"lexicon is ambiguous: {lex_errors[:5]}")

    concepts = sample_concepts(cfg["n_test"], cfg.get("n_distractor", 0), cfg.get("min_hamming", 3), seed)
    wordlist = None
    if cfg.get("wordlist"):
        wordlist = Path(cfg["wordlist"]).read_text(encoding="utf-8").split()
    names = generate_names(len(concepts), seed=cfg.get("name_seed", seed), wordlist=wordlist,
                           max_abs_symbolism=cfg.get("max_abs_symbolism", 0.2))
    assign_names(concepts, names, seed=cfg.get("name_seed", seed))
    # Absent-mode hiding can't hide shape/pattern/material, so by default the S
    # split always shows them in images; every S arm uses the same split.
    latin_square_split(concepts, seed, always_visual=cfg.get("s_always_visual", UNHIDEABLE))

    with open(out / "concepts.jsonl", "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c.to_dict()) + "\n")

    n_img, n_txt, size = cfg["images_per_concept"], cfg["texts_per_concept"], cfg.get("image_size", 448)
    all_items = []
    for regime in cfg["regimes"]:
        base, hide_mode = parse_regime(regime)
        # seeded by the base regime: the S arms get the same image seeds,
        # captions and texts, and differ only in how hidden attributes are drawn
        rng = random.Random(f"{seed}-{base}")
        jobs, items = [], []
        for c in concepts:
            p = plan(c, regime)
            if p["image"]:
                d = out / "images" / regime / c.cid
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n_img):
                    path = d / f"{i:03d}.png"
                    img_seed = rng.randrange(2**31)
                    jobs.append((c.to_dict(), sorted(p["image"]), img_seed, size, hide_mode,
                                 cfg.get("absence"), cfg.get("size_reference", "matched"), str(path)))
                    items.append({"id": f"{regime}/{c.cid}/img{i:03d}", "concept_id": c.cid, "name": c.name,
                                  "regime": regime, "hide_mode": hide_mode, "modality": "image",
                                  "image": str(path.relative_to(out)).replace("\\", "/"),
                                  "caption": caption(c.name, rng), "shows": sorted(p["image"]), "seed": img_seed})
            if p["text"]:
                texts = describe_many(c, sorted(p["text"]), n_txt, seed=rng.randrange(2**31))
                for i, t in enumerate(texts):
                    items.append({"id": f"{regime}/{c.cid}/txt{i:03d}", "concept_id": c.cid, "name": c.name,
                                  "regime": regime, "modality": "text", "text": t, "states": sorted(p["text"])})

        with ProcessPoolExecutor(max_workers=cfg.get("workers")) as ex:
            metas = dict(ex.map(_render_job, jobs, chunksize=4))
        for it in items:
            if it["modality"] == "image":
                it.update(drawn=metas[str(out / it["image"])]["drawn"],
                          reference=metas[str(out / it["image"])]["reference"])
        with open(out / f"items_{regime}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        all_items += items
        print(f"[{regime}] {sum(i['modality'] == 'image' for i in items)} images, "
              f"{sum(i['modality'] == 'text' for i in items)} texts")

    problems = audit_items(concepts, all_items)
    # In S-neutral, a text-taught attribute whose true value equals the default
    # is accidentally confirmed by the images; analyses should drop or flag these.
    coincide = [{"concept": c.cid, "attr": a} for c in concepts for a in c.split["T"]
                if c.attrs[a] == NEUTRAL_DEFAULTS[a]]
    (out / "audit.json").write_text(json.dumps({"n_items": len(all_items), "n_problems": len(problems),
                                                "problems": problems, "neutral_coincidences": coincide},
                                               indent=2), encoding="utf-8")
    print(f"audit: {len(problems)} problems across {len(all_items)} items; "
          f"{len(coincide)} text-taught attributes coincide with the neutral default")
    return concepts, all_items, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sheets", action="store_true", help="also write contact sheets")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    concepts, items, _ = build(cfg)
    if args.sheets:
        from .preview import contact_sheets
        contact_sheets(cfg["out_dir"], concepts, items)


if __name__ == "__main__":
    main()
