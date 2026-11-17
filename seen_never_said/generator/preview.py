"""Contact sheets: one row per concept, first few images, with name and attributes."""

from pathlib import Path

from PIL import Image, ImageDraw

THUMB, PER_ROW, LABEL_W = 160, 6, 300


def contact_sheets(out_dir, concepts, items, max_concepts=12):
    out = Path(out_dir)
    (out / "sheets").mkdir(exist_ok=True)
    for regime in sorted({i["regime"] for i in items}):
        imgs = [i for i in items if i["regime"] == regime and i["modality"] == "image"]
        if not imgs:
            continue
        rows = concepts[:max_concepts]
        sheet = Image.new("RGB", (LABEL_W + PER_ROW * THUMB, len(rows) * THUMB), "white")
        d = ImageDraw.Draw(sheet)
        for r, c in enumerate(rows):
            mine = [i for i in imgs if i["concept_id"] == c.cid][:PER_ROW]
            shown = mine[0]["shows"] if mine else []
            lines = [f"{c.name} ({c.cid}, {c.role})"] + [
                f"{'*' if a in shown else ' '} {a}: {v}" for a, v in c.attrs.items()]
            d.multiline_text((6, r * THUMB + 6), "\n".join(lines), fill="black", spacing=2)
            for k, it in enumerate(mine):
                im = Image.open(out / it["image"]).resize((THUMB, THUMB))
                sheet.paste(im, (LABEL_W + k * THUMB, r * THUMB))
        sheet.save(out / "sheets" / f"sheet_{regime}.png")
