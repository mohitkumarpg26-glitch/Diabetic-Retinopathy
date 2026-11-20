from pathlib import Path
from PIL import Image, ImageOps
import csv

root = Path(__file__).resolve().parents[1]
source = root / "data/raw/train_images"
target = root / "data/processed/images224"
target.mkdir(parents=True, exist_ok=True)

rows = []
for split in ("train", "validation", "test"):
    with (root / f"data/processed/{split}_split.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows.extend(csv.DictReader(f))

for i, row in enumerate(rows, 1):
    filename = row["id_code"] + ".png"
    with Image.open(source / filename) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.pad(
            image, (224, 224),
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0)
        )
        image.save(target / filename)

    if i % 250 == 0:
        print(f"Prepared {i}/{len(rows)}", flush=True)

print(f"Done: {len(rows)} images prepared.")
