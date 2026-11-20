from pathlib import Path
from collections import Counter
import csv

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
csv_path = raw / "train.csv"
images = raw / "train_images"

with csv_path.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

counts = Counter(row["diagnosis"] for row in rows)
missing = [
    row["id_code"] + ".png"
    for row in rows
    if not (images / (row["id_code"] + ".png")).is_file()
]

print("Labeled rows:", len(rows))
print("Images in train_images:", len(list(images.glob("*.png"))))
print("Grade counts:", dict(sorted(counts.items())))
print("Missing training images:", len(missing))
if missing:
    print("Examples:", missing[:5])
