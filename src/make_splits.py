from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
out = root / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)

with (raw / "train.csv").open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
seen = set()

for row in rows:
    image_id = row["id_code"].strip()
    grade = row["diagnosis"].strip()

    if image_id in seen:
        raise ValueError(f"Duplicate image ID: {image_id}")
    seen.add(image_id)

    if grade not in {"0", "1", "2", "3", "4"}:
        raise ValueError(f"Unexpected diagnosis grade: {grade}")
    if not (raw / "train_images" / f"{image_id}.png").is_file():
        raise FileNotFoundError(f"Missing image: {image_id}.png")

    groups[grade].append({"id_code": image_id, "diagnosis": grade})

rng = random.Random(42)
splits = {"train": [], "validation": [], "test": []}

for grade in sorted(groups, key=int):
    items = groups[grade]
    rng.shuffle(items)
    n_train = int(len(items) * 0.70)
    n_validation = int(len(items) * 0.15)

    splits["train"].extend(items[:n_train])
    splits["validation"].extend(items[n_train:n_train + n_validation])
    splits["test"].extend(items[n_train + n_validation:])

for name, items in splits.items():
    rng.shuffle(items)
    path = out / f"{name}_split.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(items)

    print(f"{name}: {len(items)} images; grade counts: {dict(sorted(Counter(r['diagnosis'] for r in items).items()))}")

print(f"Split files saved in: {out}")
