from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import random
import shutil

root = Path(__file__).resolve().parents[1]
out = root / "data/processed"
audit = json.loads((root / "results/image_audit.json").read_text())

with (root / "data/raw/train.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

labels = {r["id_code"]: r["diagnosis"] for r in rows}
excluded = {}

for group in audit["exact_duplicate_groups"]:
    if len({labels[i] for i in group}) > 1:
        for image_id in group:
            excluded[image_id] = "conflicting_labels"
    else:
        for image_id in sorted(group)[1:]:
            excluded[image_id] = "extra_identical_copy"

clean = [r for r in rows if r["id_code"] not in excluded]

backup = out / "original_splits"
backup.mkdir(exist_ok=True)
for name in ("train", "validation", "test"):
    source = out / f"{name}_split.csv"
    target = backup / source.name
    if source.exists() and not target.exists():
        shutil.copy2(source, target)

groups = defaultdict(list)
for row in clean:
    groups[row["diagnosis"]].append(row)

rng = random.Random(42)
splits = {"train": [], "validation": [], "test": []}
for grade in sorted(groups, key=int):
    items = groups[grade]
    rng.shuffle(items)
    a = int(len(items) * 0.70)
    b = a + int(len(items) * 0.15)
    splits["train"].extend(items[:a])
    splits["validation"].extend(items[a:b])
    splits["test"].extend(items[b:])

included = {r["id_code"] for r in clean}
assert all(
    len(included.intersection(group)) <= 1
    for group in audit["exact_duplicate_groups"]
)

for name, items in splits.items():
    rng.shuffle(items)
    with (out / f"{name}_split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(
            {"id_code": r["id_code"], "diagnosis": r["diagnosis"]}
            for r in items
        )
    print(name, len(items), "images; grades:",
          dict(sorted(Counter(r["diagnosis"] for r in items).items())))

report = {
    "original_images": len(rows),
    "retained_images": len(clean),
    "exclusion_counts": dict(Counter(excluded.values())),
    "excluded_image_ids": excluded,
    "seed": 42
}
(root / "results/split_cleaning.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("Retained images:", len(clean))
print("Excluded:", dict(Counter(excluded.values())))
print("Verified: no exact duplicates remain in the cleaned splits.")
