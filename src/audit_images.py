from pathlib import Path
from collections import defaultdict
from PIL import Image
import csv
import hashlib
import json

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
results = root / "results"
results.mkdir(exist_ok=True)

with (raw / "train.csv").open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

hashes = defaultdict(list)
unreadable = []

for i, row in enumerate(rows, 1):
    path = raw / "train_images" / (row["id_code"] + ".png")
    try:
        with Image.open(path) as image:
            image.load()
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        hashes[digest].append(row["id_code"])
    except Exception as error:
        unreadable.append({"id_code": row["id_code"], "error": str(error)})

    if i % 250 == 0:
        print(f"Checked {i}/{len(rows)}", flush=True)

duplicates = [ids for ids in hashes.values() if len(ids) > 1]
report = {
    "images_checked": len(rows),
    "unreadable_images": unreadable,
    "exact_duplicate_groups": duplicates
}
report_path = results / "image_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("Unreadable images:", len(unreadable))
print("Exact duplicate groups:", len(duplicates))
print("Report saved:", report_path)
