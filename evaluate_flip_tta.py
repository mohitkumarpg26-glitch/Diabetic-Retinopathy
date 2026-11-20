"""Compare single-view and horizontal-flip averaged inference on validation only."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from retina_project import (
    RetinaDataset,
    get_device,
    load_checkpoint,
    metrics,
    probabilities,
    read_rows,
    verify_data,
)


def main():
    splits, fingerprint = verify_data()
    run = ROOT / "runs" / "baseline_finetune"
    model, checkpoint = load_checkpoint(run / "best.pt")
    if checkpoint["config"]["split_fingerprint"] != fingerprint:
        raise RuntimeError("Checkpoint split fingerprint does not match current splits")
    device = get_device("directml")
    model.to(device).eval()
    loader = DataLoader(RetinaDataset(splits["validation"]), batch_size=4, num_workers=0)
    labels, ids = [], []
    single_prob, averaged_prob = [], []
    # DirectML's current backend has version-counter issues with inference_mode.
    with torch.no_grad():
        for batch_index, (images, targets, image_ids) in enumerate(loader, 1):
            batch_size = len(images)
            paired = torch.cat((images, images.flip(-1)), dim=0).to(device)
            logits_both = model(paired)
            p_both = probabilities(logits_both, checkpoint["config"]["mode"])
            p_single, p_flip = p_both[:batch_size], p_both[batch_size:]
            single_prob.extend(p_single.cpu().tolist())
            averaged_prob.extend(((p_single + p_flip) / 2).cpu().tolist())
            labels.extend(targets.tolist())
            ids.extend(image_ids)
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"Processed {batch_index}/{len(loader)} validation batches", flush=True)

    single_pred = [max(range(5), key=row.__getitem__) for row in single_prob]
    tta_pred = [max(range(5), key=row.__getitem__) for row in averaged_prob]
    result = {
        "run": "baseline_finetune",
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "validation",
        "images": len(labels),
        "single_view": metrics(labels, single_pred),
        "horizontal_flip_average": metrics(labels, tta_pred),
        "note": "Validation-only exploratory inference comparison; no weights or training data changed.",
    }
    out = ROOT / "results" / "decoding_improvement"
    out.mkdir(parents=True, exist_ok=True)
    (out / "flip_tta_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "flip_tta_validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "true_grade", "single_view_grade", "flip_tta_grade"])
        writer.writerows(zip(ids, labels, single_pred, tta_pred))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
