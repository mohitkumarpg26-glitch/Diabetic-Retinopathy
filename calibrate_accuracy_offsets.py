"""Explore validation-only class offsets for baseline accuracy; never retrains weights."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

from retina_project import metrics


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = np.asarray([int(r["true_grade"]) for r in rows], dtype=int)
    probs = np.asarray([[float(r[f"p_grade_{k}"]) for k in range(5)] for r in rows])
    return labels, probs


def predict(probs, offsets):
    logits = np.log(np.maximum(probs, 1e-12)) + np.r_[0.0, np.asarray(offsets)]
    return logits.argmax(axis=1)


def fit_offsets(labels, probs, seed):
    log_probs = np.log(np.maximum(probs, 1e-12))
    def loss(offsets):
        logits = log_probs + np.r_[0.0, offsets]
        return 1.0 - np.mean(logits.argmax(axis=1) == labels)
    result = differential_evolution(
        loss, [(-1.5, 1.5)] * 4, seed=seed, popsize=10,
        maxiter=80, tol=1e-5, polish=False, x0=np.zeros(4),
        updating="immediate", workers=1,
    )
    return result.x.tolist()


def main():
    val_y, val_p = read_predictions(ROOT / "runs/baseline_finetune/evaluation_validation/predictions.csv")
    test_y, test_p = read_predictions(ROOT / "runs/baseline_finetune/evaluation_test/predictions.csv")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.empty_like(val_y)
    fold_offsets = []
    for fold, (fit_idx, hold_idx) in enumerate(folds.split(val_p, val_y), 1):
        offsets = fit_offsets(val_y[fit_idx], val_p[fit_idx], seed=41 + fold)
        oof[hold_idx] = predict(val_p[hold_idx], offsets)
        fold_offsets.append(offsets)
        print(f"Finished validation fold {fold}/5", flush=True)

    frozen_offsets = fit_offsets(val_y, val_p, seed=42)
    previous_offsets = json.loads((ROOT / "results/decoding_improvement/frozen_selection.json").read_text())["params"]
    output = {
        "objective": "validation accuracy",
        "method": "five-fold out-of-fold class logit offsets; DenseNet weights unchanged",
        "seed": 42,
        "test_used_for_selection": False,
        "validation_oof": {
            "raw": metrics(val_y, val_p.argmax(axis=1)),
            "accuracy_offsets": metrics(val_y, oof),
        },
        "fold_offsets": fold_offsets,
        "frozen_offsets_fit_on_all_validation": frozen_offsets,
        "saved_qwk_offsets": previous_offsets,
        "test_check_after_freeze": {
            "raw": metrics(test_y, test_p.argmax(axis=1)),
            "accuracy_offsets": metrics(test_y, predict(test_p, frozen_offsets)),
        },
        "limitations": "Checkpoint was previously selected using this validation set. Test was previously examined in the project; this is exploratory, not an independent confirmatory comparison.",
    }
    out = ROOT / "results/decoding_improvement/accuracy_offsets_exploratory.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
