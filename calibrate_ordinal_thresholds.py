"""
Diabetic Retinopathy - Ordinal Threshold Calibration Script
Author: Mohit Kumar (2026PAD2529)

Objective:
Optimize continuous severity thresholds on the validation set using Nelder-Mead
to maximize Quadratic Weighted Kappa (QWK), then evaluate on the untouched test set.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent

def run_calibration():
    val_path = ROOT / 'runs' / 'ordinal_finetune' / 'evaluation_validation' / 'predictions.csv'
    test_path = ROOT / 'runs' / 'ordinal_finetune' / 'evaluation_test' / 'predictions.csv'

    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prediction files: {val_path} or {test_path}")

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    p_cols = ['p_grade_0', 'p_grade_1', 'p_grade_2', 'p_grade_3', 'p_grade_4']
    val_probs = val_df[p_cols].values
    test_probs = test_df[p_cols].values

    val_y = val_df['true_grade'].values
    test_y = test_df['true_grade'].values

    # Continuous expected grade score: E[grade] = sum_{k=0}^4 (k * P(grade=k))
    val_exp = val_probs @ np.arange(5)
    test_exp = test_probs @ np.arange(5)

    def calc_metrics(y_true, y_pred):
        eligible = y_true >= 2
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'quadratic_weighted_kappa': float(cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=list(range(5)))),
            'mean_absolute_grade_error': float(mean_absolute_error(y_true, y_pred)),
            'undergraded_by_2plus_all': float(((y_true - y_pred) >= 2).mean()),
            'undergraded_by_2plus_grade2plus': float(((y_true[eligible] - y_pred[eligible]) >= 2).mean()) if eligible.any() else None,
            'n_images': int(len(y_true))
        }

    # 1. Uncalibrated argmax baseline
    val_raw_preds = val_probs.argmax(1)
    test_raw_preds = test_probs.argmax(1)

    # 2. Threshold optimization on VALIDATION set strictly
    def loss_qwk(th):
        if not (th[0] < th[1] < th[2] < th[3]):
            return 999.0
        preds = np.digitize(val_exp, th)
        return -cohen_kappa_score(val_y, preds, weights='quadratic', labels=list(range(5)))

    init_th = [0.5, 1.5, 2.5, 3.5]
    res = minimize(loss_qwk, init_th, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    opt_th = res.x.tolist()

    # 3. Apply calibrated thresholds
    val_cal_preds = np.digitize(val_exp, opt_th)
    test_cal_preds = np.digitize(test_exp, opt_th)

    results = {
        'optimization': {
            'method': 'Nelder-Mead',
            'objective': 'maximize_validation_qwk',
            'initial_thresholds': init_th,
            'calibrated_thresholds': opt_th
        },
        'validation': {
            'raw_argmax': calc_metrics(val_y, val_raw_preds),
            'calibrated_expected_grade': calc_metrics(val_y, val_cal_preds)
        },
        'test': {
            'raw_argmax': calc_metrics(test_y, test_raw_preds),
            'calibrated_expected_grade': calc_metrics(test_y, test_cal_preds)
        }
    }

    out_file = ROOT / 'results' / 'calibrated_ordinal_summary.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Calibration complete! Results written to: {out_file}\n")
    print(f"Optimal Thresholds: {[round(t, 4) for t in opt_th]}")
    print("\n=== HELD-OUT TEST EVALUATION COMPARISON ===")
    print(f"{'Metric':<35} | {'Raw Argmax':<12} | {'Calibrated Thresholds':<20}")
    print("-" * 72)
    m_raw = results['test']['raw_argmax']
    m_cal = results['test']['calibrated_expected_grade']
    print(f"{'Accuracy':<35} | {m_raw['accuracy']*100:.2f}%{'':<6} | {m_cal['accuracy']*100:.2f}%")
    print(f"{'Quadratic Weighted Kappa (QWK)':<35} | {m_raw['quadratic_weighted_kappa']:.4f}{'':<6} | {m_cal['quadratic_weighted_kappa']:.4f}")
    print(f"{'Macro F1':<35} | {m_raw['macro_f1']:.4f}{'':<6} | {m_cal['macro_f1']:.4f}")
    print(f"{'Mean Absolute Error (MAE)':<35} | {m_raw['mean_absolute_grade_error']:.4f}{'':<6} | {m_cal['mean_absolute_grade_error']:.4f}")
    print(f"{'Undergraded >=2 (All Images)':<35} | {m_raw['undergraded_by_2plus_all']*100:.2f}%{'':<6} | {m_cal['undergraded_by_2plus_all']*100:.2f}%")
    print(f"{'Undergraded >=2 (Grades 2-4 Only)':<35} | {m_raw['undergraded_by_2plus_grade2plus']*100:.2f}%{'':<6} | {m_cal['undergraded_by_2plus_grade2plus']*100:.2f}%")

if __name__ == '__main__':
    run_calibration()
