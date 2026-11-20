# Diabetic retinopathy grading: preliminary research results

Author: Mohit Kumar, 2026PAD2529

## Data integrity
Byte-identical duplicate groups were checked before splitting. Extra copies were removed; all images in conflicting-label groups were excluded pending expert review. This changes the evaluation population and may make it easier; report exclusions explicitly. Near-duplicates and patient-level overlap have not been ruled out. No patient IDs or demographic attributes were available in the supplied CSV.

## Model results

### baseline_finetune
```json
{
  "accuracy": 0.8022598870056498,
  "macro_f1": 0.6147018192631142,
  "quadratic_weighted_kappa": 0.858519786580695,
  "mean_absolute_grade_error": 0.2711864406779661,
  "undergraded_by_2plus_all": 0.03766478342749529,
  "undergraded_by_2plus_grade2plus": 0.09569377990430622,
  "n_images": 531,
  "split": "test",
  "epoch": 4,
  "mode": "baseline",
  "run": "baseline_finetune"
}
```

### ordinal_finetune
```json
{
  "accuracy": 0.568738229755179,
  "macro_f1": 0.24050575062837992,
  "quadratic_weighted_kappa": 0.6270645007816217,
  "mean_absolute_grade_error": 0.783427495291902,
  "undergraded_by_2plus_all": 0.1958568738229755,
  "undergraded_by_2plus_grade2plus": 0.49760765550239233,
  "n_images": 531,
  "split": "test",
  "epoch": 2,
  "mode": "ordinal",
  "run": "ordinal_finetune"
}
```

## Interpretation and ethics
The two-or-more-grade undergrading rate is reported over all images and separately over true grades 2-4. This is an error proxy, not a measured patient outcome.
Report class-specific recall alongside macro F1, weighted kappa and mean grade error. No demographic fairness claim is supported. Softmax/ordinal scores are uncalibrated. No clinical deployment, treatment recommendation, or guarantee of screening safety is justified.
Select models using validation only. Fix the training budget before evaluating the held-out test set; do not tune models using test results. Compare matching seeds, backbone settings and epoch budgets.
A head-only run is transfer learning with a frozen backbone. Longer fine-tuning and external validation remain future work. Report negative or inconclusive comparisons honestly.

## Reference
Farag et al. (2022), IEEE Access. https://doi.org/10.1109/ACCESS.2022.3165193

This implementation uses DenseNet121 and an ordered cumulative-logit head; it does not reproduce the paper's CBAM architecture.

## AI assistance disclosure
Code and workflow were drafted with Codex. Keep the chat/prompts with the project, review the implementation, and insert actual measured results into the final submission.