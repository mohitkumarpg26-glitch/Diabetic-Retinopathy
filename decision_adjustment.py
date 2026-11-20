"""Frozen validation-selected baseline decision rule; no training or fitting here."""
import numpy as np

def adjust_scores(scores, config):
    if config['run'] != 'baseline_finetune' or config['selected'] != 'class_logit_offsets':
        raise ValueError('Unsupported score adjustment configuration')
    offsets = np.r_[0., np.asarray(config['params'], dtype=float)]
    values = np.asarray(scores, dtype=float)
    if values.shape[-1] != 5 or offsets.shape != (5,) or not np.isfinite(offsets).all():
        raise ValueError('Expected five scores and four finite offsets')
    logits = np.log(np.maximum(values, 1e-12)) + offsets
    logits -= logits.max(axis=-1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=-1, keepdims=True)
