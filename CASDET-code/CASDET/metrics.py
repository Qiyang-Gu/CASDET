# -*- coding: utf-8 -*-

"""Evaluation metrics and Page's L trend test."""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import norm, pearsonr, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

def _finite_pairs(labels: Sequence[int],
                  scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores have different lengths: %d vs %d" % (y.size, s.size))
    ok = np.isfinite(s)
    dropped = int((~ok).sum())
    if dropped:
        warnings.warn("%d samples have non-finite scores (NaN/inf); "
                      "they were excluded from the metric computation" % dropped)
    return y[ok], s[ok]

def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve."""
    y, s = _finite_pairs(labels, scores)
    if y.size == 0 or len(np.unique(y)) < 2:
        warnings.warn("AUROC is undefined when the valid labels contain a single class or no valid samples; returning NaN")
        return float("nan")
    return float(roc_auc_score(y, s))

def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve."""
    y, s = _finite_pairs(labels, scores)
    if y.size == 0 or len(np.unique(y)) < 2:
        warnings.warn("AUPRC is undefined when the valid labels contain a single class or no valid samples; returning NaN")
        return float("nan")
    return float(average_precision_score(y, s))

def pcc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    y, s = _finite_pairs(labels, scores)
    y = y.astype(np.float64)
    if y.size == 0 or np.std(y) == 0 or np.std(s) == 0:
        warnings.warn("PCC is undefined when the valid labels or scores have zero variance; returning NaN")
        return float("nan")
    r, _ = pearsonr(y, s)
    return float(r)

def compute_metrics(labels: Sequence[int],
                    scores: Sequence[float]) -> Dict[str, float]:
    """Compute AUROC, AUPRC, and PCC."""
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "pcc": pcc(labels, scores),
    }

def page_l_test(
    effects: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Page's L test for an ordered trend across conditions."""
    effects = np.asarray(effects, dtype=np.float64)
    if effects.ndim != 2:
        raise ValueError("effects must be a 2-D array [n_responses, n_conditions]")
    n, k = effects.shape
    if k < 3:
        raise ValueError(f"Page's L test requires at least 3 ordered conditions, got {k}")
    ranks = np.apply_along_axis(rankdata, axis=1, arr=effects)
    R_j = ranks.sum(axis=0)
    j = np.arange(1, k + 1, dtype=np.float64)
    L = float((j * R_j).sum())
    expected = float(n * k * (k + 1) ** 2 / 4.0)
    variance = float(n * (k ** 3 - k) ** 2 / (144.0 * (k - 1)))
    z = (L - expected - 0.5) / math.sqrt(variance)
    log_p = float(norm.logsf(z))
    p_value = float(math.exp(log_p)) if log_p > -745.0 else 0.0
    p_display = ("< 1e-300" if log_p < -690.78
                 else f"{p_value:.3g}")
    return {
        "page_l": L,
        "expected_l": expected,
        "variance_l": variance,
        "z": z,
        "p_value": p_value,
        "log_p_value": log_p,
        "p_display": p_display,
        "significant": p_value < alpha,
        "n_responses": n,
        "n_conditions": k,
    }

