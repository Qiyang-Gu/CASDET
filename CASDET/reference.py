# -*- coding: utf-8 -*-
"""Reference-calibrated single-response scoring.

The main pipeline estimates sigma_l and the relative positions on the evaluated
set.  For deployment with isolated responses, a calibration is fitted once on
an unlabeled reference set and then held fixed; each response is scored from its
own statistics and the fixed calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .casdet import (
    _layer_scale,
    influence_breadth,
    influence_depth,
    influence_intensity,
)


@dataclass(frozen=True)
class ReferenceCalibration:
    """Fixed sigma_l and sorted reference values for the three statistics."""
    sigma: np.ndarray
    breadth: np.ndarray
    depth: np.ndarray
    intensity: np.ndarray


def _sorted_finite(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sort(values[np.isfinite(values)])


def fit_reference(layer_effects: Sequence[np.ndarray]) -> ReferenceCalibration:
    """Fit sigma_l and the three reference distributions on unlabeled responses."""
    grid = list(layer_effects)
    if not grid:
        raise ValueError("reference set is empty")
    sigma = _layer_scale(grid)
    return ReferenceCalibration(
        sigma=sigma,
        breadth=_sorted_finite([influence_breadth(e) for e in grid]),
        depth=_sorted_finite([influence_depth(e) for e in grid]),
        intensity=_sorted_finite([influence_intensity(e, sigma) for e in grid]),
    )


def _risk_percentile(reference: np.ndarray, value: float) -> float:
    """Risk-oriented relative position of one value against a sorted reference."""
    n = int(reference.size)
    if n == 0 or not np.isfinite(value):
        return float("nan")
    count_ge = n - int(np.searchsorted(reference, value, side="left"))
    if n == 1:
        return 0.0
    return float(min(1.0, max(0.0, (count_ge - 1) / (n - 1))))


def score_with_reference(layer_effects: np.ndarray,
                         reference: ReferenceCalibration) -> float:
    """Score one response using only its own statistics and a fixed calibration."""
    e = np.asarray(layer_effects, dtype=np.float64)
    values = [
        _risk_percentile(reference.breadth, influence_breadth(e)),
        _risk_percentile(reference.depth, influence_depth(e)),
        _risk_percentile(reference.intensity,
                         influence_intensity(e, reference.sigma)),
    ]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_batch_with_reference(layer_effects: Sequence[np.ndarray],
                               reference: ReferenceCalibration) -> np.ndarray:
    """Score each response independently with the same fixed calibration."""
    return np.asarray([score_with_reference(e, reference) for e in layer_effects],
                      dtype=np.float64)
