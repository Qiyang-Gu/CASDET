# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CASDET.reference import (  # noqa: E402
    _risk_percentile,
    fit_reference,
    score_batch_with_reference,
    score_with_reference,
)

R1 = np.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0]])
R2 = np.array([[0.0, 1.0],
               [2.0, 3.0],
               [4.0, 5.0]])
Q1 = np.array([[2.0, 1.0],
               [2.0, 1.0],
               [2.0, 1.0]])
Q2 = np.array([[-1.0, 0.0],
               [0.0, 1.0],
               [1.0, 2.0]])


def test_risk_percentile_hand_computed():
    ref = np.array([1.0, 2.0, 3.0])
    assert _risk_percentile(ref, 1.0) == 1.0
    assert _risk_percentile(ref, 2.0) == 0.5
    assert _risk_percentile(ref, 3.0) == 0.0
    assert _risk_percentile(ref, 0.5) == 1.0
    assert _risk_percentile(ref, 3.5) == 0.0


def test_fit_reference_returns_sorted_values():
    ref = fit_reference([R1, R2])
    assert ref.breadth.size == 2
    assert ref.depth.size == 2
    assert ref.intensity.size == 2
    assert np.all(np.diff(ref.breadth) >= 0)
    assert np.all(np.diff(ref.depth) >= 0)
    assert np.all(np.diff(ref.intensity) >= 0)
    assert ref.sigma.shape == (3,)


def test_fit_reference_rejects_empty():
    with pytest.raises(ValueError):
        fit_reference([])


def test_scores_are_bounded_and_batch_matches_single():
    ref = fit_reference([R1, R2])
    single = score_with_reference(Q1, ref)
    batch = score_batch_with_reference([Q1, Q2], ref)
    assert 0.0 <= single <= 1.0
    assert batch.shape == (2,)
    assert np.all((batch >= 0.0) & (batch <= 1.0))
    assert batch[0] == pytest.approx(single)


def test_score_is_independent_of_other_evaluated_responses():
    ref = fit_reference([R1, R2])
    first = score_batch_with_reference([Q1, Q2], ref)[0]
    second = score_batch_with_reference([Q1, R2], ref)[0]
    assert first == pytest.approx(second)
