# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CASDET.casdet import CasdetResult  # noqa: E402
from CASDET.experiments import _save_layer_effects  # noqa: E402


def test_save_layer_effects_pads_ragged_responses(tmp_path):
    short = CasdetResult("a", True, np.zeros(2), np.ones((3, 2)))
    long = CasdetResult("b", False, np.zeros(4), np.full((3, 4), 2.0))

    path = tmp_path / "ragtruth_llama2_7b_layers.npz"
    _save_layer_effects(path, [short, long])

    z = np.load(path, allow_pickle=True)
    effects = z["layer_effects"]
    assert effects.shape == (2, 3, 4)
    assert np.all(effects[0, :, :2] == 1.0)
    assert np.isnan(effects[0, :, 2:]).all()
    assert np.all(effects[1] == 2.0)
    assert z["labels"].tolist() == [1.0, 0.0]


def test_save_layer_effects_ignores_empty_batch(tmp_path):
    path = tmp_path / "empty_layers.npz"
    _save_layer_effects(path, [])
    assert not path.exists()
