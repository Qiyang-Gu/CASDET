# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import warnings
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CASDET.metrics import compute_metrics, page_l_test  # noqa: E402
from CASDET.casdet import (  # noqa: E402
    _layer_scale,
    fuse_profile_statistics,
    fuse_statistics,
    influence_breadth,
    influence_depth,
    influence_intensity,
    layer_relative_change,
    profile_statistics,
    relative_position,
    token_relative_effect,
)

def _hand_built_pair():
    logits = torch.stack([torch.log(torch.tensor([0.5, 0.3, 0.2])),
                          torch.log(torch.tensor([0.1, 0.6, 0.3]))])
    logits_hat = torch.stack([torch.log(torch.tensor([0.25, 0.35, 0.4])),
                              torch.log(torch.tensor([0.4, 0.3, 0.3]))])
    token_ids = torch.tensor([0, 0])
    return logits, logits_hat, token_ids

def test_layer_relative_change_logprob_hand_computed():
    logits, logits_hat, token_ids = _hand_built_pair()
    s = layer_relative_change(logits, logits_hat, token_ids)
    assert abs(s[0].item() - math.log(2.0)) < 1e-5
    assert abs(s[1].item() - math.log(0.25)) < 1e-5

    p = torch.softmax(logits.float(), dim=-1)[:, 0]
    q = torch.softmax(logits_hat.float(), dim=-1)[:, 0]
    assert torch.allclose(s, torch.log(p) - torch.log(q), atol=1e-6)

def test_token_relative_effect_is_layer_mean():
    layer_changes = torch.tensor([[0.5, -0.2, 1.0],
                                  [1.5, 0.2, 0.0]])
    s = token_relative_effect(layer_changes)
    expect = layer_changes.mean(dim=0)
    assert torch.allclose(s, expect, atol=1e-6)

E1 = np.array([[-1.0],
               [2.0]])
E2 = np.array([[1.0],
               [4.0]])
SIGMA = np.array([1.0, 1.0])

def test_influence_breadth_hand_computed():
    assert abs(influence_breadth(E1) - 0.5) < 1e-12
    assert abs(influence_breadth(E2) - 1.0) < 1e-12

def test_influence_breadth_is_invariant_to_layer_scale():
    assert abs(influence_breadth(E1 * 1e6) - 0.5) < 1e-12
    assert abs(influence_breadth(E1 * 1e-6) - 0.5) < 1e-12

def test_influence_breadth_counts_tokens_without_any_positive_layer():
    E = np.array([[1.0, -1.0], [-1.0, -2.0]])
    assert abs(influence_breadth(E) - 0.25) < 1e-12

def test_influence_depth_hand_computed():
    assert abs(influence_depth(E1) - 1.0) < 1e-12
    assert abs(influence_depth(E2) - 0.8) < 1e-12

def test_influence_depth_is_not_scaled_by_layer_count():
    E = np.array([[2.0], [1.0], [-1.0], [0.0]])
    assert abs(influence_depth(E) - 1.0 / 3.0) < 1e-12

def test_influence_depth_weights_by_positive_magnitude():
    bottom = np.array([[3.0], [1.0]])
    top = np.array([[1.0], [3.0]])
    assert abs(influence_depth(bottom) - 0.25) < 1e-12
    assert abs(influence_depth(top) - 0.75) < 1e-12

def test_influence_depth_zero_denominator_counts_as_zero():
    E = np.array([[2.0, -1.0],
                  [3.0, -2.0],
                  [1.0, -3.0]])
    assert abs(influence_depth(E) - 5.0 / 12.0) < 1e-12

def test_influence_depth_all_negative_is_zero():
    E = np.array([[-1.0, -2.0], [-1.0, -3.0]])
    assert abs(influence_depth(E) - 0.0) < 1e-12

def test_influence_intensity_hand_computed():
    assert abs(influence_intensity(E1, SIGMA) - math.sqrt(2.5)) < 1e-12
    assert abs(influence_intensity(E2, SIGMA) - math.sqrt(8.5)) < 1e-12

def test_influence_intensity_is_sign_agnostic():
    flipped = E1.copy()
    flipped[0, 0] = 1.0
    assert abs(influence_intensity(flipped, SIGMA)
               - math.sqrt((1.0 + 4.0) / 2.0)) < 1e-12
    assert abs(influence_intensity(-E2, SIGMA) - math.sqrt(8.5)) < 1e-12

def test_influence_intensity_averages_over_tokens_before_layers():
    E = np.array([[2.0, 4.0]])
    assert abs(influence_intensity(E, np.array([1.0])) - 3.0) < 1e-12

def test_influence_intensity_reads_layer_scale():
    assert abs(influence_intensity(E2, np.array([0.5, 1.0]))
               - math.sqrt(10.0)) < 1e-12

def test_profile_statistics_bundles_three_values():
    st = profile_statistics([E1, E2])
    assert abs(st[0].breadth - 0.5) < 1e-12
    assert abs(st[1].breadth - 1.0) < 1e-12
    assert abs(st[0].depth - 1.0) < 1e-12
    assert abs(st[1].depth - 0.8) < 1e-12
    assert abs(st[0].intensity - math.sqrt(2.5)) < 1e-12
    assert abs(st[1].intensity - math.sqrt(8.5)) < 1e-12

def test_layer_scale_pools_all_tokens_of_the_evaluation_set():
    a = np.array([[2.0, 4.0]])
    b = np.array([[1.0, 1.0]])
    sigma = _layer_scale([a, b])
    assert abs(sigma[0] - math.sqrt(1.5)) < 1e-12
    assert abs(sigma[0] - 1.0) > 1e-6

def test_profile_statistics_depend_on_the_whole_batch():
    a = np.array([[2.0, -1.0], [1.0, 0.0]])
    b = np.array([[0.0, 2.0], [1.0, 1.0]])
    alone = profile_statistics([a])[0]
    together = profile_statistics([a, b])[0]
    assert abs(alone.intensity - together.intensity) > 1e-6
    assert abs(alone.breadth - together.breadth) < 1e-12
    assert abs(alone.depth - together.depth) < 1e-12

def test_profile_statistics_zero_layer_scale_is_identity():
    st = profile_statistics([np.ones((2, 2)), np.ones((2, 2))])
    assert abs(st[0].breadth - 1.0) < 1e-12
    assert abs(st[0].depth - 0.5) < 1e-12
    assert abs(st[0].intensity - 1.0) < 1e-12

def test_profile_statistics_empty_batch_is_empty():
    assert profile_statistics([]) == []

def test_profile_statistics_rejects_empty_grid():
    with pytest.raises(ValueError):
        profile_statistics([np.zeros((2, 0))])

def test_relative_position_hand_computed():
    assert np.allclose(relative_position(np.array([1.0, 2.0, 3.0])), [0.0, 0.5, 1.0])
    assert np.allclose(relative_position(np.array([3.0, 1.0, 2.0])), [1.0, 0.0, 0.5])

def test_relative_position_is_shift_and_scale_invariant():
    x = np.array([3.0, 1.0, 2.0, 4.0])
    assert np.allclose(relative_position(x + 7.0), relative_position(x))
    assert np.allclose(relative_position(x * 3.0), relative_position(x))

def test_relative_position_reverses_with_the_values():
    x = np.array([3.0, 1.0, 2.0, 4.0])
    assert np.allclose(relative_position(-x), 1.0 - relative_position(x))

def test_relative_position_keeps_nan():
    z = relative_position(np.array([1.0, np.nan, 3.0]))
    assert math.isnan(z[1])
    assert np.allclose(z[[0, 2]], [0.0, 1.0])

def test_relative_position_is_bounded():
    rng = np.random.default_rng(0)
    for scale in (1e-6, 1.0, 1e6):
        z = relative_position(rng.normal(size=50) * scale)
        assert np.all((z >= 0.0) & (z <= 1.0))

def test_relative_position_is_robust_to_outliers():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x.copy()
    y[3] = 1e9
    assert np.allclose(relative_position(y)[:3], relative_position(x)[:3])

def test_relative_position_averages_ties():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    assert np.allclose(relative_position(x), [0.0, 0.5, 0.5, 1.0])
    y = np.array([1.0, 2.0, 2.0, 4.0])
    assert np.allclose(relative_position(y), relative_position(x))
    assert np.allclose(relative_position(np.ones(5)), 0.5)

def test_fuse_profile_statistics_equal_weight_and_direction():
    b = np.array([0.1, 0.5, 0.9])
    assert np.allclose(fuse_profile_statistics(b, b, b), relative_position(-b))
    assert np.allclose(relative_position(-b), [1.0, 0.5, 0.0])

def test_fuse_statistics_subset_matches_manual():
    b = np.array([0.1, 0.5, 0.9])
    d = np.array([0.2, 0.5, 0.8])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(fuse_statistics(b), relative_position(-b))
    assert np.allclose(fuse_statistics(b, d),
                       (relative_position(-b) + relative_position(-d)) / 2)
    assert np.allclose(fuse_statistics(b, d, v),
                       fuse_profile_statistics(b, d, v))

def test_fuse_profile_statistics_skips_all_nan_component():
    b = np.array([0.1, 0.5, 0.9])
    v = np.array([1.0, 2.0, 3.0])
    fused = fuse_profile_statistics(b, np.array([np.nan] * 3), v)
    assert np.allclose(fused, (relative_position(-b) + relative_position(-v)) / 2)

def test_fuse_profile_statistics_skips_nan_per_sample():
    d = np.array([0.2, 0.5, 0.8])
    fused = fuse_profile_statistics(np.array([0.1, np.nan, 0.9]),
                                    d, np.array([1.0, 2.0, 3.0]))
    assert not np.isnan(fused[1])
    assert abs(fused[1] - relative_position(-d)[1]) < 1e-12

def test_weighted_fusion_uniform_matches_default():
    rng = np.random.default_rng(0)
    b, d, v = (rng.normal(size=40) for _ in range(3))
    default = fuse_profile_statistics(b, d, v)
    explicit = fuse_profile_statistics(b, d, v, weights=(1 / 3, 1 / 3, 1 / 3))
    assert np.allclose(default, explicit, rtol=0, atol=1e-12)
    assert np.array_equal(np.round(default, 6), np.round(explicit, 6))

def test_weighted_fusion_single_statistic_matches_relative_position():
    b = np.array([0.1, 0.5, 0.9])
    d = np.array([0.2, 0.5, 0.8])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(fuse_profile_statistics(b, d, v, weights=(1, 0, 0)),
                       relative_position(-b))
    assert np.allclose(fuse_profile_statistics(b, d, v, weights=(0, 0, 1)),
                       relative_position(-v))

def test_weighted_fusion_half_half_is_mean_of_two():
    b = np.array([0.1, 0.5, 0.9])
    d = np.array([0.2, 0.5, 0.8])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        fuse_profile_statistics(b, d, v, weights=(0.5, 0.5, 0.0)),
        (relative_position(-b) + relative_position(-d)) / 2)

@pytest.mark.parametrize("bad", [(1.0, 0.0), (0.5, 0.5, 0.5), (1 / 3,) * 4,
                                 (2 / 3, 2 / 3, -1 / 3)])
def test_weighted_fusion_rejects_bad_weights(bad):
    with pytest.raises(ValueError):
        fuse_profile_statistics(np.array([1.0, 2.0]), np.array([2.0, 1.0]),
                                np.array([1.0, 3.0]), weights=bad)

def test_weighted_fusion_renormalizes_when_a_component_is_nan():
    b = np.array([0.1, np.nan, 0.9])
    d = np.array([0.2, 0.5, 0.8])
    v = np.array([1.0, 2.0, 3.0])
    fused = fuse_profile_statistics(b, d, v, weights=(0.5, 0.3, 0.2))
    assert abs(fused[1] - relative_position(-d)[1]) < 1e-12

def test_weight_grid_step_must_be_multiple_of_three():
    from CASDET.experiments import weight_grid
    for bad in (10, 8, 7, 0, -6):
        with pytest.raises(ValueError):
            weight_grid(bad)
    pts = weight_grid(6)
    assert len(pts) == 28
    assert any(np.allclose(p, 1 / 3) for p in pts)
    assert all(abs(sum(p) - 1.0) < 1e-12 and min(p) >= 0 for p in pts)

def test_weight_sweep_merges_across_models(tmp_path):
    from CASDET.experiments import (WEIGHT_GRID_STEP, merge_weight_sweep,
                                       weight_grid)

    grid = weight_grid(WEIGHT_GRID_STEP)
    eq = grid.index((1 / 3, 1 / 3, 1 / 3))
    pts = [{"w": list(w), "metrics": {"auroc": 0.70 + (0.05 if i == eq else 0.0),
                                      "auprc": 0.60 + (0.05 if i == eq else 0.0),
                                      "pcc": 0.30}}
           for i, w in enumerate(grid)]
    for model in ("llama2_7b", "llama2_13b"):
        (tmp_path / f"ragtruth_{model}.json").write_text(json.dumps(
            {"dataset": "ragtruth", "model": model, "weight_sweep": pts}),
            encoding="utf-8")
    (tmp_path / "hallurag_x.json").write_text(json.dumps(
        {"dataset": "hallurag", "model": "x", "weight_sweep": pts}),
        encoding="utf-8")
    (tmp_path / "ragtruth_llama3_8b.json").write_text(json.dumps(
        {"dataset": "ragtruth", "model": "llama3_8b"}), encoding="utf-8")

    merged = merge_weight_sweep(tmp_path, "ragtruth")
    assert sorted(merged) == ["llama2_13b", "llama2_7b"]

def test_compute_casdet_results_warns_on_nonfinite_layer_effects(monkeypatch,
                                                                caplog):
    import logging

    from CASDET import experiments
    from CASDET.casdet import CasdetResult

    class _Sample:

        def __init__(self, sid):
            self.sample_id = sid
            self.input_ids, self.resp_token_span, self.label = [0], (0, 1), False

        def context_mask(self):
            return [1]

    def fake_run_casdet(adapter, input_ids, resp_span, ctx_mask, sample_id,
                       label):
        layer_effects = np.ones((2, 3), dtype=np.float32)
        if sample_id != "good":
            layer_effects[1, 2] = np.nan
        return CasdetResult(sample_id=sample_id, label=label,
                           token_effect=np.ones(3, dtype=np.float32),
                           layer_effects=layer_effects)

    monkeypatch.setattr(experiments, "run_casdet", fake_run_casdet)
    samples = [_Sample("good"), _Sample("bad"), _Sample("bad2")]

    with caplog.at_level(logging.WARNING, logger="CASDET"):
        results = experiments.compute_casdet_results(None, samples, log_every=0)

    assert len(results) == 3, "non-finite values must not drop samples or raise"
    warns = [r.getMessage() for r in caplog.records
             if r.levelno == logging.WARNING]
    assert any("[bad]" in m and "non-finite" in m for m in warns), warns
    assert any("2/3" in m and "2 elements" in m for m in warns), warns

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="CASDET"):
        experiments.compute_casdet_results(None, [_Sample("good")], log_every=0)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], \
        "no warning when all values are finite"

def test_page_l_perfect_increasing_trend():
    effects = np.tile(np.arange(5, dtype=np.float64), (5, 1)) * 0.25
    res = page_l_test(effects)
    assert res["page_l"] == 275.0
    assert res["p_value"] < 1e-4
    assert res["significant"] is True

def test_page_l_random_no_trend():
    rng = np.random.RandomState(1)
    effects = rng.rand(30, 5)
    res = page_l_test(effects)
    assert res["p_value"] > 0.05
    assert res["significant"] is False

def test_page_l_expectation_variance():
    n, k = 5, 5
    effects = np.zeros((n, k))
    res = page_l_test(effects)
    assert abs(res["expected_l"] - n * k * (k + 1) ** 2 / 4.0) < 1e-9
    assert abs(res["variance_l"] - n * (k ** 3 - k) ** 2 / (144.0 * (k - 1))) < 1e-9

def test_metrics_perfect_separation():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    m = compute_metrics(labels, scores)
    assert abs(m["auroc"] - 1.0) < 1e-9
    assert abs(m["auprc"] - 1.0) < 1e-9
    assert abs(m["pcc"] - 1.0) < 1e-9

def test_metrics_anticorrelation():
    labels = [0, 0, 1, 1]
    scores = [1.0, 1.0, 0.0, 0.0]
    m = compute_metrics(labels, scores)
    assert abs(m["auroc"] - 0.0) < 1e-9
    assert abs(m["pcc"] + 1.0) < 1e-9

def test_metrics_single_class_nan():
    labels = [1, 1, 1]
    scores = [0.5, 0.6, 0.7]
    m = compute_metrics(labels, scores)
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"])

def test_metrics_drop_nonfinite_scores():
    labels = [0, 1, 0, 1]
    scores = [0.1, float("nan"), 0.9, 0.2]
    m = compute_metrics(labels, scores)
    assert not math.isnan(m["auroc"])
    m2 = compute_metrics(labels, [0.1, 0.9, 0.2, 0.8])
    assert not math.isnan(m2["auroc"])
    with pytest.warns(UserWarning):
        compute_metrics(labels, scores)

def test_metrics_all_nonfinite_returns_nan():
    labels = [0, 1, 0, 1]
    scores = [float("nan")] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = compute_metrics(labels, scores)
    assert math.isnan(m["auroc"]) and math.isnan(m["auprc"]) and math.isnan(m["pcc"])

def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1])

def test_page_l_large_z_p_does_not_underflow_to_zero():
    effects = np.tile(np.arange(5, dtype=np.float64), (450, 1))
    res = page_l_test(effects)
    assert res["page_l"] == 450 * 55.0
    assert res["p_value"] == 0.0
    assert res["log_p_value"] < -300
    assert res["p_display"] == "< 1e-300"
    assert res["significant"] is True

def test_page_l_moderate_z_p_display_is_numeric():
    effects = np.tile(np.arange(5, dtype=np.float64), (5, 1)) * 0.25
    res = page_l_test(effects)
    assert res["p_display"] == "%.3g" % res["p_value"]
    assert abs(res["log_p_value"] - math.log(res["p_value"])) < 1e-9

def test_save_summary_merged_keeps_previous_combinations(tmp_path):
    from CASDET.experiments import save_summary_merged

    path = tmp_path / "summary.json"
    save_summary_merged({"results": {"a": {"auroc": 0.5}}, "n_failed": 0},
                        path, "results")
    save_summary_merged({"results": {"b": {"auroc": 0.9}}, "n_failed": 1},
                        path, "results")
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(merged["results"]) == ["a", "b"]
    assert merged["results"]["a"]["auroc"] == 0.5
    assert merged["n_failed"] == 1
    save_summary_merged({"results": {"a": {"auroc": 0.7}}}, path, "results")
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert merged["results"]["a"]["auroc"] == 0.7
    assert sorted(merged["results"]) == ["a", "b"]

    save_summary_merged(
        {"results": {"c": {"auroc": 0.6}},
         "statistic_combinations": {"c": {"kappa": 0.5}}}, path, "results")
    save_summary_merged(
        {"results": {"d": {"auroc": 0.4}},
         "statistic_combinations": {"d": {"kappa": 0.8}}}, path, "results")
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(merged["statistic_combinations"]) == ["c", "d"]
    assert merged["statistic_combinations"]["c"]["kappa"] == 0.5
    assert sorted(merged["results"]) == ["a", "b", "c", "d"]

    save_summary_merged({"results": {}, "weight_sweep": {"e": [{"w": [1, 0, 0]}]}},
                        path, "results")
    save_summary_merged({"results": {}, "weight_sweep": {"f": [{"w": [0, 1, 0]}]}},
                        path, "results")
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(merged["weight_sweep"]) == ["e", "f"]

def _completeness_cfg(models, all_models=None):
    cfg = {
        "models": {m: {"name": f"models/{m}"} for m in models},
        "datasets": [
            {"name": "ragtruth",
             "files": {m: "x" for m in (all_models or models)}},
            {"name": "hallurag", "models": ["llama2_7b", "mistral_7b"],
             "files": {m: "x" for m in (all_models or models)}},
        ],
    }
    if all_models is not None:
        cfg["_all_models"] = {m: {"name": f"models/{m}"} for m in all_models}
    return cfg

ALL_MODELS = ["llama2_7b", "llama2_13b", "llama3_8b", "mistral_7b"]

def test_completeness_single_model_window_does_not_fail_its_own_process():
    from CASDET.experiments import _check_completeness

    cfg = _completeness_cfg(["llama3_8b"], all_models=ALL_MODELS)
    got = _check_completeness(cfg, {"results": {"ragtruth/llama3_8b": {}}},
                              "results", "main")
    assert got["n_missing_own"] == 0
    assert got["n_missing"] == 5

def test_completeness_single_model_window_fails_on_its_own_gap():
    from CASDET.experiments import _check_completeness

    cfg = _completeness_cfg(["llama3_8b"], all_models=ALL_MODELS)
    got = _check_completeness(cfg, {"results": {}}, "results", "main")
    assert got["n_missing_own"] == 1
    assert got["n_missing"] == 6

def test_completeness_full_run_counts_every_gap():
    from CASDET.experiments import _check_completeness

    cfg = _completeness_cfg(ALL_MODELS)
    have = {"results": {f"{d}/{m}": {} for m in ALL_MODELS
                        for d in ("ragtruth", "hallurag")
                        if not (d == "hallurag" and m == "llama3_8b")}}
    got = _check_completeness(cfg, have, "results", "main")
    assert got == {"n_missing": 0, "n_missing_own": 0}

def test_completeness_ablation_container_keys_are_model_names():
    from CASDET.experiments import _check_completeness

    cfg = _completeness_cfg(["llama2_7b"], all_models=ALL_MODELS)
    got = _check_completeness(cfg, {"models": {"llama2_7b": {}}}, "models",
                              "ablation")
    assert got["n_missing_own"] == 0
    assert got["n_missing"] == 3

def test_save_summary_merged_accepts_count_dict(tmp_path):
    from CASDET.experiments import save_summary_merged

    path = tmp_path / "summary.json"
    merged = save_summary_merged(
        {"results": {"ragtruth/llama2_7b": {}}}, path, "results",
        completeness=lambda m: {"n_missing": 3, "n_missing_own": 0})
    assert merged["n_missing"] == 3 and merged["n_missing_own"] == 0
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["n_missing"] == 3 and on_disk["n_missing_own"] == 0
