# -*- coding: utf-8 -*-

"""Experiment pipelines for CASDET: main results, graded intervention, and ablation."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .datasets import (
    RagSample,
    load_dataset,
    selection_eval_splits,
    tokenize_and_align,
)
from .metrics import compute_metrics, page_l_test
from .model_adapter import ModelAdapter
from .casdet import (
    CasdetResult,
    attention_only_effect,
    compute_layer_effects,
    fuse_profile_statistics,
    fuse_statistics,
    graded_intervention,
    graded_intervention_effect,
    no_control_effect,
    profile_statistics,
    run_casdet,
    variant_statistics,
)
from .utils import (
    Timer,
    ensure_dir,
    file_lock,
    load_json,
    save_json,
    save_npz,
    set_seed,
)

logger = logging.getLogger("CASDET")

def _log_completion(task: str, done: int, attempted: int) -> None:
    if done < attempted:
        logger.warning("%s completed %d/%d combinations; some failed or were skipped, "
                       "see the warnings above", task, done, attempted)
    else:
        logger.info("%s completed %d/%d combinations", task, done, attempted)

def save_summary_merged(summary: Dict[str, Any], path: Path,
                        container: str,
                        completeness: Optional[Callable[
                            [Dict[str, Any]], Union[int, Dict[str, int]]]]
                        = None) -> Dict[str, Any]:
    """Merge a per-model summary into an existing summary file."""
    with file_lock(path):
        merged = _merge_into(summary, path, container)
        if completeness is not None:
            got = completeness(merged)
            merged.update(got if isinstance(got, dict) else {"n_missing": got})
            save_json(merged, path)
        return merged

def _merge_into(summary: Dict[str, Any], path: Path,
                container: str) -> Dict[str, Any]:
    prev = load_json(path, default={}) or {}
    prev_items = dict(prev.get(container) or {})
    now_items = dict(summary.get(container) or {})
    carried = sorted(set(prev_items) - set(now_items))
    merged = dict(prev)
    merged.update(summary)
    merged[container] = {**prev_items, **now_items}
    for extra in ("statistic_combinations", "weight_sweep"):
        p_extra = dict(prev.get(extra) or {})
        n_extra = dict(summary.get(extra) or {})
        if p_extra or n_extra:
            carried_extra = sorted(set(p_extra) - set(n_extra))
            merged[extra] = {**p_extra, **n_extra}
            if carried_extra:
                logger.info("summary merge: %s reuses %d entries already on disk %s",
                            extra, len(carried_extra), carried_extra)
    save_json(merged, path)
    if carried:
        logger.info("summary merge: reusing %d entries already on disk %s",
                    len(carried), carried)
    return merged

def _expected_keys(cfg: Dict[str, Any], container: str,
                   models: Optional[Sequence[str]] = None) -> List[str]:
    models = list(models) if models is not None else \
        (cfg.get("_all_models") or cfg["models"])
    keys: List[str] = []
    for ds_cfg in cfg["datasets"]:
        for model_key in models:
            if ds_cfg.get("models") and model_key not in ds_cfg["models"]:
                continue
            key = f"{ds_cfg['name']}/{model_key}" if container == "results" else model_key
            if key not in keys:
                keys.append(key)
    return keys

def _require_single_dataset(cfg: Dict[str, Any], task: str,
                            outputs: str) -> None:
    if len(cfg["datasets"]) != 1:
        raise ValueError(
            f"{task} writes outputs and summary keys per <model> only ({outputs}), "
            f"so exactly one dataset is supported; got {len(cfg['datasets'])} "
            f"({[d.get('name') for d in cfg['datasets']]}). "
            "To use multiple datasets, rename keys and files to <dataset>_<model>.")

def _check_completeness(cfg: Dict[str, Any], merged: Dict[str, Any],
                        container: str, task: str) -> Dict[str, int]:
    expected = _expected_keys(cfg, container)
    own = set(_expected_keys(cfg, container, models=list(cfg["models"])))
    have = list(merged.get(container) or {})
    missing = [k for k in expected if k not in have]
    mine = [k for k in missing if k in own]
    others = [k for k in missing if k not in own]
    extra = [k for k in have if k not in expected]
    if extra:
        logger.warning("[%s] summary contains combinations not present in the config: %s",
                       task, extra)
    if mine:
        logger.error(
            "[%s] incomplete: expected %d combinations, found %d, missing %s"
            " -- see the warnings above.",
            task, len(own), len(own) - len(mine), mine)
    if others:
        logger.info("[%s] %d combinations belong to models not run in this process: %s",
                    task, len(others), others)
    if not missing:
        logger.info("[%s] completeness check passed: all %d combinations present", task, len(expected))
    return {"n_missing": len(missing), "n_missing_own": len(mine)}

FIGURE_MODEL_ORDER: Tuple[str, ...] = ("llama2_7b", "llama2_13b",
                                       "llama3_8b", "mistral_7b")

def _ordered_keys(keys: Sequence[str]) -> List[str]:
    known = [k for k in FIGURE_MODEL_ORDER if k in keys]
    return known + sorted(k for k in keys if k not in known)

def merge_intervention_effects(out_dir: Path) -> Dict[str, np.ndarray]:
    """Merge per-model intervention effects."""
    out: Dict[str, np.ndarray] = {}
    for path in out_dir.glob("intervention_*.npz"):
        with np.load(path) as z:
            out[path.stem[len("intervention_"):]] = z["effects"]
    return {k: out[k] for k in _ordered_keys(list(out))}

def merge_ablation_metrics(out_dir: Path) -> Dict[str, Any]:
    """Merge per-model ablation metrics."""
    out: Dict[str, Any] = {}
    for path in sorted(out_dir.glob("ablation_*.json")):
        data = load_json(path, default=None) or {}
        if data.get("metrics"):
            out[path.stem[len("ablation_"):]] = data["metrics"]
    return {k: out[k] for k in _ordered_keys(list(out))}

def merge_statistic_combinations(out_dir: Path,
                                 dataset: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(out_dir.glob(f"{dataset}_*.json")):
        data = load_json(path, default=None) or {}
        model = data.get("model")
        if data.get("dataset") == dataset and model \
                and data.get("statistic_combinations"):
            out[model] = data["statistic_combinations"]
    return {k: out[k] for k in _ordered_keys(list(out))}

def build_adapter(model_cfg: Dict[str, Any]) -> ModelAdapter:
    """Build a ModelAdapter from a model configuration."""
    logger.info("Loading model %s: device=%s, %d visible GPUs",
                model_cfg["name"], model_cfg.get("device", "auto"),
                torch.cuda.device_count())
    return ModelAdapter(
        model_name_or_path=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "float16"),
    )

def prepare_samples(
    dataset_cfg: Dict[str, Any],
    model_key: str,
    tokenizer: Any,
    max_len: int = 4096,
    max_samples: Optional[int] = None,
) -> List[RagSample]:
    """Load, filter, and align samples for one dataset/model pair."""
    files = dataset_cfg.get("files", {})
    path = files.get(model_key) or dataset_cfg.get("path")
    if path is None:
        logger.warning("[%s/%s] no data file configured; skipping", dataset_cfg.get("name"),
                       model_key)
        return []
    try:
        samples = load_dataset(
            dataset_cfg["name"], path,
            max_samples=max_samples,
            split=dataset_cfg.get("split"),
            quality=dataset_cfg.get("quality", "good"),
        )
    except FileNotFoundError:
        logger.warning("[%s/%s] data file not found: %s; skipping", dataset_cfg.get("name"),
                       model_key, path)
        return []
    for s in samples:
        tokenize_and_align(s, tokenizer, max_len=max_len)
    logger.info("[%s/%s] loaded %d records (%d hallucinated / %d non-hallucinated, all splits)",
                dataset_cfg.get("name"), model_key, len(samples),
                sum(s.label for s in samples),
                sum(not s.label for s in samples))
    return samples

def _labels_and_statistics(
        results: Sequence[CasdetResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([1.0 if r.label else 0.0 for r in results])
    stats = profile_statistics([r.layer_effects for r in results])
    return (labels,
            np.asarray([s.breadth for s in stats], dtype=np.float64),
            np.asarray([s.depth for s in stats], dtype=np.float64),
            np.asarray([s.intensity for s in stats], dtype=np.float64))

STATISTIC_COMBINATIONS: Dict[str, Tuple[str, ...]] = {
    "breadth": ("breadth",),
    "depth": ("depth",),
    "intensity": ("intensity",),
    "breadth+depth": ("breadth", "depth"),
    "breadth+intensity": ("breadth", "intensity"),
    "depth+intensity": ("depth", "intensity"),
    "breadth+depth+intensity": ("breadth", "depth", "intensity"),
}

def statistic_combination_scores(
        breadth: np.ndarray, depth: np.ndarray, intensity: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Score all single, pairwise, and full statistic combinations."""
    pools = {"breadth": breadth, "depth": depth, "intensity": intensity}
    return {name: fuse_statistics(*[pools[m] for m in members])
            for name, members in STATISTIC_COMBINATIONS.items()}

WEIGHT_GRID_STEP = 12

def weight_grid(step: int = WEIGHT_GRID_STEP
                ) -> List[Tuple[float, float, float]]:
    """Enumerate weight settings on the simplex."""
    if step <= 0 or step % 3:
        raise ValueError(f"weight_grid_step={step} must be a positive multiple of 3, "
                         f"otherwise the equal-weight point 1/3 is not on the grid")
    return [(i / step, j / step, (step - i - j) / step)
            for i in range(step + 1) for j in range(step + 1 - i)]

def weight_sweep_scores(
        breadth: np.ndarray, depth: np.ndarray, intensity: np.ndarray,
        labels: np.ndarray, step: int = WEIGHT_GRID_STEP,
) -> List[Dict[str, Any]]:
    """Compute metrics over the weight grid."""
    out: List[Dict[str, Any]] = []
    for w in weight_grid(step):
        scores = fuse_profile_statistics(breadth, depth, intensity, weights=w)
        out.append({"w": list(w), "metrics": compute_metrics(labels, scores)})
    return out

def _sweep_table(per_model: Dict[str, Any], metric: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = next(iter(per_model.values()))
    wk = np.asarray([p["w"][0] for p in points], dtype=np.float64)
    wd = np.asarray([p["w"][1] for p in points], dtype=np.float64)
    vals = np.asarray(
        [[p["metrics"].get(metric, np.nan) for p in per_model[m]]
         for m in per_model], dtype=np.float64)
    mean = np.nanmean(vals, axis=0)
    n = vals.shape[0]
    err = (np.nanstd(vals, axis=0, ddof=1) / np.sqrt(n) if n > 1
           else np.zeros_like(mean))
    return wk, wd, mean, err

def merge_weight_sweep(out_dir: Path, dataset: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(out_dir.glob(f"{dataset}_*.json")):
        data = load_json(path, default=None) or {}
        model = data.get("model")
        if data.get("dataset") == dataset and model \
                and data.get("weight_sweep"):
            out[model] = data["weight_sweep"]
    return {k: out[k] for k in _ordered_keys(list(out))}

def compute_casdet_results(
    adapter: ModelAdapter,
    samples: Sequence[RagSample],
    log_every: int = 20,
) -> List[CasdetResult]:
    """Run CASDET over a list of samples."""
    results: List[CasdetResult] = []
    bad_ids: List[str] = []
    n_bad_elems = 0
    for i, s in enumerate(samples):
        r = run_casdet(
            adapter=adapter,
            input_ids=s.input_ids,
            resp_span=s.resp_token_span,
            ctx_mask=s.context_mask(),
            sample_id=s.sample_id,
            label=s.label,
        )
        bad = ~np.isfinite(r.layer_effects)
        n_bad = int(bad.sum())
        if n_bad:
            if not bad_ids:
                logger.warning(
                    "[%s] non-finite values in per-layer effects (%d/%d elements); "
                    "breadth treats them as unsupported (highest risk) while depth/intensity drop them, "
                    "so the statistics are inconsistent. Further cases are summarized at the end.",
                    s.sample_id, n_bad, bad.size)
            bad_ids.append(s.sample_id)
            n_bad_elems += n_bad
        results.append(r)
        if log_every and ((i + 1) % log_every == 0 or i + 1 == len(samples)):
            logger.info("CASDET progress %d/%d (evaluation set)", i + 1, len(samples))
    if bad_ids:
        logger.warning(
            "%d/%d responses contain non-finite S_t^l values (%d elements in total, "
            "first: %s). Metrics are still computed, but the statistics treat them differently; "
            "interpret the results accordingly.",
            len(bad_ids), len(samples), n_bad_elems, bad_ids[0])
    return results

def _save_layer_effects(path: Path, results: Sequence[CasdetResult]) -> None:
    """Cache ragged per-layer effects as a NaN-padded array for the reference check."""
    if not results:
        return
    n_layers = results[0].layer_effects.shape[0]
    n_tokens = max(r.layer_effects.shape[1] for r in results)
    effects = np.full((len(results), n_layers, n_tokens), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        t = r.layer_effects.shape[1]
        effects[i, :, :t] = r.layer_effects
    labels = np.asarray([float(r.label) for r in results], dtype=np.float64)
    save_npz(path, layer_effects=effects, labels=labels)
    logger.info("[cache] wrote %s (%d x %d x %d)", path,
                len(results), n_layers, n_tokens)

def run_main_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main experiments (Table 2)."""
    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(cfg.get("output_dir", "log/results/main"))
    max_len = cfg.get("max_len", 4096)
    max_samples = cfg.get("max_samples")
    summary: Dict[str, Any] = {"results": {}, "statistic_combinations": {},
                               "weight_sweep": {}}
    attempted = 0
    failed = 0
    adapter: Optional[ModelAdapter] = None

    for ds_cfg in cfg["datasets"]:
        for model_key, model_cfg in cfg["models"].items():
            tag = f"{ds_cfg['name']}/{model_key}"
            if ds_cfg.get("models") and model_key not in ds_cfg["models"]:
                logger.info("[%s] not in the model list for this dataset; skipping (e.g., HalluRAG "
                            "has no Llama3-8B)", tag)
                continue
            logger.info("==== Main experiment [%s] ====", tag)
            attempted += 1
            try:
                if adapter is not None:
                    del adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        logger.info("Released the previous model; allocated GPU memory: %.2f GB",
                                    torch.cuda.memory_allocated() / 1e9)
                adapter = build_adapter(model_cfg)
                samples = prepare_samples(ds_cfg, model_key, adapter.tokenizer,
                                          max_len=max_len, max_samples=max_samples)
                if not samples:
                    continue
                split = selection_eval_splits(
                    samples, sel_ratio=cfg.get("val_ratio", 0.2),
                    seed=cfg.get("seed", 42),
                    cv_folds=ds_cfg.get("cv_folds"))
                sel, eval_set = split.selection, split.evaluation
                logger.info("[%s] split: %d selection / %d evaluation (%s)"
                            " -- only the evaluation set is scored",
                            tag, len(sel), len(eval_set),
                            f"{len(split.folds)}-fold cross-validation" if split.folds
                            else "official split")
                if not eval_set:
                    logger.warning("[%s] empty evaluation set; skipping. If this is caused by max_samples "
                                   "(the first records are often in the train split), increase it or add "
                                   "split: test to the dataset config.",
                                   tag)
                    continue
                with Timer(logger, f"[{tag}] CASDET ({len(eval_set)} responses)"):
                    results = compute_casdet_results(adapter, eval_set)
                if cfg.get("save_layer_effects", False):
                    _save_layer_effects(
                        out_dir / f"{ds_cfg['name']}_{model_key}_layers.npz",
                        results)
                labels, breadth, depth, intensity = _labels_and_statistics(results)
                scores = fuse_profile_statistics(breadth, depth, intensity)
                metrics = compute_metrics(labels, scores)
                combo_scores = statistic_combination_scores(breadth, depth, intensity)
                combo_metrics = {name: compute_metrics(labels, sc)
                                 for name, sc in combo_scores.items()}
                weight_sweep = weight_sweep_scores(
                    breadth, depth, intensity, labels,
                    step=cfg.get("weight_grid_step", WEIGHT_GRID_STEP))
                logger.info("[%s] AUROC=%.4f  AUPRC=%.4f  PCC=%.4f  | combination AUROC: %s",
                            tag, metrics["auroc"], metrics["auprc"], metrics["pcc"],
                            "  ".join("%s=%.4f" % (n, m["auroc"])
                                      for n, m in combo_metrics.items()))
                save_json({"dataset": ds_cfg["name"], "model": model_key,
                           "metrics": metrics, "n_eval": len(eval_set),
                           "scores": {r.sample_id: float(s)
                                      for r, s in zip(results, scores)},
                           "statistics": {r.sample_id: {
                               "breadth": float(b),
                               "depth": float(dp),
                               "intensity": float(iv)}
                               for r, b, dp, iv in zip(results, breadth, depth, intensity)},
                           "statistic_combinations": combo_metrics,
                           "weight_sweep": weight_sweep},
                          out_dir / f"{ds_cfg['name']}_{model_key}.json")
                summary["results"][tag] = metrics
                summary["statistic_combinations"][tag] = combo_metrics
                summary["weight_sweep"][tag] = weight_sweep
                save_summary_merged(summary, out_dir / "summary.json", "results")
            except Exception as e:  # noqa: BLE001 - one failure must not stop the rest
                failed += 1
                logger.exception("[%s] main experiment failed: %s", tag, e)

    summary["n_failed"] = failed
    _log_completion("main", len(summary["results"]), attempted)
    merged = save_summary_merged(
        summary, out_dir / "summary.json", "results",
        completeness=lambda m: _check_completeness(cfg, m, "results", "main"))
    summary["n_missing"] = merged["n_missing"]
    summary["n_missing_own"] = merged["n_missing_own"]
    return summary

def run_intervention_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Graded local intervention and Page's L test (Table 1)."""
    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(cfg.get("output_dir", "log/results/intervention"))
    _require_single_dataset(cfg, "intervention", "intervention_<model>.json/.npz")
    gamma_grid = cfg.get("gamma_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    max_len = cfg.get("max_len", 4096)
    max_samples = cfg.get("max_samples")
    summary: Dict[str, Any] = {"gamma_grid": list(gamma_grid), "models": {}}
    attempted = 0
    failed = 0
    adapter: Optional[ModelAdapter] = None

    for ds_cfg in cfg["datasets"]:
        for model_key, model_cfg in cfg["models"].items():
            tag = f"{model_key}"
            logger.info("==== Intervention experiment [%s] ====", tag)
            attempted += 1
            try:
                if adapter is not None:
                    del adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        logger.info("Released the previous model; allocated GPU memory: %.2f GB",
                                    torch.cuda.memory_allocated() / 1e9)
                adapter = build_adapter(model_cfg)
                samples = prepare_samples(ds_cfg, model_key, adapter.tokenizer,
                                          max_len=max_len,
                                          max_samples=max_samples)
                if not samples:
                    continue
                split = selection_eval_splits(
                    samples, sel_ratio=cfg.get("val_ratio", 0.2),
                    seed=cfg.get("seed", 42),
                    cv_folds=ds_cfg.get("cv_folds"))
                eval_set = split.evaluation
                logger.info("[%s] split: %d evaluation responses (%s)"
                            " -- same evaluation set as the main experiment",
                            tag, len(eval_set),
                            f"{len(split.folds)}-fold cross-validation" if split.folds
                            else "official split")
                if not eval_set:
                    logger.warning("[%s] empty evaluation set; skipping.", tag)
                    continue
                all_e: List[np.ndarray] = []
                for i, s in enumerate(eval_set):
                    with adapter.capture(s.input_ids, s.resp_token_span,
                                         s.context_mask()) as layers:
                        r0, r1 = s.resp_token_span
                        w_ids = torch.tensor(s.input_ids[r0:r1],
                                             dtype=torch.long)
                        s_gamma = graded_intervention(adapter, layers, w_ids,
                                                      gamma_grid)
                    all_e.append(graded_intervention_effect(s_gamma))
                    if (i + 1) % 10 == 0 or i + 1 == len(eval_set):
                        logger.info("Intervention progress %d/%d", i + 1, len(eval_set))
                E = np.stack(all_e)
                test = page_l_test(E)
                logger.info("[%s] Page L=%.1f  p=%s  %s", tag, test["page_l"],
                            test["p_display"],
                            "significant" if test["significant"] else "not significant")
                summary["models"][tag] = {
                    "page_l_test": test,
                    "mean_effect_by_gamma": E.mean(axis=0).tolist(),
                    "n_responses": len(all_e),
                }
                save_json(summary["models"][tag],
                          out_dir / f"intervention_{model_key}.json")
                save_npz(out_dir / f"intervention_{model_key}.npz",
                         effects=E,
                         labels=np.asarray([1.0 if s.label else 0.0
                                            for s in eval_set]))
                save_summary_merged(summary, out_dir / "summary.json", "models")
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.exception("[%s] intervention experiment failed: %s", tag, e)

    summary["n_failed"] = failed
    _log_completion("intervention", len(summary["models"]), attempted)
    merged = save_summary_merged(
        summary, out_dir / "summary.json", "models",
        completeness=lambda m: _check_completeness(cfg, m, "models", "intervention"))
    summary["n_missing"] = merged["n_missing"]
    summary["n_missing_own"] = merged["n_missing_own"]
    return summary

def run_ablation_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation studies (Figure 3)."""
    set_seed(cfg.get("seed", 42))
    _require_single_dataset(cfg, "ablation", "ablation_<model>.json")
    out_dir = ensure_dir(cfg.get("output_dir", "log/results/ablation"))
    max_len = cfg.get("max_len", 4096)
    max_samples = cfg.get("max_samples")
    summary: Dict[str, Any] = {"models": {}}
    attempted = 0
    failed = 0
    adapter: Optional[ModelAdapter] = None

    for ds_cfg in cfg["datasets"]:
        for model_key, model_cfg in cfg["models"].items():
            tag = f"{model_key}"
            logger.info("==== Ablation experiment [%s] ====", tag)
            attempted += 1
            try:
                if adapter is not None:
                    del adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        logger.info("Released the previous model; allocated GPU memory: %.2f GB",
                                    torch.cuda.memory_allocated() / 1e9)
                adapter = build_adapter(model_cfg)
                samples = prepare_samples(ds_cfg, model_key, adapter.tokenizer,
                                          max_len=max_len,
                                          max_samples=max_samples)
                if not samples:
                    continue
                split = selection_eval_splits(
                    samples, sel_ratio=cfg.get("val_ratio", 0.2),
                    seed=cfg.get("seed", 42),
                    cv_folds=ds_cfg.get("cv_folds"))
                eval_set = split.evaluation
                logger.info("[%s] split: %d evaluation responses (%s)"
                            " -- only the evaluation set is scored",
                            tag, len(eval_set),
                            f"{len(split.folds)}-fold cross-validation" if split.folds
                            else "official split")
                if not eval_set:
                    continue
                labels = np.asarray([1.0 if s.label else 0.0 for s in eval_set])

                variants = ("casdet", "attention_only", "no_control")
                grids: Dict[str, List[np.ndarray]] = {v: [] for v in variants}
                for i, s in enumerate(eval_set):
                    with adapter.capture(s.input_ids, s.resp_token_span,
                                         s.context_mask()) as layers:
                        r0, r1 = s.resp_token_span
                        w_ids = torch.tensor(s.input_ids[r0:r1],
                                             dtype=torch.long)
                        _, layer_full = compute_layer_effects(
                            adapter, layers, w_ids)
                        layer_attn = attention_only_effect(adapter, layers, w_ids)
                        layer_noctrl = no_control_effect(adapter, layers, w_ids)
                    for name, mat in (("casdet", layer_full),
                                      ("attention_only", layer_attn),
                                      ("no_control", layer_noctrl)):
                        grids[name].append(mat.numpy())
                    if (i + 1) % 10 == 0 or i + 1 == len(eval_set):
                        logger.info("Ablation progress %d/%d", i + 1, len(eval_set))

                model_summary: Dict[str, Any] = {}
                for name in variants:
                    st = variant_statistics(grids[name])
                    scores = fuse_profile_statistics(
                        [x.breadth for x in st], [x.depth for x in st],
                        [x.intensity for x in st])
                    model_summary[name] = compute_metrics(labels, scores)
                logger.info("[%s] CASDET AUPRC=%.4f | attention_only=%.4f | "
                            "no_control=%.4f (evaluation set: %d responses)",
                            tag, model_summary["casdet"]["auprc"],
                            model_summary["attention_only"]["auprc"],
                            model_summary["no_control"]["auprc"], len(eval_set))
                summary["models"][tag] = {"metrics": model_summary,
                                          "n_eval": len(eval_set)}
                save_json(summary["models"][tag],
                          out_dir / f"ablation_{model_key}.json")
                save_summary_merged(summary, out_dir / "summary.json", "models")

            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.exception("[%s] ablation experiment failed: %s", tag, e)

    summary["n_failed"] = failed
    _log_completion("ablation", len(summary["models"]), attempted)
    merged = save_summary_merged(
        summary, out_dir / "summary.json", "models",
        completeness=lambda m: _check_completeness(cfg, m, "models", "ablation"))
    summary["n_missing"] = merged["n_missing"]
    summary["n_missing_own"] = merged["n_missing_own"]
    return summary

TASK_REGISTRY = {
    "main": run_main_experiment,
    "intervention": run_intervention_experiment,
    "ablation": run_ablation_experiment,
}

def run_experiment(task: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch an experiment task."""
    if task not in TASK_REGISTRY:
        raise ValueError(f"unknown task: {task}; expected one of {sorted(TASK_REGISTRY)}")
    logger.info("Running task: %s", task)
    return TASK_REGISTRY[task](cfg)
