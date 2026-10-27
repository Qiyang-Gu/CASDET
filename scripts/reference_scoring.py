# -*- coding: utf-8 -*-
"""Reproduce the fixed-reference single-response scoring check.

The script expects cached per-response layer effects as
``{dataset}_{model}_layers.npz`` with keys ``layer_effects`` and ``labels``.
For each dataset--model combination it compares full-batch scoring with
five-fold cross-fitting: sigma_l and the reference distributions are fitted on
the other four folds (no labels) and each held-out response is scored alone.
The printed full/fixed AUROC, maximum absolute changes, and minimum Spearman
correlation correspond to Table 3 of the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CASDET.casdet import (  # noqa: E402
    _layer_scale,
    fuse_statistics,
    influence_breadth,
    influence_depth,
    influence_intensity,
)
from CASDET.metrics import compute_metrics  # noqa: E402
from CASDET.reference import fit_reference, score_batch_with_reference  # noqa: E402

COMBOS = [
    ("ragtruth", "llama2_7b"),
    ("ragtruth", "llama2_13b"),
    ("ragtruth", "llama3_8b"),
    ("ragtruth", "mistral_7b"),
    ("hallurag", "llama2_7b"),
    ("hallurag", "llama2_13b"),
    ("hallurag", "mistral_7b"),
]


def load(path: Path):
    z = np.load(path, allow_pickle=True)
    labels = z["labels"].astype(np.float64)
    effects = [e[:, np.isfinite(e).any(axis=0)].astype(np.float64)
               for e in z["layer_effects"]]
    return effects, labels


def batch_scores(effects):
    sigma = _layer_scale(effects)
    breadth = np.array([influence_breadth(e) for e in effects])
    depth = np.array([influence_depth(e) for e in effects])
    intensity = np.array([influence_intensity(e, sigma) for e in effects])
    return fuse_statistics(breadth, depth, intensity)


def cross_fit(effects, labels, seed, folds):
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(labels), dtype=int)
    assignment[rng.permutation(len(labels))] = np.arange(len(labels)) % folds
    out = np.full(len(labels), np.nan)
    for f in range(folds):
        ref_idx = np.where(assignment != f)[0]
        ev_idx = np.where(assignment == f)[0]
        reference = fit_reference([effects[i] for i in ref_idx])
        out[ev_idx] = score_batch_with_reference(
            [effects[i] for i in ev_idx], reference)
    return out


def run(data_dir: Path, datasets, models, seeds, folds):
    rows = []
    for dataset, model in COMBOS:
        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue
        path = data_dir / f"{dataset}_{model}_layers.npz"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        effects, labels = load(path)
        batch = batch_scores(effects)
        batch_metrics = compute_metrics(labels, batch)
        runs = []
        for seed in range(seeds):
            scores = cross_fit(effects, labels, seed, folds)
            metrics = compute_metrics(labels, scores)
            runs.append({
                "seed": seed,
                "metrics": metrics,
                "spearman": float(spearmanr(batch, scores).correlation),
                "delta": {k: abs(metrics[k] - batch_metrics[k])
                          for k in ("auroc", "auprc", "pcc")},
            })
        rows.append({
            "dataset": dataset,
            "model": model,
            "n": int(len(labels)),
            "batch": batch_metrics,
            "fixed_reference": {
                k: float(np.mean([r["metrics"][k] for r in runs]))
                for k in ("auroc", "auprc", "pcc")
            },
            "max_abs_delta": {k: float(max(r["delta"][k] for r in runs))
                              for k in ("auroc", "auprc", "pcc")},
            "min_spearman": float(min(r["spearman"] for r in runs)),
            "runs": runs,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory containing *_layers.npz")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = run(args.data_dir, args.dataset, args.model, args.seeds, args.folds)
    print(f"{'dataset':9s} {'model':11s} {'n':>4s} "
          f"{'fullAUROC':>9s} {'fixAUROC':>8s} "
          f"{'dAUROC':>7s} {'dAUPRC':>7s} {'dPCC':>7s} {'minRho':>7s}")
    for r in rows:
        d = r["max_abs_delta"]
        print(f"{r['dataset']:9s} {r['model']:11s} {r['n']:4d} "
              f"{r['batch']['auroc']:9.4f} {r['fixed_reference']['auroc']:8.4f} "
              f"{d['auroc']:7.4f} {d['auprc']:7.4f} {d['pcc']:7.4f} "
              f"{r['min_spearman']:7.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
