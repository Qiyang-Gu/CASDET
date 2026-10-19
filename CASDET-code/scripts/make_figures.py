# -*- coding: utf-8 -*-
"""Generate the paper figures from the saved results.

Figure 1 is hand-drawn and is not generated here. Figures 2-5 are regenerated
from ``log/results`` and written as vector PDFs to ``figs_generated/``.

Usage:
    python scripts/make_figures.py [--out-dir figs_generated]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CASDET.experiments import (  # noqa: E402
    merge_ablation_metrics,
    merge_intervention_effects,
    merge_statistic_combinations,
)
from CASDET.figures_weight import make_figure  # noqa: E402
from CASDET.utils import (  # noqa: E402
    model_sort_key,
    plot_ablation_dumbbell,
    plot_intervention_trend,
    plot_statistic_combinations,
)

GAMMA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _report_figure_font() -> None:
    try:
        from matplotlib import font_manager
        for family in ("Arial", "Helvetica"):
            try:
                path = font_manager.findfont(family, fallback_to_default=False)
                print(f"Figure font: {family} ({path})")
                return
            except Exception:  # noqa: BLE001
                continue
        print("Warning: Arial/Helvetica not found; figures will use "
              "DejaVu Sans. Install Microsoft core fonts for identical "
              "typography.")
    except Exception as exc:  # noqa: BLE001
        print("Font check failed:", exc)


def _make_intervention(results: Path, out_path: Path) -> None:
    effects = merge_intervention_effects(results / "intervention")
    if not effects:
        raise FileNotFoundError("no intervention_*.npz files found")
    plot_intervention_trend(GAMMA_GRID, effects, out_path)


def _make_ablation(results: Path, out_path: Path) -> None:
    metrics = merge_ablation_metrics(results / "ablation")
    if not metrics:
        raise FileNotFoundError("no ablation_*.json files found")
    models = sorted(metrics, key=model_sort_key)
    casdet_auprc = [metrics[m]["casdet"]["auprc"] for m in models]
    variants = [
        ("w/o Attention–FFN cascade",
         [metrics[m]["attention_only"]["auprc"] for m in models]),
        ("w/o local control state",
         [metrics[m]["no_control"]["auprc"] for m in models]),
    ]
    panels = [f"({'ab'[i]}) {name}" for i, (name, _) in enumerate(variants)]
    plot_ablation_dumbbell(
        models, casdet_auprc,
        list(zip(panels, [values for _, values in variants])),
        out_path)


def _make_statistics(results: Path, out_path: Path) -> None:
    combined = {}
    for dataset in ("ragtruth", "hallurag"):
        by_model = merge_statistic_combinations(results / "main", dataset)
        for model, combos in by_model.items():
            for combo, metric_values in combos.items():
                slot = combined.setdefault(model, {}).setdefault(combo, {})
                for metric, value in metric_values.items():
                    slot.setdefault(metric, []).append(value)
    if not combined:
        raise FileNotFoundError("no statistic_combinations found")
    per_model = {
        model: {
            combo: {metric: float(np.mean(values))
                    for metric, values in metrics.items()}
            for combo, metrics in combos.items()
        }
        for model, combos in combined.items()
    }
    models = sorted(per_model, key=model_sort_key)
    combos_order = [
        "intensity", "depth", "breadth",
        "depth+intensity", "breadth+intensity", "breadth+depth",
        "breadth+depth+intensity",
    ]
    plot_statistic_combinations(
        per_model, out_path,
        combos_order=combos_order,
        y_axes={
            "auroc": ([0.5, 0.6, 0.7, 0.8], (0.47, 0.83)),
            "auprc": ([0.4, 0.5, 0.6, 0.7, 0.8], (0.38, 0.84)),
            "pcc": ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], (-0.045, 0.52)),
        },
        fig_w=5.05, fig_h=5.47,
        colors={"llama2_7b": "#0956AF", "llama2_13b": "#2995A3",
                "llama3_8b": "#6E6E6E", "mistral_7b": "#111111"},
        mean_color="#A63096",
        mean_models=models,
        label_suffix={"llama3_8b": " (RAGTruth only)"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the paper figures from log/results.")
    parser.add_argument("--out-dir", default="figs_generated")
    args = parser.parse_args()

    _report_figure_font()

    results = ROOT / "log" / "results"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _make_intervention(results, out_dir / "fig2_intervention.pdf")
    _make_ablation(results, out_dir / "fig3_ablation.pdf")
    _make_statistics(results, out_dir / "fig4_statistics.pdf")
    make_figure(out_dir / "fig5_weight_simplex.pdf")

    print("Figures written to", out_dir)


if __name__ == "__main__":
    main()
