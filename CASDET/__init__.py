# -*- coding: utf-8 -*-

"""Detecting RAG hallucinations via cascade influence from attention to feedforward networks.

Core package: model access, influence-signal computation, datasets, metrics,
experiment pipelines, and utilities."""
from .datasets import RagSample, load_dataset, tokenize_and_align
from .metrics import auprc, auroc, compute_metrics, page_l_test, pcc
from .model_adapter import LayerStates, ModelAdapter
from .casdet import (
    CasdetResult,
    ProfileStatistics,
    fuse_profile_statistics,
    fuse_statistics,
    influence_breadth,
    influence_depth,
    influence_intensity,
    layer_relative_change,
    profile_statistics,
    relative_position,
    run_casdet,
    token_relative_effect,
    variant_statistics,
)
from .experiments import run_experiment
from .reference import (
    ReferenceCalibration,
    fit_reference,
    score_batch_with_reference,
    score_with_reference,
)

__version__ = "1.1.1"

__all__ = [
    "RagSample", "load_dataset", "tokenize_and_align",
    "ModelAdapter", "LayerStates",
    "CasdetResult", "ProfileStatistics",
    "run_casdet", "profile_statistics", "variant_statistics",
    "influence_breadth", "influence_depth", "influence_intensity",
    "relative_position", "fuse_statistics", "fuse_profile_statistics",
    "ReferenceCalibration", "fit_reference",
    "score_with_reference", "score_batch_with_reference",
    "layer_relative_change", "token_relative_effect",
    "auroc", "auprc", "pcc", "compute_metrics", "page_l_test",
    "run_experiment",
    "__version__",
]
