# -*- coding: utf-8 -*-

"""Core CASDET algorithm.

Implements the local control state, the influence signal, Logit-Lens
quantification, the three influence statistics, and the response-level
hallucination risk score. Pure-math helpers are model-free; model-dependent
functions consume LayerStates captured by ModelAdapter."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from .model_adapter import LayerStates, ModelAdapter

@dataclass
class ProfileStatistics:
    """Response-level influence breadth, depth, and intensity (Eqs. 13-15)."""
    breadth: float
    depth: float
    intensity: float

@dataclass
class CasdetResult:
    """Per-response CASDET output: token-level and layer-level effects."""
    sample_id: str
    label: bool
    token_effect: np.ndarray
    layer_effects: np.ndarray

def _log_prob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Log-probability of the target tokens, computed directly from logits."""
    z = logits.float()
    tid = token_ids.to(z.device)
    col = tid.unsqueeze(1)
    return z.gather(1, col).squeeze(1) - torch.logsumexp(z, dim=-1)

def layer_relative_change(
    logits: torch.Tensor,
    logits_hat: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Local change in the target token's log-probability at one layer (Eq. 9)."""
    z = logits.float()
    z_hat = logits_hat.float()
    tid = token_ids.to(z.device)
    return _log_prob(z, tid) - _log_prob(z_hat, tid)

def token_relative_effect(layer_changes: torch.Tensor) -> torch.Tensor:
    """Average layer-wise changes into the token-level effect (Eq. 10)."""
    return layer_changes.float().mean(dim=0)

def _layer_scale(layer_effects: Sequence[np.ndarray]) -> np.ndarray:
    """Per-layer standard deviation used to normalize influence intensity."""
    pooled: List[np.ndarray] = []
    n_layers: Optional[int] = None
    for values in layer_effects:
        e = np.asarray(values, dtype=np.float64)
        if e.size == 0:
            raise ValueError("Layer effects are empty (the response has no generated tokens), "
                             "so the three statistics cannot be computed")
        if n_layers is None:
            n_layers = e.shape[0]
        elif e.shape[0] != n_layers:
            raise ValueError("Inconsistent layer counts across responses: %d vs %d"
                             % (n_layers, e.shape[0]))
        pooled.append(e)
    all_values = np.concatenate(pooled, axis=1)
    mu = np.nanmean(all_values, axis=1)
    dev = np.where(np.isfinite(all_values), all_values - mu[:, None], np.nan)
    sigma = np.sqrt(np.maximum(np.nanmean(dev ** 2, axis=1), 0.0))
    return np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1.0)

def influence_breadth(layer_effects: np.ndarray) -> float:
    """Fraction of positive (layer, token) entries (Eq. 13)."""
    e = np.asarray(layer_effects, dtype=np.float64)
    if e.size == 0:
        return float("nan")
    return float((e > 0).mean())

def influence_depth(layer_effects: np.ndarray) -> float:
    """Positive-effect-weighted average layer index (Eq. 14)."""
    e = np.asarray(layer_effects, dtype=np.float64)
    if e.size == 0 or e.ndim != 2:
        return float("nan")
    L = e.shape[0]
    if L < 2:
        return float("nan")
    idx = np.arange(L, dtype=np.float64).reshape(-1, 1)
    pos = np.clip(e, 0.0, None)
    den = pos.sum(axis=0)
    ok = den > 1e-12
    ratio = np.where(ok, (idx * pos).sum(axis=0) / np.where(ok, den, 1.0), 0.0)
    return float(ratio.mean())

def influence_intensity(layer_effects: np.ndarray, sigma: np.ndarray) -> float:
    """RMS of layer-standardized token effects (Eq. 15)."""
    e = np.asarray(layer_effects, dtype=np.float64)
    if e.size == 0 or e.ndim != 2:
        return float("nan")
    s_bar = np.nanmean(e, axis=1)
    z = s_bar / np.asarray(sigma, dtype=np.float64)
    finite = np.isfinite(z)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.nanmean(np.where(finite, z ** 2, np.nan))))

def profile_statistics(layer_effects: Sequence[np.ndarray]
                       ) -> List[ProfileStatistics]:
    """Compute the three statistics for a batch of responses (Eqs. 13-15)."""
    grid = list(layer_effects)
    if not grid:
        return []
    # sigma_l is defined over the evaluation batch, so the statistics are batch-level.
    sigma = _layer_scale(grid)
    return [ProfileStatistics(breadth=influence_breadth(e),
                              depth=influence_depth(e),
                              intensity=influence_intensity(e, sigma))
            for e in grid]

def relative_position(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank; NaN is preserved."""
    values = np.asarray(x, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    n = int(ok.sum())
    if n == 0:
        return out
    ranks = rankdata(values[ok], method="average") - 1.0
    out[ok] = ranks / max(1, n - 1)
    return out

def fuse_statistics(*statistics: Sequence[float],
                    weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Weighted sum of risk-oriented relative positions."""
    risk = [relative_position(-np.asarray(v, dtype=np.float64))
            for v in statistics]
    stack = np.stack(risk, axis=0)
    if weights is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stack, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stack.shape[0],):
        raise ValueError(f"weights has shape {w.shape}, but there are "
                         f"{stack.shape[0]} statistics")
    if (w < 0).any():
        raise ValueError(f"weights must be non-negative: {w!r}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum()!r}")
    finite = np.isfinite(stack)
    wcol = np.where(finite, w[:, None], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nansum(np.where(finite, stack, 0.0) * w[:, None], axis=0) \
            / np.where(wcol.sum(axis=0) > 0, wcol.sum(axis=0), np.nan)

def fuse_profile_statistics(breadth: Sequence[float],
                            depth: Sequence[float],
                            intensity: Sequence[float],
                            weights: Optional[Sequence[float]] = None
                            ) -> np.ndarray:
    """Response-level hallucination risk score (Eq. 16)."""
    return fuse_statistics(breadth, depth, intensity, weights=weights)

def _response_token_ids(input_ids: List[int],
                        resp_span: Tuple[int, int]) -> torch.Tensor:
    """Target token ids aligned with the response rows."""
    r_start, r_end = resp_span
    return torch.tensor(input_ids[r_start:r_end], dtype=torch.long)

def compute_layer_effects(
    adapter: ModelAdapter,
    layers: Dict[int, LayerStates],
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute S_t^l for every response token and layer (Eq. 9)."""
    L = adapter.num_layers
    R = token_ids.numel()
    token_ids_dev = token_ids.to(adapter.device)
    per_layer = torch.empty(L, R, dtype=torch.float32)
    running: Optional[torch.Tensor] = None
    for li in range(L):
        s: LayerStates = layers[li]
        x_mid = s.x_mid.to(adapter.device)
        e = s.e.to(adapter.device)
        # Replace only the FFN output; the attention contribution stays unchanged.
        x_mid_hat = x_mid - e
        f = s.f.to(adapter.device)
        f_hat = adapter.run_ffn(li, x_mid_hat).to(x_mid.device)
        x = x_mid + f
        x_hat = x_mid + f_hat
        sl = layer_relative_change(adapter.logit_lens(x),
                                   adapter.logit_lens(x_hat),
                                   token_ids_dev).cpu()
        per_layer[li] = sl
        running = sl if running is None else running + sl
    return running, per_layer

def run_casdet(
    adapter: ModelAdapter,
    input_ids: List[int],
    resp_span: Tuple[int, int],
    ctx_mask: Sequence[int],
    sample_id: str,
    label: bool,
) -> CasdetResult:
    """Run CASDET on one response."""
    with adapter.capture(input_ids, resp_span, ctx_mask) as layers:
        token_ids = _response_token_ids(input_ids, resp_span)
        s_sum, per_layer = compute_layer_effects(adapter, layers, token_ids)
    s = s_sum / adapter.num_layers
    return CasdetResult(
        sample_id=sample_id,
        label=label,
        token_effect=s.cpu().numpy(),
        layer_effects=per_layer.cpu().numpy(),
    )

def graded_intervention(
    adapter: ModelAdapter,
    layers: Dict[int, LayerStates],
    token_ids: torch.Tensor,
    gamma_grid: Sequence[float],
) -> np.ndarray:
    """Apply the graded local intervention for a given gamma (Eq. 11)."""
    L = adapter.num_layers
    R = token_ids.numel()
    token_ids_dev = token_ids.to(adapter.device)
    gamma_grid = list(gamma_grid)
    s_gamma = np.zeros((R, len(gamma_grid)), dtype=np.float64)

    x_zero_states: List[torch.Tensor] = []
    for li in range(L):
        s: LayerStates = layers[li]
        x_mid = s.x_mid.to(adapter.device)
        f_zero = adapter.run_ffn(li, x_mid - s.e.to(adapter.device)).to(x_mid.device)
        x_zero_states.append(x_mid + f_zero)

    for gi, gamma in enumerate(gamma_grid):
        if gamma == 0.0:
            continue
        running: Optional[torch.Tensor] = None
        for li in range(L):
            s: LayerStates = layers[li]
            x_mid = s.x_mid.to(adapter.device)
            e = s.e.to(adapter.device)
            x_mid_gamma = x_mid - (1.0 - gamma) * e
            f_gamma = adapter.run_ffn(li, x_mid_gamma).to(x_mid.device)
            x_gamma = x_mid + f_gamma
            sl = layer_relative_change(adapter.logit_lens(x_gamma),
                                       adapter.logit_lens(x_zero_states[li]),
                                       token_ids_dev).cpu()
            running = sl if running is None else running + sl
        s_gamma[:, gi] = running.numpy() / L
    return s_gamma

def graded_intervention_effect(
    s_gamma: np.ndarray,
) -> np.ndarray:
    """Mean absolute token-level effect at a given gamma (Eq. 12)."""
    return np.abs(s_gamma).mean(axis=0)

def attention_only_effect(
    adapter: ModelAdapter,
    layers: Dict[int, LayerStates],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Ablation: quantify only the direct attention contribution."""
    L = adapter.num_layers
    token_ids_dev = token_ids.to(adapter.device)
    per_layer: List[torch.Tensor] = []
    for li in range(L):
        s: LayerStates = layers[li]
        x_mid = s.x_mid.to(adapter.device)
        x_mid_hat = x_mid - s.e.to(adapter.device)
        sl = layer_relative_change(adapter.logit_lens(x_mid),
                                   adapter.logit_lens(x_mid_hat),
                                   token_ids_dev).cpu()
        per_layer.append(sl)
    return torch.stack(per_layer, dim=0)

def no_control_effect(
    adapter: ModelAdapter,
    layers: Dict[int, LayerStates],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Ablation: quantify the normal FFN output without a control state."""
    L = adapter.num_layers
    token_ids_dev = token_ids.to(adapter.device)
    per_layer: List[torch.Tensor] = []
    for li in range(L):
        s: LayerStates = layers[li]
        x = s.x_mid.to(adapter.device) + s.f.to(adapter.device)
        z = adapter.logit_lens(x).float()
        sl = _log_prob(z, token_ids_dev.to(z.device)).cpu()
        per_layer.append(sl)
    return torch.stack(per_layer, dim=0)

def variant_statistics(layer_effects: Sequence[torch.Tensor]
                       ) -> List[ProfileStatistics]:
    """Compute the three statistics for a batch of ablation results (Eqs. 13-15)."""
    return profile_statistics([np.asarray(e, dtype=np.float64)
                               for e in layer_effects])
