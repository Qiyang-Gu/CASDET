# -*- coding: utf-8 -*-

"""Configuration, logging, file utilities, and plotting helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (Any, Dict, Iterator, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import torch
import yaml

PathLike = Union[str, Path]
logger = logging.getLogger("CASDET")

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(prefer: Optional[str] = None) -> torch.device:
    """Select a device."""
    if prefer in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)

def setup_logging(log_file: Optional[PathLike] = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging."""
    logger = logging.getLogger("CASDET")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert isinstance(cfg, dict), f"configuration {path} must be a mapping"
    return cfg

def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    """Write JSON to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("%s.%d.tmp" % (p.name, os.getpid()))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=indent, ensure_ascii=False)
    os.replace(tmp, p)

@contextmanager
def file_lock(path: PathLike, timeout: float = 120.0,
              stale: float = 300.0) -> Iterator[None]:
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale:
                logger.warning("lock %s has existed for %.0f s; taking it over as stale", lock, age)
                try:
                    os.unlink(lock)
                except FileNotFoundError:
                    pass
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError(f"timed out after {timeout:.0f} s waiting for lock {lock}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass

def load_json(path: PathLike, default: Any = None) -> Any:
    """Load JSON from disk, returning a default on failure."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

def save_npz(path: PathLike, **arrays: np.ndarray) -> None:
    """Save NumPy arrays to a compressed .npz file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)

class Timer:

    """Context manager that logs elapsed time."""
    def __init__(self, logger: Optional[logging.Logger] = None,
                 name: str = "step"):
        self.logger = logger
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self.start
        if self.logger is not None:
            self.logger.info("%s took %.2fs", self.name, self.elapsed)

def _trim_figure_pdf(path: PathLike, pad: float = 6.0) -> None:
    import os
    import pymupdf
    doc = pymupdf.open(path)
    pg = doc[0]
    H = pg.rect.height
    pix = pg.get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)[..., :3].mean(axis=2)
    ink = arr < 245
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    S = 300.0 / 72.0
    x0, y0 = cols.min() / S, rows.min() / S
    x1, y1 = (cols.max() + 1) / S, (rows.max() + 1) / S
    bx0, by0 = x0 - pad, y0 - pad
    bx1, by1 = x1 + pad, y1 + pad
    pg.set_mediabox(pymupdf.Rect(bx0, H - by1, bx1, H - by0))
    tmp = str(path) + ".trim"
    doc.save(tmp)
    doc.close()
    os.replace(tmp, path)

def plot_intervention_trend(
    gamma_grid: Sequence[float],
    effects_by_model: Dict[str, np.ndarray],
    out_path: PathLike,
    xlabel: str = r"Intervention coefficient $\gamma$",
    ylabel: str = r"Mean $M_i(\gamma)$ over responses",
) -> None:
    """Plot the graded-intervention trend (Figure 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _pub_rc(plt)

    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.145, top=0.965)

    models = sorted(effects_by_model, key=model_sort_key)
    palette = tuple(_PUB_MODEL_F2.values())
    fs_lab = 8.7
    x0, x1 = float(gamma_grid[0]), float(gamma_grid[-1])
    val_x = x1 + 0.32
    max_end = 0.0
    for i, name in enumerate(models):
        mean = np.mean(effects_by_model[name], axis=0)
        col = _PUB_MODEL_F2.get(name, palette[i % len(palette)])
        max_end = max(max_end, float(mean[-1]))
        ax.plot(gamma_grid, mean, color=col, lw=2.0, zorder=6,
                solid_capstyle="round")
        ax.plot(gamma_grid[1:], mean[1:], "o", ms=4.4, mfc="white", mec=col,
                mew=1.25, zorder=7)
        ax.annotate(model_display_name(name), (x1, mean[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=fs_lab, color=col,
                    va="center", zorder=10)
        ax.annotate("%.4f" % mean[-1], (val_x, mean[-1]), fontsize=fs_lab,
                    color=col, ha="right", va="center", zorder=10)
    ax.plot([x0], [0.0], "o", ms=4.2, mfc="white", mec="#888", mew=1.1,
            zorder=7)
    xlo, xhi = x0 - 0.015, x1 + 0.345
    ylo, yhi = -0.002, max(0.084, max_end * 1.2)
    ymax = yhi
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    yticks = np.arange(0.0, ymax * 0.82, 0.02)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["%.2f" % t for t in yticks])
    ax.set_xticks(list(gamma_grid))
    ax.set_xticklabels(["%g" % g for g in gamma_grid])
    fs_axis = 9.6
    xlab_x = ((x0 + (x1 + 0.06)) / 2 - xlo) / (xhi - xlo)
    ylab_y = ((max_end + 0.002) / 2 - ylo) / (yhi - ylo)
    ax.set_xlabel(xlabel, fontsize=fs_axis, labelpad=7, x=xlab_x)
    ax.set_ylabel(ylabel, fontsize=fs_axis, labelpad=7, y=ylab_y)
    for sn in ("left", "bottom"):
        ax.spines[sn].set_color("#999")
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_bounds(x0, x1 + 0.06)
    ax.spines["left"].set_bounds(0.0, max_end + 0.002)
    ax.tick_params(axis="both", length=2.4, color="#999")
    ax.tick_params(axis="both", labelsize=8.5, labelcolor="#333")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, facecolor="white")
    _trim_figure_pdf(out_path, pad=6.0)
    plt.close(fig)

def _parse_model_key(key: str) -> Tuple[str, int, int]:
    m = re.match(r"^([a-z]+)(\d*)_(\d+)([a-z]+)$", str(key))
    if not m:
        return (str(key), 0, 0)
    fam, ver, size, _unit = m.groups()
    return (fam, int(ver) if ver else 0, int(size))

def model_display_name(key: str) -> str:
    fam, ver, size = _parse_model_key(key)
    m = re.match(r"^([a-z]+)(\d*)_(\d+)([a-z]+)$", str(key))
    if not m:
        return str(key)
    return f"{fam.capitalize()}{ver or ''}-{size}{m.group(4).upper()}"

def model_sort_key(key: str) -> Tuple[str, int, int]:
    return _parse_model_key(key)

_PUB = {
    "signal": "#0F4D92", "teal": "#42949E", "violet": "#9A4D8E",
    "neutral": "#767676", "neutral_l": "#CFCECE", "ink": "#272727",
}
_PUB_MODEL_F2 = {"llama2_7b": "#0956AF", "llama2_13b": "#2995A3",
                 "llama3_8b": "#6E6E6E", "mistral_7b": "#111111"}

def _pub_rc(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

def plot_ablation_dumbbell(
    models: Sequence[str],
    casdet_auprc: Sequence[float],
    variants: Sequence[Tuple[str, Sequence[float]]],
    out_path: PathLike,
    xlabel: str = "AUPRC",
) -> None:
    """Plot the ablation comparison (Figure 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _pub_rc(plt)

    models = sorted(models, key=model_sort_key)
    n = max(1, len(variants))
    fig, axes = plt.subplots(1, n, figsize=(5.77, 2.75), sharex=True,
                             gridspec_kw=dict(left=0.125, right=0.985,
                                              bottom=0.155, top=0.77,
                                              wspace=0.50))
    for ax, (panel_title, abl_vals) in zip(np.atleast_1d(axes), variants):
        ys = np.arange(len(models))[::-1]
        for y, av, tv in zip(ys, abl_vals, casdet_auprc):
            ax.plot([av, tv], [y, y], color=_PUB["neutral_l"], lw=1.6,
                    solid_capstyle="round", zorder=1)
            ax.plot([av], [y], "o", ms=5.0, mfc="white", mec=_PUB["neutral"],
                    mew=1.1, zorder=3)
            ax.plot([tv], [y], "o", ms=5.6, mfc=_PUB["signal"],
                    mec=_PUB["signal"], zorder=3)
            ax.annotate("%+.4f" % (tv - av), (tv, y), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=_PUB["signal"], fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels([model_display_name(m) for m in models])
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.set_xlim(0.45, 0.92)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8])
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, fontsize=8, color=_PUB["ink"], pad=5,
                     fontweight="bold")
        ax.set_axisbelow(True)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=5.6, mfc=_PUB["signal"],
               mec=_PUB["signal"], label="Full CASDET"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white",
               mec=_PUB["neutral"], mew=1.1, label="Ablation variant")],
        loc="lower center", bbox_to_anchor=(0.50, 0.895), ncol=2,
        handletextpad=0.4, columnspacing=1.6, fontsize=8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, facecolor="white")
    plt.close(fig)

_COMBO_SYMBOL = {"breadth": r"\kappa", "depth": "d", "intensity": r"\rho"}

_COMBO_DISPLAY_SEQ = ("intensity", "depth", "breadth")

def _combo_label(combo: str) -> str:
    members = combo.split("+")
    ordered = [p for p in _COMBO_DISPLAY_SEQ if p in members]
    return "$" + r"\!+\!".join(
        _COMBO_SYMBOL.get(p, p) for p in ordered) + "$"

_STAT_X = (0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6)
_STAT_SEP = (4.0, 8.8)
_STAT_XLIM = (-0.55, 10.45)
_STAT_GC = ((_STAT_XLIM[0] + _STAT_SEP[0]) / 2,
            (_STAT_SEP[0] + _STAT_SEP[1]) / 2,
            (_STAT_SEP[1] + _STAT_XLIM[1]) / 2)
_STAT_GROUP = ("single statistic", "pairwise", "full")
_STAT_AXIS = {"auroc": ([0.50, 0.60, 0.70, 0.80], (0.47, 0.83)),
              "auprc": ([0.55, 0.65, 0.75, 0.85], (0.51, 0.86)),
              "pcc": ([0.0, 0.2, 0.4, 0.6], (-0.05, 0.60))}

def _stat_axis_from(vals: "np.ndarray"):
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = max(hi - lo, 1e-6)
    lo, hi = lo - span * 0.13, hi + span * 0.13 * 1.6
    step = 0.1 if span > 0.35 else 0.05
    n0 = int(np.ceil(lo / step - 1e-9))
    n1 = int(np.floor(hi / step + 1e-9))
    ticks = [round(k * step, 6) for k in range(n0, n1 + 1)] or [0.0]
    return ticks, (lo, hi)

_PUB_MODEL_F5 = {"llama2_7b": "#0956AF", "llama2_13b": "#2995A3",
                 "llama3_8b": "#A63096", "mistral_7b": "#6E6E6E"}

def plot_statistic_combinations(
    per_model: Dict[str, Dict[str, Dict[str, float]]],
    out_path: PathLike,
    metrics: Sequence[str] = ("auroc", "auprc", "pcc"),
    y_axes: Optional[Dict[str, Tuple[Sequence[float],
                                     Tuple[float, float]]]] = None,
    combos_order: Optional[Sequence[str]] = None,
    mean_models: Optional[Sequence[str]] = None,
    label_suffix: Optional[Dict[str, str]] = None,
    fig_h: Optional[float] = None,
    colors: Optional[Dict[str, str]] = None,
    mean_color: str = "#111",
    fig_w: Optional[float] = None,
) -> None:
    """Plot the statistic-combination ablation (Figure 4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "font.size": 5.25,
        "lines.scale_dashes": False,
    })

    models = sorted(per_model, key=model_sort_key)
    palette = tuple(_PUB_MODEL_F5.values())
    colors = colors or _PUB_MODEL_F5
    present = set(next(iter(per_model.values())))
    if combos_order is not None:
        combos = [c for c in combos_order if c in present]
    else:
        combos = sorted(present, key=lambda c: (len(c.split("+")), c))
    xs = list(_STAT_X[:len(combos)])
    mean_cols = ([models.index(m) for m in mean_models]
                 if mean_models is not None else list(range(len(models))))

    h_in = fig_h or 5.47
    fig = plt.figure(figsize=(fig_w or 5.05, h_in))
    gs = fig.add_gridspec(2, 2, wspace=0.19, hspace=0.23)
    fig.subplots_adjust(left=0.055, right=0.995,
                        top=1 - 0.594 / h_in, bottom=0.297 / h_in)

    def _stat_x_decor(ax, ylim):
        for sep in _STAT_SEP[:len(combos) - 4]:
            ax.axvline(sep, color="#e8e8e8", lw=0.8, zorder=0)
        ax.set_xlim(*_STAT_XLIM)
        ax.set_xticks(xs)
        ax.set_xticklabels([_combo_label(c) for c in combos], fontsize=8.2)
        for gx, gt in zip(_STAT_GC, _STAT_GROUP):
            ax.text(gx, ylim[1] - 0.004 * (ylim[1] - ylim[0]), gt,
                    ha="center", va="top", fontsize=6.4, color="black")

    def _value_box(ax, txt, x_end, y_end):
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import FancyBboxPatch
        fs = 6.4
        bb = TextPath((0.0, 0.0), txt, size=fs,
                      prop=FontProperties(family="Arial")).get_extents()
        pad_pt = 0.75
        pos = ax.get_position()
        w_pt = pos.width * fig.get_figwidth() * 72.0
        h_pt = pos.height * fig.get_figheight() * 72.0
        sx = (ax.get_xlim()[1] - ax.get_xlim()[0]) / w_pt
        sy = (ax.get_ylim()[1] - ax.get_ylim()[0]) / h_pt
        w = (bb.width + 2 * pad_pt) * sx
        h = (bb.height + 2 * pad_pt) * sy
        cx = x_end - 3.5 * sx - w / 2
        cy = y_end
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=%.6f" % (pad_pt * sx),
            mutation_aspect=sy / sx,
            fc="white", ec="#bbb", lw=0.5, alpha=0.9, zorder=6))
        ax.text(cx - (bb.x0 + bb.x1) / 2 * sx, cy - (bb.y0 + bb.y1) / 2 * sy,
                txt, ha="left", va="baseline", fontsize=fs, zorder=6.5)

    mv_by_metric: Dict[str, "np.ndarray"] = {}
    for ci, met in enumerate(metrics):
        ax = fig.add_subplot(gs[ci // 2, ci % 2])
        vals = np.array([[per_model[m].get(c, {}).get(met, np.nan)
                          for m in models] for c in combos])
        for mi, m in enumerate(models):
            col = colors.get(m, palette[mi % len(palette)])
            ax.plot(xs, vals[:, mi], color=col, lw=1.1, alpha=0.65, zorder=2,
                    marker="o", ms=3.0, mec="none")
        mv = np.nanmean(vals[:, mean_cols], axis=1)
        mv_by_metric[met] = mv
        ax.plot(xs, mv, color=mean_color, lw=1.1, ls=(0, (3.2, 1.9)),
                zorder=5, marker="o", ms=3.0, mfc=mean_color, mec="none")
        ticks, ylim = (y_axes or _STAT_AXIS).get(met, _stat_axis_from(vals))
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.set_yticklabels(["%.2f" % t for t in ticks], fontsize=5.9,
                           color="#444")
        _stat_x_decor(ax, ylim)
        if np.isfinite(mv[-1]):
            _value_box(ax, "%.4f" % mv[-1], xs[-1], mv[-1])
        ax.set_title("(%s) %s" % (chr(97 + ci), met.upper()), fontsize=6.2,
                     pad=11)
        for sn in ("top", "right"):
            ax.spines[sn].set_visible(False)
        for sn in ("left", "bottom"):
            ax.spines[sn].set_color("#888")
            ax.spines[sn].set_linewidth(0.8)
        ax.tick_params(axis="x", length=2.2, color="#888")
        ax.tick_params(axis="y", length=2.2, color="#888")

    axd = fig.add_subplot(gs[1, 1])
    d_vals = np.nanmean([mv_by_metric[m] for m in metrics], axis=0)
    axd.plot(xs, d_vals, color=mean_color, lw=1.1, ls=(0, (3.2, 1.9)),
             zorder=5, marker="o", ms=3.0, mfc=mean_color, mec="none")
    ticks_d, ylim_d = _stat_axis_from(d_vals)
    axd.set_ylim(*ylim_d)
    axd.set_yticks(ticks_d)
    axd.set_yticklabels(["%.2f" % t for t in ticks_d], fontsize=5.9,
                        color="#444")
    _stat_x_decor(axd, ylim_d)
    if np.isfinite(d_vals[-1]):
        _value_box(axd, "%.4f" % d_vals[-1], xs[-1], d_vals[-1])
    axd.set_title("(d) Mean over the three metrics", fontsize=6.2, pad=11)
    for sn in ("top", "right"):
        axd.spines[sn].set_visible(False)
    for sn in ("left", "bottom"):
        axd.spines[sn].set_color("#888")
        axd.spines[sn].set_linewidth(0.8)
    axd.tick_params(axis="x", length=2.2, color="#888")
    axd.tick_params(axis="y", length=2.2, color="#888")

    n_mean = len(mean_models) if mean_models is not None else len(models)
    mean_label = "%s-model mean" % {3: "three", 4: "four"}.get(
        n_mean, str(n_mean))
    sfx = label_suffix or {}
    handles = [Line2D([], [], color=colors.get(m, palette[i % len(palette)]),
                      lw=1.1, marker="o", ms=3.0,
                      mfc=colors.get(m, palette[i % len(palette)]),
                      mec="none", label=model_display_name(m) + sfx.get(m, ""))
               for i, m in enumerate(models)]
    handles.append(Line2D([], [], color=mean_color, lw=1.1, ls=(0, (3.0, 6.4)),
                          marker="o", ms=3.0, mfc=mean_color, mec="none",
                          label=mean_label))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles) + 1,
               frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.5, 1 - 0.163 / h_in),
               handletextpad=0.5, columnspacing=1.7)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

_DATASET_DISPLAY = {"ragtruth": "RAGTruth", "hallurag": "HalluRAG"}
