# -*- coding: utf-8 -*-

"""Weight-simplex figure (paper Figure 5).

Reads the main-experiment results under ``log/results/main`` and writes the
figure as a vector PDF. Invoked by ``scripts/make_figures.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CASDET.experiments import merge_weight_sweep, _sweep_table  # noqa: E402
from CASDET.utils import _DATASET_DISPLAY  # noqa: E402

DATASETS = ("ragtruth", "hallurag")

METRICS = ("auroc", "auprc", "pcc")

HGT = np.sqrt(3) / 2

MODEL_COLORS = {"llama2_7b": "#0956AF", "llama2_13b": "#2995A3",
                "llama3_8b": "#6E6E6E", "mistral_7b": "#111111"}

MODEL_NAMES = {"llama2_7b": "Llama2-7B", "llama2_13b": "Llama2-13B",
               "llama3_8b": "Llama3-8B", "mistral_7b": "Mistral-7B"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
}

_K = 1.15

_PAD = 7.2

_T_W = 180.28 * _K

_T_H = _T_W * np.sqrt(3) / 2

_TITLE_H = 10.06 * _K

_GAP_TT = 4.09 * _K

_GUTTER = 18.56

_COL_W = _T_W + 7.60 + 6.34 + 7.00 + 21.38 * _K

_W_CANVAS = 2 * _PAD + 3 * _COL_W + 2 * _GUTTER

_AIR_ROW = 15.0

_T_H_PANEL = _T_H + 22.0

_LEG_FS = 12.5

def tern_xy(wk, wd):
    """Convert barycentric weights to 2-D simplex coordinates."""
    return wk + 0.5 * wd, HGT * wd

def load_all():
    """Load weight-sweep tables and per-model data from the main results."""
    out = ROOT / "log" / "results" / "main"
    tables, by_model = {}, {}
    for ds in DATASETS:
        bm = merge_weight_sweep(out, ds)
        assert bm, f"{ds}: no sweep data"
        by_model[ds] = bm
        tables[ds] = {m: _sweep_table(bm, m) for m in METRICS}
    return tables, by_model

def draw_triangles(fig, tables, H_TOTAL, y_off=0.0):
    """Draw the six simplex panels."""
    K = 1.15
    PAD = 7.2
    T_W = 180.28 * K
    T_H = T_W * np.sqrt(3) / 2
    CB_OFF = 7.60
    CB_W = 6.34
    CB_LBL = 7.00
    TICK_W = 21.38 * K
    GUTTER = 18.56
    TITLE_H = 10.06 * K
    GAP_TT = 4.09 * K
    AIR_ROW = 15.0
    LBL_FS = 7.0 * K
    LBL_PAD = 1.20

    row_pitch = TITLE_H + GAP_TT + T_H + AIR_ROW
    col_w = T_W + CB_OFF + CB_W + CB_LBL + TICK_W
    W_FIG = 2 * PAD + 3 * col_w + 2 * GUTTER
    assert abs(W_FIG - 810.07) < 0.01, "unexpected 3-column canvas width"

    def _rect(x_pt, y_top_pt, w_pt, h_pt):
        y_top_pt = y_top_pt + y_off
        return [x_pt / W_FIG, (H_TOTAL - y_top_pt - h_pt) / H_TOTAL,
                w_pt / W_FIG, h_pt / H_TOTAL]

    def _value_label(ax, ax_x, ax_y, txt, right):
        bb = TextPath((0.0, 0.0), txt, size=LBL_FS,
                      prop=FontProperties(family="Arial")).get_extents()
        hw = (bb.width / 2 + LBL_PAD) / T_W
        hh = (bb.height / 2 + LBL_PAD) / T_W
        cx = ax_x - hw if right else ax_x + hw
        cy = ax_y - hh
        ax.add_patch(FancyBboxPatch(
            (cx - hw, cy - hh), 2 * hw, 2 * hh,
            boxstyle="round,pad=0,rounding_size=%.5f" % (LBL_PAD / T_W),
            fc="white", ec="none", alpha=0.82, zorder=6.8))
        ax.text(cx - (bb.x0 + bb.x1) / 2 / T_W,
                cy - (bb.y0 + bb.y1) / 2 / T_W, txt,
                ha="left", va="baseline", fontsize=LBL_FS, zorder=7)

    letter = iter("abcdef")
    for r, ds in enumerate(DATASETS):
        for c, met in enumerate(METRICS):
            wk, wd, mean, _err = tables[ds][met]
            eq = int(np.argmin(np.abs(wk - 1 / 3) + np.abs(wd - 1 / 3)))
            best = int(np.nanargmax(mean))
            x, y = tern_xy(wk, wd)
            tri = Triangulation(x, y)
            vmin, vmax = float(np.nanmin(mean)), float(np.nanmax(mean))
            span = vmax - vmin
            pad = 0.03 * span if span > 1e-9 else max(0.004,
                                                     abs(vmax) * 0.01)
            vlo, vhi = vmin - pad, vmax + pad
            lv = np.linspace(vlo, vhi, 14)
            x0 = PAD + c * (col_w + GUTTER)
            y_top = PAD + r * row_pitch + TITLE_H + GAP_TT
            ax = fig.add_axes(_rect(x0, y_top, T_W, T_H))
            cf = ax.tricontourf(tri, mean, levels=lv, cmap="viridis")
            ax.tricontour(tri, mean, levels=lv, colors="white",
                          linewidths=0.35, alpha=0.65)
            for g in (1 / 3, 2 / 3):
                ax.plot([g, g + 0.5 * (1 - g)], [0, HGT * (1 - g)],
                        color="white", lw=0.5, alpha=0.35)
                ax.plot([1 - g, 0.5 * (1 - g)], [0, HGT * (1 - g)],
                        color="white", lw=0.5, alpha=0.35)
                ax.plot([1 - 0.5 * g, 0.5 * g], [HGT * g, HGT * g],
                        color="white", lw=0.5, alpha=0.35)
            ax.plot(x, y, ".", color="white", ms=1.4, alpha=0.5)
            ax.plot([0, 1, 0.5, 0], [0, 0, HGT, 0], color="black", lw=1.1)

            ex, ey = tern_xy(wk[eq], wd[eq])
            bx, by = tern_xy(wk[best], wd[best])
            ax.plot(ex, ey, marker="*", ms=13, mfc="white", mec="#222",
                    mew=0.8, zorder=6)
            ax.plot(bx, by, marker="D", ms=6, mfc="#222", mec="white",
                    mew=0.7, zorder=6)
            _value_label(ax, ex - 0.02, ey - 0.03, "%.4f" % mean[eq],
                         right=True)
            _value_label(ax, bx + 0.02, by - 0.03, "%.4f" % mean[best],
                         right=False)

            bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                        alpha=0.78)
            t = 0.14
            ux, uy = 0.8660, 0.5
            ax.text(0.5, HGT - t, r"$w_d$", ha="center", va="center",
                    fontsize=9.5 * K, bbox=bbox)
            ax.text(1 - ux * t, uy * t, r"$w_\kappa$", ha="center",
                    va="center", fontsize=9.5 * K, bbox=bbox)
            ax.text(ux * t, uy * t, r"$w_\rho$", ha="center", va="center",
                    fontsize=9.5 * K, bbox=bbox)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, HGT)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("(%s) %s — %s" % (next(letter),
                                           _DATASET_DISPLAY.get(ds, str(ds)),
                                           met.upper()), fontsize=9 * K,
                         pad=6 * K)
            cax = fig.add_axes(_rect(x0 + T_W + CB_OFF, y_top, CB_W, T_H))
            cb = fig.colorbar(cf, cax=cax)
            cb.outline.set_visible(False)
            ticks = np.linspace(vlo, vhi, 5)
            cb.set_ticks(ticks)
            cb.set_ticklabels(["%.4f" % t for t in ticks])
            cb.ax.tick_params(labelsize=7 * K)


def _draw_metric_panels(fig, tables, by_model, rects):
    """Draw the metric panels over the weight simplex."""
    plt.rcParams.update(RC)
    letters = iter("gh")
    panels = []
    for i, ds in enumerate(DATASETS):
        bm = by_model[ds]
        models = sorted(bm)
        wk, wd, mean, _ = tables[ds]["auroc"]
        eq = int(np.argmin(np.abs(wk - 1 / 3) + np.abs(wd - 1 / 3)))
        ax = fig.add_axes(rects[i], projection="3d")
        ax.computed_zorder = False
        ax.set_proj_type("ortho")
        eq_pts = []
        lo3 = np.array([np.inf, np.inf, np.inf])
        hi3 = -lo3
        for mm in models:
            a = np.array([p["metrics"]["auroc"] for p in bm[mm]])
            b = np.array([p["metrics"]["auprc"] for p in bm[mm]])
            c = np.array([p["metrics"]["pcc"] for p in bm[mm]])
            assert len(a) == 91 and np.isfinite(a).all() \
                and np.isfinite(b).all() and np.isfinite(c).all(), \
                "%s/%s: expected 91 finite values" % (ds, mm)
            lo3 = np.minimum(lo3, [a.min(), b.min(), c.min()])
            hi3 = np.maximum(hi3, [a.max(), b.max(), c.max()])
            col = MODEL_COLORS[mm]
            ax.scatter(a, b, c, c=col, s=5.5, alpha=0.5, depthshade=False,
                       zorder=4)
            ax.scatter([a[eq]], [b[eq]], [c[eq]], c=col, s=60,
                       edgecolors="white", linewidths=1.0, depthshade=False,
                       zorder=8)
            eq_pts.append((a[eq], b[eq], c[eq]))
        pad3 = 0.05 * (hi3 - lo3)
        ax.set_xlim(lo3[0] - pad3[0], hi3[0] + pad3[0])
        ax.set_ylim(lo3[1] - pad3[1], hi3[1] + pad3[1])
        ax.set_zlim(lo3[2] - pad3[2], hi3[2] + pad3[2])
        for ax_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_.set_major_locator(MaxNLocator(3))
            ax_.set_major_formatter(FormatStrFormatter("%.2f"))
        for _axname in ("x", "y", "z"):
            ax.tick_params(axis=_axname, labelsize=7 * _K, pad=0)
        ax.zaxis.pane.set_facecolor((0.87, 0.87, 0.90))
        ax.zaxis.pane.set_alpha(1.0)
        ax.view_init(elev=32, azim=-142)
        ax.set_box_aspect((1.0, 1.0, 0.85), zoom=1.03)
        ax.set_title("(%s) %s — AUROC / AUPRC / PCC"
                     % (next(letters), _DATASET_DISPLAY[ds]),
                     fontsize=9 * _K, pad=6 * _K)
        panels.append((ax, lo3, hi3))

    ov = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=40)
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.axis("off")
    fig.set_dpi(220)

    CAL_COLORS = [(255, 0, 0), (0, 190, 0), (0, 80, 255),
                  (255, 140, 0), (150, 0, 210), (220, 220, 0)]
    temp_art, cal_info = [], []
    for ax_, lo3_, hi3_ in panels:
        a_lo, b_lo, c_lo = lo3_
        a_hi, b_hi, c_hi = hi3_
        pts3 = [(a_lo, b_lo, c_lo), (a_hi, b_lo, c_lo), (a_lo, b_hi, c_lo),
                (a_lo, b_lo, c_hi), (a_hi, b_lo, c_hi), (a_lo, b_hi, c_hi)]
        uv = [proj3d.proj_transform(p[0], p[1], p[2], ax_.get_proj())[:2]
              for p in pts3]
        art = ax_.scatter([p[0] for p in pts3], [p[1] for p in pts3],
                          [p[2] for p in pts3],
                          c=[tuple(cc / 255.0 for cc in col)
                             for col in CAL_COLORS],
                          s=42, depthshade=False, zorder=30)
        temp_art.append(art)
        cal_info.append((ax_, np.asarray(uv, dtype=float), pts3, lo3_, hi3_))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    hbuf = buf.shape[0]

    for ax_, uv_arr, pts3, lo3_, hi3_ in cal_info:
        pos_ = ax_.get_position()
        px0 = int(pos_.x0 * fig.bbox.width)
        px1 = int(pos_.x1 * fig.bbox.width)
        r0 = int((1.0 - pos_.y1) * fig.bbox.height)
        r1 = int((1.0 - pos_.y0) * fig.bbox.height)
        sub = buf[r0:r1, px0:px1]
        disp = []
        for col in CAL_COLORS:
            m_ = ((abs(sub[:, :, 0] - col[0]) < 40)
                  & (abs(sub[:, :, 1] - col[1]) < 40)
                  & (abs(sub[:, :, 2] - col[2]) < 40))
            yy, xx = np.where(m_)
            disp.append([np.nan, np.nan] if len(xx) == 0
                        else [xx.mean() + px0,
                              hbuf - 1 - (yy.mean() + r0)])
        disp = np.asarray(disp, dtype=float)
        ok = ~np.isnan(disp).any(axis=1)
        Afit = np.column_stack([uv_arr[ok], np.ones(int(ok.sum()))])
        cx = np.linalg.lstsq(Afit, disp[ok, 0], rcond=None)[0]
        cy = np.linalg.lstsq(Afit, disp[ok, 1], rcond=None)[0]

        def to_disp(p3, _ax=ax_, _cx=cx, _cy=cy):
            u, v, _ = proj3d.proj_transform(p3[0], p3[1], p3[2],
                                            _ax.get_proj())
            return np.array([_cx[0] * u + _cx[1] * v + _cx[2],
                             _cy[0] * u + _cy[1] * v + _cy[2]])

        a_lo, b_lo, c_lo = lo3_
        a_hi, b_hi, c_hi = hi3_
        cen = to_disp(((a_lo + a_hi) / 2.0, (b_lo + b_hi) / 2.0,
                       (c_lo + c_hi) / 2.0))

        def put(txt, pm, pe, off, rot_lock=None):
            d_ = to_disp(pe) - to_disp(pm)
            ang = float(np.degrees(np.arctan2(d_[1], d_[0])))
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            mid_ = to_disp(pm)
            out_ = mid_ - cen
            n_ = float(np.hypot(out_[0], out_[1]))
            out_ = out_ / n_ if n_ > 1e-9 else np.array([0.0, -1.0])
            pos_ = mid_ + out_ * off
            fx, fy = fig.transFigure.inverted().transform(pos_)
            ov.text(fx, fy, txt,
                    rotation=ang if rot_lock is None else rot_lock,
                    rotation_mode="anchor", ha="center", va="center",
                    fontsize=8.5 * _K)

        put("AUPRC", (a_lo, (b_lo + b_hi) / 2.0, c_lo), (a_lo, b_hi, c_lo),
            112)
        put("AUROC", ((a_lo + a_hi) / 2.0, b_lo, c_lo), (a_hi, b_lo, c_lo),
            112)
        put("PCC", (a_lo, b_hi, (c_lo + c_hi) / 2.0), (a_lo, b_hi, c_hi),
            112, rot_lock=90)
    for art in temp_art:
        art.remove()

def _panel_i_data(tables):
    """Aggregate tuning gain and cross-benchmark transfer costs."""
    out = {}
    for met in METRICS:
        vals = {}
        for ds in DATASETS:
            wk, wd, mean, _err = tables[ds][met]
            eqi = int(np.argmin(np.abs(wk - 1 / 3) + np.abs(wd - 1 / 3)))
            pk = int(np.nanargmax(mean))
            vals[ds] = dict(eq=float(mean[eqi]), peak=float(mean[pk]),
                            gain=float(mean[pk] - mean[eqi]),
                            wpk=(float(wk[pk]), float(wd[pk])))

        def _at(ds, wpk):
            wk, wd, mean, _err = tables[ds][met]
            i = int(np.argmin(np.abs(wk - wpk[0]) + np.abs(wd - wpk[1])))
            return float(mean[i])

        out[met] = dict(
            gain=max(vals["ragtruth"]["gain"], vals["hallurag"]["gain"]),
            r2h=vals["hallurag"]["eq"] - _at("hallurag",
                                             vals["ragtruth"]["wpk"]),
            h2r=vals["ragtruth"]["eq"] - _at("ragtruth",
                                             vals["hallurag"]["wpk"]))
    return out

def draw_panel_i(fig, tables, by_model, rect, H_TOTAL):
    """Draw the tuning-gain / transfer-cost panel."""
    pad_l, pad_r, pad_b, pad_t = 34.0, 10.0, 14.0, 32.0
    x0, y0, wid, hei = rect
    ax = fig.add_axes([x0 + pad_l / _W_CANVAS, y0 + pad_b / H_TOTAL,
                       wid - (pad_l + pad_r) / _W_CANVAS,
                       hei - (pad_b + pad_t) / H_TOTAL])
    _panel_i_dots(ax, tables)
    return ax

def _panel_i_dots(ax, tables):
    """Draw panel (i) as gain and transfer-cost dots."""
    data = _panel_i_data(tables)
    X = np.arange(len(METRICS))
    C_GAIN, C_COST = "#5AA469", "#A63096"
    OFF = (-0.12, 0.0, 0.12)

    for i, m in enumerate(METRICS):
        g, a, b = data[m]["gain"], data[m]["r2h"], data[m]["h2r"]
        xs = [i + o for o in OFF]
        ax.plot(xs, [g, a, b], "-", color="#D6D6D6", lw=1.1, zorder=1)
        ax.plot([xs[0]], [g], "o", ms=6.4, mfc=C_GAIN, mec="white",
                mew=0.8, zorder=3)
        ax.plot([xs[1]], [a], "o", ms=6.4, mfc=C_COST, mec="white",
                mew=0.8, zorder=3)
        ax.plot([xs[2]], [b], "o", ms=4.6, mfc="white", mec=C_COST,
                mew=1.0, zorder=3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="None", mfc=C_GAIN, mec="white",
               mew=0.8, ms=6.4, label="tuning gain"),
        Line2D([], [], marker="o", ls="None", mfc=C_COST, mec="white",
               mew=0.8, ms=6.4, label="RAGTruth → HalluRAG transfer cost"),
        Line2D([], [], marker="o", ls="None", mfc="white", mec=C_COST,
               mew=1.0, ms=4.6, label="HalluRAG → RAGTruth transfer cost")],
        loc="upper left", frameon=False, fontsize=9 * _K, handlelength=1.4,
        handletextpad=0.5, labelspacing=0.4, borderpad=0.1,
        borderaxespad=0.3)
    ax.set_xticks(X)
    ax.set_xticklabels([m.upper() for m in METRICS], fontsize=8.5 * _K)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"], fontsize=7 * _K)
    ax.set_ylim(0, 0.24)
    ax.set_xlim(-0.55, 2.63)
    ax.set_ylabel("effect on the metric", fontsize=8.5 * _K, labelpad=3)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for xv in (0.5, 1.5):
        ax.plot([xv, xv], [0, 0.15], color="#E3E3E3", lw=0.7,
                ls=(0, (2, 3)), zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["left"].set_linewidth(0.7)
    ax.tick_params(length=2.2, color="#777")
    ax.tick_params(axis="x", pad=5.5)
    ax.set_title("(i) Tuning gain vs. transfer cost",
                 fontsize=9.5 * _K, pad=6 * _K + 31.0)

def _cell_rect(x0_pt, y_top_pt, H_TOTAL):
    """Return the axes rectangle for a panel cell."""
    return [x0_pt / _W_CANVAS,
            (H_TOTAL - y_top_pt - _T_H_PANEL) / H_TOTAL,
            _COL_W / _W_CANVAS, _T_H_PANEL / H_TOTAL]

def make_figure(out_path: Path | None = None) -> Path:
    """Generate the weight-simplex figure and return its path."""
    tables, by_model = load_all()

    row_pitch = _TITLE_H + _GAP_TT + _T_H + _AIR_ROW
    Y_OFF = 66.0
    y_top3 = Y_OFF + _PAD + 1 * row_pitch + _TITLE_H + _GAP_TT
    y_end3 = y_top3 + _T_H + 4.65
    y_title_band = y_end3 + _AIR_ROW - 4.32
    y_slot_band = y_title_band + _TITLE_H + _GAP_TT
    H_TOTAL = y_slot_band + _T_H_PANEL + 8.0 + 2.0

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(_W_CANVAS / 72.0, H_TOTAL / 72.0))
    draw_triangles(fig, tables, H_TOTAL, y_off=Y_OFF)
    rects_m = [_cell_rect(x0, y_slot_band, H_TOTAL)
               for x0 in (_PAD, _PAD + _COL_W + _GUTTER)]
    _draw_metric_panels(fig, tables, by_model, rects_m)
    rect_i = _cell_rect(_PAD + 2 * (_COL_W + _GUTTER), y_slot_band, H_TOTAL)
    draw_panel_i(fig, tables, by_model, rect_i, H_TOTAL)

    def _top_anchor(y_pt):
        return (0.5, 1.0 - y_pt / H_TOTAL)

    mkeys = ("llama2_7b", "llama2_13b", "llama3_8b", "mistral_7b")
    fig.legend(handles=[
        Line2D([], [], marker="*", linestyle="None", markersize=12,
               mfc="white", mec="#222",
               label="equal weights (1/3, 1/3, 1/3)"),
        Line2D([], [], marker="D", linestyle="None", markersize=6,
               mfc="#222", mec="white", label="peak")],
        loc="upper center", ncol=2, frameon=False, fontsize=_LEG_FS,
        bbox_to_anchor=_top_anchor(-8.5))
    fig.legend(
        handles=[Line2D([], [], marker="o", ls="None",
                        mfc=MODEL_COLORS[m], mec="none", ms=5)
                 for m in mkeys],
        labels=[MODEL_NAMES[m] for m in mkeys],
        loc="upper center", ncol=4, frameon=False, fontsize=_LEG_FS,
        columnspacing=1.8, handletextpad=0.6, handlelength=1.6,
        bbox_to_anchor=_top_anchor(12.5))
    fig.legend(
        handles=[tuple(Line2D([], [], marker="o", ls="None",
                              mfc=MODEL_COLORS[m], mec="white", mew=0.8,
                              ms=8) for m in mkeys)],
        labels=["equal weights (per model)"], loc="upper center", ncol=1,
        frameon=False, fontsize=_LEG_FS, handlelength=3.8,
        bbox_to_anchor=_top_anchor(33.5),
        handler_map={tuple: HandlerTuple(ndivide=None)})

    if out_path is None:
        out_path = ROOT / "figs_generated" / "fig5_weight_simplex.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate the weight-simplex figure (paper Figure 5).")
    parser.add_argument("output", nargs="?", default=None)
    make_figure(parser.parse_args().output)

