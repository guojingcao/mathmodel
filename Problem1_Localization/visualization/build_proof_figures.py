"""Build four publication-ready proof figures for Model 1.

The calculations remain in Python as the single source of truth.  The exported
tables are subsequently loaded into Origin by ``build_origin_project.py``.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
ROOT = P1.parent
sys.path.insert(0, str(ROOT / "CUMCM_Visualization"))
sys.path.insert(0, str(P1))

from CUMCM_Style import (  # noqa: E402
    BLUE, DARK_BLUE, DARK_GRAY, GRAY, GREEN, ORANGE, PURPLE, RED,
    polish_axes, set_cumcm_style,
)
from CUMCM_Export import save_cumcm  # noqa: E402
from CUMCM_Diagnostics import check_figure, check_output  # noqa: E402
from config import GridConfig  # noqa: E402
from model.feasible_region import calculate_feasible_region  # noqa: E402
from model.metrics import (  # noqa: E402
    calculate_metrics, maximum_diameter, minimum_enclosing_circle,
    polygon_area_centroid,
)

OUT = ROOT / "origin_figures" / "Problem1_Model1_Proof"
DATA = OUT / "origin_data"
QA = OUT / "qa"
OBS_PATH = P1 / "results" / "observations.jsonl"
SUMMARY_PATH = P1 / "results" / "experiment_summary.csv"
CONV_PATH = P1 / "results" / "grid_convergence.csv"

COLORS = [BLUE, ORANGE, GREEN, PURPLE, DARK_BLUE]


def closed(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, float)
    return np.vstack([poly, poly[0]]) if len(poly) else poly


def farthest_pair(poly: np.ndarray):
    p = np.asarray(poly, float)
    d2 = np.sum((p[:, None] - p[None, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return p[i], p[j], float(math.sqrt(d2[i, j]))


def records_by_id():
    with OBS_PATH.open(encoding="utf-8") as stream:
        return {item["experiment_id"]: item for item in map(json.loads, stream)}


def add_panel_label(ax, label):
    ax.text(-0.02, 1.025, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold")


def finish(fig, stem, title, note, diagnostics, *, top=.90, bottom=.12):
    """Export a clean figure body; figure number/title/caption belong in the paper."""
    fig.subplots_adjust(left=.075, right=.985, bottom=bottom, top=top,
                        wspace=.24, hspace=.30)
    saved = save_cumcm(fig, OUT / stem, dpi=600)
    diagnostics[stem] = check_figure(fig) + check_output(OUT / stem)
    plt.close(fig)
    return saved


def draw_measurements(ax, points, angles, target, *, extent=1500, alpha=.105,
                      labels=True, centerlines=True):
    for i, (point, angle) in enumerate(zip(points, angles)):
        color = COLORS[i % len(COLORS)]
        ax.add_patch(Wedge(point, extent, angle - 1, angle + 1,
                           facecolor=color, edgecolor="none", alpha=alpha, zorder=1))
        for da, ls, lw in ((-1, "--", .75), (0, "-", .9), (1, "--", .75)):
            if da == 0 and not centerlines:
                continue
            rad = np.deg2rad(angle + da)
            end = point + extent * np.array([np.cos(rad), np.sin(rad)])
            ax.plot([point[0], end[0]], [point[1], end[1]], color=color,
                    ls=ls, lw=lw, alpha=.9, zorder=2)
        ax.scatter(*point, marker="^", s=25, color=DARK_BLUE, edgecolor="white",
                   linewidth=.4, zorder=7)
        if labels:
            ax.annotate(f"$P_{i+1}$", point, xytext=(4, 4), textcoords="offset points",
                        fontsize=7.2, color=DARK_GRAY)
    ax.scatter(*target, marker="*", s=62, color=RED, edgecolor="white",
               linewidth=.45, zorder=9, label="真实位置")


def main_geometry_figure(records, diagnostics):
    record = records["number_014_5"]
    points = np.asarray(record["observation_points"], float)
    angles = np.asarray(record["angles"], float)
    target = np.asarray(record["true_position"], float)
    region = calculate_feasible_region(points, angles, 1.0,
                                       config=GridConfig(), return_details=True)
    metric = calculate_metrics(region, target)
    inner, outer = region.inner_hull, region.outer_hull
    center, radius = minimum_enclosing_circle(inner)
    a, b, diameter = farthest_pair(inner)
    dcenter, dradius = (a + b) / 2, diameter / 2
    ratio = 2 * radius / diameter

    fig, axes = plt.subplots(2, 2, figsize=(7.01, 6.15))
    ax = axes[0, 0]
    draw_measurements(ax, points, angles, target, extent=1500)
    ax.add_patch(Circle((0, 0), 1800, facecolor="#F3F4F5", edgecolor=GRAY,
                        lw=.75, alpha=.22, zorder=0))
    ax.set(xlim=(-1450, 1450), ylim=(-1450, 1450), xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal"); polish_axes(ax, grid=False); add_panel_label(ax, "(a) 多站示向角域")

    ax = axes[0, 1]
    draw_measurements(ax, points, angles, target, extent=1500, alpha=.12, labels=False)
    ax.add_patch(Polygon(outer, closed=True, facecolor=ORANGE, alpha=.18,
                         edgecolor=ORANGE, lw=1.0, label="$K_{out}$"))
    ax.add_patch(Polygon(inner, closed=True, facecolor=RED, alpha=.34,
                         edgecolor=RED, lw=1.25, label="$K_{in}$"))
    mid = (outer.min(0) + outer.max(0)) / 2
    span = max(np.ptp(outer, axis=0).max() * .74, 18)
    ax.set(xlim=(mid[0]-span, mid[0]+span), ylim=(mid[1]-span, mid[1]+span),
           xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal"); polish_axes(ax, grid=False)
    add_panel_label(ax, "(b) 公共交集定位区域")

    ax = axes[1, 0]
    lo, hi = outer.min(0) - 1.5, outer.max(0) + 1.5
    cells = region.cells
    mask = ((cells[:, 0] + cells[:, 2] >= lo[0]) & (cells[:, 0] - cells[:, 2] <= hi[0]) &
            (cells[:, 1] + cells[:, 2] >= lo[1]) & (cells[:, 1] - cells[:, 2] <= hi[1]))
    near = cells[mask]
    if len(near) > 1600:
        near = near[np.linspace(0, len(near)-1, 1600).astype(int)]
    segs = []
    for x, y, h in near:
        segs.extend([[(x-h, y-h), (x+h, y-h)], [(x+h, y-h), (x+h, y+h)],
                     [(x+h, y+h), (x-h, y+h)], [(x-h, y+h), (x-h, y-h)]])
    ax.add_collection(LineCollection(segs, colors="#C9CDD1", linewidths=.23, alpha=.72))
    ax.plot(*closed(outer).T, color=ORANGE, lw=1.25, label="$K_{out}$")
    ax.fill(*closed(inner).T, color=BLUE, alpha=.22, label="$K_{in}$")
    ax.plot(*closed(inner).T, color=DARK_BLUE, lw=1.15)
    ax.scatter(*target, marker="*", s=55, color=RED, zorder=8, label="真实位置")
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal"); polish_axes(ax, grid=False)
    add_panel_label(ax, "(c) 自适应网格内外包")

    ax = axes[1, 1]
    ax.fill(*closed(inner).T, color=RED, alpha=.25, edgecolor=RED, lw=1.2,
            label="定位区域")
    ax.add_patch(Circle(dcenter, dradius, fill=False, edgecolor=DARK_GRAY,
                        lw=1.0, ls="--", label="直径圆 $D/2$"))
    ax.add_patch(Circle(center, radius, fill=False, edgecolor=PURPLE,
                        lw=1.45, ls="-.", label="最小覆盖圆 $r^*$"))
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.5, zorder=5,
            label="最远点对")
    ax.scatter([a[0], b[0]], [a[1], b[1]], s=20, color="black", zorder=6)
    ax.scatter(*center, marker="+", s=60, color=PURPLE, lw=1.3, zorder=7)
    dist_d = np.linalg.norm(inner - dcenter, axis=1)
    outside = inner[dist_d > dradius + 1e-8]
    if len(outside):
        ax.scatter(outside[:, 0], outside[:, 1], s=28, facecolors="none", edgecolors=RED,
                   lw=.9, zorder=8, label="直径圆未覆盖点")
    pad = max(np.ptp(inner, axis=0).max() * .20, 3)
    y_low = inner[:,1].min()-pad
    y_high = inner[:,1].max()+pad
    # Reserve an empty information band above the circles so the metrics never
    # cover geometric evidence or collide with the panel heading.
    y_high += .22 * (y_high-y_low)
    ax.set(xlim=(inner[:,0].min()-pad, inner[:,0].max()+pad),
           ylim=(y_low, y_high),
           xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal"); polish_axes(ax, grid=False)
    area, _ = polygon_area_centroid(inner)
    ax.text(.985, .965,
            f"$A={area:.1f}$ m²,  $D={diameter:.2f}$ m,  $r^*={radius:.2f}$ m,  $2r^*/D={ratio:.4f}>1$",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    add_panel_label(ax, "(d) 直径与最小覆盖圆")

    shared_handles = [
        Line2D([], [], marker="^", ls="none", color=DARK_BLUE, markersize=4.5, label="观测站"),
        Line2D([], [], marker="*", ls="none", color=RED, markersize=6.5, label="真实位置"),
        Patch(facecolor=ORANGE, edgecolor=ORANGE, alpha=.20, label="$K_{out}$"),
        Patch(facecolor=BLUE, edgecolor=DARK_BLUE, alpha=.25, label="$K_{in}$"),
        Line2D([], [], color=DARK_GRAY, ls="--", lw=1.0, label="直径圆 $D/2$"),
        Line2D([], [], color=PURPLE, ls="-.", lw=1.4, label="最小覆盖圆 $r^*$"),
    ]
    fig.legend(handles=shared_handles, loc="lower center", bbox_to_anchor=(.53, .006),
               ncol=6, fontsize=6.2, frameon=False, handlelength=2.0, columnspacing=1.0)

    note = ("算例 number_014_5，5站、示向误差界±1°；红星仅用于离线验证。"
            f" 数值包络分辨率 {region.boundary_width:.5f} m，2r*/D={ratio:.4f}>1，直径圆不能覆盖。")
    finish(fig, "fig01_model1_geometry_proof", "图1  有界测向误差下定位区域的构造、包络验证与覆盖判据",
           note, diagnostics, top=.965, bottom=.13)

    theta = np.linspace(0, 2*np.pi, 361)
    columns = {
        "outer_x": closed(outer)[:,0], "outer_y": closed(outer)[:,1],
        "inner_x": closed(inner)[:,0], "inner_y": closed(inner)[:,1],
        "diameter_x": np.array([a[0], b[0]]), "diameter_y": np.array([a[1], b[1]]),
        "mec_x": center[0] + radius*np.cos(theta), "mec_y": center[1] + radius*np.sin(theta),
        "diam_circle_x": dcenter[0] + dradius*np.cos(theta),
        "diam_circle_y": dcenter[1] + dradius*np.sin(theta),
        "observer_x": points[:,0], "observer_y": points[:,1],
        "truth_x": np.array([target[0]]), "truth_y": np.array([target[1]]),
    }
    pd.DataFrame({k: pd.Series(v) for k, v in columns.items()}).to_csv(
        DATA / "fig01_geometry.csv", index=False, encoding="utf-8-sig")
    return region, metric, ratio


def convergence_figure(diagnostics):
    d = pd.read_csv(CONV_PATH).sort_values("requested_resolution", ascending=False)
    specs = [
        ("area_lower", "area_upper", "面积 $A$ (m²)", "A"),
        ("diameter", "diameter_upper", "直径 $D$ (m)", "D"),
        ("cover_radius_lower", "cover_radius", "覆盖半径 $r^*$ (m)", "r"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.01, 5.65), sharex=True)
    for i, (lower, upper, ylabel, key) in enumerate(specs):
        ax = axes[i]
        x, lo, hi = d.requested_resolution.to_numpy(), d[lower].to_numpy(), d[upper].to_numpy()
        ax.fill_between(x, lo, hi, color=BLUE, alpha=.18, label="数值包络")
        ax.plot(x, lo, "o-", color=BLUE, ms=3.8, lw=1.25, label="$K_{in}$ 下界")
        ax.plot(x, hi, "s-", color=ORANGE, ms=3.6, lw=1.2, label="$K_{out}$ 上界")
        rel = (hi[-1]-lo[-1]) / ((hi[-1]+lo[-1])/2) * 100
        ax.text(.985, .92, f"最终相对界宽={rel:.3f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.2, color=DARK_GRAY)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2); ax.invert_xaxis(); polish_axes(ax)
        add_panel_label(ax, f"({chr(97+i)}) {key} 的内外包收敛")
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    axes[-1].set_xlabel("终止网格边长 $h$ (m，对数尺度；由粗到细)")
    axes[-1].set_xticks(d.requested_resolution.to_numpy())
    axes[-1].set_xticklabels([f"{value:g}" for value in d.requested_resolution])
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.53, .008),
               fontsize=6.8, frameon=False)
    note = (r"同一观测算例逐级加密；阴影为 $K_{in}\subseteq\Omega\subseteq K_{out}$ 导出的确定性数值包络。"
            " 三项界宽同步收缩，说明结论不依赖偶然网格采样。")
    finish(fig, "fig02_grid_convergence", "图2  网格加密使面积、直径与最小覆盖圆半径上下界同步收敛",
           note, diagnostics, top=.965, bottom=.105)
    out = d.copy()
    for lower, upper, _, key in specs:
        out[f"rel_gap_{key}"] = (out[upper]-out[lower]) / ((out[upper]+out[lower])/2)
    out.to_csv(DATA / "fig02_convergence.csv", index=False, encoding="utf-8-sig")


def draw_cover_case(ax, poly, subtitle, *, actual=False):
    poly = np.asarray(poly, float)
    center, radius = minimum_enclosing_circle(poly)
    a, b, diameter = farthest_pair(poly)
    dcenter, dradius = (a+b)/2, diameter/2
    ax.fill(*closed(poly).T, color=BLUE if not actual else RED, alpha=.23,
            edgecolor=DARK_BLUE if not actual else RED, lw=1.25)
    ax.add_patch(Circle(dcenter, dradius, fill=False, edgecolor=DARK_GRAY, ls="--", lw=1.0))
    ax.add_patch(Circle(center, radius, fill=False, edgecolor=PURPLE, ls="-.", lw=1.45))
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.35)
    ax.scatter([a[0], b[0]], [a[1], b[1]], color="black", s=17, zorder=5)
    outside = poly[np.linalg.norm(poly-dcenter, axis=1) > dradius + 1e-8]
    if len(outside):
        ax.scatter(outside[:,0], outside[:,1], s=34, facecolors="none", edgecolors=RED, lw=1.0, zorder=7)
    xmin = min(poly[:,0].min(), center[0]-radius, dcenter[0]-dradius)
    xmax = max(poly[:,0].max(), center[0]+radius, dcenter[0]+dradius)
    ymin = min(poly[:,1].min(), center[1]-radius, dcenter[1]-dradius)
    ymax = max(poly[:,1].max(), center[1]+radius, dcenter[1]+dradius)
    pad = max(xmax-xmin, ymax-ymin) * .07
    ax.set(xlim=(xmin-pad, xmax+pad), ylim=(ymin-pad, ymax+pad))
    ax.set_aspect("equal"); polish_axes(ax, grid=False)
    ax.set_xlabel(subtitle, labelpad=5)
    ax.text(.98, .98, f"$2r^*/D={2*radius/diameter:.4f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=.22", facecolor="white", edgecolor=GRAY, alpha=.92))
    return center, radius, a, b, diameter


def theory_figure(records, diagnostics):
    rec = records["number_014_5"]
    region = calculate_feasible_region(np.asarray(rec["observation_points"]), np.asarray(rec["angles"]),
                                       1.0, config=GridConfig(), return_details=True)
    actual = region.inner_hull.copy()
    # Normalize the actual localization region to a comparable visual scale.
    actual = (actual - actual.mean(0)) / maximum_diameter(actual) * 4
    segment_like = np.array([[-2,0],[-1.1,-.35],[1.1,-.35],[2,0],[1.1,.35],[-1.1,.35]])
    equilateral = np.array([[-2, -2/math.sqrt(3)], [2, -2/math.sqrt(3)], [0, 4/math.sqrt(3)]])
    cases = [segment_like, actual, equilateral]
    subtitles = ["二点支撑：直径圆可覆盖", "实际定位区域：三点支撑", "等边三角形：荣格上界等号"]
    fig, axes = plt.subplots(1, 3, figsize=(7.01, 3.05))
    rows = {}
    for i, (ax, poly, subtitle) in enumerate(zip(axes, cases, subtitles)):
        result = draw_cover_case(ax, poly, subtitle, actual=(i == 1))
        add_panel_label(ax, f"({chr(97+i)})")
        center, radius, a, b, diameter = result
        theta = np.linspace(0, 2*np.pi, 361)
        prefix = f"case{i+1}"
        rows.update({
            f"{prefix}_poly_x": closed(poly)[:,0], f"{prefix}_poly_y": closed(poly)[:,1],
            f"{prefix}_mec_x": center[0]+radius*np.cos(theta),
            f"{prefix}_mec_y": center[1]+radius*np.sin(theta),
            f"{prefix}_diam_x": np.array([a[0], b[0]]),
            f"{prefix}_diam_y": np.array([a[1], b[1]]),
        })
        ax.set_ylabel("$y$（归一化坐标）" if i == 0 else "")
    note = ("黑实线为实现直径的最远点对，灰虚线为半径 $D/2$ 的直径圆，紫点划线为最小覆盖圆。"
            r" 平面荣格界给出 $1\leq 2r^*/D\leq 2/\sqrt{3}\approx 1.1547$。")
    finish(fig, "fig03_jung_covering", "图3  区域直径与圆覆盖并不等价：二点支撑、三点支撑与荣格极端情形",
           note, diagnostics, top=.965, bottom=.17)
    pd.DataFrame({k: pd.Series(v) for k, v in rows.items()}).to_csv(
        DATA / "fig03_theory.csv", index=False, encoding="utf-8-sig")


def validation_figure(records, diagnostics):
    d = pd.read_csv(SUMMARY_PATH)
    truth = {key: np.asarray(value["true_position"], float) for key, value in records.items()}
    d["mec_error"] = [np.linalg.norm(np.array([r.cover_center_x, r.cover_center_y]) - truth[r.experiment_id])
                      for _, r in d.iterrows()]
    d["rho_mec"] = d.mec_error / d.cover_radius
    order = ["number", "error", "angle"]
    labels = ["检测数量", "误差界", "交会角"]
    grouped = [d[d.experiment_type == key] for key in order]
    rates = [100 * g.truth_feasible.mean() for g in grouped]

    fig, axes = plt.subplots(1, 2, figsize=(7.01, 3.65))
    ax = axes[0]
    bars = ax.bar(np.arange(3), rates, width=.58, color=[BLUE, GREEN, ORANGE], edgecolor="none")
    ax.set(xticks=np.arange(3), xticklabels=labels, ylim=(96, 100.55), ylabel="真实位置包含率 (%)")
    for bar, g, rate in zip(bars, grouped, rates):
        ax.text(bar.get_x()+bar.get_width()/2, rate+.08, f"{rate:.1f}%\n$n={len(g)}$",
                ha="center", va="bottom", fontsize=7.2)
    ax.axhline(100, color=DARK_GRAY, lw=.8, ls="--")
    polish_axes(ax); add_panel_label(ax, "(a) 三类实验全部满足真值包含")

    ax = axes[1]
    dn = d[d.experiment_type == "number"]
    levels = [2,3,4,5,6]
    values = [dn.loc[dn.number_of_points == n, "rho_mec"].to_numpy() for n in levels]
    bp = ax.boxplot(values, positions=np.arange(len(levels)), widths=.55, patch_artist=True,
                    showfliers=True, medianprops=dict(color=DARK_GRAY, lw=1.15),
                    whiskerprops=dict(color=DARK_GRAY, lw=.8), capprops=dict(color=DARK_GRAY, lw=.8),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=GRAY, markeredgecolor="none"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=BLUE, alpha=.22, edgecolor=BLUE, linewidth=1.0)
    rng = np.random.default_rng(20260912)
    for i, vals in enumerate(values):
        ax.scatter(i+rng.uniform(-.11,.11,len(vals)), vals, s=8, color=BLUE, alpha=.42, edgecolors="none")
    ax.axhline(1, color=RED, lw=1.1, ls="--", label=r"理论上界 $\rho=1$")
    ax.set(xticks=np.arange(len(levels)), xticklabels=levels, ylim=(0,1.08),
           xlabel="检测点数量（个）", ylabel=r"$\rho=\Vert c^*-S\Vert_2/r^*$")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.055), fontsize=6.8); polish_axes(ax)
    add_panel_label(ax, "(b) 最小覆盖圆中心的无量纲误差")

    note = ("共450组受控仿真：每类150组，全部满足原始±1°角域约束。"
            rf" 所有算例均有 $\rho\leq 1$（最大值 {d.rho_mec.max():.4f}），直接验证外包最小圆的安全覆盖性。")
    finish(fig, "fig04_monte_carlo_validation", "图4  450组随机算例同时验证真值包含与最小覆盖圆安全性",
           note, diagnostics, top=.95, bottom=.13)
    d.to_csv(DATA / "fig04_validation_cases.csv", index=False, encoding="utf-8-sig")


def write_captions(ratio):
    text = f"""# 问题一证明型图件使用说明

## 图1 有界测向误差下定位区域的构造、包络验证与覆盖判据
图1由观测角域、公共交集、自适应网格内外包和圆覆盖判据四部分构成。算例的 $2r^*/D={ratio:.4f}>1$，因此以区域直径为直径的圆不能覆盖整个定位区域；应使用独立计算的最小覆盖圆。

## 图2 网格加密使三项几何指标上下界同步收敛
随着终止网格边长减小，面积、直径与最小覆盖圆半径的内外包界宽同步缩小，说明连续区域结论不是由某一固定网格密度偶然产生。

## 图3 区域直径与圆覆盖并不等价
二点支撑时 $2r^*/D=1$；三点支撑时该比值严格大于1；等边三角形达到平面荣格界 $2/\sqrt{{3}}\approx1.1547$。

## 图4 450组随机算例的联合验证
全部450组真实位置均落入其角域交集；对最小覆盖圆中心定义 $\rho=\|c^*-S\|_2/r^*$，全部算例满足 $\rho\le1$，验证了安全覆盖结论。

## 排版建议
正文优先放图1、图2、图4；图3紧接荣格定理。若版面受限，可将图3移至附录，但不要删除图1右下角的直径圆与最小覆盖圆对比。
"""
    (OUT / "图题与正文解读.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    fonts = set_cumcm_style()
    # CUMCM_Style reports both Times New Roman and SimSun.  Chinese-first font
    # selection prevents vector backends from replacing CJK glyphs by squares;
    # equations and numerals continue to use the configured STIX/Times math font.
    plt.rcParams["font.family"] = [fonts["chinese"], fonts["english"]]
    diagnostics = {"fonts": fonts, "source_pdf": "document_修订.pdf", "model": "问题一"}
    records = records_by_id()
    _, _, ratio = main_geometry_figure(records, diagnostics)
    convergence_figure(diagnostics)
    theory_figure(records, diagnostics)
    validation_figure(records, diagnostics)
    write_captions(ratio)
    (OUT / "figure_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "ratio": ratio, "figures": list(diagnostics)[3:]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
