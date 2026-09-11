"""既有 CUMCM 绘图规范的适配层，不改动共享绘图库。"""
import json
import logging
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon
from config import WORKSPACE

# 既有库使用平铺模块而非 __init__.py 对外导出，遵循其 README 的导入方式。
sys.path.insert(0, str(WORKSPACE / "CUMCM_Visualization"))
from CUMCM_Style import set_cumcm_style, polish_axes, BLUE, DARK_BLUE, RED, GRAY
from CUMCM_Export import save_cumcm
from CUMCM_Diagnostics import check_figure, check_output
from experiment.report import group_rows, bootstrap_mean_ci
from model.feasible_region import calculate_feasible_region
from model.metrics import calculate_metrics


def finish(fig, stem, title, note, folder, diagnostics):
    """图号标题放在坐标轴外，统一字号、脚注与矢量/600dpi导出，记录诊断。"""
    fig.suptitle(title, fontsize=10, y=.99)
    fig.text(.5, .018, note, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.86, wspace=.33)
    saved = save_cumcm(fig, folder / stem, dpi=600)
    checks = check_figure(fig) + check_output(folder / stem)
    diagnostics[stem] = checks
    logging.info("图件导出：%s", ", ".join(str(p) for p in saved))
    plt.close(fig)


def draw_wedges(ax, observers, angles, error, extent=2200):
    """按原始误差界绘制扇形及中心射线；展示长度仅为绘图截断。"""
    for index, (point, angle) in enumerate(zip(observers, angles)):
        color = BLUE if index % 2 == 0 else DARK_BLUE
        ax.add_patch(Wedge(point, extent, angle-error, angle+error,
                           facecolor=color, edgecolor=color, alpha=.14, linewidth=.6))
        direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        end = point + extent * direction
        ax.plot([point[0], end[0]], [point[1], end[1]], color=color, linewidth=.8, linestyle="--")
        ax.plot(*point, marker="s", color=color, linestyle="none", markersize=4,
                label="观测站" if index == 0 else None)
        ax.annotate(f"$P_{index+1}$", point, xytext=(5, 5), textcoords="offset points", fontsize=8)


def plot_geometry(record, config, folder, diagnostics):
    """图1原比例扇形及横向放大视窗；图2全局测向与可行域局部放大。"""
    points = np.array(record["observation_points"])
    angles = np.array(record["angles"])
    target = np.array(record["true_position"])
    fig, axes = plt.subplots(1, 2, figsize=(7.01, 4.1))
    for ax in axes:
        draw_wedges(ax, points[:1], angles[:1], 1, 1400)
        ax.scatter(*target, marker="*", color=RED, s=65, zorder=5, label="真实干扰源")
        ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)")
        ax.set_aspect("equal")
        polish_axes(ax, grid=False)
    low = np.minimum(points[0], target) - 150
    high = np.maximum(points[0], target) + 150
    span = max(high - low)
    midpoint = (low + high) / 2
    axes[0].set(xlim=(midpoint[0]-span/2, midpoint[0]+span/2), ylim=(midpoint[1]-span/2, midpoint[1]+span/2))
    axes[0].legend(loc="best")
    axes[1].set(xlim=(target[0]-35, target[0]+35), ylim=(target[1]-35, target[1]+35))
    axes[0].text(.02, .97, "(a) 单站测向几何", transform=axes[0].transAxes, va="top")
    axes[1].text(.02, .97, "(b) 真值附近放大", transform=axes[1].transAxes, va="top")
    finish(fig, "fig01_single_wedge", "图1  单站测向误差约束区域",
           "扇形半角为真实的1°，不人为夸大；虚线为测得方向，扇形长度仅用于显示。", folder, diagnostics)

    region = calculate_feasible_region(points, angles, 1, config=config, return_details=True)
    metric = calculate_metrics(region, target)
    centroid = np.array([metric["centroid_x"], metric["centroid_y"]])
    fig, axes = plt.subplots(1, 2, figsize=(7.01, 4.1))
    draw_wedges(axes[0], points, angles, 1, 1700)
    for ax in axes:
        ax.scatter(*target, marker="*", color=RED, s=65, zorder=6, label="真实位置")
        ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)")
        ax.set_aspect("equal")
        polish_axes(ax, grid=False)
    axes[0].set(xlim=(target[0]-1200, target[0]+1200), ylim=(target[1]-1200, target[1]+1200))
    axes[0].legend(loc="lower left")
    outer, inner = region.outer_hull, region.inner_hull
    axes[1].add_patch(Polygon(outer, facecolor="none", edgecolor=GRAY, linewidth=1, label="网格外包"))
    axes[1].add_patch(Polygon(inner, facecolor=BLUE, alpha=.3, edgecolor=DARK_BLUE, label="可行域内包"))
    axes[1].scatter(*centroid, color=DARK_BLUE, marker="+", s=50, zorder=7, label="区域质心估计")
    mid = (outer.min(axis=0) + outer.max(axis=0)) / 2
    extent = max(np.ptp(outer, axis=0).max() * .75, 3)
    axes[1].set(xlim=(mid[0]-extent, mid[0]+extent), ylim=(mid[1]-extent, mid[1]+extent))
    axes[1].legend(loc="lower left", fontsize=6.5)
    axes[0].text(.02, .97, "(a) 多站几何", transform=axes[0].transAxes, va="top")
    axes[1].text(.02, .97, "(b) 角域交集放大", transform=axes[1].transAxes, va="top")
    finish(fig, "fig02_feasible_region", "图2  多站测向约束下干扰源可行区域",
           f"三站测向误差界±1°；面积包络 [{metric['area_lower']:.2f}, {metric['area_upper']:.2f}] m²。", folder, diagnostics)
    return region, metric


def plot_experiment(rows, kind, folder, diagnostics):
    """画均值和场景自举区间；灰色细线显示配对重复，避免只展示平滑均值。"""
    specs = {"number": (3, "检测数量对定位区域面积的影响", "检测点数量（个）", "area", "区域面积 (m²)"),
             "error": (4, "测向误差界的敏感性分析", "角度误差界（°）", "area", "区域面积 (m²)"),
             "angle": (5, "交会角对定位误差的影响", "交会角（°）", "localization_error", "区域质心定位误差 (m)")}
    number, title, xlabel, metric, ylabel = specs[kind]
    groups = group_rows(rows, kind)
    x = np.array(list(groups))
    values = np.array([[r[metric] for r in subset] for subset in groups.values()])
    means = values.mean(axis=1)
    ci = np.array([bootstrap_mean_ci(v) for v in values])
    fig, ax = plt.subplots(figsize=(7.01, 3.62))
    for replicate in values.T:
        ax.plot(x, replicate, color=GRAY, alpha=.20, linewidth=.5)
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=BLUE, alpha=.20, label="均值的95%自举区间")
    ax.plot(x, means, marker="o", color=BLUE, label="重复实验均值")
    ax.set(xlabel=xlabel, ylabel=ylabel, xticks=x)
    if kind == "angle":
        best = int(np.argmin(means))
        ax.scatter(x[best], means[best], color=RED, s=27, zorder=5, label="样本均值最小")
    ax.legend(loc="best")
    polish_axes(ax)
    note = f"每组 {values.shape[1]} 次配对重复；灰线为单次实验；阴影不表示位置置信域。"
    finish(fig, f"fig{number:02d}_{kind}", f"图{number}  {title}", note, folder, diagnostics)


def plot_all(rows, records, config, folder, output):
    """统一设置 CUMCM 风格，生成全部五图及原始示例可行点。"""
    fonts = set_cumcm_style()
    diagnostics = {"fonts": fonts}
    representative = next(r for r in records if r["experiment_id"] == "number_000_3")
    region, metric = plot_geometry(representative, config, folder, diagnostics)
    np.savez_compressed(output / "example_region.npz", valid_points=region.valid_points,
                        inner_hull=region.inner_hull, outer_hull=region.outer_hull, cells=region.cells)
    (output / "example_metrics.json").write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind in ("number", "error", "angle"):
        plot_experiment(rows, kind, folder, diagnostics)
    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
