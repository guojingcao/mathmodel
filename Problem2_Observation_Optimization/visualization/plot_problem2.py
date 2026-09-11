"""用既有 CUMCM_Visualization 生成问题二五张图。"""
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from config import WORKSPACE
from experiment.report import bootstrap_ci
from model.localization import localize

sys.path.insert(0, str(WORKSPACE / "CUMCM_Visualization"))
from CUMCM_Style import set_cumcm_style, polish_axes, BLUE, DARK_BLUE, RED, GRAY  # noqa: E402
from CUMCM_Export import save_cumcm  # noqa: E402
from CUMCM_Diagnostics import check_figure, check_output  # noqa: E402


def finish_figure(fig, stem, title, note, folder, diagnostics):
    """添加轴外图题和实验说明，统一导出三种格式并记录质量诊断。"""
    fig.suptitle(title, fontsize=10, y=.99)
    fig.text(.5, .018, note, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.86, wspace=.30)
    saved = save_cumcm(fig, folder / stem, dpi=600)
    diagnostics[stem] = check_figure(fig) + check_output(folder / stem)
    logging.info("图件导出：%s", ", ".join(str(path) for path in saved))
    plt.close(fig)


def curve_data(rows, experiment, xfield, metric):
    """将明细整理为横轴、重复矩阵、均值和均值自举区间。"""
    subset = [row for row in rows if row["experiment_type"] == experiment]
    x = sorted({row[xfield] for row in subset})
    values = np.array([[row[metric] for row in subset if row[xfield] == value] for value in x])
    means = values.mean(axis=1)
    intervals = np.array([bootstrap_ci(value) for value in values])
    return np.asarray(x, float), values, means, intervals


def draw_region(ax, record, config, title):
    """重放测量并调用问题一，在目标局部坐标中绘制 Ω 内包。"""
    points = np.asarray(record["observation_points"], dtype=float)
    target = np.asarray(record["target"], dtype=float)
    errors = np.asarray(record["measurement_errors"], dtype=float)
    region, _, _ = localize(points, target, angle_error=config.error_bound_deg,
        measurement_errors=errors, grid_resolution=config.evaluation_resolution_m,
        initial_grid=config.initial_grid, arena_radius=config.arena_radius,
        truth_for_evaluation=target)
    local_hull = region.inner_hull - target
    ax.add_patch(Polygon(local_hull, facecolor=BLUE, edgecolor=DARK_BLUE,
                         alpha=.30, linewidth=1))
    ax.scatter(0, 0, marker="*", color=RED, s=50, zorder=5)
    ax.text(.02, .97, title, transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set(xlabel="$x-x_0$ (m)", ylabel="$y-y_0$ (m)")
    ax.set_aspect("equal")
    polish_axes(ax, grid=False)


def plot_all(rows, records, candidates, optimized, optimizer_trace, config, folder, output):
    """绘制布局、区域比较、点数、交会角及贪心过程五张图。"""
    folder.mkdir(parents=True, exist_ok=True)
    diagnostics = {"fonts": set_cumcm_style()}

    fig, ax = plt.subplots(figsize=(7.01, 3.62))
    ax.add_patch(Circle((0, 0), config.arena_radius, facecolor="none",
                        edgecolor=GRAY, linewidth=.8, label="搜索区域边界"))
    ax.scatter(candidates[:, 0], candidates[:, 1], facecolors="white",
               edgecolors=BLUE, s=24, linewidth=.8, label="候选检测点")
    ax.plot(optimized[:, 0], optimized[:, 1], color=RED, linewidth=.8, alpha=.65)
    ax.scatter(optimized[:, 0], optimized[:, 1], color=RED, s=30,
               zorder=5, label="最终选择点")
    ax.scatter(0, 0, marker="+", color=DARK_BLUE, s=50, label="目标参考中心")
    for order, point in enumerate(optimized, 1):
        ax.annotate(str(order), point, xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)", xlim=(-1900, 1900), ylim=(-1900, 1900))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", ncol=2)
    polish_axes(ax, grid=False)
    finish_figure(fig, "fig01_optimized_layout", "图1  优化后的观测点布局",
                  "24个候选点位于半径1000 m圆周；数字表示贪心选择顺序。",
                  folder, diagnostics)

    methods = [("random", "随机布局"), ("uniform", "均匀圆周布局"),
               ("optimized", "本文优化布局")]
    fig, axes = plt.subplots(1, 3, figsize=(7.01, 70/25.4))
    layout_rows = [row for row in rows if row["experiment_type"] == "layout"]
    area_by_key = {(row["method"], row["repetition"]): row["area"] for row in layout_rows}
    differences = np.array([[area_by_key[("optimized", repetition)]-area_by_key[("random", repetition)],
                             area_by_key[("optimized", repetition)]-area_by_key[("uniform", repetition)]]
                            for repetition in range(max(row["repetition"] for row in layout_rows)+1)])
    center = differences.mean(axis=0)
    scale = np.maximum(differences.std(axis=0, ddof=1), 1e-9)
    representative_repetition = int(np.argmin(np.sum(((differences-center)/scale)**2, axis=1)))
    representative = {method: next(record for record in records
                      if record["experiment_id"] ==
                      f"layout_{method}_{representative_repetition:03d}_n4")
                      for method, _ in methods}
    for ax, (method, label) in zip(axes, methods):
        draw_region(ax, representative[method], config, label)
    all_hulls = []
    for method, _ in methods:
        record = representative[method]
        points = np.asarray(record["observation_points"])
        target = np.asarray(record["target"])
        region, _, _ = localize(points, target, angle_error=config.error_bound_deg,
            measurement_errors=record["measurement_errors"],
            grid_resolution=config.evaluation_resolution_m, initial_grid=config.initial_grid,
            arena_radius=config.arena_radius)
        all_hulls.append(region.inner_hull-target)
    limit = max(np.abs(np.concatenate(all_hulls)).max()*1.08, 2)
    for ax in axes:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    finish_figure(fig, "fig02_layout_regions", "图2  不同观测布局的定位区域比较",
                  f"第{representative_repetition}次配对场景最接近总体面积差均值；蓝色为问题一Ω内包，红星为真值。",
                  folder, diagnostics)

    for figure_number, experiment, xfield, metric, title, xlabel, ylabel in [
        (3, "number", "number_points", "area", "检测点数量与定位区域面积",
         "检测点数量（个）", "可行区域面积 (m²)"),
        (4, "angle", "intersection_angle", "localization_error", "交会角与定位误差关系",
         "交会角（°）", "区域质心定位误差 (m)")]:
        x, values, means, intervals = curve_data(rows, experiment, xfield, metric)
        fig, ax = plt.subplots(figsize=(7.01, 3.62))
        for replicate in values.T:
            ax.plot(x, replicate, color=GRAY, alpha=.18, linewidth=.5)
        ax.fill_between(x, intervals[:, 0], intervals[:, 1], color=BLUE,
                        alpha=.20, label="均值的95%自举区间")
        ax.plot(x, means, color=BLUE, marker="o", label="重复实验均值")
        if experiment == "angle":
            best = int(np.argmin(means))
            ax.scatter(x[best], means[best], color=RED, s=30,
                       zorder=5, label="样本均值最小")
        ax.set(xlabel=xlabel, ylabel=ylabel, xticks=x)
        ax.legend(loc="best")
        polish_axes(ax)
        finish_figure(fig, f"fig{figure_number:02d}_{experiment}",
                      f"图{figure_number}  {title}",
                      f"每个水平{values.shape[1]}次配对重复；灰线为单次实验，阴影不是位置置信域。",
                      folder, diagnostics)

    trace = sorted(optimizer_trace, key=lambda row: row["round"])
    x = np.array([row["round"] for row in trace])
    area = np.array([row["area"] for row in trace])
    lower = np.array([row["area_lower"] for row in trace])
    upper = np.array([row["area_upper"] for row in trace])
    fig, ax = plt.subplots(figsize=(7.01, 3.62))
    ax.fill_between(x, lower, upper, color=BLUE, alpha=.2, label="问题一数值包络")
    ax.plot(x, area, color=BLUE, marker="o", label="Ω面积")
    ax.set(xlabel="已选择检测点数量（个）", ylabel="可行区域面积 (m²，对数轴)",
           xticks=x, yscale="log")
    ax.legend(loc="best")
    polish_axes(ax)
    finish_figure(fig, "fig05_greedy_convergence", "图5  观测点贪心选择过程",
                  "每轮枚举可达候选点，面积外包上界最小者入选；纵轴采用对数刻度。",
                  folder, diagnostics)

    (output / "figure_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
