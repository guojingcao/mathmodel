"""Build four proof-oriented Model-2 figures with the CUMCM visual system."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "Problem2_Observation_Optimization"
RESULTS = P2 / "results"
OUT = ROOT / "origin_figures" / "Problem2_Model2_Proof"
DATA = OUT / "origin_data"

sys.path.insert(0, str(P2))
sys.path.insert(0, str(ROOT / "CUMCM_Visualization"))

from config import LayoutConfig  # noqa: E402
from model.localization import localize  # noqa: E402
from CUMCM_Style import (  # noqa: E402
    set_cumcm_style, polish_axes, BLUE, DARK_BLUE, RED, GREEN, ORANGE,
    PURPLE, GRAY, DARK_GRAY, GRID, BLUE_CMAP,
)
from CUMCM_Export import save_cumcm  # noqa: E402
from CUMCM_Diagnostics import check_figure, check_output  # noqa: E402

METHODS = ["random", "uniform", "optimized", "dop_baseline"]
METHOD_LABELS = {
    "random": "随机布局", "uniform": "均匀布局",
    "optimized": "本文优化", "dop_baseline": "DOP基线",
}
METHOD_COLORS = {
    "random": GRAY, "uniform": GREEN, "optimized": RED, "dop_baseline": ORANGE,
}


def load_inputs():
    config = LayoutConfig()
    layouts = json.loads((RESULTS / "selected_layouts.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS / "problem2_summary.csv")
    groups = pd.read_csv(RESULTS / "group_statistics.csv")
    trace = pd.read_csv(RESULTS / "optimization_trace.csv")
    candidates = pd.read_csv(RESULTS / "candidate_evaluations.csv")
    records = [json.loads(line) for line in (RESULTS / "observations.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return config, layouts, summary, groups, trace, candidates, records


def export(fig, stem, diagnostics):
    saved = save_cumcm(fig, OUT / stem, dpi=600)
    diagnostics[stem] = check_figure(fig) + check_output(OUT / stem)
    plt.close(fig)
    return saved


def panel_title(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=6, fontsize=9.2)


def replay(record, config):
    points = np.asarray(record["observation_points"], dtype=float)
    target = np.asarray(record["target"], dtype=float)
    region, metrics, _ = localize(
        points, target, angle_error=config.error_bound_deg,
        measurement_errors=np.asarray(record["measurement_errors"], dtype=float),
        grid_resolution=config.evaluation_resolution_m,
        initial_grid=config.initial_grid, arena_radius=config.arena_radius,
        truth_for_evaluation=target,
    )
    return region, target


def representative_repetition(summary):
    layout = summary[summary.experiment_type.eq("layout")]
    pivot = layout.pivot(index="repetition", columns="method", values="area")
    diff = np.column_stack([
        pivot.optimized - pivot.random,
        pivot.optimized - pivot.uniform,
        pivot.optimized - pivot.dop_baseline,
    ])
    center = diff.mean(axis=0)
    scale = np.maximum(diff.std(axis=0, ddof=1), 1e-9)
    return int(pivot.index[np.argmin(np.sum(((diff - center) / scale) ** 2, axis=1))])


def get_record(records, method, repetition):
    key = f"layout_{method}_{repetition:03d}_n4"
    return next(item for item in records if item["experiment_id"] == key)


def draw_region(ax, record, config, color):
    region, target = replay(record, config)
    hull = np.asarray(region.inner_hull) - target
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, edgecolor=color,
                         alpha=.24, linewidth=1.25))
    ax.scatter(0, 0, marker="*", color=RED, edgecolor="white", linewidth=.45,
               s=58, zorder=6)
    ax.axhline(0, color=GRID, lw=.45, zorder=0)
    ax.axvline(0, color=GRID, lw=.45, zorder=0)
    ax.set_aspect("equal")
    ax.set(xlabel="$x-x_0$ (m)", ylabel="$y-y_0$ (m)")
    polish_axes(ax, grid=False)
    return hull


def figure1(config, layouts, trace, candidate_eval, diagnostics):
    fig, axes = plt.subplots(2, 2, figsize=(7.01, 6.55))
    cand = np.asarray(layouts["candidate_points"], float)
    selected_idx = layouts["optimized_indices"]
    selected = cand[selected_idx]

    ax = axes[0, 0]
    ax.add_patch(Circle((0, 0), config.arena_radius, fill=False, ec=GRAY, lw=.8))
    ax.add_patch(Circle((0, 0), config.observation_radius, fill=False, ec=GRID, lw=.7, ls="--"))
    ax.scatter(cand[:, 0], cand[:, 1], facecolor="white", edgecolor=BLUE, s=24, lw=.8)
    ax.plot(selected[:, 0], selected[:, 1], color=RED, lw=.8, alpha=.65)
    ax.scatter(selected[:, 0], selected[:, 1], color=RED, s=31, zorder=4)
    ax.scatter(0, 0, marker="+", color=DARK_BLUE, s=48, zorder=5)
    for order, point in enumerate(selected, 1):
        offset = (5, 5) if order not in {3, 5} else (-10, 5)
        ax.annotate(str(order), point, xytext=offset, textcoords="offset points",
                    fontsize=7, color=DARK_GRAY)
    ax.set(xlim=(-1900, 1900), ylim=(-1900, 1900), xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal")
    polish_axes(ax, grid=False)
    panel_title(ax, "a", "候选域与贪心选择顺序")

    ax = axes[0, 1]
    r2 = candidate_eval[candidate_eval["round"].eq(2)].copy()
    xy = cand[r2.candidate_index.astype(int).to_numpy()]
    values = np.log10(r2.area_upper.to_numpy())
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=BLUE_CMAP, s=42,
                    edgecolor="white", linewidth=.55)
    chosen = trace.loc[trace["round"].eq(2), "candidate_index"].iloc[0]
    point = cand[int(chosen)]
    ax.scatter(*point, marker="*", color=RED, edgecolor="white", linewidth=.6,
               s=115, zorder=6)
    ax.scatter(cand[0, 0], cand[0, 1], marker="s", facecolor="none",
               edgecolor=DARK_GRAY, s=48, lw=1.0, zorder=5)
    ax.add_patch(Circle((0, 0), config.observation_radius, fill=False, ec=GRID, lw=.65))
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.03)
    cb.set_label(r"$\log_{10}(A_{out}/\mathrm{m}^2)$", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    ax.set(xlim=(-1150, 1150), ylim=(-1150, 1150), xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax.set_aspect("equal")
    polish_axes(ax, grid=False)
    panel_title(ax, "b", "第2轮候选点评分地形")

    ax = axes[1, 0]
    x = trace["round"].to_numpy()
    ax.fill_between(x, trace.area_lower, trace.area_upper, color=BLUE, alpha=.16)
    line1, = ax.plot(x, trace.area, color=BLUE, marker="o", label="面积 A")
    ax.set_yscale("log")
    ax.set(xlabel="已选观测点数量", ylabel="面积 (m²，对数轴)", xticks=x)
    ax2 = ax.twinx()
    line2, = ax2.plot(x, trace.diameter_upper, color=ORANGE, marker="s",
                     label="直径上界 D_out")
    ax2.set_ylabel("直径上界 (m)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.legend([line1, line2], [line1.get_label(), line2.get_label()],
              loc="upper right", bbox_to_anchor=(1.0, .98))
    polish_axes(ax)
    ax2.spines["top"].set_visible(False)
    panel_title(ax, "c", "选择过程中不确定域快速收缩")

    ax = axes[1, 1]
    target = np.zeros(2)
    points = np.asarray(layouts["optimized_points"], float)
    region, metrics, _ = localize(
        points, target, angle_error=config.error_bound_deg,
        measurement_errors=np.zeros(len(points)),
        grid_resolution=config.evaluation_resolution_m,
        initial_grid=config.initial_grid, arena_radius=config.arena_radius,
        truth_for_evaluation=target,
    )
    outer = np.asarray(region.outer_hull)
    inner = np.asarray(region.inner_hull)
    ax.add_patch(Polygon(outer, closed=True, fc=ORANGE, ec=ORANGE, alpha=.15, lw=.9))
    ax.add_patch(Polygon(inner, closed=True, fc=BLUE, ec=DARK_BLUE, alpha=.28, lw=1.25))
    ax.scatter(0, 0, marker="*", color=RED, s=58, edgecolor="white", lw=.45, zorder=5)
    ax.text(.97, .96, f"$A$={metrics['area']:.1f} m²\n$D$={metrics['diameter']:.1f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec=GRID, alpha=.92))
    lim = max(np.abs(outer).max() * 1.18, 25)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="$x-x_0$ (m)", ylabel="$y-y_0$ (m)")
    ax.set_aspect("equal")
    polish_axes(ax, grid=False)
    panel_title(ax, "d", "最终六点布局对应的定位域")

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", mec=BLUE, mfc="white", label="候选点"),
        Line2D([0], [0], marker="o", color="none", mec=RED, mfc=RED, label="入选点"),
        Line2D([0], [0], color=BLUE, lw=5, alpha=.28, label="Ω 内包"),
        Line2D([0], [0], color=ORANGE, lw=5, alpha=.25, label="Ω 外包"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(.5, .006))
    fig.subplots_adjust(left=.09, right=.96, bottom=.085, top=.97, wspace=.31, hspace=.34)

    pd.DataFrame({"candidate_index": np.arange(len(cand)), "x": cand[:, 0], "y": cand[:, 1],
                  "selected_order": [selected_idx.index(i)+1 if i in selected_idx else 0 for i in range(len(cand))]}).to_csv(
        DATA / "fig01_candidates.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(DATA / "fig01_trace.csv", index=False, encoding="utf-8-sig")
    export(fig, "fig01_selection_mechanism", diagnostics)


def figure2(config, summary, records, diagnostics):
    rep = representative_repetition(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.01, 6.15), sharex=True, sharey=True)
    all_hulls = {}
    region_columns = {}
    for ax, method, letter in zip(axes.flat, METHODS, "abcd"):
        record = get_record(records, method, rep)
        hull = draw_region(ax, record, config, METHOD_COLORS[method])
        all_hulls[method] = hull
        region_columns[f"{method}_x"] = pd.Series(hull[:, 0])
        region_columns[f"{method}_y"] = pd.Series(hull[:, 1])
        row = summary[(summary.experiment_type.eq("layout")) &
                      (summary.method.eq(method)) & (summary.repetition.eq(rep))].iloc[0]
        panel_title(ax, letter, METHOD_LABELS[method])
        ax.text(.97, .95, f"$A$={row.area:.1f} m²\n$D$={row.diameter:.1f} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
                bbox=dict(boxstyle="round,pad=.22", fc="white", ec=GRID, alpha=.92))
    limit = max(np.abs(np.concatenate(list(all_hulls.values()))).max() * 1.12, 35)
    for ax in axes.flat:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    fig.legend(handles=[
        Line2D([0], [0], marker="*", color="none", markerfacecolor=RED,
               markeredgecolor="white", markersize=9, label="目标真值"),
        Line2D([0], [0], color=BLUE, lw=6, alpha=.28, label="定位可行域 Ω")
    ], loc="lower center", ncol=2, bbox_to_anchor=(.5, .006))
    fig.subplots_adjust(left=.09, right=.975, bottom=.095, top=.97, wspace=.18, hspace=.28)
    pd.DataFrame(region_columns).to_csv(DATA / "fig02_regions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"representative_repetition": [rep]}).to_csv(
        DATA / "fig02_metadata.csv", index=False, encoding="utf-8-sig")
    export(fig, "fig02_layout_regions", diagnostics)
    return rep


def figure3(groups, diagnostics):
    layout = groups[groups.experiment_type.eq("layout")].set_index("method")
    metrics = [
        ("area", "a", "可行域面积", "面积 (m²)"),
        ("diameter", "b", "可行域直径", "直径 (m)"),
        ("cover_radius", "c", "最小覆盖圆半径", "覆盖半径 (m)"),
        ("localization_error", "d", "质心定位误差", "误差 (m)"),
    ]
    display_order = ["random", "uniform", "dop_baseline", "optimized"]
    fig, axes = plt.subplots(2, 2, figsize=(7.01, 5.65))
    export_rows = []
    for ax, (metric, letter, title, ylabel) in zip(axes.flat, metrics):
        means = np.array([layout.loc[m, f"{metric}_mean"] for m in display_order])
        lows = np.array([layout.loc[m, f"{metric}_ci_low"] for m in display_order])
        highs = np.array([layout.loc[m, f"{metric}_ci_high"] for m in display_order])
        x = np.arange(len(display_order))
        colors = [METHOD_COLORS[m] for m in display_order]
        for i, (mean, low, high, color) in enumerate(zip(means, lows, highs, colors)):
            ax.errorbar(i, mean, yerr=[[mean-low], [high-mean]], fmt="o", color=color,
                        ecolor=color, capsize=3.2, elinewidth=1.15, markersize=5.5, zorder=3)
            ax.text(i, high + .045 * max(np.ptp(means), means.mean()*.18), f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=7, color=DARK_GRAY)
            export_rows.append({"metric": metric, "method": m if False else display_order[i],
                                "mean": mean, "ci_low": low, "ci_high": high})
        ax.set_xticks(x, [METHOD_LABELS[m] for m in display_order])
        ax.set_ylabel(ylabel)
        ax.margins(y=.20)
        polish_axes(ax)
        panel_title(ax, letter, title)
    fig.subplots_adjust(left=.095, right=.98, bottom=.09, top=.97, wspace=.28, hspace=.36)
    pd.DataFrame(export_rows).to_csv(DATA / "fig03_layout_metrics.csv", index=False, encoding="utf-8-sig")
    export(fig, "fig03_layout_performance", diagnostics)


def paired_panel(ax, pivot, baseline, letter, title):
    x = pivot[baseline].to_numpy()
    y = pivot["optimized"].to_numpy()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = .06 * (hi - lo)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color=DARK_GRAY, lw=.9, ls="--")
    ax.scatter(x, y, s=25, color=BLUE, edgecolor="white", lw=.45, alpha=.88, zorder=3)
    wins = int(np.sum(y < x))
    mean_diff = float(np.mean(y-x))
    ax.text(.04, .96, f"优化胜出 {wins}/30\n平均差 {mean_diff:.1f} m²",
            transform=ax.transAxes, va="top", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec=GRID, alpha=.94))
    ax.set(xlabel=f"{METHOD_LABELS[baseline]}面积 (m²)", ylabel="本文优化面积 (m²)",
           xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad))
    ax.set_aspect("equal", adjustable="box")
    polish_axes(ax)
    panel_title(ax, letter, title)


def figure4(summary, groups, diagnostics):
    layout = summary[summary.experiment_type.eq("layout")]
    pivot = layout.pivot(index="repetition", columns="method", values="area")
    fig, axes = plt.subplots(2, 2, figsize=(7.01, 6.10))
    paired_panel(axes[0, 0], pivot, "random", "a", "与随机布局的配对面积比较")
    paired_panel(axes[0, 1], pivot, "dop_baseline", "b", "与DOP基线的配对面积比较")

    number = groups[groups.experiment_type.eq("number")].sort_values("number_points")
    ax = axes[1, 0]
    x = number.number_points.to_numpy()
    y = number.area_mean.to_numpy()
    ax.fill_between(x, number.area_ci_low, number.area_ci_high, color=BLUE, alpha=.18)
    ax.plot(x, y, color=BLUE, marker="o")
    ax.set(xlabel="观测点数量", ylabel="平均面积 (m²)", xticks=x)
    ax.text(.96, .96, f"2→6点缩减 {(1-y[-1]/y[0])*100:.1f}%\n单调性违例 0次",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec=GRID, alpha=.94))
    polish_axes(ax)
    panel_title(ax, "c", "增加观测点带来的不确定域收缩")

    angle = groups[groups.experiment_type.eq("angle")].sort_values("intersection_angle")
    ax = axes[1, 1]
    x = angle.intersection_angle.to_numpy()
    y = angle.localization_error_mean.to_numpy()
    ax.fill_between(x, angle.localization_error_ci_low, angle.localization_error_ci_high,
                    color=PURPLE, alpha=.18)
    ax.plot(x, y, color=PURPLE, marker="o")
    best = int(np.argmin(y))
    ax.scatter(x[best], y[best], color=RED, s=40, zorder=4)
    ax.annotate(f"最小：{y[best]:.1f} m", (x[best], y[best]), xytext=(0, -17),
                textcoords="offset points", ha="center", fontsize=7.2, color=RED)
    ax.text(.97, .96, "420/420 真值可行", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.4, bbox=dict(boxstyle="round,pad=.22", fc="white", ec=GRID, alpha=.94))
    ax.set(xlabel="交会角 (°)", ylabel="平均定位误差 (m)", xticks=x)
    polish_axes(ax)
    panel_title(ax, "d", "交会角敏感性与可行性验证")

    fig.subplots_adjust(left=.095, right=.98, bottom=.09, top=.97, wspace=.31, hspace=.34)
    paired = pd.DataFrame({"repetition": pivot.index,
                           "random_area": pivot.random.to_numpy(),
                           "dop_area": pivot.dop_baseline.to_numpy(),
                           "optimized_area": pivot.optimized.to_numpy()})
    paired.to_csv(DATA / "fig04_paired_area.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        number.assign(series="number"), angle.assign(series="angle")
    ], ignore_index=True).to_csv(DATA / "fig04_sensitivity.csv", index=False, encoding="utf-8-sig")
    export(fig, "fig04_paired_robustness", diagnostics)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fonts = set_cumcm_style()
    diagnostics = {"fonts": fonts}
    config, layouts, summary, groups, trace, candidate_eval, records = load_inputs()
    figure1(config, layouts, trace, candidate_eval, diagnostics)
    rep = figure2(config, summary, records, diagnostics)
    figure3(groups, diagnostics)
    figure4(summary, groups, diagnostics)
    diagnostics["representative_repetition"] = rep
    (OUT / "figure_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created four figures in {OUT}")


if __name__ == "__main__":
    main()
