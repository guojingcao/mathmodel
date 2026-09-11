"""A restrained, publication-oriented plotting toolkit for CUMCM papers.

The module has no dependency beyond NumPy and Matplotlib.  All public plotting
functions return ``(figure, axes)`` so that a manuscript author can add the
few annotations that are genuinely needed for a particular result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

MM = 1 / 25.4
FIGSIZE_SINGLE = (89 * MM, 62 * MM)
FIGSIZE_DOUBLE = (178 * MM, 92 * MM)
FIGSIZE_WIDE = (178 * MM, 70 * MM)

BLUE = "#5B7FA3"
DARK_BLUE = "#36566F"
RED = "#B6534B"
GREEN = "#668B74"
ORANGE = "#C08B52"
PURPLE = "#806A8A"
GRAY = "#9AA0A6"
DARK_GRAY = "#555555"
TEXT = "#333333"
GRID = "#D9D9D9"
ACCENT = RED
PALETTE = (BLUE, RED, GREEN, ORANGE, PURPLE, DARK_BLUE, GRAY)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "cumcm_blue", ["#F7FAFC", "#D5E2EC", "#93B2C8", DARK_BLUE]
)

_CHINESE_FONTS = ("SimSun", "STSong", "Microsoft YaHei", "SimHei", "Arial Unicode MS")
_ENGLISH_FONTS = ("Times New Roman", "Times", "DejaVu Serif")


def check_fonts() -> dict[str, str | bool]:
    """Return the font choices available in the current Matplotlib installation."""
    installed = {item.name for item in font_manager.fontManager.ttflist}
    chinese = next((name for name in _CHINESE_FONTS if name in installed), "DejaVu Sans")
    english = next((name for name in _ENGLISH_FONTS if name in installed), "DejaVu Serif")
    return {
        "english": english,
        "chinese": chinese,
        "times_new_roman_available": "Times New Roman" in installed,
        "simsun_available": "SimSun" in installed,
    }


def set_cumcm_style() -> dict[str, str | bool]:
    """Apply the CUMCM style to Matplotlib rcParams and return font diagnostics."""
    fonts = check_fonts()
    mpl.rcParams.update({
        "font.family": [fonts["english"], fonts["chinese"], "DejaVu Serif"],
        "font.serif": [fonts["english"], "Times New Roman", "DejaVu Serif"],
        "font.sans-serif": [fonts["chinese"], "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.6,
        "axes.labelcolor": TEXT, "text.color": TEXT,
        "axes.edgecolor": DARK_GRAY, "axes.linewidth": 0.8,
        "xtick.color": DARK_GRAY, "ytick.color": DARK_GRAY,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "lines.linewidth": 1.45, "lines.markersize": 4.2,
        "legend.frameon": False, "legend.handlelength": 2.0,
    })
    return fonts


def create_figure(figsize: tuple[float, float] = FIGSIZE_SINGLE,
                  *, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Create a white figure, or validate and reuse a supplied axes."""
    set_cumcm_style()
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize, layout="constrained")


def polish_axes(ax: plt.Axes, *, grid: bool = True) -> plt.Axes:
    """Apply the shared low-ink axis treatment to ``ax``."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", pad=2.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.9)
    return ax


def save_figure(fig: plt.Figure, filename: str | None, output_dir: str | Path = "figures",
                formats: Sequence[str] = ("pdf", "svg", "png"), dpi: int = 600) -> list[Path]:
    """Save PDF/SVG/PNG versions without letting one failed format stop others."""
    if not filename:
        return []
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    saved: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        if extension not in {"pdf", "svg", "png"}:
            raise ValueError("formats must contain only pdf, svg, or png")
        path = folder / f"{stem}.{extension}"
        try:
            fig.savefig(path, dpi=dpi if extension == "png" else None,
                        bbox_inches="tight", pad_inches=0.035)
            saved.append(path)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Warning: could not save {path}: {exc}")
    return saved


def _finish(fig: plt.Figure, ax: plt.Axes, filename: str | None, output_dir: str | Path) -> tuple[plt.Figure, plt.Axes]:
    if filename:
        save_figure(fig, filename, output_dir)
    return fig, ax


def _values(values: Iterable[float], name: str = "values") -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    return array


def cumcm_bar(categories: Sequence[str], values: Iterable[float], *, xlabel: str = "",
              ylabel: str = "", highlight: str | int | None = None,
              show_values: bool = False, mean_line: bool = False,
              figsize: tuple[float, float] = FIGSIZE_SINGLE, filename: str | None = None,
              output_dir: str | Path = "figures", ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw a restrained bar chart; highlight accepts ``max``, ``min`` or an index."""
    data = _values(values)
    if len(categories) != len(data):
        raise ValueError("categories and values must have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    colors = [BLUE] * len(data)
    if highlight is not None:
        index = int(np.argmax(data) if highlight == "max" else np.argmin(data) if highlight == "min" else highlight)
        if not 0 <= index < len(data):
            raise ValueError("highlight index is outside values")
        colors[index] = RED
    bars = ax.bar(np.arange(len(data)), data, width=0.62, color=colors, edgecolor="none")
    ax.set_xticks(np.arange(len(data)), categories)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if mean_line:
        ax.axhline(data.mean(), color=DARK_GRAY, linewidth=0.8, linestyle="--", label="Mean")
        ax.legend(loc="best")
    if show_values:
        spread = max(np.ptp(data), abs(data.max()) * 0.12, 1.0)
        for bar, value in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, value + spread * 0.025, f"{value:.2g}",
                    ha="center", va="bottom", fontsize=7.2, color=TEXT)
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_line(x: Iterable[float], y: Iterable[float], *, xlabel: str = "", ylabel: str = "",
               label: str | None = None, marker: str = "o", linewidth: float = 1.45,
               show_grid: bool = True, figsize: tuple[float, float] = FIGSIZE_SINGLE,
               filename: str | None = None, output_dir: str | Path = "figures",
               ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw one primary-series line chart."""
    xx, yy = _values(x, "x"), _values(y, "y")
    if len(xx) != len(yy): raise ValueError("x and y must have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    ax.plot(xx, yy, color=BLUE, marker=marker, linewidth=linewidth, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if label: ax.legend(loc="best")
    polish_axes(ax, grid=show_grid)
    return _finish(fig, ax, filename, output_dir)


def cumcm_multi_line(x: Iterable[float], series: Mapping[str, Iterable[float]], *, xlabel: str = "",
                     ylabel: str = "", ncol: int = 1, figsize: tuple[float, float] = FIGSIZE_SINGLE,
                     filename: str | None = None, output_dir: str | Path = "figures",
                     ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw named model or scenario series using the fixed paper palette."""
    xx = _values(x, "x")
    if not series: raise ValueError("series must not be empty")
    fig, ax = create_figure(figsize, ax=ax)
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, (name, y) in enumerate(series.items()):
        yy = _values(y, name)
        if len(yy) != len(xx): raise ValueError(f"series '{name}' must have the same length as x")
        ax.plot(xx, yy, color=PALETTE[index % len(PALETTE)], marker=markers[index % len(markers)], label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(ncol=ncol, loc="best")
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_scatter(x: Iterable[float], y: Iterable[float], *, xlabel: str = "", ylabel: str = "",
                  fit: bool = True, show_r2: bool = True, show_rmse: bool = False,
                  figsize: tuple[float, float] = FIGSIZE_SINGLE, filename: str | None = None,
                  output_dir: str | Path = "figures", ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw a scatter plot with an optional linear fit and compact fit diagnostics."""
    xx, yy = _values(x, "x"), _values(y, "y")
    if len(xx) != len(yy): raise ValueError("x and y must have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    ax.scatter(xx, yy, s=24, color=BLUE, edgecolors="white", linewidths=0.45, alpha=0.9, zorder=3)
    if fit:
        slope, intercept = np.polyfit(xx, yy, 1)
        fitted = slope * xx + intercept
        order = np.argsort(xx)
        ax.plot(xx[order], fitted[order], color=RED, linewidth=1.7, label="Linear fit")
        metrics = [f"y = {slope:.3g}x {intercept:+.3g}"]
        if show_r2:
            total = np.sum((yy - yy.mean()) ** 2)
            metrics.append(f"$R^2$ = {1 - np.sum((yy-fitted)**2)/total:.3f}" if total else "$R^2$ = N/A")
        if show_rmse: metrics.append(f"RMSE = {np.sqrt(np.mean((yy-fitted)**2)):.3g}")
        ax.text(0.03, 0.97, "\n".join(metrics), transform=ax.transAxes, va="top", fontsize=7.4,
                bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.25", "alpha": 0.92})
        ax.legend(loc="lower right")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_box(data: Sequence[Iterable[float]], labels: Sequence[str], *, xlabel: str = "", ylabel: str = "",
              figsize: tuple[float, float] = FIGSIZE_SINGLE, filename: str | None = None,
              output_dir: str | Path = "figures", ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw a styled box plot for distributions or model errors."""
    if len(data) != len(labels) or not data: raise ValueError("data and labels must be non-empty and have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    box = ax.boxplot(data, patch_artist=True, widths=0.52,
        medianprops={"color": RED, "linewidth": 1.35},
        whiskerprops={"color": DARK_GRAY, "linewidth": 0.85}, capprops={"color": DARK_GRAY, "linewidth": 0.85},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GRAY, "markeredgewidth": 0})
    for item in box["boxes"]:
        item.set(facecolor="#C9D9E6", edgecolor=BLUE, linewidth=0.9)
    ax.set_xticks(range(1, len(labels)+1), labels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_heatmap(data: Iterable[Iterable[float]], *, xlabels: Sequence[str] | None = None,
                  ylabels: Sequence[str] | None = None, annotate: bool = True, colorbar: bool = True,
                  fmt: str = ".2f", figsize: tuple[float, float] = FIGSIZE_SINGLE,
                  filename: str | None = None, output_dir: str | Path = "figures",
                  ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw a blue sequential heatmap for matrices, weights, and correlations."""
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or not matrix.size: raise ValueError("data must be a non-empty two-dimensional matrix")
    fig, ax = create_figure(figsize, ax=ax)
    image = ax.imshow(matrix, cmap=BLUE_CMAP, aspect="auto")
    if xlabels is not None:
        if len(xlabels) != matrix.shape[1]: raise ValueError("xlabels length must equal matrix columns")
        ax.set_xticks(range(matrix.shape[1]), xlabels)
    if ylabels is not None:
        if len(ylabels) != matrix.shape[0]: raise ValueError("ylabels length must equal matrix rows")
        ax.set_yticks(range(matrix.shape[0]), ylabels)
    if annotate:
        threshold = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=7,
                    color="white" if value > threshold else TEXT)
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(length=2, labelsize=7)
        cbar.outline.set_linewidth(0.55)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    return _finish(fig, ax, filename, output_dir)


def cumcm_sensitivity(variables: Sequence[str], effects: Iterable[float], *, xlabel: str = "Sensitivity effect",
                      ylabel: str = "", figsize: tuple[float, float] = FIGSIZE_SINGLE,
                      filename: str | None = None, output_dir: str | Path = "figures",
                      ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw an absolute-effect-sorted tornado plot; negative and positive effects differ."""
    values = _values(effects, "effects")
    if len(variables) != len(values): raise ValueError("variables and effects must have the same length")
    order = np.argsort(np.abs(values))
    values, names = values[order], np.asarray(variables)[order]
    fig, ax = create_figure(figsize, ax=ax)
    ax.barh(range(len(values)), values, color=[BLUE if v < 0 else RED for v in values], height=0.62)
    ax.axvline(0, color=DARK_GRAY, linewidth=0.75)
    ax.set_yticks(range(len(values)), names)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_model_compare(models: Sequence[str], scores: Iterable[float], *, metric: str = "Score",
                        higher_is_better: bool = True, figsize: tuple[float, float] = FIGSIZE_SINGLE,
                        filename: str | None = None, output_dir: str | Path = "figures",
                        ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Compare models and highlight the best result according to metric direction."""
    values = _values(scores, "scores")
    if len(models) != len(values): raise ValueError("models and scores must have the same length")
    best = int(np.argmax(values) if higher_is_better else np.argmin(values))
    return cumcm_bar(models, values, ylabel=metric, highlight=best, show_values=True, figsize=figsize,
                     filename=filename, output_dir=output_dir, ax=ax)


def cumcm_residual(predicted: Iterable[float], residual: Iterable[float], *, xlabel: str = "Predicted value",
                   ylabel: str = "Residual", figsize: tuple[float, float] = FIGSIZE_SINGLE,
                   filename: str | None = None, output_dir: str | Path = "figures",
                   ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw residuals against predictions with an unobtrusive zero reference."""
    xx, yy = _values(predicted, "predicted"), _values(residual, "residual")
    if len(xx) != len(yy): raise ValueError("predicted and residual must have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    ax.scatter(xx, yy, s=22, color=BLUE, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axhline(0, color=RED, linewidth=0.85, linestyle="--")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_prediction(x: Iterable[float], actual: Iterable[float], predicted: Iterable[float], *, xlabel: str = "",
                     ylabel: str = "", actual_label: str = "Actual", predicted_label: str = "Predicted",
                     figsize: tuple[float, float] = FIGSIZE_SINGLE, filename: str | None = None,
                     output_dir: str | Path = "figures", ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw actual (dark solid circles) and predicted (red dashed squares) results."""
    xx, aa, pp = _values(x, "x"), _values(actual, "actual"), _values(predicted, "predicted")
    if len(xx) != len(aa) or len(xx) != len(pp): raise ValueError("x, actual, and predicted must have the same length")
    fig, ax = create_figure(figsize, ax=ax)
    ax.plot(xx, aa, color=DARK_GRAY, marker="o", label=actual_label)
    ax.plot(xx, pp, color=RED, marker="s", linestyle="--", label=predicted_label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best")
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_prediction_band(x: Iterable[float], actual: Iterable[float], predicted: Iterable[float],
                          lower: Iterable[float], upper: Iterable[float], *, xlabel: str = "", ylabel: str = "",
                          figsize: tuple[float, float] = FIGSIZE_SINGLE, filename: str | None = None,
                          output_dir: str | Path = "figures", ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Draw actual/predicted values and a deliberately light prediction interval."""
    xx, aa, pp, lo, hi = (_values(v, n) for v, n in ((x, "x"), (actual, "actual"), (predicted, "predicted"), (lower, "lower"), (upper, "upper")))
    if len({len(xx), len(aa), len(pp), len(lo), len(hi)}) != 1: raise ValueError("all input sequences must have the same length")
    if np.any(lo > hi): raise ValueError("lower must not exceed upper")
    fig, ax = create_figure(figsize, ax=ax)
    ax.fill_between(xx, lo, hi, color=BLUE, alpha=0.16, linewidth=0, label="Prediction interval")
    ax.plot(xx, aa, color=DARK_GRAY, marker="o", label="Actual")
    ax.plot(xx, pp, color=RED, marker="s", linestyle="--", label="Predicted")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best")
    polish_axes(ax)
    return _finish(fig, ax, filename, output_dir)


def cumcm_multi_panel(nrows: int = 2, ncols: int = 2, *, figsize: tuple[float, float] = FIGSIZE_DOUBLE,
                      labels: bool = True) -> tuple[plt.Figure, np.ndarray]:
    """Create a paper-ready multi-panel canvas with optional (a), (b), ... labels."""
    if nrows < 1 or ncols < 1: raise ValueError("nrows and ncols must be positive")
    set_cumcm_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout="constrained")
    for index, axis in enumerate(axes.flat):
        polish_axes(axis)
        if labels:
            axis.text(-0.02, 1.03, f"({chr(97 + index)})", transform=axis.transAxes,
                      fontsize=8.6, fontweight="normal", ha="left", va="bottom")
    return fig, axes


# V2 extensions.  They live here too so ``from CUMCM_Style import *`` remains
# the stable, competition-friendly public entry point.
def cumcm_metric_compare(models: Sequence[str], metrics: Mapping[str, Iterable[float]], *,
                         normalize: bool = True, directions: Mapping[str, str] | None = None, annotate: bool = True,
                         figsize: tuple[float, float] = FIGSIZE_DOUBLE, filename: str | None = None,
                         output_dir: str | Path = "figures", ax: plt.Axes | None = None):
    """Compare models × metrics as an information-dense heatmap.

    When ``normalize`` is true each metric column is min-max normalized, which
    prevents mixing unlike units (for example R² and RMSE) on one visual scale.
    """
    if not metrics: raise ValueError("metrics must not be empty")
    matrix=np.column_stack([_values(v, name) for name,v in metrics.items()])
    if matrix.shape[0] != len(models): raise ValueError("every metric must have one value per model")
    shown=matrix.copy()
    if directions is not None:
        invalid={key:value for key,value in directions.items() if value not in {"min","max"}}
        if invalid: raise ValueError("metric directions must be 'min' or 'max'")
        for column,name in enumerate(metrics):
            if directions.get(name,"max") == "min":
                shown[:,column] = -shown[:,column]
    if normalize:
        lo,hi=np.nanmin(shown,axis=0),np.nanmax(shown,axis=0); den=hi-lo
        shown=np.divide(shown-lo,den,out=np.zeros_like(shown),where=den!=0)
    return cumcm_heatmap(shown,xlabels=list(metrics),ylabels=models,annotate=annotate,fmt=".2f",figsize=figsize,filename=filename,output_dir=output_dir,ax=ax)


def cumcm_convergence(iteration: Iterable[float], objective: Iterable[float] | Mapping[str, Iterable[float]], *,
                      best_so_far: bool = True, minimize: bool = True, log_scale: bool = False,
                      xlabel: str = "Iteration", ylabel: str = "Objective value", figsize=FIGSIZE_SINGLE,
                      filename=None, output_dir="figures", ax=None):
    """Plot iterative optimisation progress, optionally with a best-so-far envelope."""
    xx=_values(iteration,"iteration"); series=objective if isinstance(objective,Mapping) else {"Objective":objective}
    fig,ax=create_figure(figsize,ax=ax)
    for i,(name,value) in enumerate(series.items()):
        yy=_values(value,name)
        if len(yy)!=len(xx): raise ValueError("iteration and each objective must have the same length")
        color=PALETTE[i%len(PALETTE)]; ax.plot(xx,yy,color=color,linewidth=1.0,alpha=.55,label=name)
        if best_so_far:
            best=np.minimum.accumulate(yy) if minimize else np.maximum.accumulate(yy)
            ax.plot(xx,best,color=color,linewidth=1.65,label=f"{name} best-so-far")
            ax.plot(xx[-1],best[-1],"o",color=ACCENT if i==0 else color,markersize=4)
    if log_scale: ax.set_yscale("log")
    ax.set(xlabel=xlabel,ylabel=ylabel); ax.legend(ncol=2,loc="best"); polish_axes(ax)
    return _finish(fig,ax,filename,output_dir)


def cumcm_pareto(x: Iterable[float], y: Iterable[float], *, x_direction="min", y_direction="min",
                 xlabel="Objective 1", ylabel="Objective 2", figsize=FIGSIZE_SINGLE, filename=None,
                 output_dir="figures", ax=None):
    """Plot 2-D candidates and their non-dominated Pareto front for min/max objectives."""
    xx,yy=_values(x,"x"),_values(y,"y")
    if len(xx)!=len(yy): raise ValueError("x and y must have the same length")
    if x_direction not in {"min","max"} or y_direction not in {"min","max"}: raise ValueError("directions must be min or max")
    u=-xx if x_direction=="max" else xx; v=-yy if y_direction=="max" else yy
    front=np.array([not any((u<=u[i])&(v<=v[i])&((u<u[i])|(v<v[i]))) for i in range(len(xx))])
    fig,ax=create_figure(figsize,ax=ax); ax.scatter(xx,yy,s=20,color=GRAY,alpha=.7,edgecolor="white",linewidth=.4,label="Candidate")
    order=np.argsort(xx[front]); ax.plot(xx[front][order],yy[front][order],color=ACCENT,linewidth=1.6,marker="o",label="Pareto front")
    ax.set(xlabel=xlabel,ylabel=ylabel); ax.legend(loc="best"); polish_axes(ax); return _finish(fig,ax,filename,output_dir)


def cumcm_feature_importance(features: Sequence[str], importance: Iterable[float], *, top_n: int = 10,
                             xlabel="Importance", figsize=FIGSIZE_SINGLE, filename=None, output_dir="figures", ax=None):
    """Draw a sorted horizontal importance plot; signed values retain their direction."""
    values=_values(importance,"importance")
    if len(features)!=len(values): raise ValueError("features and importance must have the same length")
    if top_n<1: raise ValueError("top_n must be positive")
    order=np.argsort(np.abs(values))[-top_n:]; values,names=values[order],np.asarray(features)[order]
    fig,ax=create_figure(figsize,ax=ax); ax.barh(range(len(values)),values,color=[BLUE if v>=0 else RED for v in values],height=.62)
    if np.any(values<0): ax.axvline(0,color=DARK_GRAY,linewidth=.75)
    ax.set_yticks(range(len(values)),names); ax.set(xlabel=xlabel); polish_axes(ax); return _finish(fig,ax,filename,output_dir)


def cumcm_distribution(values: Iterable[float], *, kind="hist", xlabel="Value", ylabel="Frequency", bins="auto",
                       figsize=FIGSIZE_SINGLE, filename=None, output_dir="figures", ax=None):
    """Draw one distribution view (histogram, box, or violin), never a cluttered hybrid."""
    data=_values(values,"values"); fig,ax=create_figure(figsize,ax=ax)
    if kind=="hist": ax.hist(data,bins=bins,color=BLUE,edgecolor="white",linewidth=.5)
    elif kind=="box":
        ax.boxplot(data,vert=False,patch_artist=True,boxprops={"facecolor":"#C9D9E6","edgecolor":BLUE},medianprops={"color":RED}); ylabel=""
    elif kind=="violin":
        part=ax.violinplot(data,showmedians=True); [body.set_facecolor(BLUE) for body in part["bodies"]]; [body.set_alpha(.75) for body in part["bodies"]]
    else: raise ValueError("kind must be hist, box, or violin")
    ax.set(xlabel=xlabel,ylabel=ylabel); polish_axes(ax); return _finish(fig,ax,filename,output_dir)


def cumcm_residual_diagnostics(predicted: Iterable[float], residual: Iterable[float], *, figsize=FIGSIZE_DOUBLE,
                               filename=None, output_dir="figures"):
    """Create residual-vs-fitted, residual distribution and normal QQ diagnostic panels."""
    xx,rr=_values(predicted,"predicted"),_values(residual,"residual")
    if len(xx)!=len(rr): raise ValueError("predicted and residual must have the same length")
    fig,axes=cumcm_multi_panel(1,3,figsize=figsize); a,b,c=axes.flat
    cumcm_residual(xx,rr,ax=a)
    b.hist(rr,bins="auto",color=BLUE,edgecolor="white"); b.set(xlabel="Residual",ylabel="Frequency"); polish_axes(b)
    q=np.sort(rr); probs=(np.arange(1,len(q)+1)-.5)/len(q); normal=np.sqrt(2)*np.vectorize(lambda p: __import__('statistics').NormalDist().inv_cdf(float(p)))(probs)
    c.scatter(normal,q,s=18,color=BLUE,edgecolor="white",linewidth=.35); c.plot([normal.min(),normal.max()],[q.min(),q.max()],"--",color=ACCENT,linewidth=.8); c.set(xlabel="Theoretical quantiles",ylabel="Sample quantiles"); polish_axes(c)
    return _finish(fig,axes.flat[0],filename,output_dir)


def cumcm_scatter(x, y, *, xlabel="", ylabel="", fit=True, show_equation=True, show_r2=True, show_rmse=True,
                  show_mae=False, show_n=False, show_ci=True, fit_type="linear", figsize=FIGSIZE_SINGLE,
                  filename=None, output_dir="figures", ax=None):
    """Scatter with polynomial/linear fit and optional 95% mean-response confidence band."""
    from CUMCM_Statistics import r2_score, rmse, mae
    xx,yy=_values(x,"x"),_values(y,"y")
    if len(xx)!=len(yy): raise ValueError("x and y must have the same length")
    if fit_type not in {"linear","polynomial"}: raise ValueError("fit_type must be linear or polynomial")
    degree=1 if fit_type=="linear" else 2; fig,ax=create_figure(figsize,ax=ax); ax.scatter(xx,yy,s=24,color=BLUE,edgecolors="white",linewidths=.45,zorder=3)
    if fit:
        coeff=np.polyfit(xx,yy,degree); fitted=np.polyval(coeff,xx); grid=np.linspace(xx.min(),xx.max(),160); line=np.polyval(coeff,grid)
        ax.plot(grid,line,color=RED,linewidth=1.7,label=f"{fit_type.title()} fit")
        if show_ci and len(xx)>degree+1:
            se=np.sqrt(np.sum((yy-fitted)**2)/(len(xx)-degree-1)); xm=xx.mean(); denom=np.sum((xx-xm)**2)
            band=1.96*se*np.sqrt(1/len(xx)+(grid-xm)**2/denom) if denom else np.zeros_like(grid)
            ax.fill_between(grid,line-band,line+band,color=RED,alpha=.12,linewidth=0,label="95% CI")
        notes=[]
        if show_equation: notes.append(f"y = {coeff[-2]:.3g}x {coeff[-1]:+.3g}" if degree==1 else "Quadratic fit")
        if show_r2: notes.append(f"$R^2$ = {r2_score(yy,fitted):.3f}")
        if show_rmse: notes.append(f"RMSE = {rmse(yy,fitted):.3g}")
        if show_mae: notes.append(f"MAE = {mae(yy,fitted):.3g}")
        if show_n: notes.append(f"n = {len(xx)}")
        if notes: ax.text(.03,.97,"\n".join(notes),transform=ax.transAxes,va="top",fontsize=7.2,bbox={"facecolor":"white","edgecolor":GRID,"boxstyle":"round,pad=.25","alpha":.92})
        ax.legend(loc="lower right")
    ax.set(xlabel=xlabel,ylabel=ylabel); polish_axes(ax); return _finish(fig,ax,filename,output_dir)


def cumcm_model_compare(models, scores, *, metric="Score", direction=None, higher_is_better=True, sort=True,
                        figsize=FIGSIZE_SINGLE, filename=None, output_dir="figures", ax=None):
    """Compare a metric with explicit ``direction='max'`` or ``'min'`` (V1 flag still works)."""
    values=_values(scores,"scores")
    if len(models)!=len(values): raise ValueError("models and scores must have the same length")
    if direction is None: direction="max" if higher_is_better else "min"
    if direction not in {"max","min"}: raise ValueError("direction must be 'max' or 'min'")
    order=np.argsort(values); order=order[::-1] if direction=="max" else order
    if not sort: order=np.arange(len(values))
    names=np.asarray(models)[order]; data=values[order]; best=int(np.argmax(data) if direction=="max" else np.argmin(data))
    return cumcm_bar(names,data,ylabel=metric,highlight=best,show_values=True,figsize=figsize,filename=filename,output_dir=output_dir,ax=ax)


try:
    from CUMCM_Layout import cumcm_panel
except ImportError:  # direct execution fallback
    cumcm_panel = cumcm_multi_panel
