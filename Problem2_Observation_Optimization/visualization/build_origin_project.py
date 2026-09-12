"""Assemble Model-2 proof figures and source data in an Origin project."""
from __future__ import annotations

import csv
from pathlib import Path

import originpro as op

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "origin_figures" / "Problem2_Model2_Proof"
DATA = OUT / "origin_data"
NATIVE = OUT / "origin_native"
OPJU = OUT / "问题二_证明型图件_Origin工程.opju"

BLUE = "#5B7FA3"
RED = "#B6534B"
ORANGE = "#C08B52"
PURPLE = "#806A8A"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            raw = row[key]
            try:
                values.append(float(raw) if raw != "" else float("nan"))
            except ValueError:
                values.append(raw)
        columns[key] = values
    return columns


def add_book(name, path):
    columns = read_csv(path)
    book = op.new_book("w", name)
    sheet = book[0]
    sheet.name = name[:20]
    for index, (key, values) in enumerate(columns.items()):
        axis = "X" if key in {"x", "round", "number_points", "intersection_angle",
                              "random_area", "dop_area"} or key.endswith("_x") else "Y"
        sheet.from_list(index, values, lname=key, axis=axis)
    return sheet, columns


def style(plot, color, symbol=3):
    plot.color = color
    plot.set_float("line.width", 1.4)
    plot.symbol_kind = symbol
    plot.symbol_size = 4.0


def make_trace(sheet, cols):
    graph = op.new_graph("M2_贪心收敛")
    layer = graph[0]
    keys = list(cols)
    plot = layer.add_plot(sheet, coly=keys.index("area"), colx=keys.index("round"), type="linesymbol")
    style(plot, BLUE, 3)
    layer.yscale = "log10"
    layer.rescale()
    layer.axis("x").title = "Number of selected observation points"
    layer.axis("y").title = "Feasible-region area (m^2)"
    graph.save_fig(str(NATIVE / "origin_fig01_trace.png"), type="png", width=2200)


def make_paired(sheet, cols):
    graph = op.new_graph("M2_配对面积比较")
    layer = graph[0]
    keys = list(cols)
    plot = layer.add_plot(sheet, coly=keys.index("optimized_area"), colx=keys.index("random_area"), type="scatter")
    style(plot, BLUE, 3)
    layer.rescale()
    layer.axis("x").title = "Random-layout area (m^2)"
    layer.axis("y").title = "Optimized-layout area (m^2)"
    graph.save_fig(str(NATIVE / "origin_fig04_paired.png"), type="png", width=2200)


def make_sensitivity(sheet, cols):
    keys = list(cols)
    if "number_points" not in keys or "area_mean" not in keys:
        return
    graph = op.new_graph("M2_点数敏感性")
    layer = graph[0]
    plot = layer.add_plot(sheet, coly=keys.index("area_mean"), colx=keys.index("number_points"), type="linesymbol")
    style(plot, PURPLE, 3)
    layer.rescale()
    layer.axis("x").title = "Number of observation points"
    layer.axis("y").title = "Mean feasible-region area (m^2)"
    graph.save_fig(str(NATIVE / "origin_fig04_sensitivity.png"), type="png", width=2200)


def import_final_images():
    for stem, name in [
        ("fig01_selection_mechanism", "最终图1_选点机制"),
        ("fig02_layout_regions", "最终图2_区域对照"),
        ("fig03_layout_performance", "最终图3_性能比较"),
        ("fig04_paired_robustness", "最终图4_配对与鲁棒性"),
    ]:
        page = op.new_image(name)
        page.from_file(str(OUT / f"{stem}.png"))


def main():
    NATIVE.mkdir(parents=True, exist_ok=True)
    op.set_show(False)
    op.new()
    add_book("图1_候选点", DATA / "fig01_candidates.csv")
    trace_sheet, trace_cols = add_book("图1_优化轨迹", DATA / "fig01_trace.csv")
    add_book("图2_区域顶点", DATA / "fig02_regions.csv")
    add_book("图3_性能统计", DATA / "fig03_layout_metrics.csv")
    paired_sheet, paired_cols = add_book("图4_配对面积", DATA / "fig04_paired_area.csv")
    sensitivity_sheet, sensitivity_cols = add_book("图4_敏感性", DATA / "fig04_sensitivity.csv")
    make_trace(trace_sheet, trace_cols)
    make_paired(paired_sheet, paired_cols)
    make_sensitivity(sensitivity_sheet, sensitivity_cols)
    import_final_images()
    op.save(str(OPJU))
    if not OPJU.exists() or OPJU.stat().st_size < 1000:
        raise RuntimeError(f"Origin project save failed: {OPJU}")
    print(f"SAVED {OPJU} ({OPJU.stat().st_size} bytes)")
    op.exit()


if __name__ == "__main__":
    main()
