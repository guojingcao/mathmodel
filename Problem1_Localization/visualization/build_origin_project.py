"""Assemble Model-1 proof figures and their data in a reusable Origin project."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import originpro as op

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "origin_figures" / "Problem1_Model1_Proof"
DATA = OUT / "origin_data"
OPJU = OUT / "问题一_证明型图件_Origin工程.opju"
NATIVE = OUT / "origin_native"

BLUE = "#5B7FA3"
ORANGE = "#C08B52"
RED = "#B6534B"
PURPLE = "#806A8A"
DARK = "#333333"


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
        axis = "X" if key.endswith("_x") or key in {"requested_resolution", "number_of_points"} else "Y"
        sheet.from_list(index, values, lname=key, axis=axis)
    return sheet, columns


def style_plot(plot, color, *, width=1.4, symbol=None):
    plot.color = color
    plot.set_float("line.width", width)
    if symbol is not None:
        plot.symbol_kind = symbol
        plot.symbol_size = 4.0


def make_geometry_graph(sheet, cols):
    graph = op.new_graph("M1_区域直径与覆盖圆")
    layer = graph[0]
    pairs = [
        ("outer_x", "outer_y", ORANGE, 1.0, None),
        ("inner_x", "inner_y", RED, 1.6, None),
        ("diam_circle_x", "diam_circle_y", DARK, 1.0, None),
        ("mec_x", "mec_y", PURPLE, 1.5, None),
        ("diameter_x", "diameter_y", DARK, 1.8, 3),
        ("observer_x", "observer_y", BLUE, 0.0, 8),
        ("truth_x", "truth_y", RED, 0.0, 18),
    ]
    keys = list(cols)
    for xkey, ykey, color, width, symbol in pairs:
        plot = layer.add_plot(sheet, coly=keys.index(ykey), colx=keys.index(xkey),
                              type="linesymbol" if symbol else "line")
        style_plot(plot, color, width=width, symbol=symbol)
    layer.rescale()
    layer.axis("x").title = "x (m)"
    layer.axis("y").title = "y (m)"
    layer.add_label("Model 1: diameter circle versus minimum enclosing circle")
    graph.save_fig(str(NATIVE / "origin_fig01_geometry.png"), type="png", width=2200)
    return graph


def make_convergence_graph(sheet, cols):
    graph = op.new_graph("M1_网格收敛_相对界宽")
    layer = graph[0]
    keys = list(cols)
    for ykey, color, symbol in (("rel_gap_A", BLUE, 3), ("rel_gap_D", ORANGE, 4), ("rel_gap_r", PURPLE, 5)):
        plot = layer.add_plot(sheet, coly=keys.index(ykey), colx=keys.index("requested_resolution"),
                              type="linesymbol")
        style_plot(plot, color, width=1.4, symbol=symbol)
    layer.xscale = "log10"
    layer.rescale()
    layer.axis("x").title = "Terminal grid size h (m)"
    layer.axis("y").title = "Relative enclosure width"
    layer.add_label("Model 1: simultaneous convergence of A, D and r*")
    graph.save_fig(str(NATIVE / "origin_fig02_convergence.png"), type="png", width=2200)
    return graph


def make_validation_graph(sheet, cols):
    graph = op.new_graph("M1_覆盖圆无量纲误差")
    layer = graph[0]
    keys = list(cols)
    plot = layer.add_plot(sheet, coly=keys.index("rho_mec"), colx=keys.index("number_of_points"), type="scatter")
    style_plot(plot, BLUE, width=0.0, symbol=3)
    layer.rescale()
    layer.axis("x").title = "Number of observation points"
    layer.axis("y").title = "rho = ||c*-S|| / r*"
    layer.add_label("All 450 cases satisfy rho <= 1")
    graph.save_fig(str(NATIVE / "origin_fig04_validation.png"), type="png", width=2200)
    return graph


def import_final_images():
    for stem, long_name in [
        ("fig01_model1_geometry_proof", "最终图1_几何证据链"),
        ("fig02_grid_convergence", "最终图2_网格收敛"),
        ("fig03_jung_covering", "最终图3_荣格覆盖"),
        ("fig04_monte_carlo_validation", "最终图4_随机验证"),
    ]:
        page = op.new_image(long_name)
        page.from_file(str(OUT / f"{stem}.png"))


def main():
    NATIVE.mkdir(parents=True, exist_ok=True)
    op.set_show(False)
    op.new()
    geometry_sheet, geometry_cols = add_book("图1_几何源数据", DATA / "fig01_geometry.csv")
    convergence_sheet, convergence_cols = add_book("图2_收敛源数据", DATA / "fig02_convergence.csv")
    add_book("图3_理论源数据", DATA / "fig03_theory.csv")
    validation_sheet, validation_cols = add_book("图4_仿真源数据", DATA / "fig04_validation_cases.csv")
    make_geometry_graph(geometry_sheet, geometry_cols)
    make_convergence_graph(convergence_sheet, convergence_cols)
    make_validation_graph(validation_sheet, validation_cols)
    import_final_images()
    op.save(str(OPJU))
    if not OPJU.exists() or OPJU.stat().st_size < 1000:
        raise RuntimeError(f"Origin project was not saved correctly: {OPJU}")
    print(f"SAVED {OPJU} ({OPJU.stat().st_size} bytes)")
    op.exit()


if __name__ == "__main__":
    main()
