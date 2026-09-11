"""正式问题二结果的独立一致性、几何包络、图件与复现校验。"""
import ast
import csv
import hashlib
import io
import json
import unittest
import numpy as np
from PIL import Image

from config import ROOT, RESULTS, FIGURES
from model.localization import ensure_problem1_path
ensure_problem1_path()
from Problem1_Localization.model.geometry import valid_mask, wedge_halfplanes  # noqa: E402
from Problem1_Localization.model.metrics import (polygon_area_centroid, maximum_diameter,
                                                  minimum_enclosing_circle)  # noqa: E402


def reference_polygon(observers, angles, error):
    """用独立半平面多边形裁剪复算受控实验Ω；本实验交集远离圆盘边界。"""
    polygon = np.array([[-1800., -1800.], [1800., -1800.],
                        [1800., 1800.], [-1800., 1800.]])
    normals, offsets = wedge_halfplanes(observers, angles, error)
    for normal, offset in zip(normals, offsets):
        clipped = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            first, second = start @ normal-offset, end @ normal-offset
            if first >= 0:
                clipped.append(start)
            if (first >= 0) != (second >= 0):
                clipped.append(start + first/(first-second)*(end-start))
        polygon = np.asarray(clipped, float).reshape(-1, 2)
    return polygon


def validate():
    """运行全部复核并写 validation.json；任何不一致立即抛出异常。"""
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert test_result.wasSuccessful(), stream.getvalue()
    (RESULTS / "unit_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    with (RESULTS / "problem2_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [json.loads(line) for line in (RESULTS / "observations.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == len(records) == 420
    assert len({row["experiment_id"] for row in rows}) == 420
    assert len({record["experiment_id"] for record in records}) == 420
    record_map = {record["experiment_id"]: record for record in records}
    reference_cases, max_centroid_gap = 0, 0.0
    for row in rows:
        record = record_map[row["experiment_id"]]
        points, angles, target = record["observation_points"], record["angles"], record["target"]
        assert valid_mask([target], points, angles, 1.0, 1800)[0]
        polygon = reference_polygon(points, angles, 1.0)
        assert len(polygon) >= 3 and np.linalg.norm(polygon, axis=1).max() < 1800
        area, centroid = polygon_area_centroid(polygon)
        diameter = maximum_diameter(polygon)
        _, radius = minimum_enclosing_circle(polygon)
        assert float(row["area_lower"])-1e-6 <= area <= float(row["area_upper"])+1e-6
        assert float(row["diameter"])-1e-6 <= diameter <= float(row["diameter_upper"])+1e-6
        assert float(row["cover_radius_lower"])-1e-6 <= radius <= float(row["cover_radius"])+1e-6
        estimate_error = float(row["localization_error"])
        exact_error = float(np.linalg.norm(centroid-np.asarray(target)))
        gap = abs(exact_error-estimate_error)
        max_centroid_gap = max(max_centroid_gap, gap)
        assert gap <= float(row["centroid_error_bound"])+1e-6
        reference_cases += 1
    # 配对实验：同一重复的公共变量必须完全相同。
    for repetition in range(30):
        layout = [record_map[f"layout_{method}_{repetition:03d}_n4"]
                  for method in ("random", "uniform", "optimized", "dop_baseline")]
        assert all(item["target"] == layout[0]["target"] for item in layout)
        assert all(item["measurement_errors"] == layout[0]["measurement_errors"] for item in layout)
        number = [record_map[f"number_optimized_{repetition:03d}_n{n}"] for n in range(2, 7)]
        for shorter, longer in zip(number, number[1:]):
            assert shorter["observation_points"] == longer["observation_points"][:len(shorter["observation_points"])]
            assert shorter["measurement_errors"] == longer["measurement_errors"][:len(shorter["measurement_errors"])]
        angle = [record_map[f"angle_controlled_{repetition:03d}_angle{value}"]
                 for value in (30, 45, 60, 90, 120)]
        assert all(item["target"] == angle[0]["target"] for item in angle)
        assert all(item["measurement_errors"] == angle[0]["measurement_errors"] for item in angle)
    trace = list(csv.DictReader((RESULTS / "optimization_trace.csv").open(encoding="utf-8-sig")))
    assert [int(row["round"]) for row in trace] == list(range(1, 7))
    assert np.all(np.diff([float(row["area_upper"]) for row in trace]) <= 1e-8)
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    figure_checks = []
    for png in sorted(FIGURES.glob("*.png")):
        with Image.open(png) as image:
            dpi = image.info.get("dpi")
            assert dpi and min(dpi) >= 599.9
            figure_checks.append({"file": png.name, "pixels": list(image.size), "dpi": list(dpi)})
        assert png.with_suffix(".pdf").stat().st_size > 1000
        assert png.with_suffix(".svg").stat().st_size > 1000
    assert len(figure_checks) == 5
    missing_docstrings = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                missing_docstrings.append(f"{path.name}:{node.name}")
    assert not missing_docstrings, missing_docstrings
    result = {"status":"PASS", "unit_tests":test_result.testsRun,
              "problem1_reference_geometry_cases":reference_cases,
              "unique_experiment_ids":"PASS", "paired_inputs":"PASS",
              "source_hashes":"PASS", "function_docstrings":"PASS",
              "maximum_localization_error_discretization_gap_m":max_centroid_gap,
              "figures":figure_checks}
    (RESULTS / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    validate()
