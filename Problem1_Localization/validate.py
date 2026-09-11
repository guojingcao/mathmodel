"""完整结果复核：单元测试、450组解析参考、配对输入和收敛表。"""
import ast
import csv
import hashlib
import io
import json
import unittest
import numpy as np
from PIL import Image
from config import ROOT, RESULTS, FIGURES, GridConfig
from experiment.report import write_csv
from model.feasible_region import calculate_feasible_region
from model.metrics import calculate_metrics, polygon_area_centroid, maximum_diameter, minimum_enclosing_circle
from model.geometry import valid_mask
from tests.test_model import independent_polygon


def validate():
    """独立检查已生成结果，写 validation.json 与收敛 CSV；任一失败抛异常。"""
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert test_result.wasSuccessful(), stream.getvalue()
    (RESULTS / "unit_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    with (RESULTS / "experiment_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [json.loads(line) for line in (RESULTS / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    record_map = {r["experiment_id"]: r for r in records}
    bounds_checked, max_centroid_error = 0, 0.0
    for row in rows:
        raw = record_map[row["experiment_id"]]
        polygon = independent_polygon(raw["observation_points"], raw["angles"], raw["error_bound"])
        # 本受控实验的交集远离场地边界，因此方框裁剪可作为圆盘交集的精确参考。
        assert len(polygon) >= 3 and np.linalg.norm(polygon, axis=1).max() < 1800
        area, centroid = polygon_area_centroid(polygon)
        diameter = maximum_diameter(polygon)
        _, radius = minimum_enclosing_circle(polygon)
        assert float(row["area_lower"])-1e-6 <= area <= float(row["area_upper"])+1e-6
        assert float(row["diameter"])-1e-6 <= diameter <= float(row["diameter_upper"])+1e-6
        assert float(row["cover_radius_lower"])-1e-6 <= radius <= float(row["cover_radius"])+1e-6
        computed = np.array([float(row["centroid_x"]), float(row["centroid_y"])])
        error = float(np.linalg.norm(centroid-computed))
        max_centroid_error = max(max_centroid_error, error)
        assert error <= float(row["centroid_error_bound"])+1e-6
        cover_center = np.array([float(row["cover_center_x"]), float(row["cover_center_y"])])
        assert np.linalg.norm(polygon-cover_center, axis=1).max() <= float(row["cover_radius"])+1e-6
        assert valid_mask([raw["true_position"]], raw["observation_points"], raw["angles"], raw["error_bound"], 1800)[0]
        bounds_checked += 1
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    repetitions = manifest["repetitions"]
    assert len(rows) == len(records) == len(record_map) == repetitions*15
    for rep in range(repetitions):
        base = record_map[f"number_{rep:03d}_6"]
        for n in range(2, 6):
            record = record_map[f"number_{rep:03d}_{n}"]
            assert record["observation_points"] == base["observation_points"][:n]
            assert record["angles"] == base["angles"][:n]
        base = record_map[f"error_{rep:03d}_0.2"]
        for bound in (.5, 1.0, 1.5, 2.0):
            record = record_map[f"error_{rep:03d}_{bound}"]
            assert record["observation_points"] == base["observation_points"]
            assert record["angles"] == base["angles"]
        base = record_map[f"angle_{rep:03d}_90"]
        for angle in (30, 45, 60, 120):
            record = record_map[f"angle_{rep:03d}_{angle}"]
            assert record["realized_errors"] == base["realized_errors"]
            assert record["true_position"] == base["true_position"]
    raw = record_map["number_000_3"]
    convergence = []
    for resolution in (.28125, .140625, .0703125, .03515625, .017578125):
        region = calculate_feasible_region(raw["observation_points"], raw["angles"], raw["error_bound"],
                             config=GridConfig(resolution=resolution), return_details=True)
        metric = calculate_metrics(region, raw["true_position"])
        convergence.append({"requested_resolution": resolution, **metric})
    widths = [r["area_upper"]-r["area_lower"] for r in convergence]
    assert np.all(np.diff(widths) < 0)
    write_csv(RESULTS / "grid_convergence.csv", convergence)
    figure_checks = []
    for png in sorted(FIGURES.glob("*.png")):
        with Image.open(png) as im:
            dpi = im.info.get("dpi")
            assert dpi and min(dpi) >= 599.9
            figure_checks.append({"file": png.name, "pixels": list(im.size), "dpi": list(dpi)})
        assert png.with_suffix(".pdf").stat().st_size > 1000
        assert png.with_suffix(".svg").stat().st_size > 1000
    assert len(figure_checks) == 5
    missing_comments = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                missing_comments.append(f"{path.name}:{node.name}")
    assert not missing_comments, missing_comments
    result = {"status": "PASS", "unit_tests": test_result.testsRun, "reference_geometry_cases": bounds_checked,
              "paired_input_checks": "PASS", "source_hash_checks": "PASS",
              "maximum_centroid_discretization_error_m": max_centroid_error,
              "convergence_area_gaps_m2": widths, "function_docstrings": "PASS", "figures": figure_checks}
    (RESULTS / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    validate()
