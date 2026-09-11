"""问题二入口：枚举贪心布局→问题一 Ω 定位→配对实验→CUMCM图。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import platform
from pathlib import Path

import numpy as np

from config import LayoutConfig, RESULTS, FIGURES, ROOT, SEED, REPETITIONS
from experiment import compare_layout, point_number, angle_analysis
from experiment.report import write_csv, write_report
from model.candidate_points import generate_candidate_points
from model.layout_optimizer import optimize_layout, optimize_dop_baseline


def main():
    """运行可复现实验并保存完整输入、选择轨迹、统计、图件和源码哈希。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--evaluation-resolution", type=float,
                        default=LayoutConfig().evaluation_resolution_m)
    parser.add_argument("--optimization-resolution", type=float,
                        default=LayoutConfig().optimization_resolution_m)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions 至少为2")
    config = LayoutConfig(evaluation_resolution_m=args.evaluation_resolution,
                          optimization_resolution_m=args.optimization_resolution)
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output / "run.log", mode="w", encoding="utf-8")], force=True)
    console = logging.StreamHandler(); console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(console)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    candidates = generate_candidate_points(radius=config.observation_radius,
        angular_step=config.angular_step_deg, arena_radius=config.arena_radius)
    logging.info("开始Ω面积贪心：%d个候选点，选择6点", len(candidates))
    optimized_result = optimize_layout(candidates, [0, 0], 6, seed=args.seed,
        angle_error=config.error_bound_deg, grid_resolution=config.optimization_resolution_m,
        arena_radius=config.arena_radius, max_step=config.max_step_m)
    dop_layout, dop_indices = optimize_dop_baseline(candidates, [0, 0], 6,
        initial_index=optimized_result.selected_indices[0], arena_radius=config.arena_radius,
        max_step=config.max_step_m)
    write_csv(args.output / "optimization_trace.csv", optimized_result.trace)
    write_csv(args.output / "candidate_evaluations.csv", optimized_result.candidate_evaluations)
    layout_data = {"candidate_points": candidates.tolist(),
        "optimized_indices": optimized_result.selected_indices,
        "optimized_points": optimized_result.selected_points.tolist(),
        "dop_baseline_indices": dop_indices, "dop_baseline_points": dop_layout.tolist()}
    (args.output / "selected_layouts.json").write_text(
        json.dumps(layout_data, ensure_ascii=False, indent=2), encoding="utf-8")

    rows, records = [], []
    modules = [(compare_layout, (candidates, optimized_result.selected_points, dop_layout)),
               (point_number, (optimized_result.selected_points,)), (angle_analysis, ())]
    for module, extra in modules:
        logging.info("开始实验模块：%s", module.__name__)
        part, raw = module.run(config, args.repetitions, args.seed, *extra)
        rows.extend(part); records.extend(raw)
    write_csv(args.output / "problem2_summary.csv", rows)
    with (args.output / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(rows, args.output, args.figures, args.repetitions, config,
                 optimized_result.selected_indices, dop_indices)
    if not args.no_plots:
        from visualization.plot_problem2 import plot_all
        plot_all(rows, records, candidates, optimized_result.selected_points,
                 optimized_result.trace, config, args.figures, args.output)
    manifest = {"model": "候选点枚举+Ω面积贪心；定位继承问题一", "seed": args.seed,
        "repetitions": args.repetitions, "config": asdict(config), "runs": len(rows),
        "python": platform.python_version(), "numpy": np.__version__,
        "problem1_dependency": str((ROOT.parent / "Problem1_Localization").resolve()),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(ROOT.rglob("*.py"))}}
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("完成：%d次问题一定位评价；结果目录 %s", len(rows), args.output.resolve())


if __name__ == "__main__":
    main()
