"""问题一离线实验入口：生成数据→角域网格求解→统计→CUMCM论文图。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import platform
from pathlib import Path
import numpy as np
from config import GridConfig, ROOT, RESULTS, FIGURES, REPETITIONS, SEED
from experiment import experiment_number, experiment_error, experiment_angle
from experiment.report import write_csv, write_report


def main():
    """命令行参数可调整分辨率与重复次数；结果保留全量输入以便确定性重放。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--resolution", type=float, default=GridConfig().resolution)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("统计实验 repetitions 至少为2")
    cfg = GridConfig(initial_grid=args.grid, resolution=args.resolution)
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "run.log", encoding="utf-8", mode="w")], force=True)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(console)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    rows, records = [], []
    for module in (experiment_number, experiment_error, experiment_angle):
        logging.info("开始模块：%s", module.__name__)
        part, raw = module.run(cfg, args.repetitions, args.seed)
        rows.extend(part)
        records.extend(raw)
    write_csv(args.output / "experiment_summary.csv", rows)
    with (args.output / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(rows, args.output, args.figures, args.repetitions, cfg)
    if not args.no_plots:
        from visualization.plot_results import plot_all
        plot_all(rows, records, cfg, args.figures, args.output)
    manifest = {"model": "确定性有界误差角域交集，自适应网格", "seed": args.seed,
                "repetitions": args.repetitions, "grid": asdict(cfg), "python": platform.python_version(),
                "numpy": np.__version__, "figure_directory": str(args.figures.resolve()),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(ROOT.rglob("*.py"))}}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("完成：%d 次离线实验；结果目录 %s", len(rows), args.output.resolve())


if __name__ == "__main__":
    main()
