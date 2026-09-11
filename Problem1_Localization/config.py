"""问题一统一配置；仅离线区域定位，不连接模拟器、不执行搜索调度。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results"
FIGURES = WORKSPACE / "origin_figures" / "Problem1_Localization"
SEED = 202604004013
REPETITIONS = 30


@dataclass(frozen=True)
class GridConfig:
    """初始方格数与边界细分精度；最终边界格边长不超过 resolution。"""

    radius: float = 1800.0
    initial_grid: int = 200
    resolution: float = 0.03515625
    max_cells: int = 2_000_000
