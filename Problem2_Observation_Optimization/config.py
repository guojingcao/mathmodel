"""问题二统一配置：只优化观测布局，定位完全复用问题一。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROBLEM1 = WORKSPACE / "Problem1_Localization"
RESULTS = ROOT / "results"
FIGURES = WORKSPACE / "origin_figures" / "Problem2_Observation_Optimization"
SEED = 202604004013
REPETITIONS = 30


@dataclass(frozen=True)
class LayoutConfig:
    """候选布局、移动可达性及问题一网格精度参数。"""

    arena_radius: float = 1800.0
    observation_radius: float = 1000.0
    angular_step_deg: float = 15.0
    error_bound_deg: float = 1.0
    max_step_m: float = 2000.0
    initial_grid: int = 200
    optimization_resolution_m: float = 0.28125
    evaluation_resolution_m: float = 0.0703125

