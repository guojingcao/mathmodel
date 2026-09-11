"""问题一适配器：唯一的 Ω 计算入口，问题二不复制角域交集代码。"""
import sys
from pathlib import Path
import numpy as np


def ensure_problem1_path():
    """将共同工作区加入路径，以命名空间导入问题一并避免 model 同名冲突。"""
    path = str(Path(__file__).resolve().parents[2])
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_problem1_path()
from Problem1_Localization.config import GridConfig  # noqa: E402
from Problem1_Localization.model.geometry import angle_between_points  # noqa: E402
from Problem1_Localization.model.feasible_region import calculate_feasible_region  # noqa: E402
from Problem1_Localization.model.metrics import calculate_metrics  # noqa: E402


def localize(observation_points, target_reference, *, angle_error=1.0,
             measurement_errors=None, grid_resolution=0.0703125,
             initial_grid=200, arena_radius=1800.0, truth_for_evaluation=None):
    """生成仿真测向角并调用问题一，返回区域、指标和实际输入角。

    target_reference 在布局规划中应为问题一当前区域中心或场景假设点；
    仿真时可令其等于真值来生成可控测量。truth_for_evaluation 只参与事后误差。
    """
    points = np.asarray(observation_points, dtype=float).reshape(-1, 2)
    target = np.asarray(target_reference, dtype=float)
    if target.shape != (2,) or len(points) == 0:
        raise ValueError("需要二维目标参考点和至少一个观测点")
    exact = np.array([angle_between_points(point, target) for point in points])
    errors = np.zeros(len(points)) if measurement_errors is None else np.asarray(measurement_errors, float)
    if errors.shape != (len(points),) or not np.isfinite(errors).all():
        raise ValueError("measurement_errors 必须与观测点一一对应且有限")
    angles = (exact + errors) % 360
    grid = GridConfig(radius=arena_radius, initial_grid=initial_grid, resolution=grid_resolution)
    region = calculate_feasible_region(points, angles, angle_error, config=grid, return_details=True)
    metrics = calculate_metrics(region, truth_for_evaluation)
    return region, metrics, angles
