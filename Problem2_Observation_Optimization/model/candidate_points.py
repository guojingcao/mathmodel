"""在圆形场地内生成有限候选检测位置。"""
import numpy as np


def generate_candidate_points(center=(0.0, 0.0), *, radius=1000.0,
                              angular_step=15.0, arena_radius=1800.0):
    """在假设目标周围等角度生成圆周候选点，并裁去场地外位置。

    center 来自问题一当前可行域的代表中心；仿真实验中用真值代替，
    仅为隔离观测几何影响，不表示实战时已知真实干扰源。
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center 必须是有限二维坐标")
    if not np.isfinite([radius, angular_step, arena_radius]).all():
        raise ValueError("候选点参数必须有限")
    if radius <= 0 or arena_radius <= 0 or not 0 < angular_step <= 180:
        raise ValueError("半径必须为正，角步长须在 (0,180] 度内")
    count = int(np.floor(360.0 / angular_step + 1e-12))
    angles = np.deg2rad(np.arange(count) * angular_step)
    points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    points = points[np.linalg.norm(points, axis=1) <= arena_radius + 1e-9]
    if len(points) < 2:
        raise ValueError("场地内候选点不足；请调整中心、观测半径或角步长")
    return points


def check_route_feasibility(points, *, arena_radius=1800.0, max_step=2000.0):
    """检查布局点位于场地且给定选择顺序的相邻移动不超过上限。"""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    inside = bool(np.all(np.linalg.norm(points, axis=1) <= arena_radius + 1e-9))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([])
    reachable = bool(np.all(steps <= max_step + 1e-9)) if max_step is not None else True
    return inside and reachable

