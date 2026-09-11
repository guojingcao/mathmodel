"""布局几何与问题一 Ω 指标的薄封装，避免复制定位算法。"""
import numpy as np
from model.localization import ensure_problem1_path

ensure_problem1_path()
from Problem1_Localization.model.metrics import calculate_metrics as _calculate_metrics  # noqa: E402


def calculate_area(region):
    """返回问题一连续区域面积的网格包络中点及上下界，单位 m²。"""
    metric = _calculate_metrics(region)
    return metric["area"], metric["area_lower"], metric["area_upper"]


def calculate_diameter(region):
    """返回问题一可行域直径的内包值与外包上界，单位 m。"""
    metric = _calculate_metrics(region)
    return metric["diameter"], metric["diameter_upper"]


def pairwise_intersection_angles(observation_points, target_reference):
    """计算所有测向线的条件交会角 min(γ,180°-γ)，范围 [0°,90°]。

    γ 是从目标参考点指向两个观测点的较小夹角。180°反向共线与0°
    同样退化，因此以锐化后的条件角衡量几何独立性。
    """
    points = np.asarray(observation_points, dtype=float).reshape(-1, 2)
    target = np.asarray(target_reference, dtype=float)
    vectors = points - target
    lengths = np.linalg.norm(vectors, axis=1)
    if target.shape != (2,) or len(points) < 2 or np.any(lengths <= 0):
        raise ValueError("至少需要两个不与目标参考点重合的观测点")
    unit = vectors / lengths[:, None]
    raw = np.degrees(np.arccos(np.clip(unit @ unit.T, -1.0, 1.0)))
    i, j = np.triu_indices(len(points), 1)
    return np.minimum(raw[i, j], 180.0 - raw[i, j])


def calculate_intersection_angle(observation_points, target_reference):
    """返回布局的最小条件交会角；只作诊断，不替代 Ω 面积目标。"""
    return float(pairwise_intersection_angles(observation_points, target_reference).min())


def bearing_dop(observation_points, target_reference):
    """计算无量纲方向几何 DOP 诊断值 sqrt(trace((HᵀH)⁻¹)))。

    H 的每行是视线法向量。矩阵秩不足时返回无穷大。该局部随机误差指标
    不含有界角域、距离尺度或场地边界，因此不作为本文最终优化目标。
    """
    points = np.asarray(observation_points, float).reshape(-1, 2)
    target = np.asarray(target_reference, float)
    vectors = target - points
    lengths = np.linalg.norm(vectors, axis=1)
    if len(points) < 2 or np.any(lengths <= 0):
        return float("inf")
    unit = vectors / lengths[:, None]
    h = np.column_stack([-unit[:, 1], unit[:, 0]])
    information = h.T @ h
    if np.linalg.matrix_rank(information, tol=1e-10) < 2:
        return float("inf")
    return float(np.sqrt(np.trace(np.linalg.inv(information))))
