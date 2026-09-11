"""布局变换、配对测量和统一评价。"""
import logging
import time
import numpy as np
from model.candidate_points import check_route_feasibility
from model.geometry_metric import calculate_intersection_angle, bearing_dop
from model.localization import localize


def transform_layout(template, target, rotation_deg):
    """将原点模板整体旋转和平移；保持距离与所有交会角不变。"""
    angle = np.deg2rad(rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(template, float) @ rotation.T + np.asarray(target, float)


def scenario_randomness(seed, repetition, max_points=6):
    """生成场景真值、候选格旋转及配对测量误差，随机性仅用于离线实验。"""
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition, 2]))
    polar = rng.uniform(0, 2*np.pi)
    target = 250*np.sqrt(rng.random())*np.array([np.cos(polar), np.sin(polar)])
    rotation = int(rng.integers(24))*15.0
    errors = rng.uniform(-1, 1, max_points)
    return target, rotation, errors, rng


def uniform_indices(candidate_count, number_points):
    """从同一离散圆周候选集中选取尽可能等角间隔的布局。"""
    indices = np.floor(np.arange(number_points)*candidate_count/number_points + .5).astype(int) % candidate_count
    if len(np.unique(indices)) != number_points:
        raise ValueError("候选角分辨率不足以构造均匀布局")
    return indices.tolist()


def evaluate(experiment, method, repetition, points, target, errors, config,
             intersection_angle=None):
    """将布局输入问题一，分别记录 Ω 面积、直径、覆盖圆、误差和几何诊断。"""
    if not check_route_feasibility(points, arena_radius=config.arena_radius,
                                   max_step=config.max_step_m):
        raise AssertionError("实验布局违反场地或相邻移动上限")
    started = time.perf_counter()
    region, metrics, angles = localize(points, target, angle_error=config.error_bound_deg,
        measurement_errors=errors, grid_resolution=config.evaluation_resolution_m,
        initial_grid=config.initial_grid, arena_radius=config.arena_radius,
        truth_for_evaluation=target)
    runtime = time.perf_counter()-started
    if region.status != "sampled":
        raise AssertionError("误差未越界的合成数据必须得到含样本的可行区域")
    minimum_angle = calculate_intersection_angle(points, target)
    setting = f"angle{intersection_angle:g}" if intersection_angle is not None else f"n{len(points)}"
    row = {"experiment_id": f"{experiment}_{method}_{repetition:03d}_{setting}",
           "experiment_type": experiment, "method": method, "repetition": repetition,
           "number_points": len(points), "intersection_angle": intersection_angle,
           "area": metrics["area"], "area_lower": metrics["area_lower"],
           "area_upper": metrics["area_upper"], "diameter": metrics["diameter"],
           "diameter_upper": metrics["diameter_upper"], "cover_radius": metrics["cover_radius"],
           "cover_radius_lower": metrics["cover_radius_lower"],
           "localization_error": metrics["localization_error"],
           "centroid_error_bound": metrics["centroid_error_bound"],
           "min_intersection_angle": minimum_angle, "dop": bearing_dop(points, target),
           "runtime": runtime, "status": region.status, "truth_feasible": True}
    record = {"experiment_id": row["experiment_id"], "target": target.tolist(),
              "observation_points": np.asarray(points).tolist(), "angles": angles.tolist(),
              "measurement_errors": np.asarray(errors).tolist()}
    logging.info("%s | A=%.3f m²，D=%.3f m，e=%.3f m，γmin=%.1f°",
                 row["experiment_id"], row["area"], row["diameter"],
                 row["localization_error"], minimum_angle)
    return row, record
