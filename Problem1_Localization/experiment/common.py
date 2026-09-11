"""统一生成配对实验输入；求解器不使用真值。"""
import logging
import time
import numpy as np
from model.geometry import angle_between_points, valid_mask
from model.feasible_region import calculate_feasible_region
from model.metrics import calculate_metrics


def make_scenario(seed, repetition, directions, noise_scale=1.0):
    """固定距离 1000 m；位置、整体旋转和标准化误差随重复编号变化。

    同一重复内改变站点数量/误差界/交会角时，复用这些随机量。
    随机性属于离线验证数据，不属于确定性定位模型。
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
    polar = rng.uniform(0, 2 * np.pi)
    target = 300 * np.sqrt(rng.random()) * np.array([np.cos(polar), np.sin(polar)])
    rotation = rng.uniform(0, 360)
    bearings = np.deg2rad(np.asarray(directions) + rotation)
    observers = target + 1000 * np.column_stack([np.cos(bearings), np.sin(bearings)])
    errors = rng.uniform(-1, 1, 6)[:len(directions)] * noise_scale
    exact = np.array([angle_between_points(p, target) for p in observers])
    measured = (exact + errors) % 360
    return target, observers, measured, errors


def evaluate(kind, repetition, setting, target, observers, angles, errors, bound, intersection, config):
    """执行单次确定性求解并记录输入和结果；真值只用于事后检验与误差统计。"""
    started = time.perf_counter()
    region = calculate_feasible_region(observers, angles, bound, config=config, return_details=True)
    metrics = calculate_metrics(region, target)
    feasible_truth = bool(valid_mask([target], observers, angles, bound, config.radius)[0])
    row = {"experiment_id": f"{kind}_{repetition:03d}_{setting}", "experiment_type": kind,
           "repetition": repetition, "number_of_points": len(observers), "angle_error": bound,
           "intersection_angle": intersection, **metrics, "truth_feasible": feasible_truth,
           "runtime_s": time.perf_counter() - started}
    record = {"experiment_id": row["experiment_id"], "true_position": target.tolist(),
              "observation_points": observers.tolist(), "angles": angles.tolist(),
              "realized_errors": errors.tolist(), "error_bound": bound}
    logging.info("%s | 面积 %.4f [%.4f, %.4f] m² | 定位误差 %.4f m | %s",
                 row["experiment_id"], row["area"] or 0, row["area_lower"], row["area_upper"],
                 row["localization_error"] or 0, row["status"])
    if not feasible_truth or region.status != "sampled":
        raise AssertionError(f"合成实验应包含真值且获得可行点：{row['experiment_id']}")
    return row, record
