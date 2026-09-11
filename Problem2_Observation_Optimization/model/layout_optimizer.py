"""候选点枚举与逐点贪心：面积优先，不使用加权混合目标。"""
from dataclasses import dataclass
import logging
import numpy as np
from model.candidate_points import check_route_feasibility
from model.geometry_metric import calculate_intersection_angle, bearing_dop
from model.localization import localize

LOG = logging.getLogger(__name__)


@dataclass
class LayoutResult:
    """选择顺序、逐轮收敛记录及所有候选点评分。"""

    selected_points: np.ndarray
    selected_indices: list
    trace: list
    candidate_evaluations: list


def optimize_layout(candidate_points, target_reference, number_points, *, seed=202604004013,
                    angle_error=1.0, grid_resolution=0.28125,
                    arena_radius=1800.0, max_step=2000.0, initial_index=None):
    """固定一个可复现初始点，之后穷举可达候选并贪心加入。

    排序采用字典序：(area_upper, diameter_upper, -最小条件交会角, 候选编号)。
    面积外包优先可避免用不同量纲硬加权；直径和角度仅在面积近似并列时
    提供稳定判据。每个候选的 Ω 均由问题一完整计算。
    """
    candidates = np.asarray(candidate_points, float).reshape(-1, 2)
    target = np.asarray(target_reference, float)
    if not 2 <= number_points <= len(candidates):
        raise ValueError("number_points 必须在 [2,候选点数] 内")
    if initial_index is None:
        initial_index = int(np.random.default_rng(seed).integers(len(candidates)))
    if not 0 <= initial_index < len(candidates):
        raise ValueError("initial_index 越界")
    selected = [initial_index]
    initial_region, initial_metrics, _ = localize(candidates[selected], target,
        angle_error=angle_error, grid_resolution=grid_resolution, arena_radius=arena_radius)
    trace = [{"round": 1, "selected_index": initial_index,
              "candidate_index": initial_index, "area": initial_metrics["area"],
              "area_lower": initial_metrics["area_lower"], "area_upper": initial_metrics["area_upper"],
              "diameter": initial_metrics["diameter"], "diameter_upper": initial_metrics["diameter_upper"],
              "min_intersection_angle": None, "dop": float("inf")}]
    evaluations = []
    while len(selected) < number_points:
        scored = []
        for index in range(len(candidates)):
            if index in selected:
                continue
            trial_indices = selected + [index]
            trial = candidates[trial_indices]
            if not check_route_feasibility(trial, arena_radius=arena_radius, max_step=max_step):
                continue
            region, metrics, _ = localize(trial, target, angle_error=angle_error,
                                          grid_resolution=grid_resolution,
                                          arena_radius=arena_radius)
            if region.status != "sampled":
                continue
            angle = calculate_intersection_angle(trial, target)
            item = {"round": len(trial_indices), "candidate_index": index,
                    "area": metrics["area"], "area_lower": metrics["area_lower"],
                    "area_upper": metrics["area_upper"], "diameter": metrics["diameter"],
                    "diameter_upper": metrics["diameter_upper"],
                    "min_intersection_angle": angle, "dop": bearing_dop(trial, target)}
            evaluations.append(item)
            scored.append((metrics["area_upper"], metrics["diameter_upper"], -angle, index, item))
        if not scored:
            raise RuntimeError("剩余候选点均不满足移动/场地/定位约束")
        _, _, _, winner, item = min(scored)
        selected.append(winner)
        trace.append({**item, "selected_index": winner})
        LOG.info("贪心第%d点：候选%d，面积 %.3f m²，直径≤%.3f m，最小交会角 %.1f°",
                 len(selected), winner, item["area"], item["diameter_upper"], item["min_intersection_angle"])
    return LayoutResult(candidates[selected], selected, trace, evaluations)


def optimize_dop_baseline(candidate_points, target_reference, number_points, *,
                          initial_index=0, arena_radius=1800.0, max_step=2000.0):
    """构造仅供诊断的DOP贪心基线；最终仍由问题一 Ω 指标评价。

    与本文方法共享初始点和候选集，每轮最小化DOP，并以较大最小交会角及
    较小候选编号破除并列。该函数不会替代 optimize_layout。
    """
    candidates = np.asarray(candidate_points, float).reshape(-1, 2)
    target = np.asarray(target_reference, float)
    if not 2 <= number_points <= len(candidates):
        raise ValueError("number_points 必须在 [2,候选点数] 内")
    selected = [initial_index]
    while len(selected) < number_points:
        scored = []
        for index in range(len(candidates)):
            if index in selected:
                continue
            trial = candidates[selected + [index]]
            if not check_route_feasibility(trial, arena_radius=arena_radius, max_step=max_step):
                continue
            dop = bearing_dop(trial, target)
            angle = calculate_intersection_angle(trial, target)
            scored.append((dop, -angle, index))
        if not scored:
            raise RuntimeError("DOP基线无可达候选点")
        selected.append(min(scored)[2])
    return candidates[selected], selected
