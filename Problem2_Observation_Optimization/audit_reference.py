# -*- coding: utf-8 -*-
"""问题二"选点信息可获得性"审计(P0-1)。

结论要点(由本脚本给出数据):
  A. 现行实现在选点时使用**参考点 Ŝ**(main.py 传入 [0,0]), 用 localize(trial, Ŝ)
     合成候选点的示向 -> **没有使用真值**, 但**假定了"未来观测指向 Ŝ"**;
  B. 因此它只有在 S = Ŝ 时才完全精确; S ≠ Ŝ 时, 选点所用的"未来观测"是模型预测而非真实观测;
  C. 本脚本量化两件事:
     ① 参考点偏差代价: 用 Ŝ=原点 选出的布局, 当真实源 S 偏离原点时, 实际 Ω 面积如何变化;
     ② 真值辅助的理想化收益上限: 若允许用 S 作为 Ŝ 选点(不可执行), 面积能好多少。
     二者之差 = "理想化布局实验"与"可执行策略"的差距。

用法: python audit_reference.py [重复次数, 默认 30]
"""
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # 使 model/、config 可导入
from config import LayoutConfig, SEED                 # noqa: E402
from model.candidate_points import generate_candidate_points      # noqa: E402
from model.layout_optimizer import optimize_layout                # noqa: E402
from model.localization import localize                          # noqa: E402


def get_cfg():
    """与 main.py 相同的配置对象。"""
    return LayoutConfig()


def scene(seed, repetition):
    """与 experiment/common.py 完全相同的场景抽样(真值在半径 250 m 内)。"""
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition, 2]))
    polar = rng.uniform(0, 2*np.pi)
    target = 250*np.sqrt(rng.random())*np.array([np.cos(polar), np.sin(polar)])
    errors = rng.uniform(-1, 1, 24)
    return target, errors


def area_of(points, target, errors, cfg):
    region, metrics, _ = localize(np.asarray(points, float), np.asarray(target, float),
                                  angle_error=cfg.error_bound_deg,
                                  measurement_errors=errors[:len(points)],
                                  grid_resolution=cfg.evaluation_resolution_m,
                                  initial_grid=cfg.initial_grid,
                                  arena_radius=cfg.arena_radius,
                                  truth_for_evaluation=target)
    return metrics["area"], metrics["area_upper"], metrics["diameter"]


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = LayoutConfig()
    seed = int(SEED)
    cands = generate_candidate_points(radius=cfg.observation_radius,
                                      arena_radius=cfg.arena_radius)
    init = int(np.random.default_rng(seed).integers(len(cands)))
    rows = []
    print(f"候选点 {len(cands)} 个; 固定初始点 index={init}; 重复 {n_rep} 次")
    print("%-6s%12s%12s%12s%12s%12s" % ("rep", "|S|(m)", "Ŝ=原点", "Ŝ=S(理想)", "Δ理想-可执行",
                                        "S偏500m"))
    for rep in range(n_rep):
        S, errors = scene(seed, rep)
        # 臂1: 可执行口径(Ŝ=原点)
        lay_o = optimize_layout(cands, [0, 0], 6, seed=seed, initial_index=init,
                                angle_error=cfg.error_bound_deg,
                                grid_resolution=cfg.evaluation_resolution_m,
                                arena_radius=cfg.arena_radius,
                                max_step=cfg.max_step_m).selected_points
        a_o, _, _ = area_of(lay_o, S, errors, cfg)
        # 臂2: 真值辅助理想化(Ŝ=S) —— 仅作上界, 不可执行
        lay_s = optimize_layout(cands, S, 6, seed=seed, initial_index=init,
                                angle_error=cfg.error_bound_deg,
                                grid_resolution=cfg.evaluation_resolution_m,
                                arena_radius=cfg.arena_radius,
                                max_step=cfg.max_step_m).selected_points
        a_s, _, _ = area_of(lay_s, S, errors, cfg)
        # 偏差探针: 布局按原点选好后, 真值再偏 500 m
        S2 = S + 500.0*np.array([np.cos(np.arctan2(S[1], S[0]) or 0.0),
                                 np.sin(np.arctan2(S[1], S[0]) or 0.0)])
        a_d, _, _ = area_of(lay_o, S2, errors, cfg)
        rows.append((np.linalg.norm(S), a_o, a_s, a_d))
        print("%-6d%12.1f%12.1f%12.1f%+12.1f%12.1f" % (
            rep+1, np.linalg.norm(S), a_o, a_s, a_s-a_o, a_d))
    arr = np.array(rows, float)
    d = arr[:, 2] - arr[:, 1]
    print("\n汇总(面积均值 m²):")
    print(f"  可执行口径(Ŝ=原点)        : {arr[:,1].mean():.1f}")
    print(f"  理想化口径(Ŝ=S, 不可执行) : {arr[:,2].mean():.1f}")
    print(f"  理想化收益                : {d.mean():+.1f} m²  (配对 95% CI "
          f"[{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}])")
    print(f"  真值再偏 500 m 后的面积   : {arr[:,3].mean():.1f} "
          f"(相对可执行口径 {100*(arr[:,3].mean()/arr[:,1].mean()-1):+.1f}%)")
    print("\n结论: 选点不使用真值(无泄漏), 但依赖『未来观测指向 Ŝ』的假设; "
          "上表第二行是该假设完全成立时的上界, 第三行是假设被破坏时的代价。")


if __name__ == "__main__":
    main()
