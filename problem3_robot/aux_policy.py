# -*- coding: utf-8 -*-
"""问题三: 覆盖成本与后续节省的**权衡分析**(Q1) + **条件辅助观测**策略(Q2)。

背景(实机暴露的权衡): 记 ΔT = ΔT_覆盖 − 后续定位/清除节省。当场景本来容易定位、源数少、
或旧方案偶然获得好交会几何时, 后续节省不足以抵消额外的覆盖成本 —— 于是"平均更快"的方案
仍会在不少具体场景变慢。故本脚本回答两个问题, 全部只用**运行时可观测量**(不读隐藏真值):

  Q1 新方案在哪类场景获益/退步: 按"基础扫描后需补测频道数"(可观测)分组, 把 ΔT 精确分解为
     覆盖侧增量(基础环变化 + 辅助点扫描)与后续侧节省(补测/顺路/队列/归航)。
  Q2 额外观测是否必须每局都支付: 保留**经证明的基础覆盖环**(证书不变, 辅助点不参与覆盖/排除
     判据), 仅当"需补测频道数 >= K"时才访问辅助点。由于辅助调度只依赖基础扫描结果,
     同一案例的"辅助增量" Δ_case 与是否触发无关, 故阈值策略的代价可由同一批配对数据**精确合成**:
         T_policy(K) = T_base + Σ_cases Δ_case · 1{supp_task_count >= K}
     并给出"完美预知"下界 Σ min(0, Δ_case) 作为任何条件策略的理论下限。

  Q1+Q2 均不改变最终清除保证: 基础环独立承担覆盖证明, 辅助点只改善定位几何。

用法: python aux_policy.py [配对案例数=1000] [种子=5150]
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import simlib                                           # noqa: E402
from simlib import (case_env, sim_time, check_clearance, config_scope,
                    frozen3, phase_time)                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
from verify_ring_supp import FixedEnv                   # noqa: E402

DEG = math.pi/180.0
AUX3 = [(1200.0*math.cos(a*DEG), 1200.0*math.sin(a*DEG)) for a in (30.0, 150.0, 270.0)]
PHASES = ("coverage", "aux_coverage", "supplement", "on_way", "on_way_homing",
          "queue_clear", "queue_homing", "homing")

ARMS = [
    ("base6(1200x6,无辅助)", dict(RING_R=1200.0, RING_N=6, AUX_PTS=None, AUX_TRIGGER_K=None)),
    ("aux9(基础+3辅助点,总访问)", dict(RING_R=1200.0, RING_N=6, AUX_PTS=AUX3,
                                       AUX_TRIGGER_K=None)),
    ("adopted(1150x9)", dict(RING_R=1150.0, RING_N=9, AUX_PTS=None, AUX_TRIGGER_K=None)),
]


def run_arm(cfg, n_case, seed):
    rows = []
    scope = config_scope((rb.Problem3Robot, frozen3(**cfg)))
    scope.__enter__()
    try:
        for k in range(n_case):
            base = case_env(seed, k, directional=False)
            env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=False,
                           scene_key=(seed, k))
            env.sources = base.sources
            env.ch_by_id = base.ch_by_id
            env.cleared = set()
            cli = simlib.SimClient(env)
            robot = rb.Problem3Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = robot.run()
            chk = check_clearance(n_ret, cli, env)
            rows.append(dict(
                T=sim_time(cli), dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch,
                clr=cli.n_clear, fail=cli.fail, full=chk["case_full_clear"],
                n_src=chk["n_src"],
                need=getattr(robot, "aux_need", None), fired=getattr(robot, "aux_fired", None),
                resolved=getattr(robot, "aux_resolved", None),
                **{p: phase_time(cli, p) for p in PHASES}))
    finally:
        scope.__exit__(None, None, None)
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5150
    print(f"配对案例 {n_case} 例(种子 {seed}), 同一场景 + 固定误差场; "
          f"辅助点 = r=1200 m 的 30/150/270 度(基础环 60 度等分)")
    res = {}
    for name, cfg in ARMS:
        rows = run_arm(cfg, n_case, seed)
        res[name] = rows
        T = np.array([r["T"] for r in rows], float)
        full = sum(r["full"] for r in rows)
        miss = sum(r["n_src"] - r["clr"] for r in rows)
        print(f"\n=== {name} ===")
        print("  T 均值 %7.1f s  P95 %7.0f  移动 %6.0f m  检测 %6.1f  调频 %6.1f  "
              "失败 %4.2f  全清 %d/%d  漏清源 %d"
              % (T.mean(), np.percentile(T, 95), np.mean([r["dist"] for r in rows]),
                 np.mean([r["meas"] for r in rows]), np.mean([r["sw"] for r in rows]),
                 np.mean([r["fail"] for r in rows]), full, len(rows), miss))
        ph = {p: np.mean([r[p] for r in rows]) for p in PHASES}
        print("  阶段均值(s): " + "  ".join(f"{p}={ph[p]:.0f}" for p in PHASES if ph[p] > 0.05))
        need = [r["need"] for r in rows]
        print("  可观测: 需补测频道数 均值 %.2f  分布 %s"
              % (np.mean(need), {int(v): need.count(v) for v in sorted(set(need))}))
        if any(r["fired"] is not None for r in rows):
            print("  辅助访问率 %.1f%%  平均消除补测任务 %.2f 个"
                  % (100*np.mean([1.0 if r["fired"] else 0.0 for r in rows]),
                     np.mean([r["resolved"] or 0 for r in rows])))

    base = res[ARMS[0][0]]
    Tb = np.array([r["T"] for r in base], float)
    print("\n=== Q1: 新方案在哪类场景获益/退步(配对, 按**可观测**的需补测频道数分组) ===")
    for name in (ARMS[1][0], ARMS[2][0]):
        Ta = np.array([r["T"] for r in res[name]], float)
        d = Ta - Tb
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"\n  【{name} vs base6】配对 ΔT = {d.mean():+.1f} s "
              f"({100*d.mean()/Tb.mean():+.2f} %)  95% CI [{d.mean()-1.96*se:+.0f}, "
              f"{d.mean()+1.96*se:+.0f}]  变快比例 {100*(d<0).mean():.1f} %")
        dcov = np.array([(res[name][i]["coverage"] + res[name][i]["aux_coverage"])
                         - (base[i]["coverage"] + base[i]["aux_coverage"]) for i in range(len(d))])
        dsup = np.array([res[name][i]["supplement"] - base[i]["supplement"] for i in range(len(d))])
        donw = np.array([res[name][i]["on_way"] - base[i]["on_way"] for i in range(len(d))])
        dque = np.array([res[name][i]["queue_clear"] - base[i]["queue_clear"] for i in range(len(d))])
        dhom = np.array([(res[name][i]["homing"] + res[name][i]["on_way_homing"]
                          + res[name][i]["queue_homing"])
                         - (base[i]["homing"] + base[i]["on_way_homing"]
                            + base[i]["queue_homing"]) for i in range(len(d))])
        print("  机制分解: 覆盖侧 %+.0f s | 补测 %+.0f s | 顺路 %+.0f s | 队列 %+.0f s "
              "| 归航 %+.0f s (合计 %+.0f s)"
              % (dcov.mean(), dsup.mean(), donw.mean(), dque.mean(), dhom.mean(),
                 (dcov+dsup+donw+dque+dhom).mean()))
        need = np.array([r["need"] for r in base], float)
        print("  %-10s%8s%12s%12s%10s" % ("需补测数", "局数", "ΔT均值(s)", "变快比例", "覆盖侧(s)"))
        for v in sorted(set(need.tolist())):
            m = need == v
            print("  %-10s%8d%12.1f%11.1f%%%10.0f"
                  % (int(v), int(m.sum()), d[m].mean(), 100*(d[m] < 0).mean(), dcov[m].mean()))
        cc = float(np.corrcoef(need, d)[0, 1])
        print(f"  相关系数 corr(需补测数, ΔT) = {cc:+.3f}  (负值表示'越难定位, 新方案越有利')")

    print("\n=== Q2: 条件辅助观测策略(阈值策略代价由配对数据精确合成) ===")
    name = ARMS[1][0]
    d = np.array([res[name][i]["T"] - base[i]["T"] for i in range(len(base))], float)
    need = np.array([r["need"] for r in base], float)
    print(f"  {'阈值K':>7}{'访问率':>9}{'ΔT均值(s)':>13}{'ΔT占总时间':>12}{'变快比例':>10}")
    best = None
    for K in (0, 1, 2, 3, 4, 5, 6, 8, 10, 10**9):
        m = need >= K if K < 10**9 else np.zeros(len(d), bool)
        dd = d*m
        rate = 100*m.mean()
        print(f"  {('总是' if K==0 else ('从不' if K>=10**9 else K)):>7}{rate:>8.1f}%"
              f"{dd.mean():>13.1f}{100*dd.mean()/Tb.mean():>11.2f}%{100*(dd<0).mean():>9.1f}%")
        if best is None or dd.mean() < best[1]:
            best = (K, dd.mean(), rate)
    oracle = np.minimum(d, 0.0)
    print(f"  最优阈值 K*={best[0]}(访问率 {best[2]:.1f}%) → ΔT {best[1]:+.1f} s "
          f"({100*best[1]/Tb.mean():+.2f} %)")
    print(f"  **完美预知下界**(每案只在该案确实划算时才访问): {oracle.mean():+.1f} s "
          f"({100*oracle.mean()/Tb.mean():+.2f} %)")
    Ta = np.array([r["T"] for r in res[ARMS[2][0]]], float)
    if best[0] < 10**9 and best[0] != 0:
        m = need >= best[0]
        pol = Tb + d*m
        dd = pol - Ta
        se = dd.std(ddof=1)/math.sqrt(len(dd))
        print(f"  最优阈值策略 vs adopted(1150x9): ΔT = {dd.mean():+.1f} s "
              f"({100*dd.mean()/Ta.mean():+.2f} %)  CI [{dd.mean()-1.96*se:+.0f}, "
              f"{dd.mean()+1.96*se:+.0f}]")

    # ---- 逐案明细入库(含可观测特征, 供复核与后续分析) ----
    out = os.path.join(HERE, "results", "aux_policy_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 问题三 权衡分析逐案明细(同一场景 + 固定误差场; need = 基础扫描后需补测频道数, "
                 "运行时可观测)\n")
        fh.write("k,n_src,need_base6,T_base6,T_aux9,T_adopted,dT_adopted,dT_aux9,"
                 "cov_base6,cov_adopted,sup_base6,sup_adopted,fired,resolved\n")
        for i in range(len(base)):
            fh.write("%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s,%s\n" % (
                i, base[i]["n_src"], base[i]["need"], base[i]["T"],
                res[ARMS[1][0]][i]["T"], res[ARMS[2][0]][i]["T"],
                res[ARMS[2][0]][i]["T"] - base[i]["T"],
                res[ARMS[1][0]][i]["T"] - base[i]["T"],
                base[i]["coverage"], res[ARMS[2][0]][i]["coverage"],
                base[i]["supplement"], res[ARMS[2][0]][i]["supplement"],
                res[ARMS[1][0]][i]["fired"], res[ARMS[1][0]][i]["resolved"]))
    print(f"\n逐案明细已写入 {out}")

    # ---- 若允许"逐案自适应选环"(完美预知): 价值上界 ----
    d9 = np.array([res[ARMS[2][0]][i]["T"] - base[i]["T"] for i in range(len(base))], float)
    print("=== 上界: 若可逐案自适应选择(6 点环 / 9 点环) ===")
    print(f"  总是用 9 点环: {d9.mean():+.1f} s ({100*d9.mean()/Tb.mean():+.2f} %)")
    print(f"  总是用 6 点环: {0.0:+.1f} s")
    print(f"  **完美预知逐案选环**: {np.minimum(d9, 0.0).mean():+.1f} s "
          f"({100*np.minimum(d9, 0.0).mean()/Tb.mean():+.2f} %)  "
          f"→ 相对【总是 9 点环】再多省 {np.minimum(d9, 0.0).mean() - d9.mean():.1f} s")
    print(f"  其中 9 点环更慢的案例占 {100*(d9 > 0).mean():.1f} %(平均慢 "
          f"{d9[d9 > 0].mean():.0f} s), 更快案例占 {100*(d9 < 0).mean():.1f} %")


if __name__ == "__main__":
    main()
