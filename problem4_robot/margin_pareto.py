# -*- coding: utf-8 -*-
"""P4-E: 站集"安全余量"帕累托 —— 证书可行窗口与代价。

V4 的 27 点同时受**两条连续证书约束**夹住, 把站集整体缩放 λ 倍即可同时压/放两侧余量:

  * λ 变大: 凸包变大 -> 支撑边余量(凸包严格包含圆盘)变大, 但**盘内最大边也随之变大**,
    一旦超过 1000 m, "源在三角形内则三顶点均可接收"的前提失效 -> 保证作废;
  * λ 变小: 盘内最大边变小(接收侧更安全), 但凸包收缩 -> 支撑边余量趋零, 圆盘露出凸包 ->
    三角形覆盖不再包含整个圆盘 -> 保证作废。

因此"安全余量"不是一个数, 而是一条**可行窗口**: λ ∈ [λ_min, λ_max]。本脚本扫出该窗口,
并在窗口内给出虚拟时间代价(逐场景配对), 回答"V4 究竟有多大的几何缓冲"。

用法: python margin_pareto.py [配对案例数=60] [配对种子=4242]
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import robot4 as r4                                     # noqa: E402
import selfcheck4 as sc4                                # noqa: E402
import station_opt as so                                # noqa: E402
from ablation4_mecfreeze import FixedEnv                # noqa: E402
from simlib import case_env, sim_time, check_clearance  # noqa: E402

P_DIR = 0.5
LAMBDAS = [0.985, 0.990, 0.992, 0.995, 1.000, 1.010, 1.020, 1.030, 1.040, 1.045, 1.050, 1.060]


def scale(pts, lam):
    return [(x*lam, y*lam) for x, y in pts]


def run_case(seed, k):
    base = case_env(seed, k, directional=True, p_dir=P_DIR)
    env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=True,
                   p_dir=P_DIR, scene_key=(seed, k))
    env.sources = base.sources
    env.ch_by_id = base.ch_by_id
    env.cleared = set()
    cli = sc4.MockClient(env)
    rb = r4.Problem4Robot(cli)
    with contextlib.redirect_stdout(io.StringIO()):
        got = rb.run()
    chk = check_clearance(got, cli, env)
    return sim_time(cli), cli.n_measure, cli.fail, chk["case_full_clear"]


def evaluate(pts, seed, n_case):
    r4.Problem4Robot.MESH_PTS_OVERRIDE = [tuple(p) for p in pts]
    T, meas, fail, miss = [], 0.0, 0.0, 0
    for k in range(n_case):
        t, m, f, ok = run_case(seed, k)
        T.append(t); meas += m; fail += f
        miss += 0 if ok else 1
    T = np.array(T)
    return dict(T=T, mean=float(T.mean()), meas=meas/n_case,
                fail=fail/n_case, miss=miss, n=n_case)


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 4242
    r4.Problem4Robot.ORDER_OVERRIDE = None
    base_pts = [tuple(p) for p in r4.Problem4Robot.MESH_DESIGN_PTS]
    print(f"V4 站集 {len(base_pts)} 点, 整体缩放扫窗; 配对 n={n_case}(种子 {seed})")
    print("证书口径: ① 凸包包含圆盘(支撑边余量 >= 0)  ② 盘内最大边 <= 1000 m(保证三顶点均可接收)")
    print(f"{'lambda':>7}{'支撑余量(m)':>12}{'盘内最大边(m)':>14}{'边余量(m)':>11}"
          f"{'面积缺口':>11}{'重叠(m^2)':>10}{'巡回(m)':>9}{'T(s)':>8}{'检测/例':>9}{'失败/例':>8}{'保证':>7}")
    rows = []
    for lam in LAMBDAS:
        pts = scale(base_pts, lam)
        ok, info, viol = so.certificate(pts, max_edge_lim=1000.0, support_lim=0.0)
        L = so.tsp_len(pts)
        r = evaluate(pts, seed, n_case)
        rows.append((lam, info, L, r, ok))
        print(f"{lam:>7.3f}{info['support_margin']:>12.2f}{info['max_edge_disk']:>14.1f}"
              f"{1000.0-info['max_edge_disk']:>11.1f}{info['gap']:>11.2e}{info['ov']:>10.1e}"
              f"{L:>9.0f}{r['mean']:>8.0f}{r['meas']:>9.1f}{r['fail']:>8.2f}"
              f"{('通过' if ok else '不通过'):>7}", flush=True)
    feas = [x for x in rows if x[4]]
    print("\n可行窗口(保证成立): λ ∈ [%.4f, %.4f]  (共 %d/%d 档通过)"
          % (min(x[0] for x in feas), max(x[0] for x in feas), len(feas), len(rows)))
    for lam, info, L, r, ok in rows:
        if not ok:
            why = []
            if info["support_margin"] < 0:
                why.append("圆盘露出凸包(支撑余量 %.2f m)" % info["support_margin"])
            if info["max_edge_disk"] > 1000.0:
                why.append("盘内最大边 %.1f m 超过 1000 m(三顶点接收前提失效)" % info["max_edge_disk"])
            if info["gap"] > 1e-6*max(1.0, info["a_hull"]):
                why.append("三角剖分出现面积缺口 %.3e m^2(网格构造容差 MESH_A=970 的邻接阈值约 "
                           "999.1 m 先于 1000 m 接收上限失效)" % info["gap"])
            if info["ov"] > 1e-6*max(1.0, info["a_hull"]):
                why.append("三角形两两内部重叠 %.3e m^2" % info["ov"])
            print("   λ=%.3f 不通过: %s" % (lam, "; ".join(why) if why else "未归类违规"))
    b = [x for x in rows if abs(x[0]-1.0) < 1e-9][0]
    print("\n相对 λ=1.000 的配对差(仅可行档):")
    for lam, info, L, r, ok in rows:
        if not ok or abs(lam-1.0) < 1e-9:
            continue
        d = r["T"] - b[3]["T"]
        se = d.std(ddof=1)/math.sqrt(len(d))
        print("   λ=%.3f: ΔT %+7.1f s (%+.2f%%)  CI [%+.0f,%+.0f]  支撑余量 %+.2f m  "
              "盘内最大边 %+.1f m  巡回 %+.0f m"
              % (lam, d.mean(), 100*d.mean()/b[3]["T"].mean(), d.mean()-1.96*se,
                 d.mean()+1.96*se, info["support_margin"]-b[1]["support_margin"],
                 info["max_edge_disk"]-b[1]["max_edge_disk"], L-b[2]))
    r4.Problem4Robot.MESH_PTS_OVERRIDE = base_pts


if __name__ == "__main__":
    main()
