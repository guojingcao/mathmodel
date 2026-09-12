# -*- coding: utf-8 -*-
"""单环"半径–覆盖余量–时间"帕累托: 用最便宜的方式买覆盖余量。

背景(修正一处细节): 均匀 n 点环的**环单独**最优半径是 r = 1800cos(π/n)(最坏 1800sin(π/n)),
但我们的覆盖集**还含原点**。此时 ρ<=1000 由原点负责, 对 ρ∈[1000,1800] 最近环点距离为
    d(ρ) = sqrt(ρ² + r² - 2ρr·cos(π/n)),
它在 ρ*=r·cos(π/n) 处取最小、在区间两端取最大; 两端相等时最坏最小:
    d(1000) = d(1800)  =>  r = (1800²-1000²) / (2·(1800-1000)·cos(π/n)) = 1400/cos(π/n)
(n=9: r≈1490 m, 最坏 647.8 m)。
本脚本对 n=9 扫 r, 逐点给出(最坏接收, 余量, 开路巡回)并用 300 例配对实测总时间。

用法: python ring_radius_margin_ab.py [案例数=300] [种子=8452]
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
from simlib import case_env, sim_time, check_clearance, phase_time   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
from verify_ring_supp import FixedEnv                   # noqa: E402

R_AREA, R_GUAR = 1800.0, 1000.0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 9      # 环点数(第3个参数; 现行部署为 8)
PHASES = ("coverage", "supplement", "on_way", "queue_clear", "homing",
          "on_way_homing", "queue_homing", "recovery")


def ring_set(r, n=N, origin=True):
    pts = ([(0.0, 0.0)] if origin else [])
    for k in range(n):
        a = (360.0/n)*k*math.pi/180.0
        pts.append((r*math.cos(a), r*math.sin(a)))
    return pts


def worst_analytic(r, n=N):
    """{原点} ∪ n 点环 的盘内最坏接收距离(闭式): max over ρ∈[0,1800]。"""
    c = math.cos(math.pi/n)
    if 1000.0 <= r*c <= R_AREA:                     # 极小点在区间内 -> 端点为最大
        return max(R_AREA*0 + math.sqrt(R_AREA**2 + r*r - 2*R_AREA*r*c),
                   math.sqrt(1000.0**2 + r*r - 2*1000.0*r*c))
    return max(math.sqrt(R_AREA**2 + r*r - 2*R_AREA*r*c),
               math.sqrt(1000.0**2 + r*r - 2*1000.0*r*c))


def worst_sampled(pts, n_ang=720, n_rad=450):
    worst = 0.0
    for ia in range(n_ang):
        a = 2*math.pi*ia/n_ang
        ca, sa = math.cos(a), math.sin(a)
        for ir in range(1, n_rad+1):
            rho = R_AREA*ir/n_rad
            gx, gy = rho*ca, rho*sa
            d = min(math.hypot(gx-p[0], gy-p[1]) for p in pts)
            if d > worst:
                worst = d
    return worst


def route_len(pts):
    seq = rb.Problem3Robot._optimal_open_path(list(pts), (0.0, 0.0))
    L = math.hypot(pts[seq[0]][0], pts[seq[0]][1])
    for a, b in zip(seq, seq[1:]):
        L += math.hypot(pts[b][0]-pts[a][0], pts[b][1]-pts[a][1])
    return L


def run_arm(pts, n_case, seed):
    old = getattr(rb.Problem3Robot, "COVER_PTS", None)
    rb.Problem3Robot.COVER_PTS = [tuple(p) for p in pts] if pts else None
    rows = []
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
            rows.append(dict(T=sim_time(cli), dist=cli.dist, meas=cli.n_measure,
                             sw=cli.n_switch, clr=cli.n_clear, fail=cli.fail,
                             full=chk["case_full_clear"], n_src=chk["n_src"],
                             **{p: phase_time(cli, p) for p in PHASES}))
    finally:
        rb.Problem3Robot.COVER_PTS = old
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 8452
    r_bal = 1400.0/math.cos(math.pi/N)
    print(f"n={N} 单环 + 原点; 两端等值的最优半径 r = 1400/cos(pi/{N}) = {r_bal:.1f} m")
    if len(sys.argv) > 3:            # n 显式给定 -> 扫"向下"(省时间)与适度向上的档位
        rs = [985.0, 1005.0, 1030.0, 1075.0, 1150.0, 1200.0]
    else:
        rs = [1030.0, 1200.0, 1300.0, r_bal, 1600.0, 1800.0*math.cos(math.pi/N)]
    print(f"{'r(m)':>8}{'最坏(闭式)':>11}{'最坏(采样)':>11}{'余量(m)':>9}{'巡回(m)':>9}"
          f"{'T(s)':>8}{'移动':>8}{'检测':>7}{'失败':>6}{'全清':>10}")
    res = {}
    for r in rs:
        pts = ring_set(r)
        wa, ws = worst_analytic(r), worst_sampled(pts)
        rows = run_arm(pts, n_case, seed)
        res[r] = rows
        print(f"{r:>8.0f}{wa:>11.2f}{ws:>11.2f}{R_GUAR-wa:>9.1f}{route_len(pts):>9.0f}"
              f"{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['dist'] for x in rows]):>8.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
    b = np.array([x["T"] for x in res[1030.0]], float)
    print(f"\n相对现行(r=1030)的配对差:")
    for r in rs[1:]:
        a = np.array([x["T"] for x in res[r]], float)
        d = a - b
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"  r={r:>6.0f}  余量 {R_GUAR-worst_analytic(r):>6.1f} m  ΔT {d.mean():+8.1f} s "
              f"({100*d.mean()/b.mean():+6.2f} %)  CI [{d.mean()-1.96*se:+7.0f},"
              f"{d.mean()+1.96*se:+7.0f}]  变快 {100*(d<0).mean():4.1f} %")
    out = os.path.join(HERE, "results", "ring_radius_margin_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 单环半径-余量-时间 帕累托(配对逐案)\n")
        fh.write("r,k,n_src,T,dist,meas,switch,clear,fail,full," + ",".join(PHASES) + "\n")
        for r in rs:
            for k, x in enumerate(res[r]):
                fh.write("%.0f,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%s\n" % (
                    r, k, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"], x["clr"],
                    x["fail"], x["full"], ",".join("%.1f" % x[p] for p in PHASES)))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
