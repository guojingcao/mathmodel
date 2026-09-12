# -*- coding: utf-8 -*-
"""同心双环覆盖集: 设计、覆盖证书复核、巡回长度、与现行方案的配对实验。

设计(用户给定参数): 外环 r1=1250 m × 8 点(0/45/...); 内环 r2=650 m × 8 点(22.5/67.5/...);
共 16 点, **不含原点**。

覆盖证书 = "盘内任一点到最近站点距离 <= 1000 m"。本脚本:
  1) 闭式核对三类最坏点(外环弦中点、内环弦中点、环间) + 1440×900 密集采样复核;
  2) 给出该集合的最短开放巡回(NN + 2-opt, 与机器人实际取点顺序一致);
  3) 配对实验: base(1030x8 环 + 透镜 + 贝叶斯) vs 双环(base 同配置, 只换覆盖集)。

用法: python double_ring_ab.py [案例数=300] [种子=7311]
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
PHASES = ("coverage", "supplement", "on_way", "queue_clear", "homing",
          "queue_homing", "on_way_homing", "recovery")


def double_ring(r1=1250.0, n1=8, r2=650.0, n2=8, phase2_deg=22.5):
    pts = []
    for k in range(n1):
        a = (360.0/n1)*k*math.pi/180.0
        pts.append((r1*math.cos(a), r1*math.sin(a)))
    for k in range(n2):
        a = (phase2_deg + (360.0/n2)*k)*math.pi/180.0
        pts.append((r2*math.cos(a), r2*math.sin(a)))
    return pts


def worst_reception(pts, n_ang=1440, n_rad=900):
    """盘内最坏接收距离(密集采样)。"""
    worst, arg = 0.0, None
    for ia in range(n_ang):
        a = 2*math.pi*ia/n_ang
        ca, sa = math.cos(a), math.sin(a)
        for ir in range(1, n_rad+1):
            rho = R_AREA*ir/n_rad
            gx, gy = rho*ca, rho*sa
            d = min(math.hypot(gx-p[0], gy-p[1]) for p in pts)
            if d > worst:
                worst, arg = d, (gx, gy)
    return worst, arg


def closed_forms(r1=1250.0, n1=8, r2=650.0, n2=8):
    """三类候选最坏点的闭式值(用于与采样互证)。"""
    out = {}
    # 外环相邻点弦中点(在边界 rho=1800 上): 弦半长 + 径向差(弦近似) 与精确圆几何两种
    out["外环弦中点(边界, 精确圆几何)"] = math.sqrt(
        R_AREA**2 + r1**2 - 2*R_AREA*r1*math.cos(math.pi/n1))
    out["外环弦中点(边界, 弦近似)"] = math.sqrt(
        (2*r1*math.sin(math.pi/n1)/2)**2 + (R_AREA - r1*math.cos(math.pi/n1))**2)
    out["内环弦中点(边界, 精确圆几何)"] = math.sqrt(
        R_AREA**2 + r2**2 - 2*R_AREA*r2*math.cos(math.pi/n2))
    # 内环三角形(原点, 相邻内环点) 的外接圆半径 = 内环内最坏点
    out["内环内部最坏(三角形外接圆)"] = (2*r2*math.sin(math.pi/n2))/(2*math.sin(2*math.pi/n2))
    # 环间: 半径取两环中点, 角向取外环弦中点
    rm = 0.5*(r1 + r2)
    out["环间(半径中点, 外环弦中点方向)"] = math.sqrt(
        rm**2 + r1**2 - 2*rm*r1*math.cos(math.pi/n1))
    return out


def route_len(pts):
    seq = rb.Problem4Robot._order_points if False else None
    rb0 = rb.Problem3Robot.__new__(rb.Problem3Robot)
    order = None
    # 复用机器人自己的取点顺序(NN + 开放 2-opt), 保证与实测一致
    old = getattr(rb.Problem3Robot, "COVER_PTS", None)
    rb.Problem3Robot.COVER_PTS = list(pts)
    try:
        seq = rb.Problem3Robot._optimal_open_path(list(pts), (0.0, 0.0))
        L = math.hypot(pts[seq[0]][0], pts[seq[0]][1])
        for a, b in zip(seq, seq[1:]):
            L += math.hypot(pts[b][0]-pts[a][0], pts[b][1]-pts[a][1])
    finally:
        rb.Problem3Robot.COVER_PTS = old
    return L


def run_arm(use_double, n_case, seed):
    old = getattr(rb.Problem3Robot, "COVER_PTS", None)
    rb.Problem3Robot.COVER_PTS = double_ring() if use_double else None
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
                             status=getattr(robot, "exit_status", None),
                             **{p: phase_time(cli, p) for p in PHASES}))
    finally:
        rb.Problem3Robot.COVER_PTS = old
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7311
    dr = double_ring()
    print("=== 同心双环设计复核 ===")
    print(f"  外环 r=1250 × 8 点 + 内环 r=650 × 8 点(交错 22.5°), 共 {len(dr)} 点, 不含原点")
    w, arg = worst_reception(dr)
    print(f"  盘内最坏接收距离(1440×900 密集采样) = {w:.2f} m  余量 {R_GUAR-w:.2f} m  "
          f"最坏点 ({arg[0]:.0f},{arg[1]:.0f})  -> {'通过' if w <= R_GUAR else '不通过'}")
    for k, v in closed_forms().items():
        print(f"  闭式候选 {k:<34} = {v:.2f} m")
    cur = [(0.0, 0.0)] + [(1150.0*math.cos(k*40*math.pi/180), 1150.0*math.sin(k*40*math.pi/180))
                          for k in range(9)]
    cur = rb.Problem3Robot.search_points()
    wc, argc = worst_reception(cur, 720, 450)
    print(f"  现行覆盖集({len(cur)} 点)最坏接收 = {wc:.2f} m 余量 {R_GUAR-wc:.2f} m")
    print(f"  巡回(开放, 从原点): 双环 {route_len(dr):.0f} m  vs  现行 {route_len(cur):.0f} m")

    print(f"\n=== 配对实验({n_case} 例, 种子 {seed}, 同场景 + 固定误差场) ===")
    print(f"{'臂':<16}{'T(s)':>8}{'移动':>9}{'检测':>7}{'换频':>7}{'失败':>6}{'全清':>10}")
    res = {}
    for name, flag in (("base(现行)", False), ("double_ring", True)):
        rows = run_arm(flag, n_case, seed)
        res[name] = rows
        print(f"{name:<16}{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['dist'] for x in rows]):>9.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['sw'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
    b = np.array([x["T"] for x in res["base(现行)"]], float)
    a = np.array([x["T"] for x in res["double_ring"]], float)
    d = a - b
    se = d.std(ddof=1)/math.sqrt(len(d))
    print(f"\n配对: ΔT {d.mean():+.1f} s ({100*d.mean()/b.mean():+.2f} %)  "
          f"CI [{d.mean()-1.96*se:+.0f},{d.mean()+1.96*se:+.0f}]  变快 {100*(d<0).mean():.1f} %  "
          f"中位 {np.median(d):+.0f} s")
    for name in res:
        rows = res[name]
        print(f"  {name:<16} 阶段(s): " + "  ".join(
            f"{p}={np.mean([x[p] for x in rows]):.0f}" for p in PHASES
            if np.mean([x[p] for x in rows]) > 1.0))
    out = os.path.join(HERE, "results", "double_ring_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 同心双环 vs 现行 配对实验逐案\n")
        fh.write("arm,k,n_src,T,dist,meas,switch,clear,fail,full,status,"
                 + ",".join(PHASES) + "\n")
        for name in res:
            for k, x in enumerate(res[name]):
                fh.write("%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%s,%s\n" % (
                    name, k, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"], x["clr"],
                    x["fail"], x["full"], x["status"],
                    ",".join("%.1f" % x[p] for p in PHASES)))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
