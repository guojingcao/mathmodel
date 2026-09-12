# -*- coding: utf-8 -*-
"""问题三: "减少覆盖点数(k)"是否值得 —— 在**新透镜判据**下重测环族最优点。

下界分析(lower_bound.py)显示: 本方案的检测项是主要可压缩项(184 次 vs 下界 68 次),
而检测次数上限 ≈ 20×k(k = 覆盖点数)。环族覆盖最优解是 n=7(r≈997 m, 开路 6 189 m,
检测上限 160)而非当前 n=9(r=1150, 开路 7 443, 上限 200)。但"小环"的定位几何更差
(补测/归航更多) —— 旧结论"几何惩罚 > 检测节省"是在**透镜判据改进之前**测的, 故需重测。

用法: python ring_k_test.py [案例数=1000] [种子=8821]
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

PHASES = ("coverage", "supplement", "on_way", "queue_clear", "homing",
          "queue_homing", "on_way_homing")
# (名称, 环半径, 环点数, 覆盖最优半径?) —— r=997/1123/1150/1200
ARMS = [("n=7, r=997(覆盖最优)", 997.2, 7),
        ("n=6, r=1123(覆盖边界)", 1123.0, 6),
        ("n=9, r=1150(当前候选)", 1150.0, 9),
        ("n=8, r=938(覆盖最优)", 938.1, 8)]


def run_arm(r, n, n_case, seed):
    rows = []
    old_sp = rb.Problem3Robot.__dict__.get("search_points")

    def pts_of(rr, nn):
        out = [(0.0, 0.0)]
        for k in range(nn):
            a = k*(360.0/nn)*math.pi/180.0
            out.append((rr*math.cos(a), rr*math.sin(a)))
        return out
    rb.Problem3Robot.search_points = staticmethod(lambda P=pts_of(r, n): list(P))
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
        if old_sp is not None:
            rb.Problem3Robot.search_points = old_sp
        else:
            delattr(rb.Problem3Robot, "search_points")
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 8821
    print(f"透镜判据已开启(DOP_PRESCREEN={rb.Problem3Robot.DOP_PRESCREEN}, "
          f"LENS_TRAVEL_W={rb.Problem3Robot.LENS_TRAVEL_W}); 配对 {n_case} 例(种子 {seed})")
    print(f"{'臂':<24}{'点':>4}{'T(s)':>8}{'移动':>8}{'检测':>7}{'换频':>7}{'覆盖':>7}"
          f"{'补测':>7}{'顺路':>7}{'归航':>7}{'失败':>6}{'全清':>9}")
    res = {}
    for name, r, n in ARMS:
        rows = run_arm(r, n, n_case, seed)
        res[name] = rows
        T = np.mean([x["T"] for x in rows])
        print(f"{name:<24}{n+1:>4}{T:>8.0f}{np.mean([x['dist'] for x in rows]):>8.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}{np.mean([x['sw'] for x in rows]):>7.1f}"
              f"{np.mean([x['coverage'] for x in rows]):>7.0f}"
              f"{np.mean([x['supplement'] for x in rows]):>7.1f}"
              f"{np.mean([x['on_way'] for x in rows]):>7.0f}"
              f"{np.mean([x['homing']+x['queue_homing']+x['on_way_homing'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>6d}/{len(rows)}", flush=True)
    base = np.array([x["T"] for x in res[ARMS[2][0]]], float)     # 当前候选 n=9 为基准
    print(f"\n相对当前候选(n=9, r=1150)的配对差:")
    for name, r, n in ARMS:
        if name == ARMS[2][0]:
            continue
        a = np.array([x["T"] for x in res[name]], float)
        d = a - base
        se = d.std(ddof=1)/math.sqrt(len(d))
        dm = (np.mean([x["meas"] for x in res[name]])
              - np.mean([x["meas"] for x in res[ARMS[2][0]]]))
        print(f"  {name:<24} ΔT {d.mean():+7.1f} s ({100*d.mean()/base.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+6.0f},{d.mean()+1.96*se:+6.0f}]  变快 "
              f"{100*(d<0).mean():4.1f} %  Δ检测 {dm:+5.1f} 次")
    out = os.path.join(HERE, "results", "ring_k_test.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 环族 k(点数)重测(透镜已开): 逐案\n")
        fh.write("arm,k,n_src,T,dist,meas,switch,clear,fail,coverage,supplement,on_way,homing\n")
        for name, r, n in ARMS:
            for k, x in enumerate(res[name]):
                fh.write("%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f\n" % (
                    name, n+1, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"],
                    x["clr"], x["fail"], x["coverage"], x["supplement"], x["on_way"],
                    x["homing"]+x["queue_homing"]+x["on_way_homing"]))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
