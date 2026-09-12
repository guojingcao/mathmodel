# -*- coding: utf-8 -*-
"""问题三: V2 执行方案 vs 现行方案(B3 路线) 的配对实验。

臂:
  base  = 现行方案(1030x8 环 + 鲁棒透镜 + 贝叶斯零增益/计数证书), 即 pre-new-scheme-v1 的默认;
  v2    = 本方案(概率发现 + GDOP/MPC + TSP + 七点兜底);
  v2_nofb = 消融: V2 但把七点兜底环换成现行 1030x8 环(用于归因——收益来自概率层还是兜底几何);
  v2_ringr = 消融: V2 但兜底环半径改为 1200(9 点, 行程更短)。

全部同场景 + 固定误差场配对。

用法: python scheme_v2_ab.py [案例数=300] [种子=6412]
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
sv2 = simlib.load_module("sv2", os.path.join(HERE, "scheme_v2.py"))
sv2.R = rb                                # 让 V2 复用同一份 robot 模块(避免类身份分裂)
from verify_ring_supp import FixedEnv                   # noqa: E402

PHASES = ("prob_discovery", "localize", "cover_fallback", "on_way", "queue_clear", "exit")


def run_case(seed, k, mode, cover_r=None):
    base = case_env(seed, k, directional=False)
    env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=False,
                   scene_key=(seed, k))
    env.sources = base.sources
    env.ch_by_id = base.ch_by_id
    env.cleared = set()
    cli = simlib.SimClient(env)
    if mode == "base":
        robot = rb.Problem3Robot(cli)
    else:
        robot = sv2.SchemeV2Robot(cli)
        if cover_r is not None:
            robot.COVER_R = cover_r          # 实例属性: 必须在 run() 之前设置且不能被还原
    with contextlib.redirect_stdout(io.StringIO()):
        n_ret = robot.run()
    chk = check_clearance(n_ret, cli, env)
    return dict(T=sim_time(cli), dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch,
                clr=cli.n_clear, fail=cli.fail, full=chk["case_full_clear"],
                n_src=chk["n_src"], status=getattr(robot, "exit_status", None),
                **{p: phase_time(cli, p) for p in PHASES})


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 6412
    arms = [("base(现行)", "base", None),
            ("v2(七点 r=1500)", "v2", 1500.0),
            ("v2(兜底 r=1200)", "v2", 1200.0),
            ("v2(兜底 r=1030)", "v2", 1030.0)]
    print(f"配对 {n_case} 例(种子 {seed}), 同场景 + 固定误差场")
    print(f"{'臂':<18}{'T(s)':>8}{'移动':>9}{'检测':>7}{'换频':>7}{'失败':>6}{'全清':>10}"
          f"{'状态':>12}")
    res = {}
    for name, mode, cr in arms:
        rows = [run_case(seed, k, mode, cr) for k in range(n_case)]
        res[name] = rows
        st = sorted(set(str(x["status"]) for x in rows))
        print(f"{name:<18}{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['dist'] for x in rows]):>9.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['sw'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}"
              f"{str(st):>12}", flush=True)
    b = np.array([x["T"] for x in res[arms[0][0]]], float)
    print("\n相对现行方案的配对差:")
    for name, _, _ in arms[1:]:
        a = np.array([x["T"] for x in res[name]], float)
        d = a - b
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"  {name:<18} ΔT {d.mean():+8.1f} s ({100*d.mean()/b.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+7.0f},{d.mean()+1.96*se:+7.0f}]  变快 "
              f"{100*(d<0).mean():4.1f} %")
    # 阶段分解(V2 侧) + 兜底覆盖率
    for name, mode, cr in arms[1:]:
        rows = res[name]
        print(f"\n{name} 阶段均值(s): " + "  ".join(
            f"{p}={np.mean([x[p] for x in rows]):.0f}" for p in PHASES
            if np.mean([x[p] for x in rows]) > 0.5))
        print("  检测 %.1f 次/例 = 概率层+定位+兜底; 每源 %.1f s"
              % (np.mean([x["meas"] for x in rows]),
                 np.sum([x["T"] for x in rows])/np.sum([x["clr"] for x in rows])))
    out = os.path.join(HERE, "results", "scheme_v2_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# V2 方案配对实验逐案\n")
        fh.write("arm,k,n_src,T,dist,meas,switch,clear,fail,full,status,"
                 + ",".join(PHASES) + "\n")
        for name, _, _ in arms:
            for k, x in enumerate(res[name]):
                fh.write("%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%s,%s\n" % (
                    name, k, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"], x["clr"],
                    x["fail"], x["full"], x["status"],
                    ",".join("%.1f" % x[p] for p in PHASES)))
    print(f"\n逐案明细 -> {out}")


if __name__ == "__main__":
    main()
