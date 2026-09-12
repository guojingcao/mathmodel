# -*- coding: utf-8 -*-
"""问题四"顺路 LS 失败后暂缓归航"配对实验(固定误差场)。

臂: A = 当前基线(顺路失败即归航); B = 暂缓归航(继续扫描, 扫描后重新定位)。
两臂共用固定误差场 err(scene,ch,x,y)(与动作顺序无关), 场景按案例编号派生 => 逐案例真配对。

输出(按采纳条件组织): 全清率/未解决、总时间均值与配对差(CI/胜率)、P90/P95/最大、
清除尝试与失败分类、顺路失败源数、暂缓次数、移动与检测、每源口径。

用法: python ablation4_defer.py [n=400]
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                  # noqa: E402

import robot4 as r4                                 # noqa: E402
import selfcheck4 as sc4                            # noqa: E402
from ablation4_mecfreeze import FixedEnv            # noqa: E402
from simlib import case_env, sim_time, check_clearance   # noqa: E402


def run_arm(defer, p_dir, n, seed):
    rows = []
    old = r4.Problem4Robot.DEFER_ONWAY_HOMING
    r4.Problem4Robot.DEFER_ONWAY_HOMING = defer
    try:
        for k in range(n):
            base = case_env(seed, k, directional=True, p_dir=p_dir)
            env = FixedEnv(np.random.default_rng(0), n_src=base.n_src,
                           directional=True, p_dir=p_dir, scene_key=(seed, k))
            env.sources = base.sources
            env.ch_by_id = base.ch_by_id
            env.cleared = set()
            cli = sc4.MockClient(env)
            rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                got = rb.run()
            chk = check_clearance(got, cli, env)
            cd = list(getattr(cli, "clear_diag", []))
            onway = [d for d in cd if str(d.get("src", "")).startswith("on_way")]
            ow_fail = [d for d in onway if d.get("result") != "success"]
            first = [d for d in cd if str(d.get("src", "")).startswith("on_way")
                     or str(d.get("src", "")).startswith("queue")]
            rows.append(dict(
                T=sim_time(cli), cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                n_src=chk["n_src"], dist=cli.dist, meas=cli.n_measure, fail=cli.fail,
                clear_n=cli.n_clear, defer=rb.n_defer,
                onway_fail=len(ow_fail),
                unresolved=len(chk.get("missing_src") or []),
                consistent=chk["consistent"]))
    finally:
        r4.Problem4Robot.DEFER_ONWAY_HOMING = old
    return rows


def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    n = len(d)
    m = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd/math.sqrt(n) if n else 0.0
    return dict(mean=m, lo=m-1.96*se, hi=m+1.96*se, win=float((d < 0).mean()),
                med=float(np.median(d)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = 3026
    print(f"问题四 · 顺路 LS 失败后暂缓归航 · 固定误差场配对实验 (n={n}/档)\n")
    for p_dir in (0.5, 1.0):
        res = {False: run_arm(False, p_dir, n, seed), True: run_arm(True, p_dir, n, seed)}
        A, B = res[False], res[True]
        p = paired([r["T"] for r in B], [r["T"] for r in A])
        for tag, rows in (("A 当前基线", A), ("B 暂缓归航", B)):
            T = np.array([r["T"] for r in rows])
            print("%-12s 全清 %6.2f%%  漏清 %2d  均值 %7.0f s  P90 %7.0f  P95 %7.0f  最大 %7.0f  "
                  "移动 %6.0f m  检测 %6.1f  清除 %5.1f  失败 %5.1f  顺路失败 %4.1f  暂缓 %4.1f"
                  % (tag, 100*np.mean([r["cr"] for r in rows]),
                     sum(r["miss"] for r in rows), T.mean(),
                     np.percentile(T, 90), np.percentile(T, 95), T.max(),
                     np.mean([r["dist"] for r in rows]), np.mean([r["meas"] for r in rows]),
                     np.mean([r["clear_n"] for r in rows]), np.mean([r["fail"] for r in rows]),
                     np.mean([r["onway_fail"] for r in rows]),
                     np.mean([r["defer"] for r in rows])))
        print("   配对: Δ = %+.1f s (%.2f%%)  95%%CI [%.0f,%.0f]  变快比例 %.1f%%  中位 Δ %+.0f s"
              % (p["mean"], 100*p["mean"]/np.mean([r["T"] for r in A]), p["lo"], p["hi"],
                 100*p["win"], p["med"]))
        print("   未解决频道: A %d 个, B %d 个; 互核不一致: A %d, B %d\n"
              % (sum(r["unresolved"] for r in A), sum(r["unresolved"] for r in B),
                 sum(0 if r["consistent"] else 1 for r in A),
                 sum(0 if r["consistent"] else 1 for r in B)))
    print("采纳条件: 全清率不下降 + 困难源恢复时间显著下降 + 最大时间不恶化。")


if __name__ == "__main__":
    main()
