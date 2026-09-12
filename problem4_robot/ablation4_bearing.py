# -*- coding: utf-8 -*-
"""问题四"归航示向排序"配对实验(固定误差场)。

臂: A = acq(按获取顺序, 原行为); B = err(按 横向误差上界 d̂·sin1° 再按移动代价 排序)。

输出(按采纳条件组织): 全清率/未解决、总时间均值与配对差(CI/胜率)、P90/P95/最大、
清除次数与失败次数、归航 episode 的数量/移动/检测/耗时、首条示向命中率、
"换了后续示向才成功"的比例。

用法: python ablation4_bearing.py [n=400]
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


def _ep_sum(eps, key):
    v = [e.get(key) for e in eps if isinstance(e.get(key), (int, float))]
    return float(sum(v)) if v else 0.0


def run_arm(order, p_dir, n, seed):
    rows = []
    old = r4.Problem4Robot.BEARING_ORDER
    r4.Problem4Robot.BEARING_ORDER = order
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
            eps = list(getattr(cli, "homing_diag", []))
            tried = [e for e in eps if e.get("bearings_tried")]
            rows.append(dict(
                T=sim_time(cli), cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                n_src=chk["n_src"], dist=cli.dist, meas=cli.n_measure, fail=cli.fail,
                clear_n=cli.n_clear, unresolved=len(chk.get("missing_src") or []),
                consistent=chk["consistent"], eps=len(eps),
                ep_move=_ep_sum(eps, "movement_distance_m"),
                ep_meas=_ep_sum(eps, "measure_count"),
                ep_time=_ep_sum(eps, "virtual_time_s"),
                first_ok=sum(1 for e in tried if e.get("cleared_at_bearing") == 0),
                later_ok=sum(1 for e in tried
                             if isinstance(e.get("cleared_at_bearing"), int)
                             and e["cleared_at_bearing"] > 0)))
    finally:
        r4.Problem4Robot.BEARING_ORDER = old
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
    print(f"问题四 · 归航示向排序 · 固定误差场配对实验 (n={n}/档)\n")
    for p_dir in (0.5, 1.0):
        res = {"acq": run_arm("acq", p_dir, n, seed), "err": run_arm("err", p_dir, n, seed)}
        A, B = res["acq"], res["err"]
        p = paired([r["T"] for r in B], [r["T"] for r in A])
        for tag, rows in (("A 获取顺序(acq)", A), ("B 横向误差排序(err)", B)):
            T = np.array([r["T"] for r in rows])
            print("%-22s 全清 %6.2f%%  漏清 %2d  均值 %7.0f s  P90 %7.0f  P95 %7.0f  最大 %7.0f"
                  % (tag, 100*np.mean([r["cr"] for r in rows]),
                     sum(r["miss"] for r in rows), T.mean(),
                     np.percentile(T, 90), np.percentile(T, 95), T.max()))
            print("%-22s   清除/例 %5.1f  失败/例 %5.1f  归航ep/例 %4.1f  归航耗时 %5.0f s  "
                  "归航移动 %6.0f m  归航检测 %5.1f  首条示向成功 %4.1f  后续示向成功 %4.1f"
                  % ("", np.mean([r["clear_n"] for r in rows]),
                     np.mean([r["fail"] for r in rows]), np.mean([r["eps"] for r in rows]),
                     np.mean([r["ep_time"] for r in rows])/max(1, len(rows)),
                     np.mean([r["ep_move"] for r in rows])/max(1, len(rows)),
                     np.mean([r["ep_meas"] for r in rows])/max(1, len(rows)),
                     np.mean([r["first_ok"] for r in rows]),
                     np.mean([r["later_ok"] for r in rows])))
        print("   配对: Δ = %+.1f s (%.2f%%)  95%%CI [%.0f,%.0f]  变快比例 %.1f%%  中位 Δ %+.0f s"
              % (p["mean"], 100*p["mean"]/np.mean([r["T"] for r in A]), p["lo"], p["hi"],
                 100*p["win"], p["med"]))
        print("   未解决: A %d, B %d; 互核不一致: A %d, B %d\n"
              % (sum(r["unresolved"] for r in A), sum(r["unresolved"] for r in B),
                 sum(0 if r["consistent"] else 1 for r in A),
                 sum(0 if r["consistent"] else 1 for r in B)))
    print("采纳条件: 全清率不下降 + 归航开销(时间/移动/检测/失败次数)下降 + 最大时间不恶化。")


if __name__ == "__main__":
    main()
