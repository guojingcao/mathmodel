# -*- coding: utf-8 -*-
"""问题四"邻域试探"四臂比较(固定误差场配对)。

A 当前12点      NEIGHBOR_MODE=rings   (8 m/15 m 两圈各 6 点)
B 仅两侧2点     NEIGHBOR_MODE=normal2, NORMAL2_STOP=True  (失败即结束本次归航)
C 不试邻域      NEIGHBOR_MODE=none    (归航点失败立即换下一条示向)
D 两侧2点后换示向 NEIGHBOR_MODE=normal2, NORMAL2_STOP=False

输出: 全清率/漏清/未解决、总时间均值与对 A 的配对差(CI/胜率)、P90/P95/最大、
清除/失败次数、以及按"由谁清除"的来源分布(顺路/队列/归航点/邻域或法线)。

用法: python ablation4_neighbor.py [n=300]
"""
import contextlib
import collections
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


def src_class(src):
    s = str(src)
    if "邻域" in s or "法线" in s:
        return "邻域/法线"
    if "归航" in s or "homing" in s:
        return "归航点"
    if s.startswith("on_way"):
        return "顺路"
    return "队列"


def run_arm(mode, stop, p_dir, n, seed):
    rows = []
    old = (r4.Problem4Robot.NEIGHBOR_MODE, r4.Problem4Robot.NORMAL2_STOP)
    r4.Problem4Robot.NEIGHBOR_MODE, r4.Problem4Robot.NORMAL2_STOP = mode, stop
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
            cls = collections.Counter(src_class(d.get("src")) for d in cd)
            rows.append(dict(
                T=sim_time(cli), cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                n_src=chk["n_src"], dist=cli.dist, meas=cli.n_measure, fail=cli.fail,
                clear_n=cli.n_clear, unresolved=len(chk.get("missing_src") or []),
                consistent=chk["consistent"], by=cls))
    finally:
        r4.Problem4Robot.NEIGHBOR_MODE, r4.Problem4Robot.NORMAL2_STOP = old
    return rows


def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    n = len(d)
    m = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd/math.sqrt(n) if n else 0.0
    return dict(mean=m, lo=m-1.96*se, hi=m+1.96*se, win=float((d < 0).mean()),
                med=float(np.median(d)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = 3026
    arms = [("A 当前12点", "rings", False), ("B 仅两侧2点", "normal2", True),
            ("C 不试邻域", "none", False), ("D 两侧2点后换示向", "normal2", False)]
    print(f"问题四 · 邻域试探四臂比较 · 固定误差场配对 (n={n}/档)\n")
    for p_dir in (0.5, 1.0):
        res = {name: run_arm(m, s, p_dir, n, seed) for name, m, s in arms}
        base = res[arms[0][0]]
        print("=== 定向比例 %.0f%% ===" % (100*p_dir))
        print("%-20s%8s%6s%9s%10s%16s%9s%9s%9s%7s%7s" % (
            "臂", "全清率", "漏清", "均值(s)", "vs A", "95%CI", "P90", "P95", "最大",
            "清除", "失败"))
        for name, _m, _s in arms:
            rows = res[name]
            T = np.array([r["T"] for r in rows])
            p = paired([r["T"] for r in rows], [r["T"] for r in base])
            tag = "—" if name == arms[0][0] else "%+10.1f" % p["mean"]
            ci = "—" if name == arms[0][0] else "[%.0f,%.0f]" % (p["lo"], p["hi"])
            print("%-20s%7.1f%%%6d%9.0f%10s%16s%9.0f%9.0f%9.0f%7.1f%7.1f" % (
                name, 100*np.mean([r["cr"] for r in rows]),
                sum(r["miss"] for r in rows), T.mean(), tag, ci,
                np.percentile(T, 90), np.percentile(T, 95), T.max(),
                np.mean([r["clear_n"] for r in rows]), np.mean([r["fail"] for r in rows])))
        for name, _m, _s in arms:
            rows = res[name]
            c = collections.Counter()
            for r in rows:
                c.update(r["by"])
            tot = sum(c.values())
            print("   %-20s 清除来源: %s  未解决 %d  互核异常 %d" % (
                name, " ".join(f"{k}={v}({100*v/tot:.0f}%)" for k, v in c.most_common()),
                sum(r["unresolved"] for r in rows),
                sum(0 if r["consistent"] else 1 for r in rows)))
        for name, _m, _s in arms[1:]:
            p = paired([r["T"] for r in res[name]], [r["T"] for r in base])
            print("   配对(A vs %s): Δ %+.1f s (%.2f%%)  胜率 %.1f%%  中位 %+.0f s" % (
                name.split()[0], p["mean"], 100*p["mean"]/np.mean([r["T"] for r in base]),
                100*p["win"], p["med"]))
        print()
    print("采纳条件: 全清率不下降 + 失败次数下降 + 最大时间不恶化 + 总时间配对差不劣于 -0.5%。")


if __name__ == "__main__":
    main()
