# -*- coding: utf-8 -*-
"""δ（顺路插入阈值）的 2x2 交叉验证: 旧配置(7 点 1200x6, 无透镜/无贝叶斯) vs 现行配置,
各自在 δ=300 与 δ=500 下, 用**同一批场景**(同种子)配对测。

目的: 判定"δ=500 更优 −140~−171 s"这一旧结论 (仅有论文陈述、无存档证据文件) 是否
     在旧配置下可复现, 以及现行配置下 δ 的最优是否已移动。

用法: python delta_cross_ab.py [案例数=300] [种子=9573]
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
from simlib import case_env, sim_time, check_clearance   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
from verify_ring_supp import FixedEnv                   # noqa: E402

OLD = dict(RING_R=1200.0, RING_N=6, DOP_PRESCREEN=False, LENS_TRAVEL_W=None,
           PROB_SKIP_IG=False, PROB_COUNT_CERT=False)
NEW = dict(RING_R=1030.0, RING_N=8, DOP_PRESCREEN=True, LENS_TRAVEL_W=0.05,
           PROB_SKIP_IG=True, PROB_COUNT_CERT=True)
FLAGS = tuple(OLD.keys()) + ("ON_WAY_DELTA",)


def run_arm(cfg, delta, n_case, seed):
    old = {k: getattr(rb.Problem3Robot, k, None) for k in FLAGS}
    for k, v in cfg.items():
        setattr(rb.Problem3Robot, k, v)
    rb.Problem3Robot.ON_WAY_DELTA = delta
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
                             fail=cli.fail, full=chk["case_full_clear"],
                             n_src=chk["n_src"]))
    finally:
        for k, v in old.items():
            if v is None:
                try:
                    delattr(rb.Problem3Robot, k)
                except AttributeError:
                    pass
            else:
                setattr(rb.Problem3Robot, k, v)
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9573
    print(f"配对 {n_case} 例(同场景 + 固定误差场, 种子 {seed})")
    print(f"{'配置':<22}{'δ':>6}{'T(s)':>8}{'移动':>9}{'检测':>7}{'全清':>10}")
    res = {}
    for cname, cfg in (("旧配置 1200x6 无透镜/无贝叶斯", OLD), ("现行 1030x8 + 透镜 + 贝叶斯", NEW)):
        for d in (300.0, 500.0):
            rows = run_arm(cfg, d, n_case, seed)
            res[(cname, d)] = rows
            print(f"{cname:<22}{d:>6.0f}{np.mean([x['T'] for x in rows]):>8.0f}"
                  f"{np.mean([x['dist'] for x in rows]):>9.0f}"
                  f"{np.mean([x['meas'] for x in rows]):>7.1f}"
                  f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
    print("\nδ=500 相对 δ=300 的配对差(逐配置):")
    for cname in ("旧配置 1200x6 无透镜/无贝叶斯", "现行 1030x8 + 透镜 + 贝叶斯"):
        a = np.array([x["T"] for x in res[(cname, 300.0)]], float)
        b = np.array([x["T"] for x in res[(cname, 500.0)]], float)
        d = b - a
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"  {cname:<22} ΔT {d.mean():+8.1f} s ({100*d.mean()/a.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+7.0f},{d.mean()+1.96*se:+7.0f}]  "
              f"变快 {100*(d<0).mean():4.1f} %  中位 {np.median(d):+.0f} s")
    out = os.path.join(HERE, "results", "delta_cross_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# δ 2x2 交叉验证逐案\nconfig,delta,k,n_src,T,dist,meas,fail,full\n")
        for (cname, d), rows in res.items():
            for k, x in enumerate(rows):
                fh.write("%s,%.0f,%d,%d,%.1f,%.0f,%d,%d,%d\n" % (
                    cname.replace(",", " "), d, k, x["n_src"], x["T"], x["dist"],
                    x["meas"], x["fail"], x["full"]))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
