# -*- coding: utf-8 -*-
"""base + TSP 微调 的配对实验(用户优先级 ①)。

三个可调旋钮(全部默认关/保持原值, 只在本实验里开关):
  A. TSP_OROPT      : 末端清除巡回在 2-opt 之后追加 Or-opt(段长 1..3 的迁移);
  B. ON_WAY_DELTA   : 顺路插入阈值 δ(决定多少清除被吸收进扫描、从而缩短末端巡回);
  C. 组合。

用法: python tsp_tune_ab.py [案例数=300] [种子=9573]
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
          "queue_homing", "on_way_homing", "recovery")

ARMS = [
    ("base(δ=300)", dict(ON_WAY_DELTA=300.0, TSP_OROPT=False)),
    ("+Oropt", dict(ON_WAY_DELTA=300.0, TSP_OROPT=True)),
    ("δ=400", dict(ON_WAY_DELTA=400.0, TSP_OROPT=False)),
    ("δ=500", dict(ON_WAY_DELTA=500.0, TSP_OROPT=False)),
    ("+Oropt+δ=500", dict(ON_WAY_DELTA=500.0, TSP_OROPT=True)),
]
FLAGS = ("ON_WAY_DELTA", "TSP_OROPT", "TSP_OROPT_SEG")


def run_arm(cfg, n_case, seed):
    old = {k: getattr(rb.Problem3Robot, k, None) for k in FLAGS}
    for k, v in cfg.items():
        setattr(rb.Problem3Robot, k, v)
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
    print(f"配对 {n_case} 例(种子 {seed}), 同场景 + 固定误差场; 环境 = 现行方案")
    print(f"{'臂':<16}{'T(s)':>8}{'移动':>9}{'检测':>7}{'失败':>6}{'全清':>10}"
          f"{'覆盖':>7}{'顺路':>7}{'队列清':>8}")
    res = {}
    for name, cfg in ARMS:
        rows = run_arm(cfg, n_case, seed)
        res[name] = rows
        print(f"{name:<16}{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['dist'] for x in rows]):>9.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}"
              f"{np.mean([x['coverage'] for x in rows]):>7.0f}"
              f"{np.mean([x['on_way'] for x in rows]):>7.0f}"
              f"{np.mean([x['queue_clear'] for x in rows]):>8.0f}", flush=True)
    b = np.array([x["T"] for x in res[ARMS[0][0]]], float)
    print("\n相对 base 的配对差:")
    for name, _ in ARMS[1:]:
        a = np.array([x["T"] for x in res[name]], float)
        d = a - b
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"  {name:<16} ΔT {d.mean():+8.1f} s ({100*d.mean()/b.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+7.0f},{d.mean()+1.96*se:+7.0f}]  "
              f"变快 {100*(d<0).mean():4.1f} %  中位 {np.median(d):+.0f} s")
    out = os.path.join(HERE, "results", "tsp_tune_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# base + TSP 微调 配对逐案\n")
        fh.write("arm,k,n_src,T,dist,meas,switch,clear,fail,full," + ",".join(PHASES) + "\n")
        for name, _ in ARMS:
            for k, x in enumerate(res[name]):
                fh.write("%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%s\n" % (
                    name, k, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"], x["clr"],
                    x["fail"], x["full"], ",".join("%.1f" % x[p] for p in PHASES)))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
