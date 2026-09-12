# -*- coding: utf-8 -*-
"""问题三: **贝叶斯概率图**(模块P)的配对实验。

三个可测杠杆(全部默认关, 见 robot.py):
  PROB_ORDER      覆盖点顺序按"存在后验 × 该点可收到概率"贪心(保证不变, 只改顺序);
  PROB_CHAN_ORDER 点内频道顺序同样按概率图(影响换频);
  PROB_COUNT_CERT 计数证书: 已确认源数达题设上界 16 -> 其余频道确定性判空。

对照组 = 当前采纳配置(1030x8 + 透镜)。全部同场景 + 固定误差场配对。

**机制上限(先说明, 便于读结果)**: 覆盖率扫描要求"每个未清除频道在每个覆盖点上各测一次",
故概率图能省的只有"源被更早发现/清除后省下的剩余测量"。当前 1 000 例实测检测 166 次,
而"全测"上限 = 20×9 = 180 次 ⟹ **排序类杠杆的天花板约 14 次 ≈ 70 s(1.7 %)**。

用法: python prob_map_ab.py [案例数=400] [种子=9163]
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
from simlib import case_env, sim_time, check_clearance, config_scope, frozen3, phase_time

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
from verify_ring_supp import FixedEnv                   # noqa: E402

PHASES = ("coverage", "supplement", "on_way", "queue_clear", "homing",
          "queue_homing", "on_way_homing")
ARMS = [
    ("base(当前采纳)", dict(PROB_ORDER=False, PROB_CHAN_ORDER=False, PROB_COUNT_CERT=False)),
    ("skip_ig(零增益不测)", dict(PROB_SKIP_IG=True)),
    ("count_cert(计数证书)", dict(PROB_COUNT_CERT=True)),
    ("skip_ig+cert", dict(PROB_SKIP_IG=True, PROB_COUNT_CERT=True)),
    ("order w=3+cert", dict(PROB_ORDER=True, PROB_TRAVEL_W=3.0, PROB_COUNT_CERT=True)),
]


def run_arm(cfg, n_case, seed):
    rows = []
    scope = config_scope((rb.Problem3Robot, frozen3(**cfg)))
    scope.__enter__()
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
                             cert=getattr(robot, "prob_cert_fired", 0),
                             minpres=getattr(robot, "prob_min_presence", None),
                             **{p: phase_time(cli, p) for p in PHASES}))
    finally:
        scope.__exit__(None, None, None)
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9163
    print(f"配对 {n_case} 例(种子 {seed}), 同场景 + 固定误差场; 环境 = 当前采纳配置"
          f"(环 {rb.Problem3Robot.RING_R:.0f}x{rb.Problem3Robot.RING_N}, "
          f"透镜={rb.Problem3Robot.DOP_PRESCREEN})")
    print(f"{'臂':<24}{'T(s)':>8}{'移动':>8}{'检测':>7}{'换频':>7}{'失败':>6}"
          f"{'证书触发':>9}{'最小存在后验':>13}{'全清':>10}")
    res = {}
    for name, cfg in ARMS:
        rows = run_arm(cfg, n_case, seed)
        res[name] = rows
        mp = [x["minpres"] for x in rows if x["minpres"] is not None]
        print(f"{name:<24}{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['dist'] for x in rows]):>8.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['sw'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{np.mean([x['cert'] for x in rows]):>9.2f}"
              f"{(np.mean(mp) if mp else float('nan')):>13.5f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
    b = np.array([x["T"] for x in res[ARMS[0][0]]], float)
    print(f"\n相对当前采纳配置的配对差:")
    for name, _ in ARMS[1:]:
        a = np.array([x["T"] for x in res[name]], float)
        d = a - b
        se = d.std(ddof=1)/math.sqrt(len(d))
        dm = (np.mean([x["meas"] for x in res[name]])
              - np.mean([x["meas"] for x in res[ARMS[0][0]]]))
        ds = (np.mean([x["sw"] for x in res[name]])
              - np.mean([x["sw"] for x in res[ARMS[0][0]]]))
        print(f"  {name:<24} ΔT {d.mean():+7.1f} s ({100*d.mean()/b.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+6.0f},{d.mean()+1.96*se:+6.0f}]  变快 "
              f"{100*(d<0).mean():4.1f} %  Δ检测 {dm:+5.1f}  Δ换频 {ds:+6.1f}")
    out = os.path.join(HERE, "results", "prob_map_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 模块P(贝叶斯概率图)配对实验逐案\n")
        fh.write("arm,k,n_src,T,dist,meas,switch,clear,fail,cert,minpres,coverage,supplement\n")
        for name, _ in ARMS:
            for k, x in enumerate(res[name]):
                fh.write("%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%s,%.1f,%.1f\n" % (
                    name, k, x["n_src"], x["T"], x["dist"], x["meas"], x["sw"], x["clr"],
                    x["fail"], x["cert"],
                    "nan" if x["minpres"] is None else "%.6f" % x["minpres"],
                    x["coverage"], x["supplement"]))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
