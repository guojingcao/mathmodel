# -*- coding: utf-8 -*-
"""P3-B/P3-C: **鲁棒透镜补测点** + **补测阶段分组路线** 的配对实验。

背景(修正认知): 现有 `_supplement_point` 是"从首次示向点沿垂直方向偏移 300/500 m, 取离机器人
最近的一侧"——与鲁棒性无关。仓库里其实**已实现**鲁棒透镜判据 `_dop_supplement_point`(模块B,
`DOP_PRESCREEN`): 候选点 ±60/±90/±120 度 × 300/500/700 m -> 交会角 >=30 度预筛 ->
按"两站 ±1° 楔形交的**最坏直径**"极小极大 + 路程择优。但**它从未被配对测试过**(论文中所有
DOP 讨论都属于问题二, P3 消融证据中没有它)。本脚本补齐这项测试, 并测试新增的补测分组模块G
(`SUPP_GROUP_R`: 相距 <= R 的补测点合并为同一停靠点, 合并后仍逐频道测量)。

四臂: base(现状) / lens(模块B) / group(模块G, R=300 与 100) / lens+group
两个环境: 旧冻结环 1200x6 (补测阶段活跃, 可观测 need 均值约 2.4) 与候选环 1150x9 (need 约 0.2)
另加**困难子集**(源贴边界/最坏接收两类, 各 200 例), 因为补测成本集中在那里。

用法: python supp_lens_group.py [每臂案例数=1000] [种子=6320]
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
from simlib import (case_env, sim_time, check_clearance, config_scope,
                    frozen3, phase_time)                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
sc3 = simlib.load_module("sc3", os.path.join(HERE, "selfcheck.py"))
from verify_ring_supp import FixedEnv                   # noqa: E402

PHASES = ("coverage", "aux_coverage", "supplement", "on_way", "on_way_homing",
          "queue_clear", "queue_homing", "homing")

ARMS = [
    ("base(现状: 垂直偏移)", dict(DOP_PRESCREEN=False, SUPP_GROUP_R=None)),
    ("lens(模块B: 字典序)", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=None, SUPP_GROUP_R=None)),
    ("lens+lam0.02", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=0.02, SUPP_GROUP_R=None)),
    ("lens+lam0.05", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=0.05, SUPP_GROUP_R=None)),
    ("lens+lam0.10", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=0.10, SUPP_GROUP_R=None)),
    ("lens+lam0.20", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=0.20, SUPP_GROUP_R=None)),
    ("group600(模块G: R=600m)", dict(DOP_PRESCREEN=False, SUPP_GROUP_R=600.0)),
    ("group1000(模块G: R=1000m)", dict(DOP_PRESCREEN=False, SUPP_GROUP_R=1000.0)),
    ("lens+lam0.05+group600", dict(DOP_PRESCREEN=True, LENS_TRAVEL_W=0.05,
                                   SUPP_GROUP_R=600.0)),
]
ENVS = [("旧冻结环 1200x6", dict(RING_R=1200.0, RING_N=6)),
        ("候选环 1150x9", dict(RING_R=1150.0, RING_N=9))]


def run_arm(env_cfg, cfg, n_case, seed):
    rows = []
    scope = config_scope((rb.Problem3Robot, frozen3(**env_cfg, **cfg)))
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
            rows.append(dict(
                T=sim_time(cli), dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch,
                clr=cli.n_clear, fail=cli.fail, full=chk["case_full_clear"],
                n_src=chk["n_src"], need=getattr(robot, "aux_need", None),
                merged=getattr(robot, "supp_group_merged", 0),
                **{p: phase_time(cli, p) for p in PHASES}))
    finally:
        scope.__exit__(None, None, None)
    return rows


def run_patho(cfg, kinds, n_rep, seed=99):
    """困难子集: 复用 selfcheck 的病理场景构造(源贴边界/最坏接收)。"""
    rows = []
    scope = config_scope((rb.Problem3Robot, frozen3(RING_R=1200.0, RING_N=6, **cfg)))
    scope.__enter__()
    try:
        for label, kind in kinds:
            rng = np.random.default_rng(seed)
            for i in range(n_rep):
                specs = sc3.patho_sources(rng, kind, 13)
                b = sc3.make_env(rng, specs)
                env = FixedEnv(np.random.default_rng(0), n_src=b.n_src, directional=False,
                               scene_key=(seed, kind, i))
                env.sources = b.sources
                env.ch_by_id = b.ch_by_id
                env.cleared = set()
                cli = simlib.SimClient(env)
                robot = rb.Problem3Robot(cli)
                with contextlib.redirect_stdout(io.StringIO()):
                    n_ret = robot.run()
                chk = check_clearance(n_ret, cli, env)
                rows.append(dict(kind=label, T=sim_time(cli), dist=cli.dist,
                                 meas=cli.n_measure, fail=cli.fail,
                                 full=chk["case_full_clear"], n_src=chk["n_src"],
                                 need=getattr(robot, "aux_need", None),
                                 supp=phase_time(cli, "supplement"),
                                 hom=phase_time(cli, "homing")))
    finally:
        scope.__exit__(None, None, None)
    return rows


def brief(rows):
    T = np.array([r["T"] for r in rows], float)
    return dict(T=T, mean=T.mean(),
                dist=np.mean([r["dist"] for r in rows]),
                meas=np.mean([r["meas"] for r in rows]),
                fail=np.mean([r["fail"] for r in rows]),
                full=sum(r["full"] for r in rows), n=len(rows),
                supp=np.mean([r["supplement"] for r in rows]),
                queue=np.mean([r["queue_clear"] for r in rows]),
                hom=np.mean([r["homing"] + r["on_way_homing"] + r["queue_homing"]
                             for r in rows]),
                need=np.mean([r["need"] for r in rows]),
                merged=np.mean([r["merged"] for r in rows]))


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 6320
    print(f"配对案例 {n_case} 例(种子 {seed}), 同一场景 + 固定误差场")
    out_rows = {}
    for env_name, env_cfg in ENVS:
        print(f"\n########## 环境: {env_name} ##########")
        res = {}
        print(f"  {'臂':<24}{'T(s)':>8}{'移动(m)':>9}{'检测':>7}{'补测(s)':>9}{'队列(s)':>9}"
              f"{'归航(s)':>8}{'失败':>6}{'合并点/例':>10}{'全清':>9}")
        for name, cfg in ARMS:
            rows = run_arm(env_cfg, cfg, n_case, seed)
            res[name] = rows
            s = brief(rows)
            out_rows[(env_name, name)] = rows
            print(f"  {name:<24}{s['mean']:>8.1f}{s['dist']:>9.0f}{s['meas']:>7.1f}"
                  f"{s['supp']:>9.1f}{s['queue']:>9.1f}{s['hom']:>8.1f}{s['fail']:>6.2f}"
                  f"{s['merged']:>10.2f}{s['full']:>6d}/{s['n']}", flush=True)
        b = np.array([r["T"] for r in res[ARMS[0][0]]], float)
        bsup = np.array([r["supplement"] for r in res[ARMS[0][0]]], float)
        print(f"  --- 相对 base 的配对差 ---")
        for name, _ in ARMS[1:]:
            a = np.array([r["T"] for r in res[name]], float)
            d = a - b
            se = d.std(ddof=1)/math.sqrt(len(d))
            sa = np.array([r["supplement"] for r in res[name]], float)
            ds = sa - bsup
            print(f"  {name:<24} ΔT {d.mean():+7.1f} s ({100*d.mean()/b.mean():+6.2f} %) "
                  f"CI [{d.mean()-1.96*se:+6.0f},{d.mean()+1.96*se:+6.0f}]  "
                  f"变快 {100*(d<0).mean():4.1f} %  Δ补测 {ds.mean():+7.1f} s")

    print("\n########## 困难子集(旧冻结环 1200x6; 每类 200 例) ##########")
    kinds = [("全源贴近边界", "edge"), ("最坏接收(全1000m)", "worst_rx")]
    sub = {}
    lens_arms = [a for a in ARMS if a[0].startswith("lens")]
    for name, cfg in [ARMS[0]] + lens_arms + [ARMS[-1]]:
        r = run_patho(cfg, kinds, 200)
        sub[name] = r
        for label, _ in kinds:
            rr = [x for x in r if x["kind"] == label]
            T = np.array([x["T"] for x in rr], float)
            print(f"  {name:<24}{label:<18} T {T.mean():7.1f} s  补测 "
                  f"{np.mean([x['supp'] for x in rr]):6.1f} s  检测 "
                  f"{np.mean([x['meas'] for x in rr]):5.1f}  失败 "
                  f"{np.mean([x['fail'] for x in rr]):4.2f}  全清 "
                  f"{sum(x['full'] for x in rr)}/{len(rr)}", flush=True)
    for name in list(sub.keys())[1:]:
        d = np.array([sub[name][i]["T"] - sub[ARMS[0][0]][i]["T"] for i in range(len(sub[name]))],
                     float)
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"  配对(困难子集, n={len(d)}): {name} vs base  ΔT {d.mean():+.1f} s "
              f"({100*d.mean()/np.mean([x['T'] for x in sub[ARMS[0][0]]]):+.2f} %)  "
              f"CI [{d.mean()-1.96*se:+.0f},{d.mean()+1.96*se:+.0f}]  "
              f"变快 {100*(d<0).mean():.1f} %")

    # 逐案明细入库
    out = os.path.join(HERE, "results", "supp_lens_group_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# P3 补测点规则/分组路线 配对实验逐案明细(固定误差场, 同场景)\n")
        fh.write("env,arm,k,n_src,need,T,dist,meas,fail,full,supplement,queue_clear,homing,merged\n")
        for (env_name, arm), rows in out_rows.items():
            for k, r in enumerate(rows):
                fh.write("%s,%s,%d,%d,%d,%.1f,%.0f,%d,%d,%d,%.1f,%.1f,%.1f,%d\n" % (
                    env_name, arm, k, r["n_src"], r["need"] or 0, r["T"], r["dist"],
                    r["meas"], r["fail"], r["full"], r["supplement"], r["queue_clear"],
                    r["homing"] + r["on_way_homing"] + r["queue_homing"], r["merged"]))
    print(f"\n逐案明细已写入 {out}")


if __name__ == "__main__":
    main()
