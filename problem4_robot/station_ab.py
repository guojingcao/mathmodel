# -*- coding: utf-8 -*-
"""站集联合优化的配对验证: 当前 27 点格点块 vs 优化得到的 27 点非规则布局。

两臂均在最终配置下(MEC 冻结 + 暂缓归航), 固定误差场(误差与动作顺序无关), 同场景配对。
注意部署口径: 新设计需要 MESH_A=970, 使 build_triangles 的邻接容差 1.03*970=999.1 m
覆盖其最大边 959.4 m; 当前设计的 MESH_A=910(容差 937 m)。

用法: python station_ab.py [n=200]
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                  # noqa: E402

import robot4 as r4                                 # noqa: E402
import selfcheck4 as sc4                            # noqa: E402
from ablation4_mecfreeze import FixedEnv            # noqa: E402
from simlib import case_env, sim_time, check_clearance   # noqa: E402

BEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "results", "station_opt_best.txt")


def load_best():
    pts = []
    for ln in open(BEST, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            x, y = ln.split(",")
            pts.append((float(x), float(y)))
    return pts


def arm(mesh_a, pts, p_dir, n, seed):
    old_a, old_ov = r4.MESH_A, r4.Problem4Robot.MESH_PTS_OVERRIDE
    r4.MESH_A = mesh_a
    r4.Problem4Robot.MESH_PTS_OVERRIDE = list(pts)
    Ts, full, meas, mv, fail = [], 0, 0.0, 0.0, 0.0
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
            Ts.append(sim_time(cli))
            full += 1 if chk["case_full_clear"] else 0
            meas += cli.n_measure
            mv += cli.dist
            fail += cli.fail
    finally:
        r4.MESH_A, r4.Problem4Robot.MESH_PTS_OVERRIDE = old_a, old_ov
    return dict(T=np.array(Ts), full=full, n=n, meas=meas/n, mv=mv/n, fail=fail/n)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = 3026
    cur = list(r4.Problem4Robot.MESH_DESIGN_PTS)
    new = load_best()
    print(f"配对验证: 当前 {len(cur)} 点格点块(MESH_A=910) vs 优化 {len(new)} 点非规则布局"
          f"(MESH_A=970)   n={n}/臂\n")
    for p_dir in (0.5, 1.0):
        A = arm(910.0, cur, p_dir, n, seed)
        B = arm(970.0, new, p_dir, n, seed)
        d = B["T"] - A["T"]
        print("=== 定向比例 %.0f%% ===" % (100*p_dir))
        for tag, r in (("A 当前格点块", A), ("B 非规则优化布局", B)):
            print("  %-18s 均值 %7.0f s  P90 %7.0f  P95 %7.0f  最大 %7.0f  全清 %d/%d  "
                  "检测 %5.1f  移动 %6.0f m  失败 %4.2f" % (
                      tag, r["T"].mean(), np.percentile(r["T"], 90),
                      np.percentile(r["T"], 95), r["T"].max(), r["full"], r["n"],
                      r["meas"], r["mv"], r["fail"]))
        se = d.std(ddof=1)/np.sqrt(len(d))
        print("  配对: Δ = %+.1f s (%.2f%%)  95%%CI [%.0f,%.0f]  变快比例 %.1f%%  中位 %+.0f s\n"
              % (d.mean(), 100*d.mean()/A["T"].mean(), d.mean()-1.96*se, d.mean()+1.96*se,
                 100*(d < 0).mean(), np.median(d)))


if __name__ == "__main__":
    main()
