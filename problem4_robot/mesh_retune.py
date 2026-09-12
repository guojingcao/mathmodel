# -*- coding: utf-8 -*-
"""问题四: 在"MEC 冻结后的真实成本结构"下重搜网格设计, 并离线配对验证。

为什么重搜: 原 27 点设计是在"每站测 20 频道"的成本模型 C=巡回/5+100·n 下选出的;
MEC 冻结 + 证书早排除后, 实测每站测量数降到约 13.7 次(368.8 次/局 ÷ 27 站),
故成本模型应改为 C=巡回/5+5·m̂·n(m̂ 由实测标定) —— 测量项变小、旅行项主导,
最优点会向"更大边长/更少站"移动。

本脚本:
  1) 用标定后的成本在格点块族上重搜(含 a 到 970: 受 build_triangles 的 1.03 邻接容差限制,
     实际最大边必须 <=1000 m);
  2) 对前若干候选做**离线配对**(固定误差场, 同场景)对比当前 27 点设计, 报 ΔT 与胜率。

用法: python mesh_retune.py [m_hat=13.7] [topk=6] [n_case=200]
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
from mesh_design import lattice, max_edge, uncovered, samples, tour_len  # noqa: E402
from simlib import case_env, sim_time, check_clearance  # noqa: E402

R = r4.R_AREA


def cost(pts, tris, m_hat):
    return tour_len(pts, tris)/5.0 + 5.0*m_hat*len(pts)


def search(m_hat, smp, topk):
    cands = []
    for a in (910.0, 930.0, 950.0, 965.0):
        for th in (0, 15, 30, 45):
            for frac in (2.4, 2.5, 2.6, 2.7, 2.8, 2.9):
                for off in ((a*0.0, 0.0), (a*0.2, 0.0), (a*0.4, 0.0), (a*0.5, 0.0),
                            (a*0.25, a*0.25)):
                    pts = lattice(a, th, a*frac, off)
                    if len(pts) > 40 or len(pts) < 8:
                        continue
                    tris = r4.build_triangles(pts, a=a)
                    if max_edge(pts, tris) > 1000.0:
                        continue
                    if uncovered(pts, tris, smp):
                        continue
                    cands.append((cost(pts, tris, m_hat), len(pts), a, th, frac, off,
                                  pts, tris))
    cands.sort(key=lambda c: c[0])
    print(f"可行设计 {len(cands)} 个(成本模型 m̂={m_hat} 次测量/站)")
    print("%-6s%8s%9s%8s%8s%10s%12s" % ("排名", "成本(s)", "点数", "a", "θ", "外扩", "巡回(m)"))
    for i, c in enumerate(cands[:topk], 1):
        print("%-6d%8.0f%9d%8.0f%8d%10.0f%12.0f" %
              (i, c[0], c[1], c[2], c[3], c[2]*c[4], tour_len(c[6], c[7])))
    return cands


def pilot(pts, n_case, p_dir=0.5, seed=3026):
    """固定误差场离线跑一臂, 返回 (T 列表, 全清数, 平均检测数, 平均移动)。"""
    old = r4.Problem4Robot.MESH_PTS_OVERRIDE
    r4.Problem4Robot.MESH_PTS_OVERRIDE = list(pts)
    Ts, full, meas, mv = [], 0, 0.0, 0.0
    try:
        for k in range(n_case):
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
    finally:
        r4.Problem4Robot.MESH_PTS_OVERRIDE = old
    return Ts, full, meas/n_case, mv/n_case


def main():
    m_hat = float(sys.argv[1]) if len(sys.argv) > 1 else 13.7
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_case = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    smp = samples(3000, 2000, seed=31)
    cands = search(m_hat, smp, topk)
    print(f"\n=== 离线配对(固定误差场, n={n_case}/臂, 定向比例 50%) ===")
    base_pts = list(r4.Problem4Robot.MESH_DESIGN_PTS)
    Tb, fb, mb, vb = pilot(base_pts, n_case)
    print("当前 27 点设计: 均值 %.0f s  全清 %d/%d  检测 %.1f  移动 %.0f m"
          % (np.mean(Tb), fb, n_case, mb, vb))
    print("%-8s%10s%10s%12s%10s%9s%9s" % ("候选", "点数", "巡回(m)", "均值(s)", "Δ(s)", "胜率", "全清"))
    for i, c in enumerate(cands[:topk], 1):
        pts = c[6]
        if len(pts) == len(base_pts) and all(abs(x-y) < 1e-6 for p, q in zip(pts, base_pts)
                                             for x, y in zip(p, q)):
            continue
        T, f, m, v = pilot(pts, n_case)
        d = np.array(T) - np.array(Tb)
        print("%-8s%10d%10.0f%12.0f%+10.0f%8.1f%%%7d/%d" % (
            f"#{i}", len(pts), tour_len(pts, r4.build_triangles(pts, a=c[2])),
            np.mean(T), d.mean(), 100*(d < 0).mean(), f, n_case))
    print("\n注: 采纳需通过全部门禁(覆盖核验/病理 2000 例/--cert/在环 12 局)。")


if __name__ == "__main__":
    main()
