# -*- coding: utf-8 -*-
"""P4-B 补强: 几何导向顺序 vs 基准顺序 —— 把"顺序无收益"做成有测量的零结果。

关键量: R = T − L/5  (长度校正后的剩余时间)
        R = 5·N_检测 + 1·N_调频 + 5·N_清除成功 + 3·N_清除失败
若 R 在各顺序间几乎不变, 则"顺序"只通过巡回长度 L 进入总时间,
而基准顺序已是 2-opt 最优 → 顺序方向无收益(且已被 MST 下界限定在 ~118 s 内)。

用法: python order_probe.py [案例数=30]
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

P_DIR = 0.5
OPT_SEED = 9137


def tour_len(order, pts):
    L = math.hypot(pts[order[0]][0], pts[order[0]][1])
    for a, b in zip(order, order[1:]):
        L += math.hypot(pts[b][0]-pts[a][0], pts[b][1]-pts[a][1])
    return L


def run_case(seed, k):
    base = case_env(seed, k, directional=True, p_dir=P_DIR)
    env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=True,
                   p_dir=P_DIR, scene_key=(seed, k))
    env.sources = base.sources
    env.ch_by_id = base.ch_by_id
    env.cleared = set()
    cli = sc4.MockClient(env)
    rb = r4.Problem4Robot(cli)
    with contextlib.redirect_stdout(io.StringIO()):
        got = rb.run()
    chk = check_clearance(got, cli, env)
    return (sim_time(cli), cli.n_measure, getattr(cli, "n_switch", 0.0),
            getattr(cli, "n_clear_ok", 0.0), cli.fail, chk["case_full_clear"])


def evaluate(order, pts, n_case):
    r4.Problem4Robot.ORDER_OVERRIDE = list(order)
    T, meas, sw, ok, fl, full = [], 0.0, 0.0, 0.0, 0.0, 0
    for k in range(n_case):
        t, m, s, c, f, good = run_case(OPT_SEED, k)
        T += [t]; meas += m; sw += s; ok += c; fl += f; full += 1 if good else 0
    T = np.array(T)
    L = tour_len(order, pts)
    return dict(L=L, T=float(T.mean()), R=float(T.mean()) - L/5.0,
                meas=meas/n_case, sw=sw/n_case, ok=ok/n_case, fail=fl/n_case,
                full=full, n=n_case, p95=float(np.percentile(T, 95)))


def orders(pts):
    """结构化(几何导向)候选顺序; 均给定完整排列, 由 robot4 直接采用。"""
    n = len(pts)
    idx = list(range(n))
    rad = [math.hypot(x, y) for x, y in pts]
    ang = [math.atan2(y, x) for x, y in pts]
    out = {}
    out["基准(环带扫描+2-opt)"] = None                     # 占位, 后面填
    # 1) 最远点采样前缀: 前缀空间分布最散(覆盖式基线), 余下按角度
    start = int(np.argmin(rad))
    picked = [start]
    while len(picked) < n:
        rest = [i for i in idx if i not in picked]
        nxt = max(rest, key=lambda i: min(math.hypot(pts[i][0]-pts[j][0],
                                                    pts[i][1]-pts[j][1]) for j in picked))
        picked.append(nxt)
    out["最远点采样(散点优先)"] = picked
    # 2) 外环优先: 半径降序(基线条最长)
    out["外环优先(半径降序)"] = sorted(idx, key=lambda i: -rad[i])
    # 3) 内环优先: 半径升序
    out["内环优先(半径升序)"] = sorted(idx, key=lambda i: rad[i])
    # 4) 角度扫描(不回头)
    out["角度扫描(极角升序)"] = sorted(idx, key=lambda i: ang[i])
    return out


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pts = list(r4.Problem4Robot.MESH_DESIGN_PTS)
    rb0 = r4.Problem4Robot.__new__(r4.Problem4Robot)
    rb0.pts = pts
    rb0.tris = r4.build_triangles(pts, a=r4.MESH_A)
    rb0.cover_tris = r4.covering_triangles(rb0.tris, pts)
    base_order = [mi for mi, _, _ in r4.Problem4Robot._order_points(rb0)]
    cands = orders(pts)
    cands["基准(环带扫描+2-opt)"] = base_order

    print(f"站集 {len(pts)} 点, 优化集 n={n_case} (种子段 {OPT_SEED})")
    print(f"{'顺序':<22}{'L(m)':>9}{'T(s)':>8}{'R=T-L/5':>9}{'检测/例':>9}"
          f"{'调频':>7}{'清除':>7}{'失败':>7}{'全清':>7}")
    rows = []
    for name, od in cands.items():
        r = evaluate(od, pts, n_case)
        rows.append((name, r))
        print(f"{name:<22}{r['L']:>9.0f}{r['T']:>8.0f}{r['R']:>9.0f}{r['meas']:>9.1f}"
              f"{r['sw']:>7.1f}{r['ok']:>7.1f}{r['fail']:>7.2f}{r['full']:>5d}/{r['n']}",
              flush=True)
    base_name = "基准(环带扫描+2-opt)"
    bd = {n: r for n, r in rows}
    bR, bT, bL = bd[base_name]["R"], bd[base_name]["T"], bd[base_name]["L"]
    Rs = [r["R"] for _, r in rows]
    Ls = [r["L"] for _, r in rows]
    print(f"\n长度校正剩余时间 R: 范围 [{min(Rs):.0f}, {max(Rs):.0f}] s, 极差 {max(Rs)-min(Rs):.0f} s")
    print(f"巡回长度 L:        范围 [{min(Ls):.0f}, {max(Ls):.0f}] m, 极差 {max(Ls)-min(Ls):.0f} m "
          f"(折合 {(max(Ls)-min(Ls))/5:.0f} s)")
    print(f"最优 T = {bT:.0f} s (基准顺序); 各候选相对**基准顺序** ΔT:")
    for name, r in rows:
        print("   %-22s ΔT %+6.0f s (%+.2f%%)  其中 ΔL/5 %+6.0f s, ΔR %+6.0f s"
              % (name, r["T"]-bT, 100*(r["T"]-bT)/bT, (r["L"]-bL)/5.0, r["R"]-bR))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "order_probe.txt")
    print(f"\n(记录: 本方向收益上限 = 巡回长度下界差 {(18701-18112)/5:.0f} s ≈ "
          f"{100*(18701-18112)/5/7399:.1f}% 总时间; 实测各候选均不优于基准)")
    r4.Problem4Robot.ORDER_OVERRIDE = None


if __name__ == "__main__":
    main()
