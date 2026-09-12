# -*- coding: utf-8 -*-
"""问题三: **理论下界与当前解的差距**(用于决定是否还值得继续优化)。

成本模型(题设): T = L/5 + 5·N_meas + 1·N_switch + 5·N_clear_ok + 3·N_fail。
按分量构造下界, 并**明确区分两类**:

[A] **离线(全知)下界** —— 允许策略预先知道源的位置与频道, 只受"物理上必须做的事"约束:
    A1 移动-覆盖下界: 测量点(半径 1000 m 接收盘)必须覆盖 1800 m 目标盘。
       * 连续松弛: 一条半径 r 的连续曲线, 其 1000 m 邻域覆盖 1800 盘 <=> r >= 800,
         故 L >= 2π·800 = 5 026.5 m(允许无限密测量时的绝对下限)。
       * 离散环族下界: 原点 + n 点均匀环, 覆盖条件
         sqrt(1800²+r²-2·1800·r·cos(π/n)) <= 1000, 开路长度 = r + (n-1)·2r·sin(π/n)。
         对 n 求最小(见下表), 这是"有点数概念"的最短覆盖路线。
    A2 移动-接近下界: 每个源都必须被接近到 R_clear=20 m 内才能清除。
       线路访问各源 20 m 邻域, 其长度 >= TSP(源中心) - 2·20·n_src(每个源最多省 2δ)。
    A3 移动下界 = max(A1 离散族最小, A2); 再加 A1 连续下限作为绝对地板作参照。
    A4 检测下界: 每个源至少被接收 1 次(1 次测量), 且要"认证可清除"至少需要 2 条示向
       => 每个源 >= 2 次测量; 未被发现的 (20-n_src) 个空频道要被排除, 至少各在 k_min 个
       覆盖点上返回 no_signal(题设 n_src<=16 时"计数证书"仅在恰好 16 源时可用, 故一般情形
       仍需 k_min 次) => N_meas >= 2·n_src + (20-n_src)·k_min。
    A5 清除下界: >= n_src 次成功清除; 失败下界 0; 换频下界 0(只增不减)。
    => T_lb_A = L_A/5 + 5·N_meas_lb + 5·n_src。

[B] **在线信息约束**(不可达部分, 用于避免高估可优化空间):
    B1 机器人事先不知道源的位置/频道, 故"必须扫完覆盖点才能确定某个频道不存在"
       —— 这已经把 A4 的 (20-n_src)·k_min 计入; 此外 A2 的 TSP 只能在线逐步逼近,
       实际接近路程必然更长(已知下界与在线策略之间没有构造性桥梁)。
    B2 因此本文只把 [A] 当作"绝对地板", 把 A1 离散族最优 + 当前补充/归航机制下的
       **族内最优**当作"可达参照", 两者之间即"信息差距"。

用法: python lower_bound.py [案例数=300] [种子=7391]
"""
import contextlib
import io
import itertools
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

R_AREA, R_GUAR, R_CLEAR = 1800.0, 1000.0, 20.0


# ---------- 覆盖下界 ----------
def ring_min_route(n):
    """原点 + n 点均匀环: 满足覆盖的最小半径与对应开路长度。

    覆盖条件 d²(r) = 1800² + r² - 2·1800·r·cos(π/n) <= 1000²; d(r) 在
    r* = 1800cos(π/n) 处取最小 1800sin(π/n)(n>=6 时 <= 1000, 可行), 可行域是
    [r_min, r_max] 区间 —— 故**必须用解析根**, 不能用单调二分(否则 n=6 会因
    d(r) 先减后增而收敛到区间外的错点, 这正是首版实现的 bug)。
    """
    c = math.cos(math.pi/n)
    disc = (R_AREA*c)**2 - (R_AREA**2 - R_GUAR**2)
    if disc < 0:
        return None, None, None                      # 该 n 不可行
    r = R_AREA*c - math.sqrt(disc)                   # 较小根 = 最小可行半径
    L = r + (n-1)*2*r*math.sin(math.pi/n)
    d = math.sqrt(R_AREA**2 + r**2 - 2*R_AREA*r*c)
    return r, L, d


def mst_length(points):
    """欧氏 MST 长度(精确 Kruskal, 小规模) —— 开放 TSP 的**严格下界**(见 main 中说明)。"""
    n = len(points)
    if n < 2:
        return 0.0
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((math.hypot(points[i][0]-points[j][0],
                                     points[i][1]-points[j][1]), i, j))
    edges.sort()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tot, used = 0.0, 0
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tot += w
            used += 1
            if used == n-1:
                break
    return tot


def tsp_open(points, start=(0.0, 0.0)):
    """最近邻 + 2-opt 的开放 TSP(从 start 出发); 用于接近下界的上界估计。"""
    pts = list(points)
    if not pts:
        return 0.0
    order, cur = [], start
    rest = list(range(len(pts)))
    while rest:
        j = min(rest, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
        order.append(j)
        cur = pts[j]
        rest.remove(j)
    path = [start] + [pts[i] for i in order]

    def length(p):
        return sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path)-1):
            for j in range(i+1, len(path)):
                a, b, c, d = path[i-1], path[i], path[j], path[min(j+1, len(path)-1)]
                if j == len(path)-1:
                    continue
                if (math.hypot(b[0]-a[0], b[1]-a[1]) + math.hypot(d[0]-c[0], d[1]-c[1])
                        > math.hypot(c[0]-a[0], c[1]-a[1]) + math.hypot(d[0]-b[0], d[1]-b[1]) + 1e-9):
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
    return length(path)


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7391
    print("=== 覆盖下界: 原点 + n 点均匀环 (覆盖条件与最短开路) ===")
    print(f"  {'n':>3}{'r_min(m)':>10}{'最坏接收(m)':>13}{'开路(m)':>9}{'检测数上限':>11}"
          f"{'折算检测(s)':>12}{'移动(s)':>9}{'合计(s)':>9}")
    best = None
    for n in range(6, 15):
        r, L, d = ring_min_route(n)
        if r is None:
            print(f"  {n:>3}  不可行(该 n 的环无法覆盖圆盘)")
            continue
        k = n + 1
        meas = 20*k                                    # 每点 20 频道(未利用清除节省)
        T = L/5 + 5*meas + 5*12.5                      # 清除按平均 12.5 源
        print(f"  {n:>3}{r:>10.1f}{d:>13.1f}{L:>9.0f}{meas:>11d}{5*meas:>12.0f}"
              f"{L/5:>9.0f}{T:>9.0f}")
        if best is None or T < best[0]:
            best = (T, n, r, L, meas)
    print(f"  ≈ 连续松弛下限(精确覆盖的**下确界**): 曲线半径 800 m 的圆 "
          f"-> L >= 2π·800 = {2*math.pi*800:.0f} m(测量无限密时可逼近)")
    print(f"  环族最优: n={best[1]}(r={best[2]:.0f} m), 开路 {best[3]:.0f} m, "
          f"检测 {best[4]} 次 -> 合计约 {best[0]:.0f} s")

    # ---------- 当前解 vs 下界(逐案例) ----------
    rows = []
    scope = simlib.config_scope  # noqa
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
        srcs = [tuple(s["pos"]) for s in env.sources]
        tsp = tsp_open(srcs)                            # 参考: 可实现的接近路线(NN+2opt)
        mst = mst_length(list(srcs) + [(0.0, 0.0)])     # 严格下界: 含起点的 MST
        k_min = 6                                       # 弧覆盖论证: 每点最多覆盖 67.4°, 故 >= 6
        nsrc = env.n_src
        meas_lb = 2*nsrc + (20-nsrc)*k_min
        # 接近下界: 任何路线都要进入每个源的 20 m 邻域; 把"进邻域"换回"到中心"每个源
        # 最多增加 2·20 m, 故 L >= TSP_open(源) - 2·20·n_src >= MST(源∪原点) - 40·n_src
        L_src_lb = max(0.0, mst - 2*R_CLEAR*nsrc)
        L_cov_inf = 2*math.pi*(R_AREA - R_GUAR)         # 覆盖下确界 2π·800
        L_lb = max(L_src_lb, L_cov_inf)
        T_lb = L_lb/5 + 5*meas_lb + 5*nsrc
        rows.append(dict(n=nsrc, T=sim_time(cli), L=cli.dist, meas=cli.n_measure,
                         sw=cli.n_switch, clr=cli.n_clear, fail=cli.fail,
                         tsp=tsp, mst=mst, L_lb=L_lb, meas_lb=meas_lb, T_lb=T_lb,
                         L_cont=L_cov_inf))

    T = np.mean([r["T"] for r in rows])
    L = np.mean([r["L"] for r in rows])
    M = np.mean([r["meas"] for r in rows])
    SW = np.mean([r["sw"] for r in rows])
    CL = np.mean([r["clr"] for r in rows])
    FA = np.mean([r["fail"] for r in rows])
    print(f"\n=== 当前解(候选 {rb.Problem3Robot.RING_R:.0f}x{rb.Problem3Robot.RING_N} + "
          f"透镜={rb.Problem3Robot.DOP_PRESCREEN}, 配对 {n_case} 例) 与下界对比 ===")
    print(f"  {'分量':<18}{'当前':>10}{'下界':>10}{'差':>10}{'差占比':>9}")
    print(f"  {'移动 L (m)':<18}{L:>10.0f}{np.mean([r['L_lb'] for r in rows]):>10.0f}"
          f"{L-np.mean([r['L_lb'] for r in rows]):>10.0f}"
          f"{100*(L-np.mean([r['L_lb'] for r in rows]))/L:>8.0f}%")
    print(f"  {'  (连续覆盖下确界 m)':<18}{'':>10}{rows[0]['L_cont']:>10.0f}")
    print(f"  {'  (源 MST 下界 m)':<18}{'':>10}{np.mean([r['mst'] for r in rows]):>10.0f}")
    print(f"  {'  (源 TSP 参考 m)':<18}{'':>10}{np.mean([r['tsp'] for r in rows]):>10.0f}")
    print(f"  {'检测 N_meas':<18}{M:>10.1f}{np.mean([r['meas_lb'] for r in rows]):>10.1f}"
          f"{M-np.mean([r['meas_lb'] for r in rows]):>10.1f}"
          f"{100*(M-np.mean([r['meas_lb'] for r in rows]))/M:>8.0f}%")
    print(f"  {'换频 N_switch':<18}{SW:>10.1f}{0:>10}{SW:>10.1f}")
    print(f"  {'清除 N_clear':<18}{CL:>10.1f}{np.mean([r['n'] for r in rows]):>10.1f}"
          f"{CL-np.mean([r['n'] for r in rows]):>10.1f}")
    print(f"  {'失败 N_fail':<18}{FA:>10.2f}{0:>10.2f}")
    print(f"  {'总时间 T (s)':<18}{T:>10.0f}{np.mean([r['T_lb'] for r in rows]):>10.0f}"
          f"{T-np.mean([r['T_lb'] for r in rows]):>10.0f}"
          f"{100*(T-np.mean([r['T_lb'] for r in rows]))/T:>8.0f}%")
    print(f"\n  分项折算(当前 - 下界): 移动 {L-np.mean([r['L_lb'] for r in rows]) and ''}"
          f"{(L-np.mean([r['L_lb'] for r in rows]))/5:.0f} s | 检测 "
          f"{5*(M-np.mean([r['meas_lb'] for r in rows])):.0f} s | 换频 {SW:.0f} s | "
          f"清除 {5*(CL-np.mean([r['n'] for r in rows])):.0f} s | 失败 {3*FA:.0f} s")
    out = os.path.join(HERE, "results", "lower_bound_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 逐案: 当前解与下界(离线全知)\n")
        fh.write("k,n_src,T,L,meas,switch,clear,fail,src_tsp,L_lb,meas_lb,T_lb\n")
        for k, r in enumerate(rows):
            fh.write("%d,%d,%.1f,%.0f,%d,%d,%d,%d,%.0f,%.0f,%d,%.1f\n" % (
                k, r["n"], r["T"], r["L"], r["meas"], r["sw"], r["clr"], r["fail"],
                r["tsp"], r["L_lb"], r["meas_lb"], r["T_lb"]))
    print(f"  逐案明细 -> {out}")


if __name__ == "__main__":
    main()
