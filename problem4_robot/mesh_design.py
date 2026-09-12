"""问题四"最小覆盖顶点集"设计器。

原理: 证书推理(以及"任意源必被检出")成立的前提是**圆盘被三角剖分完全覆盖**。
      对**完整三角格点块**而言, 其三角剖分的并 = 该点块的凸包(格点间距恰为 a ⇒ 每个格
      三角形都被 build_triangles 连出), 因此只要凸包 ⊇ 圆盘, 就**不存在空洞**——
      这比过去"稀疏外环 + 事后补点"的做法从根本上避免了最外环薄空洞。

搜索: 三角格点(边长 a, 旋转 θ, 平移 off)落在半径 r_out 内的全部点 -> 最小顶点数且覆盖圆盘。
输出: results/mesh_design_best.txt (每行 "x,y"), 供 robot4.MESH_PTS_OVERRIDE 使用。

用法:
    python mesh_design.py --search            # 搜索并保存最优设计
    python mesh_design.py --verify            # 核验已保存设计(覆盖/最大边/巡回/方向判据)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                      # noqa: E402

import robot4 as r4                     # noqa: E402

R = r4.R_AREA
BEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "mesh_design_best.txt")


def in_tri(p, a, b, c):
    def cr(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
    d1 = cr(a, b, p); d2 = cr(b, c, p); d3 = cr(c, a, p)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def samples(n_uni=4000, n_ring=3000, seed=31):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_uni):
        r = R*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    for _ in range(n_ring):
        r = rng.uniform(1650.0, R); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    return out


def lattice(a, theta_deg, r_out, off=(0.0, 0.0)):
    """旋转三角格点块: 全部落在半径 r_out 内的格点。"""
    th = math.radians(theta_deg)
    e1 = (a*math.cos(th), a*math.sin(th))
    e2 = (a*math.cos(th+math.pi/3), a*math.sin(th+math.pi/3))
    n = int(r_out/a) + 2
    pts = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            x = off[0] + i*e1[0] + j*e2[0]
            y = off[1] + i*e1[1] + j*e2[1]
            if math.hypot(x, y) <= r_out:
                pts.append((round(x, 3), round(y, 3)))
    return pts


def max_edge(pts, tris):
    m = 0.0
    for t in tris:
        for i in range(3):
            for j in range(i+1, 3):
                m = max(m, math.hypot(pts[t[i]][0]-pts[t[j]][0], pts[t[i]][1]-pts[t[j]][1]))
    return m


def uncovered(pts, tris, smp):
    bad = 0
    for p in smp:
        if not any(in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]) for t in tris):
            bad += 1
    return bad


def tour_len(pts, tris=None):
    """覆盖巡回长度: 用 robot4 的真实取点顺序(最近邻 + 开放式 2-opt), 与实跑一致。"""
    rb = r4.Problem4Robot.__new__(r4.Problem4Robot)
    rb.pts = [tuple(p) for p in pts]
    rb.tris = r4.build_triangles(rb.pts, a=r4.MESH_A)
    rb.cover_tris = r4.covering_triangles(rb.tris, rb.pts)
    seq = r4.Problem4Robot._order_points(rb)
    L = math.hypot(seq[0][1], seq[0][2])
    for p, q in zip(seq, seq[1:]):
        L += math.hypot(q[1]-p[1], q[2]-p[2])
    return L


def cost_seconds(pts, tris):
    """扫描成本(秒) ≈ 移动时间 + 20 频道 × 5 s 测量; 这是网格设计的真正目标函数。"""
    n = len(pts)
    return tour_len(pts, tris)/5.0 + 5.0*20*n


def prune(pts, smp, a=None):
    """删点剪枝: 逐个尝试删除"对覆盖无贡献"的点(留一验证)。"""
    a = a or r4.MESH_A
    pts = list(pts)
    changed = True
    while changed:
        changed = False
        for i in range(len(pts)):
            cand = pts[:i] + pts[i+1:]
            tris = r4.build_triangles(cand, a=a)
            if max_edge(cand, tris) > 1000.5:
                continue
            if uncovered(cand, tris, smp) == 0:
                pts = cand
                changed = True
                break
    return pts


def search(a_list, theta_list, r_frac_list, off_list, smp, do_prune=True, topk=5):
    """搜索最小**扫描成本**设计: 成本 = 巡回/5 + 20 频道×5 s (点数与巡回同时计入)。"""
    cands = []
    print(f"搜索: a={a_list} θ={theta_list} r_out/a={r_frac_list} off={len(off_list)} 种; "
          f"采样 {len(smp)} 点; 目标 = 巡回/5 + 100×点数(秒)")
    for a in a_list:
        for th in theta_list:
            for frac in r_frac_list:
                r_out = a*frac
                for off in off_list:
                    pts = lattice(a, th, r_out, off)
                    if len(pts) > 44 or len(pts) < 8:
                        continue
                    tris = r4.build_triangles(pts, a=a)
                    mx = max_edge(pts, tris)
                    if mx > 1000.0:
                        continue
                    if uncovered(pts, tris, smp):
                        continue
                    cands.append((cost_seconds(pts, tris), len(pts), a, th, frac, off,
                                  pts, tris))
    if not cands:
        print("未找到可行设计")
        return None
    cands.sort(key=lambda c: c[0])
    print(f"可行设计 {len(cands)} 个; 前 {topk} 名(粗采样):")
    for c in cands[:topk]:
        print(f"  成本 {c[0]:.0f} s; {c[1]} 点/{len(c[7])} 三角; a={c[2]:.0f} θ={c[3]} "
              f"r_out={c[2]*c[4]:.0f} off=({c[5][0]:.0f},{c[5][1]:.0f}); "
              f"最大边 {max_edge(c[6], c[7]):.0f} m")
    best = cands[0]
    if do_prune:                       # 对最优候选剪枝后再比一次
        pts2 = prune(best[6], smp, best[2])
        tris2 = r4.build_triangles(pts2, a=best[2])
        if max_edge(pts2, tris2) <= 1000.0 and not uncovered(pts2, tris2, smp):
            c2 = (cost_seconds(pts2, tris2), len(pts2), best[2], best[3], best[4],
                  best[5], pts2, tris2)
            print(f"  剪枝后: 成本 {c2[0]:.0f} s; {len(pts2)} 点 (原 {best[1]} 点)")
            if c2[0] < best[0]:
                best = c2
    cost, n, a, th, frac, off, pts, tris = best
    with open(BEST_FILE, "w", encoding="utf-8") as f:
        for x, y in pts:
            f.write(f"{x},{y}\n")
    print(f"\n最优设计: {n} 点 / {len(tris)} 三角; a={a} θ={th} off={off}; "
          f"最大边 {max_edge(pts, tris):.0f} m; 巡回 {tour_len(pts, tris):.0f} m; "
          f"扫描成本 ≈ {cost:.0f} s; 已写入 {BEST_FILE}")
    return best


def verify(path=BEST_FILE):
    pts = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            x, y = ln.split(",")
            pts.append((float(x), float(y)))
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    print(f"设计: {len(pts)} 点 / {len(tris)} 三角形; 最大边 {max_edge(pts, tris):.1f} m; "
          f"巡回 {tour_len(pts):.0f} m")
    for tag, n_uni, n_ring in (("粗采样", 4000, 3000), ("细采样", 40000, 40000)):
        smp = samples(n_uni, n_ring, seed=31 if tag == "粗采样" else 97)
        bad = uncovered(pts, tris, smp)
        print(f"  [{tag}] {len(smp)} 点: 未覆盖 {bad} ({100.0*bad/len(smp):.5f}%)")
    # 方向鲁棒判据 S∈conv Q(S) 抽查
    rng = np.random.default_rng(5)
    ok = 0; N = 5000; worst = 0.0
    for _ in range(N):
        r = R*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        S = (r*math.cos(a), r*math.sin(a))
        Q = [p for p in pts if math.hypot(p[0]-S[0], p[1]-S[1]) <= 1000.0]
        if len(Q) >= 3:
            # 简单判据: S 在 Q 的凸包内 <=> 存在三点三角形包含 S(此处只做必要条件抽查)
            hit = any(in_tri(S, Q[i], Q[j], Q[k])
                      for i in range(len(Q)) for j in range(i+1, len(Q))
                      for k in range(j+1, len(Q)))
            ok += 1 if hit else 0
        else:
            worst = max(worst, 1.0)
    print(f"  方向判据抽查: {ok}/{N} 点的 conv Q(S) 覆盖 S; |Q(S)| 不足的点 {int(worst)}")
    return pts, tris


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    elif "--search" in sys.argv:
        smp = samples(1500, 1000, seed=31)      # 粗筛用; 优胜者随后细采样复核
        offs = []
        for a in (760.0, 820.0, 880.0, 920.0, 960.0):
            for fx in (0.0, 0.25, 0.5):
                for fy in (0.0, 0.25):
                    offs.append((a*fx, a*fy))
        search([760.0, 820.0, 880.0, 910.0, 940.0, 960.0], [0, 15, 30, 45],
               [2.45, 2.55, 2.65, 2.75, 2.85, 2.95], offs, smp)
    else:
        print(__doc__)
