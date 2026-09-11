"""问题四"证书可靠性不变量"审计。

不变量(证书推理的前提):
    圆盘内**任意一点 P** 都必须落在某个 cover_tris 三角形内。
否则 "所有 cover_tris 三角形被证伪 ⇒ 该频道不存在" 不成立——
存在某些源位置,其所在三角形根本不在证书集合里,于是它永远不会被证伪,
机器人却因为其余三角形全被证伪而把它判为"已排除"。

本脚本:
  1) 对比三种三角形集合: 全部三角剖分 / cover_tris(现用形心规则) / 精确与圆盘相交集;
  2) 找出"精确相交但未被 cover_tris 收录"的三角形(即证书空洞), 打印其几何;
  3) 用均匀采样 + **边界环加权采样** 统计落在 cover_tris 之外的点数;
  4) 复现病理例中的具体漏清源位置。

用法: python cover_audit.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                      # noqa: E402

import robot4 as r4                     # noqa: E402

R = r4.R_AREA


def in_tri(p, a, b, c):
    def cr(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
    d1 = cr(a, b, p); d2 = cr(b, c, p); d3 = cr(c, a, p)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    L2 = dx*dx + dy*dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy)/L2))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))


def tri_origin_dist(a, b, c):
    """原点到三角形的最近距离(含内部/边/顶点)。"""
    if in_tri((0.0, 0.0), a, b, c):
        return 0.0
    return min(seg_dist((0.0, 0.0), a, b), seg_dist((0.0, 0.0), b, c),
               seg_dist((0.0, 0.0), c, a))


def circumradius(a, b, c):
    la = math.hypot(b[0]-c[0], b[1]-c[1])
    lb = math.hypot(a[0]-c[0], a[1]-c[1])
    lc = math.hypot(a[0]-b[0], a[1]-b[1])
    s = 2.0*(la*lb*lc)
    if s == 0:
        return float("inf")
    ar = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))/2.0
    return (la*lb*lc)/(4.0*ar) if ar > 0 else float("inf")


def main():
    rb = r4.Problem4Robot(None)
    pts, tris, cov = rb.pts, rb.tris, rb.cover_tris
    print(f"网格: {len(pts)} 点, 三角剖分 {len(tris)} 个, cover_tris(证书集合) {len(cov)} 个")
    print(f"MESH_A={r4.MESH_A}, MESH_MARGIN={r4.MESH_MARGIN}, "
          f"形心阈值 R+0.6a = {R + r4.MESH_A*0.6:.1f} m")

    covset = {tuple(sorted(t)) for t in cov}
    exact, exact_not_cov = [], []
    for t in tris:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        if tri_origin_dist(a, b, c) <= R:
            exact.append(t)
            if tuple(sorted(t)) not in covset:
                exact_not_cov.append(t)
    print(f"精确与圆盘相交的三角形: {len(exact)} 个; "
          f"其中**未被 cover_tris 收录** {len(exact_not_cov)} 个 "
          f"-> 这些是证书空洞" if exact_not_cov else "-> 无空洞")

    if exact_not_cov:
        print("\n空洞三角形几何(按形心半径排序):")
        rows = []
        for t in exact_not_cov:
            a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
            cen = ((a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3)
            rows.append((math.hypot(*cen), t, a, b, c,
                         max(math.hypot(b[0]-c[0], b[1]-c[1]),
                             math.hypot(a[0]-c[0], a[1]-c[1]),
                             math.hypot(a[0]-b[0], a[1]-b[1])),
                         circumradius(a, b, c), tri_origin_dist(a, b, c)))
        for rc, t, a, b, c, mx, cr, d0 in sorted(rows):
            print(f"  {tuple(int(i) for i in t)}: 形心半径 {rc:.1f} m, 最大边 {mx:.0f} m, "
                  f"外接半径 {cr:.0f} m, 到原点最近距离 {d0:.1f} m")

    # 采样: 均匀 + 边界环
    def covered(p, keep):
        for t in keep:
            if in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]):
                return True
        return False

    rng = np.random.default_rng(11)
    N = 20000
    stats = {}
    for tag in ("均匀", "边界环1700-1800", "极边环1780-1800"):
        bad_cov = 0; bad_exact = 0; worst = 0.0; example = None
        for _ in range(N):
            if tag == "均匀":
                r = R*math.sqrt(rng.uniform())
            elif tag == "边界环1700-1800":
                r = rng.uniform(1700, 1800)
            else:
                r = rng.uniform(1780, 1800)
            ang = rng.uniform(0, 2*math.pi)
            p = (r*math.cos(ang), r*math.sin(ang))
            if not covered(p, cov):
                bad_cov += 1
                if r > worst:
                    worst = r; example = p
            if not covered(p, exact):
                bad_exact += 1
        stats[tag] = (bad_cov, bad_exact, worst, example)
        print(f"\n[{tag}] n={N}: 落在 cover_tris 之外 {bad_cov} 个 "
              f"({100.0*bad_cov/N:.3f}%), 落在精确集之外 {bad_exact} 个; "
              f"最远漏点半径 {worst:.1f} m")
        if example:
            print(f"    示例漏点 ({example[0]:.1f}, {example[1]:.1f})")

    # 病理例的具体源
    src = (-1150.5, -1376.7)
    print(f"\n复现病理源 ({src[0]}, {src[1]}), |r|={math.hypot(*src):.1f} m:")
    hit = [t for t in tris if in_tri(src, pts[t[0]], pts[t[1]], pts[t[2]])]
    for t in hit:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        cen = ((a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3)
        print(f"  所在三角形 {tuple(int(i) for i in t)}: 形心半径 {math.hypot(*cen):.1f} m, "
              f"在 cover_tris 内 = {tuple(sorted(t)) in covset}, "
              f"顶点半径 {[round(math.hypot(*pts[i]),0) for i in t]}")
    if not hit:
        print("  未被任何三角形包含(剖分空洞)")


if __name__ == "__main__":
    main()
