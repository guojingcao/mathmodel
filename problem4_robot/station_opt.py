# -*- coding: utf-8 -*-
"""问题四: 站集与访问顺序的**联合优化**(自由坐标), 证书为连续可证形式。

与 mesh_design.py 的区别: 那里只搜"完整三角格点块"这一族(规则布局);
这里直接在**连续坐标**上做随机局部搜索, 可产生**非格点**布局, 并直接以实测成本为目标:

    成本 C(S) = TSP(S)/5 + 5·m̂·|S|      (m̂≈13.7: MEC 冻结后实测每站测量次数)

硬约束(证书, 全部为**连续判据**, 不依赖采样):
    ① 凸包 ⊇ 圆盘: 原点在凸包内 且 每条支撑边到原点距离 >= R;
    ② 无孔: Σ三角形面积 = 凸包面积(与 ①、③ 合起来即"并集 = 凸包");
    ③ 两两内部不重叠(Sutherland-Hodgman 交面积 ≈ 0);
    ④ 凡与圆盘相交的三角形, 其最大边 <= 1000 m(保证"源在三角形内 ⟹ 三顶点均可接收")。
满足 ①–④ 即得: ∀G∈D, G ∈ conv(S ∩ B(G,1000)), 且 G 所在三角形的 3 顶点都可用 —— 这就是
方向鲁棒覆盖证书(§6.1 判据)的连续证明。

用法: python station_opt.py [迭代数=4000] [随机种子=1]
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robot4 as r4                     # noqa: E402

R = r4.R_AREA
A_TOL = 970.0                           # build_triangles 的邻接容差参数(1.03*970=999.1<=1000)
M_HAT = 13.7                            # 实测每站测量次数(MEC 冻结后)
EDGE_LIM = 960.0                        # 盘内最大边上限(部署留 40 m 余量)
SUPPORT_LIM = 10.0                      # 支撑边到原点需 >= R + 10 m


def area2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def hull(points):
    ps = sorted(set((round(x, 9), round(y, 9)) for x, y in points))

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2-x1)*(p[1]-y1) - (y2-y1)*(p[0]-x1) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out
    return half(ps)[:-1] + half(ps[::-1])[:-1]


def poly_area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % len(poly)]
        s += x1*y2 - x2*y1
    return abs(s)/2.0


def seg_dist(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    L2 = dx*dx + dy*dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L2))
    return math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy))


def clip_area(poly, clip):
    def cr(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    out = list(poly)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i+1) % len(clip)]
        if not out:
            return 0.0
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j+1) % len(out)]
            ip_, iq_ = cr(a, b, p) >= -1e-12, cr(a, b, q) >= -1e-12
            if ip_:
                new.append(p)
            if ip_ != iq_:
                dx, dy = q[0]-p[0], q[1]-p[1]
                d1, d2 = cr(a, b, p), cr(a, b, q)
                t = d1/(d1-d2) if abs(d1-d2) > 1e-18 else 0.0
                new.append((p[0]+t*dx, p[1]+t*dy))
        out = new
    return poly_area(out) if len(out) >= 3 else 0.0


def tri_intersects_disk(a, b, c, m=12):
    """三角形是否与圆盘相交(顶点在内 / 边与圆相交 / 原点在内 三判据)。"""
    for p in (a, b, c):
        if math.hypot(*p) <= R:
            return True
    if math.hypot(0.0, 0.0) <= R:                     # 原点在三角形内
        d1, d2, d3 = area2(a, b, (0, 0)), area2(b, c, (0, 0)), area2(c, a, (0, 0))
        if not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0))):
            return True
    for p, q in ((a, b), (b, c), (c, a)):
        if seg_dist((0.0, 0.0), p, q) <= R:
            return True
    return False


def certificate(pts, max_edge_lim=1000.0, support_lim=0.0):
    """连续证书核验(可带鲁棒余量要求)。返回 (是否通过, 指标 dict, 违规量)。

    max_edge_lim: 盘内最大边上限(部署需留余量, 建议 960);
    support_lim : 支撑边到原点距离需 >= R + support_lim(建议 10 m)。
    """
    pts = [tuple(p) for p in pts]
    if len(pts) < 4:
        return False, {}, 1e9
    tris = [tuple(t) for t in r4.build_triangles(pts, a=A_TOL)]
    hp = hull(pts)
    a_hull = poly_area(hp)
    a_sum = sum(abs(area2(pts[t[0]], pts[t[1]], pts[t[2]]))/2.0 for t in tris)
    inside = all(area2(hp[i], hp[(i+1) % len(hp)], (0.0, 0.0)) >= -1e-9
                 for i in range(len(hp)))
    dmin = min(seg_dist((0.0, 0.0), hp[i], hp[(i+1) % len(hp)])
               for i in range(len(hp))) if len(hp) >= 3 else -1e9
    ov = 0.0
    polys = []
    for t in tris:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        polys.append([a, b, c] if area2(a, b, c) > 0 else [a, c, b])
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            v = clip_area(polys[i], polys[j])
            if v > ov:
                ov = v
    max_edge_disk = 0.0
    for t in tris:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        if not tri_intersects_disk(a, b, c):
            continue
        mx = max(math.hypot(b[0]-a[0], b[1]-a[1]),
                 math.hypot(c[0]-b[0], c[1]-b[1]),
                 math.hypot(a[0]-c[0], a[1]-c[1]))
        max_edge_disk = max(max_edge_disk, mx)
    gap_area = abs(a_sum - a_hull)
    viol = 0.0
    if not inside:
        viol += 1e6
    if dmin < R + support_lim:
        viol += (R + support_lim - dmin)*1e3
    if gap_area > 1e-6*max(1.0, a_hull):
        viol += gap_area*1e-2
    if ov > 1e-6*max(1.0, a_hull):
        viol += ov*1e-2
    if max_edge_disk > max_edge_lim:
        viol += (max_edge_disk - max_edge_lim)*1e2
    ok = (viol <= 0.0)
    return ok, dict(n=len(pts), n_tri=len(tris), a_hull=a_hull, gap=gap_area, ov=ov,
                    support_margin=dmin - R, max_edge_disk=max_edge_disk,
                    hull_v=len(hp)), viol


def tsp_len(pts):
    rb = r4.Problem4Robot.__new__(r4.Problem4Robot)
    rb.pts = [tuple(p) for p in pts]
    rb.tris = r4.build_triangles(rb.pts, a=A_TOL)
    rb.cover_tris = r4.covering_triangles(rb.tris, rb.pts)
    seq = r4.Problem4Robot._order_points(rb)
    L = math.hypot(seq[0][1], seq[0][2])
    for p, q in zip(seq, seq[1:]):
        L += math.hypot(q[1]-p[1], q[2]-p[2])
    return L


def cost(pts):
    return tsp_len(pts)/5.0 + 5.0*M_HAT*len(pts)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rnd = random.Random(seed)
    cur = [tuple(p) for p in r4.Problem4Robot.MESH_DESIGN_PTS]
    def cert(p):
        return certificate(p, max_edge_lim=EDGE_LIM, support_lim=SUPPORT_LIM)
    ok, info, viol = cert(cur)
    cur_c = cost(cur) + 1e4*viol
    best = (cur_c, list(cur), info, viol)
    print(f"起点: 27 点格点块, 证书 {'通过' if ok else '不通过'}, 成本 {cost(cur):.0f} s "
          f"(TSP {tsp_len(cur):.0f} m)")
    T0, Tend = 3.0, 0.02
    for it in range(iters):
        cand = [list(p) for p in cur]
        move = rnd.random()
        if move < 0.72:                                    # 抖动一个点
            i = rnd.randrange(len(cand))
            cand[i][0] += rnd.gauss(0, 60)
            cand[i][1] += rnd.gauss(0, 60)
        elif move < 0.86 and len(cand) > 18:               # 删点
            cand.pop(rnd.randrange(len(cand)))
        else:                                              # 加点(在圆盘内随机)
            a = rnd.uniform(0, 2*math.pi); rr = R*math.sqrt(rnd.random())
            cand.append([rr*math.cos(a), rr*math.sin(a)])
        cand = [tuple(p) for p in cand]
        ok2, info2, viol2 = cert(cand)
        c2 = cost(cand) + 1e4*viol2
        T = T0*(Tend/T0)**(it/max(1, iters-1))
        if c2 < cur_c or rnd.random() < math.exp(-(c2-cur_c)/max(1e-9, T*100)):
            cur, cur_c = cand, c2
            if c2 < best[0]:
                best = (c2, list(cand), info2, viol2)
                print(f"  [{it}] 新最优: 成本 {cost(cand):.0f} s, {len(cand)} 点, "
                      f"TSP {tsp_len(cand):.0f} m, 证书{'通过' if ok2 else '不通过'}"
                      f"(支撑余量 {info2.get('support_margin', 0):.1f} m, "
                      f"盘内最大边 {info2.get('max_edge_disk', 0):.0f} m)", flush=True)
    bc, bpts, binfo, bviol = best
    ok, info, viol = certificate(bpts)
    print(f"\n最优: {len(bpts)} 点, 成本 {cost(bpts):.0f} s (TSP {tsp_len(bpts):.0f} m), "
          f"证书 {'通过' if ok else '不通过'}")
    print("  连续证书指标:", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in info.items()})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "station_opt_best.txt")
    with open(out, "w", encoding="utf-8") as f:
        for x, y in bpts:
            f.write(f"{x:.3f},{y:.3f}\n")
    print(f"  已写入 {out}")
    print(f"  当前设计对照: 27 点, 成本 {cost(r4.Problem4Robot.MESH_DESIGN_PTS):.0f} s "
          f"(TSP {tsp_len(r4.Problem4Robot.MESH_DESIGN_PTS):.0f} m)")


if __name__ == "__main__":
    main()
