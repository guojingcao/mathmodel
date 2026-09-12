# -*- coding: utf-8 -*-
"""P4-D 严格证明: 充要判据的**精确算术**复核 + "每个站点都承重"的显式违例证书。

两个命题(全部用 fractions.Fraction 精确算术, 不含浮点判定):

  命题 A(充分性, V4 满足充要判据) —— 机器辅助精确证明:
     前提(逐项精确复核): ① 三角形顶点 ⊂ 点集; ② 凸包由 Fraction 叉积**重新构造**(非浮点),
     且 27 点全部位于每条有向边的同一侧、顶点严格左转; ③ 37 个三角形**两两内部不重叠**
     (精确凸多边形裁剪求交面积 = 0); ④ Σ|2S_三角形| = |2S_凸包|(精确相等);
     ⑤ 凸包每条边到原点距离 >= 1800; ⑥ 与圆盘相交的每个三角形最大边 <= 1000 m。
     证明链:
       - ①+②: 每个三角形 ⊂ 凸包(凸包凸性);
       - ③+④: 三角形**互不重叠**且面积之和恰等于凸包面积;
       - 三角形与凸包均为**闭集**, 故并集闭; 由 ③ 得 area(∪T_i)=Σarea(T_i)=area(conv S),
         即 conv(S)\∪T_i 是开集且测度为 0 ⟹ 无内点 ⟹ 为空(并集闭) ⟹ **∪T_i = conv(S)**;
         结合 ⑤ 得 ∪T_i ⊇ 圆盘。
       - 取任意 G∈D: G 落在某三角形 T 内, 其顶点 A,B,C ∈ S 且由 ⑥ |GA|,|GB|,|GC| <= 1000
         ⟹ A,B,C ∈ S∩B(G,1000); 而 G ∈ T = conv{A,B,C} ⟹ G ∈ conv(S∩B(G,1000)). ∎
     注意: 若不做 ③(不重叠), 仅 ④ 的面积相等**不能**推出并集 = 凸包(重叠可与缺口相抵),
     故 ③ 是证明链的必需环节。

  命题 B(V4 为**包含极小**可行站集):
     对每个 i, 给出**显式证书** (G, u): G ∈ D 且对全部 s ∈ S_i ∩ B(G,1000) 有
     <s−G, u> < 0 (u 为有理向量), 即 G ∉ conv(S_i ∩ B(G,1000)) ⟹ 删除第 i 站后充要判据失效。
     每条证书逐项用精确算术验证(距离比较用平方比较, 不引入开方)。
     **适用范围**: 仅覆盖 V4 的 27 个"单站删除"子集, 不能推出"任何 26 点布局不可行",
     也不构成"27 点为全局最少站数"的证明。

用法: python hull_cert_proof.py
"""
import math
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robot4 as r4                                     # noqa: E402
from hull_margin import hull, margin_of                 # noqa: E402

R2 = F(1000)**2                    # 接收半径平方(精确)
RA2 = F(1800)**2                   # 目标区半径平方(精确)


def ex(v):
    """float -> 精确有理数(十进制字符串解析, 避免二进制浮点误差)。"""
    return F(repr(float(v)))


def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def tri_intersects_disk(a, b, c, lim):
    """三角形是否与半径 lim 的圆盘相交(精确判据: 顶点在内 / 边到原点距离平方 <= lim²)。"""
    for p in (a, b, c):
        if p[0]**2 + p[1]**2 <= lim*lim:
            return True
    for p, q in ((a, b), (b, c), (c, a)):
        d = q[0]-p[0], q[1]-p[1]
        L2 = d[0]**2 + d[1]**2
        if L2 == 0:
            continue
        t = -((p[0]*d[0] + p[1]*d[1])/L2)
        if t < 0:
            t = F(0)
        elif t > 1:
            t = F(1)
        fx, fy = p[0]+t*d[0], p[1]+t*d[1]
        if fx*fx + fy*fy <= lim*lim:
            return True
    return False


def hull_exact(pts):
    """用 Fraction 叉积重新构造凸包(CCW, 去掉共线点) —— 不依赖浮点 hull()。"""
    ps = sorted(set(pts))

    def half(q):
        out = []
        for p in q:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(ps)[:-1] + half(ps[::-1])[:-1]


def clip_convex(subject, clip):
    """Sutherland-Hodgman 精确裁剪(clip 为 CCW 凸多边形, 内部 = 每条有向边左侧)。"""
    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i+1) % len(clip)]
        if not out:
            return []
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j+1) % len(out)]
            cp = cross(a, b, p)
            cq = cross(a, b, q)
            if cp >= 0:
                new.append(p)
            if (cp > 0 and cq < 0) or (cp < 0 and cq > 0):
                # 交点参数 t = cp/(cp-cq), 用 Fraction 精确表示
                t = cp/(cp-cq)
                new.append((p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1])))
        out = new
    return out


def poly_area2(poly):
    """2×面积(精确); 逆序时取绝对值。"""
    if len(poly) < 3:
        return F(0)
    s = F(0)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % len(poly)]
        s += x1*y2 - x2*y1
    return abs(s)


def ccw_tri(a, b, c):
    return [a, b, c] if cross(a, b, c) > 0 else [a, c, b]


def prove_A(pts, tris):
    """精确复核命题 A 的六个前提, 返回 (是否成立, 指标)。"""
    Ph = [tuple(ex(c) for c in p) for p in pts]
    # ② 凸包由 Fraction 叉积重新构造, 并精确验证"凸性 + 全部点在同侧"
    hpx = hull_exact(Ph)
    ok_hull = len(hpx) >= 3
    for i in range(len(hpx)):
        a, b, c = hpx[i], hpx[(i+1) % len(hpx)], hpx[(i+2) % len(hpx)]
        if cross(a, b, c) <= 0:                        # 严格左转(凸且 CCW)
            ok_hull = False
    for i in range(len(hpx)):                          # 所有点位于每条有向边左侧或边上
        a, b = hpx[i], hpx[(i+1) % len(hpx)]
        for p in Ph:
            if cross(a, b, p) < 0:
                ok_hull = False
    # ④ 面积加性
    s2_tri = F(0)
    ts = []
    for t in tris:
        a, b, c = Ph[t[0]], Ph[t[1]], Ph[t[2]]
        s2_tri += abs(cross(a, b, c))
        ts.append(ccw_tri(a, b, c))
    s2_hull = abs(sum(cross(hpx[i], hpx[(i+1) % len(hpx)], hpx[0])
                      for i in range(len(hpx))))
    ok_area = (s2_tri == s2_hull)
    # ③ 两两内部不重叠(精确求交面积)
    ov_pairs, ov_max = 0, F(0)
    n = len(ts)
    for i in range(n):
        for j in range(i+1, n):
            inter = clip_convex(ts[i], ts[j])
            a2 = poly_area2(inter)
            if a2 > 0:
                ov_pairs += 1
                ov_max = max(ov_max, a2)
    ok_nonoverlap = (ov_pairs == 0)
    # ⑤ 支撑边到原点距离 >= 1800:  |cross| = dist * |edge|
    marginal = None
    ok_support = True
    for i in range(len(hpx)):
        a, b = hpx[i], hpx[(i+1) % len(hpx)]
        cr = a[0]*b[1] - a[1]*b[0]
        L2 = dist2(a, b)
        if cr*cr < RA2*L2:
            ok_support = False
        m = math.sqrt(float(cr*cr/L2)) - 1800.0
        marginal = m if marginal is None else min(marginal, m)
    # ⑥ 与圆盘相交的三角形的最大边(精确平方比较)
    ok_edge = True
    worst_edge = F(0)
    for t in tris:
        a, b, c = Ph[t[0]], Ph[t[1]], Ph[t[2]]
        if not tri_intersects_disk(a, b, c, F(1800)):
            continue
        e2 = max(dist2(a, b), dist2(b, c), dist2(c, a))
        worst_edge = max(worst_edge, e2)
        if e2 > R2:
            ok_edge = False
    # ① 顶点来自点集
    ok_verts = all(0 <= i < len(Ph) for t in tris for i in t)
    return (ok_verts and ok_hull and ok_nonoverlap and ok_area and ok_support and ok_edge), dict(
        ok_verts=ok_verts, ok_hull=ok_hull, hull_v=len(hpx), ok_nonoverlap=ok_nonoverlap,
        ov_pairs=ov_pairs, ov_max=float(ov_max)/2,
        ok_area=ok_area, s2_tri=float(s2_tri)/2, s2_hull=float(s2_hull)/2,
        ok_support=ok_support, support_margin=marginal, ok_edge=ok_edge,
        max_edge_disk=math.sqrt(float(worst_edge)))


def witness_exact(pts, G, u):
    """精确验证一条违例证书: G ∈ D 且 ∀s∈S∩B(G,R): <s−G,u> < 0。返回 (是否成立, 明细)。"""
    Gx, Gy = ex(G[0]), ex(G[1])
    ux, uy = ex(u[0]), ex(u[1])
    inside = (Gx*Gx + Gy*Gy <= RA2)
    sub, worst = [], None
    for p in pts:
        px, py = ex(p[0]), ex(p[1])
        if (px-Gx)**2 + (py-Gy)**2 <= R2:              # 平方比较, 无开方
            sub.append((px, py))
            dp = (px-Gx)*ux + (py-Gy)*uy
            worst = dp if worst is None else max(worst, dp)
    ok = inside and (worst is None or worst < 0)
    return ok, dict(n_sub=len(sub), inside=inside,
                    max_dot=(float(worst) if worst is not None else None),
                    dist_to_hull=(float(math.hypot(u[0], u[1])) if worst is not None else None))


def closest_point_on_hull(G, sub):
    """数值求 G 到 conv(sub) 的最近点(用于生成证书方向; 精确性由 witness_exact 保证)。"""
    hp = hull(sub)
    if len(hp) == 0:
        return None
    if len(hp) == 1:
        return hp[0]
    best, bp = None, None
    segs = [(hp[i], hp[(i+1) % len(hp)]) for i in range(len(hp))] if len(hp) > 2 else [tuple(hp)]
    for a, b in segs:
        dx, dy = b[0]-a[0], b[1]-a[1]
        L2 = dx*dx + dy*dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((G[0]-a[0])*dx + (G[1]-a[1])*dy)/L2))
        q = (a[0]+t*dx, a[1]+t*dy)
        d = math.hypot(G[0]-q[0], G[1]-q[1])
        if best is None or d < best:
            best, bp = d, q
    return bp


def find_witness(pts, seed_pts, n_dir=96, n_rad=120, R=1000.0):
    """找一条违例证书 (G, u): 仅当 G 在 conv(S_G) **外部**才构造证书
    (u = G − 凸包上最近点, 由变分不等式 <s−G,u> <= −|G−q|² < 0 保证严格性)。"""
    from hull_margin import signed_margin

    def probe(G):
        sub = [p for p in pts if (p[0]-G[0])**2 + (p[1]-G[1])**2 <= R*R]
        return signed_margin(G, sub), sub

    cands = []
    for (ax, ay) in seed_pts:
        for k in range(16):
            th = 2*math.pi*k/16
            for rr in (25.0, 60.0, 120.0):
                cands.append((ax + rr*math.cos(th), ay + rr*math.sin(th)))
    best, bG, bsub = None, None, None
    for G in cands:
        if math.hypot(*G) > 1800.0:
            continue
        m, sub = probe(G)
        if best is None or m < best:
            best, bG, bsub = m, G, sub
    if best is None or best >= 0.0:                     # 种子邻域内无违例 -> 粗网格兜底
        for i in range(n_rad):
            r = 1800.0*(i+1)/n_rad
            for j in range(n_dir):
                th = 2*math.pi*j/n_dir
                G = (r*math.cos(th), r*math.sin(th))
                m, sub = probe(G)
                if m < 0.0 and (best is None or m < best):
                    best, bG, bsub = m, G, sub
    if bG is None or best >= 0.0:
        return None
    q = closest_point_on_hull(bG, bsub)
    u = (bG[0]-q[0], bG[1]-q[1])
    if u == (0.0, 0.0):
        return None
    return bG, u, best


def main():
    pts = [tuple(p) for p in r4.Problem4Robot.MESH_DESIGN_PTS]
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    print(f"V4: {len(pts)} 点, {len(tris)} 三角形; 全部判定使用 fractions.Fraction 精确算术")
    ok, info = prove_A(pts, tris)
    print("\n=== 命题 A: V4 满足充要判据(六个前提逐项精确复核) ===")
    print(f"  ① 三角形顶点均取自点集: {info['ok_verts']}")
    print(f"  ② 凸包由 Fraction 叉积重新构造(非浮点)且严格凸: {info['ok_hull']}"
          f"  (凸包 {info['hull_v']} 顶点; 27 点均位于每条有向边同一侧)")
    print(f"  ③ 37 个三角形两两内部不重叠(精确求交面积): {info['ok_nonoverlap']}"
          f"  (重叠对数 {info['ov_pairs']}, 最大交面积 {info['ov_max']:.3e} m²)")
    print(f"  ④ 面积加性 Σ|2S_三角形| = |2S_凸包|: {info['ok_area']}"
          f"  ({info['s2_tri']:.3f} = {info['s2_hull']:.3f} m²)")
    print(f"  ⑤ 凸包每条边到原点距离 >= 1800 m: {info['ok_support']}"
          f"  (最紧支撑边余量 {info['support_margin']:.4f} m)")
    print(f"  ⑥ 与圆盘相交的三角形最大边 <= 1000 m: {info['ok_edge']}"
          f"  (最大边 {info['max_edge_disk']:.4f} m)")
    print(f"  ==> 命题 A {'成立(闭集 + 不重叠 + 面积相等 ⟹ 三角形并集 = 凸包 ⊇ 圆盘)' if ok else '不成立'}")

    print("\n=== 命题 B: 逐个删除站点后的显式违例证书(精确验证) ===")
    m0, g0, k0 = margin_of(pts, n_dir=240, n_rad=200)
    print(f"  (V4 自身在充要判据下的数值余量 = {m0:+.2f} m, 最紧处 G=({g0[0]:.0f},{g0[1]:.0f}), "
          f"子集 {k0} 点)")
    seeds = [g0] + [p for p in pts]
    allok, bad = True, []
    print(f"  {'删除站点':>8}{'证书 G':>22}{'|S_G|':>7}{'最大内积':>12}{'违例距离(m)':>13}{'精确验证':>10}")
    for i in range(len(pts)):
        red = pts[:i] + pts[i+1:]
        w = find_witness(red, seeds)
        if w is None:
            allok = False
            bad.append(i)
            print(f"  {i:>8}{'-':>22}{'-':>7}{'-':>12}{'-':>13}{'未找到':>10}")
            continue
        G, u, m = w
        okw, det = witness_exact(red, G, u)
        allok = allok and okw
        if not okw:
            bad.append(i)
        print(f"  {i:>8}{f'({G[0]:.2f},{G[1]:.2f})':>22}{det['n_sub']:>7}"
              f"{det['max_dot']:>12.4f}{m:>13.2f}{('PASS' if okw else 'FAIL'):>10}", flush=True)
    print(f"\n  ==> 命题 B {'成立: 27/27 删除方案均给出精确违例证书 ⟹ V4 为包含极小(inclusion-minimal)可行站集' if allok else '未全部通过: ' + str(bad)}")
    print("\n结论: ① 命题 A 为**机器辅助精确证明**(六前提 + 闭集/不重叠/面积相等的并集=凸包论证), "
          "即 V4 满足充要判据;")
    print("      ② 命题 B 证明 V4 为**包含极小**可行站集(27 条显式证书, 逐条精确验证): "
          "从 V4 删除任何单个站点都会失效。")
    print("      ③ **适用范围(须严格限定)**: 命题 B 只覆盖 V4 的 27 个单站删除子集, "
          "**不能**推出'任何 26 点布局均不可行', 也不构成'27 点为全局最少站数'的证明;")
    print("         坐标完全不同的 22-26 点可行布局是否存在, 本文未证明。")


if __name__ == "__main__":
    main()
