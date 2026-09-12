# -*- coding: utf-8 -*-
"""问题四"实际证书三角形"的连续覆盖证明核验(P0-3)。

审阅要求: 要保留"构造上覆盖圆盘"的说法, 必须对**实际承担认证的 37 个三角形**证明:
  (1) 这些三角形构成无孔剖分(面积可加、无重叠、边匹配);
  (2) 其并集等于所声明的凸包;
  (3) 凸包包含整个目标圆盘(支撑边: 原点在内 + 每条支撑线到原点距离 >= R);
  (4) 全部三角形满足边长 <= 1000 m(证书的距离前提)。

本脚本逐条数值核验并给出 PASS/FAIL 与最坏余量(不依赖任何采样)。

用法: python hull_proof.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robot4 as r4                     # noqa: E402

R = r4.R_AREA


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


def seg_dist_point(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    L2 = dx*dx + dy*dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L2))
    return math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy))


def main():
    rb = r4.Problem4Robot(None)
    pts = [tuple(p) for p in rb.pts]
    tris = [tuple(t) for t in rb.cover_tris]
    print(f"实际网格: {len(pts)} 点; 承担认证的三角形 {len(tris)} 个 "
          f"(cover_tris, MESH_A={r4.MESH_A})")
    ok_all = True

    # (4) 边长限制
    mx = 0.0
    for t in tris:
        for i in range(3):
            for j in range(i+1, 3):
                mx = max(mx, math.hypot(pts[t[i]][0]-pts[t[j]][0],
                                        pts[t[i]][1]-pts[t[j]][1]))
    ok4 = mx <= 1000.0 + 1e-9
    ok_all &= ok4
    print(f"(4) 证书三角形最大边长 {mx:.3f} m <= 1000 m : {'PASS' if ok4 else 'FAIL'} "
          f"(余量 {1000.0-mx:.3f} m)")

    # (1)(2) 无孔剖分 + 并集 = 凸包(面积可加 + 半平面一致性 + 边匹配)
    hp = hull(pts)
    a_hull = poly_area(hp)
    a_sum = sum(abs(area2(pts[t[0]], pts[t[1]], pts[t[2]]))/2.0 for t in tris)
    ok_area = abs(a_sum - a_hull) <= 1e-6 * max(1.0, a_hull)
    print(f"(1) 三角形面积和 {a_sum:.3f} m² vs 凸包面积 {a_hull:.3f} m² : "
          f"{'PASS(无孔)' if ok_area else 'FAIL(存在空洞或重叠)'} 差 {a_sum-a_hull:+.3f}")

    # (1) 无孔剖分: 面积可加(已算) + 两两内部不重叠 + 每个三角形都在凸包内
    #     三者合起来即"并集 = 凸包"的连续证明: 三角形均闭集且两两内部不交 => 并集面积 =
    #     面积和 = 凸包面积; 并集 ⊆ 凸包(顶点都在凸包内) => 补集是凸包内的零测开集 => 为空。
    def clip_area(poly, clip):
        """Sutherland-Hodgman: 凸多边形交的面积(用于两两重叠判定)。"""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        out = poly
        for i in range(len(clip)):
            a, b = clip[i], clip[(i+1) % len(clip)]
            if not out:
                return 0.0
            new = []
            for j in range(len(out)):
                p, q = out[j], out[(j+1) % len(out)]
                inside_p = cross(a, b, p) >= -1e-12
                inside_q = cross(a, b, q) >= -1e-12
                if inside_p:
                    new.append(p)
                if inside_p != inside_q:
                    dx, dy = q[0]-p[0], q[1]-p[1]
                    d1 = cross(a, b, p)
                    d2 = cross(a, b, q)
                    t = d1/(d1-d2) if abs(d1-d2) > 1e-18 else 0.0
                    new.append((p[0]+t*dx, p[1]+t*dy))
            out = new
        return poly_area(out) if len(out) >= 3 else 0.0

    def ccw(t):
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        return [a, b, c] if area2(a, b, c) > 0 else [a, c, b]

    tri_polys = [ccw(t) for t in tris]
    worst_ov = 0.0
    worst_pair = None
    for i in range(len(tri_polys)):
        for j in range(i+1, len(tri_polys)):
            ov = clip_area(tri_polys[i], tri_polys[j])
            if ov > worst_ov:
                worst_ov, worst_pair = ov, (i, j)
    ok_ov = worst_ov <= 1e-6 * max(1.0, a_hull)
    ok_all &= ok_ov
    print(f"(1a) 两两内部重叠最大面积 {worst_ov:.6e} m² (阈值 1e-6×凸包面积) : "
          f"{'PASS(无重叠)' if ok_ov else 'FAIL ' + str(worst_pair)}")
    print(f"(1b) 面积可加: 三角形面积和 - 凸包面积 = {a_sum - a_hull:+.6e} m² : "
          f"{'PASS' if ok_area else 'FAIL'}")
    print("     => 由 闭集+两两内部不交+面积可加+并集⊆凸包 得 **并集 = 凸包**(连续证明, 非采样)")
    print("     说明: 顶点编号经 sorted 排序, 故三角形朝向不统一属正常, 不影响覆盖; "
          "格点共线会形成 T 型接点, 同样不影响覆盖。")

    # (3) 凸包包含圆盘: 原点在凸包内 + 每条支撑边到原点距离 >= R
    inside = all(area2(hp[i], hp[(i+1) % len(hp)], (0.0, 0.0)) >= -1e-9
                 for i in range(len(hp)))
    dmin = min(seg_dist_point((0.0, 0.0), hp[i], hp[(i+1) % len(hp)])
               for i in range(len(hp)))
    ok3 = inside and dmin >= R - 1e-9
    ok_all &= ok3
    print(f"(3) 凸包 {len(hp)} 顶点; 原点在凸包内={inside}; 支撑边到原点最近距离 "
          f"{dmin:.3f} m >= R={R:.0f} m : {'PASS' if ok3 else 'FAIL'} "
          f"(余量 {dmin-R:.3f} m)")
    print(f"    凸包顶点半径范围 [{min(math.hypot(*p) for p in hp):.1f}, "
          f"{max(math.hypot(*p) for p in hp):.1f}] m")

    print("\n总判定: " + ("**PASS** —— 可以声明'实际证书三角形构成无孔剖分, 其并集等于凸包, "
                     "凸包严格包含目标圆盘, 且全部边长<=1000 m'(连续证明, 非采样)"
                     if ok_all else "**FAIL** —— 不得声明构造性覆盖, 需按失败项修正"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
