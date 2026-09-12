"""问题四"方向鲁棒覆盖"判据核验: S ∈ conv Q(S)。

判据(附件给出的准确几何表达):
    对假设源位置 S, 取"距 S 不超过 1000 m 且对该频道完成了有效无信号检测"的站点集 Q(S)。
    若 S ∈ conv Q(S), 则无论源朝哪个方向, 至少一个站点落在其 180° 发射半平面内,
    而该站点又必然在接收半径内 ⇒ 应收到信号, 与"无信号"矛盾 ⇒ S 处不可能存在未清除源。

本脚本核验:
  (1) 用**全部网格点**构成 Q(S)(理想情形, 上界)时, 圆盘内采样点满足 S ∈ conv Q(S) 的比例;
  (2) 用**包含 S 的三角形三顶点**构成 Q(S)(即 robot4 实际使用的证书集合)时同样比例;
  (3) 最坏情形: 不满足的点到 conv Q(S) 的距离, 以及它们到最近网格点的距离;
  (4) 结论对覆盖率(§6.2 的空洞修复)是否敏感。

用法: python dircov_audit.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                      # noqa: E402

import robot4 as r4                     # noqa: E402

R = r4.R_AREA
RX_MIN = 1000.0     # 接收半径下界: 站距源 <=1000 m 时必然可接收


def in_tri(p, a, b, c):
    def cr(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
    d1 = cr(a, b, p); d2 = cr(b, c, p); d3 = cr(c, a, p)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def hull(points):
    """单调链求凸包(逆时针), 返回顶点列表。"""
    ps = sorted(set((round(x, 9), round(y, 9)) for x, y in points))
    if len(ps) <= 2:
        return ps

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


def in_conv(p, hp):
    """点是否在凸包内(含边界); hp 为凸包顶点(逆时针)。"""
    if len(hp) < 3:
        return False
    n = len(hp)
    for i in range(n):
        a, b = hp[i], hp[(i+1) % n]
        if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) < -1e-9:
            return False
    return True


def dist_to_hull(p, hp):
    """点若在凸包外, 到凸包的最小距离(用于报告最坏情形)。"""
    if len(hp) < 2:
        return float("inf") if not hp else math.hypot(p[0]-hp[0][0], p[1]-hp[0][1])
    if in_conv(p, hp):
        return 0.0
    best = float("inf")
    n = len(hp)
    for i in range(n):
        a, b = hp[i], hp[(i+1) % n]
        dx, dy = b[0]-a[0], b[1]-a[1]
        L2 = dx*dx + dy*dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L2))
        best = min(best, math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy)))
    return best


def main():
    rb = r4.Problem4Robot(None)
    pts = [tuple(p) for p in rb.pts]
    print(f"网格: {len(pts)} 点 / {len(rb.tris)} 三角形; 接收半径下界 {RX_MIN:.0f} m")
    rng = np.random.default_rng(23)
    N = 40000
    stats = {}
    for tag, lo in (("均匀", None), ("环1700-1800", 1700.0), ("环1780-1800", 1780.0)):
        ok_all = ok_tri = 0
        nq = []
        worst_all = worst_tri = 0.0
        worst_pt = None
        for _ in range(N):
            r = R*math.sqrt(rng.uniform()) if lo is None else rng.uniform(lo, R)
            a = rng.uniform(0, 2*math.pi)
            S = (r*math.cos(a), r*math.sin(a))
            # (1) Q(S) = 全部在 1000 m 内的网格点
            Q = [p for p in pts if math.hypot(p[0]-S[0], p[1]-S[1]) <= RX_MIN]
            nq.append(len(Q))
            hp = hull(Q)
            if in_conv(S, hp):
                ok_all += 1
            else:
                d = dist_to_hull(S, hp)
                if d > worst_all:
                    worst_all, worst_pt = d, S
            # (2) Q(S) = 包含 S 的三角形三顶点(robot4 实际证书集合)
            hit = None
            for t in rb.tris:
                if in_tri(S, pts[t[0]], pts[t[1]], pts[t[2]]):
                    hit = [pts[t[0]], pts[t[1]], pts[t[2]]]
                    break
            if hit is not None and in_conv(S, hull(hit)):
                ok_tri += 1
            elif hit is not None:
                worst_tri = max(worst_tri, dist_to_hull(S, hull(hit)))
        stats[tag] = (ok_all, ok_tri, worst_all, worst_tri, worst_pt, np.mean(nq))
        print(f"\n[{tag}] n={N}")
        print(f"  (1) Q(S)=全部≤1000 m 网格点   : 满足 S∈conv Q(S) {ok_all}/{N} "
              f"({100.0*ok_all/N:.4f}%); 不满足者到凸包最远 {worst_all:.2f} m")
        print(f"  (2) Q(S)=包含 S 的三角形三顶点: 满足 {ok_tri}/{N} "
              f"({100.0*ok_tri/N:.4f}%); 不满足者最远 {worst_tri:.2f} m")
        print(f"  平均 |Q(S)| = {np.mean(nq):.1f} 个站点"
              + (f"; 最坏点示例 ({worst_pt[0]:.1f},{worst_pt[1]:.1f})" if worst_pt else ""))
    print("\n结论: 只要 S 落在某个三角形内, 其三个顶点都在 1000 m 内且 S∈conv(三顶点) "
          "=> 方向鲁棒覆盖条件自动满足; 因此该判据与本文'三角形证书 + 最大边≤920 m'等价, "
          "而它的成立恰好依赖 §6.2 修复后的**完全覆盖**(空洞处 S 不在任何三角形内, 条件失效)。")


if __name__ == "__main__":
    main()
