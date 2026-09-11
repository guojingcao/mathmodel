# -*- coding: utf-8 -*-
"""
问题1 检验脚本
交会定位法：由多个检测点示向度求"定位区域"多边形及其直径，
并验证"以定位区域直径为直径的圆能否覆盖该区域"(Jung 定理)。
"""
import numpy as np
from itertools import combinations

DEG = np.pi / 180.0
EPS = 1.0 * DEG          # 示向度误差 ±1°
R_AREA = 1800.0          # 目标区域半径

# ---------------------------------------------------------------
# 1. 半平面表示：ax + by <= c   (法向量指向楔形内部)
# ---------------------------------------------------------------
def wedge_halfplanes(S, theta_deg):
    """检测点 S, 示向度 theta(deg) -> 两个半平面 (n1, c1), (n2, c2)  (n·P >= c)"""
    th = theta_deg * DEG
    # 上边界射线方向 theta+eps, 法向量逆时针旋转 -90°(即顺时针90°)，指向楔形内部
    n1 = np.array([np.cos(th + EPS - np.pi/2), np.sin(th + EPS - np.pi/2)])
    # 下边界射线方向 theta-eps, 法向量逆时针旋转 +90°，指向楔形内部
    n2 = np.array([np.cos(th - EPS + np.pi/2), np.sin(th - EPS + np.pi/2)])
    return [(n1, float(n1 @ S)), (n2, float(n2 @ S))]

def clip_polygon(poly, n, c):
    """用半平面 n·P >= c 裁剪凸多边形 poly(顶点列表,顺时针或逆时针均可)"""
    if len(poly) == 0:
        return []
    out = []
    m = len(poly)
    for i in range(m):
        A = poly[i]
        B = poly[(i + 1) % m]
        dA = float(n @ A) - c
        dB = float(n @ B) - c
        Ain = dA >= -1e-9
        Bin = dB >= -1e-9
        if Ain:
            out.append(A)
        if Ain != Bin:
            # 交点
            t = dA / (dA - dB)
            out.append(A + t * (B - A))
    return out

def localization_polygon(stations, thetas_deg, box=2.0*R_AREA):
    """返回定位区域多边形顶点列表(凸包)。"""
    # 初始为大方框
    poly = [np.array([-box, -box]), np.array([box, -box]),
            np.array([box, box]), np.array([-box, box])]
    for S, th in zip(stations, thetas_deg):
        for n, c in wedge_halfplanes(np.array(S, float), th):
            poly = clip_polygon(poly, n, c)
            if len(poly) == 0:
                return []
    return poly

def polygon_diameter(vertices):
    """凸多边形直径 = 顶点对最大距离(暴力, 顶点数少)。"""
    best = 0.0
    best_pair = None
    for a, b in combinations(vertices, 2):
        d = float(np.linalg.norm(a - b))
        if d > best:
            best = d
            best_pair = (a, b)
    return best, best_pair

def minimal_enclosing_circle(points):
    """O(m^3) 最小覆盖圆：枚举2点圆(直径)与3点圆(外接圆)。"""
    pts = [np.array(p, float) for p in points]
    n = len(pts)
    if n == 0:
        return None
    if n == 1:
        return pts[0], 0.0
    best_r = np.inf
    best_c = None
    def covers(c, r):
        return all(np.linalg.norm(p - c) <= r + 1e-9 for p in pts)
    # 两点圆
    for i, j in combinations(range(n), 2):
        c = (pts[i] + pts[j]) / 2
        r = np.linalg.norm(pts[i] - pts[j]) / 2
        if covers(c, r) and r < best_r:
            best_r, best_c = r, c
    # 三点外接圆
    for i, j, k in combinations(range(n), 3):
        A, B, C = pts[i], pts[j], pts[k]
        ax, ay = A; bx, by = B; cx, cy = C
        d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-12:
            continue
        ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by))/d
        uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax))/d
        c = np.array([ux, uy])
        r = np.linalg.norm(A - c)
        if covers(c, r) and r < best_r:
            best_r, best_c = r, c
    return best_c, best_r

# ---------------------------------------------------------------
# 检验示例：真源 G 附近布置检测点，示向度含 ±1° 误差
# ---------------------------------------------------------------
def run_case(tag, G, stations, deltas_deg):
    print("=" * 60)
    print(f"[{tag}] 真源 G = ({G[0]:.1f}, {G[1]:.1f})")
    thetas = []
    for S in stations:
        d = np.array(G) - np.array(S)
        th0 = np.degrees(np.arctan2(d[1], d[0])) % 360
        thetas.append(th0)
    # 加入误差(这里取误差上界附近, 构造较"松"的定位区域)
    thetas_err = [(th + dl) % 360 for th, dl in zip(thetas, deltas_deg)]

    poly = localization_polygon(stations, thetas_err)
    if len(poly) == 0:
        print("  定位区域为空(检测点配置不合理)")
        return
    D, pair = polygon_diameter(poly)
    # 验证 G 是否在区域内
    inside = all(n @ (np.array(G) - S) >= -1e-9
                 for S, th in zip(stations, thetas_err)
                 for n, c in wedge_halfplanes(np.array(S, float), th))
    print(f"  检测点数 = {len(stations)}, 多边形顶点数 = {len(poly)}")
    print(f"  真源 G 在定位区域内? {inside}")
    print(f"  定位区域直径 D = {D:.2f} m")
    print(f"  直径端点 = ({pair[0][0]:.1f},{pair[0][1]:.1f}) - ({pair[1][0]:.1f},{pair[1][1]:.1f})")
    # 顶点两两距离的最大值核对
    dmax = max(np.linalg.norm(a-b) for a, b in combinations(poly, 2))
    print(f"  顶点两两最大距离(核对) = {dmax:.2f} m  (与 D 一致? {abs(dmax-D)<1e-6})")
    # 最小覆盖圆
    c, r = minimal_enclosing_circle(poly)
    print(f"  最小覆盖圆半径 r* = {r:.2f} m   (D/2 = {D/2:.2f}, D/√3 = {D/np.sqrt(3):.2f})")
    print(f"  半径 D/2 的圆能否覆盖? {'能' if r <= D/2+1e-6 else '不能'}  "
          f"({'r*=D/2' if abs(r-D/2)<1e-6 else ('r*>D/2' if r>D/2 else 'r*<D/2')})")
    return dict(poly=poly, D=D, r=r, inside=inside)

if __name__ == "__main__":
    np.random.seed(0)
    G = np.array([0.0, 0.0])
    # 情形A：两个检测点，接近正交交会 -> 近似矩形(可被 D/2 圆覆盖)
    run_case("A-正交交会", G,
             [(-800.0, -600.0), (900.0, -500.0)],
             [1.0, -1.0])
    # 情形B：三个检测点均匀分布 -> 近似正六边形(饱满, r*>D/2)
    run_case("B-三站包围(近等边)",
             G,
             [(1000, 0.0), (-500, 866.0), (-500, -866.0)],
             [1.0, 0.0, -1.0])
    # 情形C：两站近乎共线(小交会角) -> 狭长区域, 直径很大
    run_case("C-小交会角", G,
             [(1000.0, 0.0), (1100.0, 60.0)],
             [0.0, 0.0])
    # 情形D：三站环绕但距离差异大 -> 饱满多边形, r* 明显大于 D/2
    run_case("D-三站距离悬殊(饱满)",
             G,
             [(1400.0, 0.0), (-400.0, 500.0), (-500.0, -400.0)],
             [1.0, -1.0, 0.0])
    # 情形E：直接演示"等边三角形"这一通用反例(非定位区域, 仅说明 Jung 上界可达)
    tri = [np.array([0.0, 0.0]), np.array([100.0, 0.0]), np.array([50.0, 86.6025])]
    Dt, _ = polygon_diameter(tri)
    ct, rt = minimal_enclosing_circle(tri)
    print("=" * 60)
    print(f"[E-等边三角形(通用反例)] 边长(直径)D = {Dt:.2f}")
    print(f"  最小覆盖圆半径 r* = {rt:.2f} = D/√3 = {Dt/np.sqrt(3):.2f};  D/2 = {Dt/2:.2f}")
    print(f"  半径 D/2 的圆能否覆盖? {'能' if rt<=Dt/2+1e-6 else '不能'}  (r* = D/√3 > D/2)")
