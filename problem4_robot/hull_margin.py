# -*- coding: utf-8 -*-
"""P4-D: 直接凸包判据(充要条件)的连续余量与变站数帕累托。

判据(§6.7 已证等价): 某频道可被确定性判为空  <=>  ∀G∈D: G ∈ conv(S ∩ B(G,R_c)), R_c=1000 m。
部署实现用的是**充分**判据(三角形 + 盘内最大边 ≤ 1000 m), 本节换用**充要**判据本身, 度量:

    margin(S) = min_{G∈D} dist_signed(G, conv(S ∩ B(G,R_c)))
               正值 = G 在子集凸包内部(余量多少米), 负值 = 违例(离凸包多远)。
    margin(S) > 0  <=> 充要判据成立。

由此可问: 站数从 27 往下减, margin 何时变负? 减少的站能省多少时间? —— 即"站数下界"的
实证版本(§6.7 曾只给出保守判据下的 20-24 站与 2.2 % 收益上界)。

另给**逐方向精确核验**(不依赖采样): 固定方向 u 后, 违例条件退化为 1-D:
    ∃y: [−W(y),W(y)] 不被 ∪_{s: h(s)>=y} {x: (x−x_s)^2+(y−y_s)^2 <= R^2} 覆盖,
其中 h,x 为沿 u/u⊥ 的坐标, W(y)=sqrt(R_area^2−y^2)。端点的相交事件只发生在
圆-圆交点与 圆-边界交点处(闭式), 故把 y 按这些临界值分段后, 每段内"覆盖结构"不变,
取中点判定即为该方向下的**精确**结论。

用法: python hull_margin.py [阶段] [案例数=60]
  阶段: margin(默认, 只算 V4 余量) | pareto(贪心删站 + 时间配对) | exact(逐方向精确核验)
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import robot4 as r4                                     # noqa: E402
import selfcheck4 as sc4                                # noqa: E402
from ablation4_mecfreeze import FixedEnv                # noqa: E402
from simlib import case_env, sim_time, check_clearance  # noqa: E402

R_AREA, R_C = 1800.0, 1000.0
P_DIR = 0.5


# ---------------- 几何: 凸包与带符号距离 ----------------
def hull(pts):
    ps = sorted(set((round(x, 9), round(y, 9)) for x, y in pts))

    def half(q):
        out = []
        for p in q:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2-x1)*(p[1]-y1) - (y2-y1)*(p[0]-x1) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(p)
        return out
    return half(ps)[:-1] + half(ps[::-1])[:-1]


def seg_dist(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    L2 = dx*dx + dy*dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L2))
    return math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy))


def signed_margin(G, sub):
    """G 到 conv(sub) 的带符号距离: 内部为正(到边最小距离), 外部为负, 退化子集按外部处理。"""
    if len(sub) == 1:
        return -math.hypot(G[0]-sub[0][0], G[1]-sub[0][1])
    if len(sub) == 2:
        return -seg_dist(G, sub[0], sub[1])
    hp = hull(sub)
    if len(hp) < 3:
        return -min(seg_dist(G, hp[i], hp[(i+1) % len(hp)]) for i in range(len(hp)))
    # 有向面积判内外(允许退化: 用最小有向边距)
    dmin = None
    inside = True
    for i in range(len(hp)):
        a, b = hp[i], hp[(i+1) % len(hp)]
        cr = (b[0]-a[0])*(G[1]-a[1]) - (b[1]-a[1])*(G[0]-a[0])
        L = math.hypot(b[0]-a[0], b[1]-a[1])
        d = cr/L if L > 0 else 0.0                     # 正值表示 G 在边左侧(逆时针凸包内侧)
        if cr < 0:
            inside = False
        dmin = d if dmin is None else min(dmin, d)
    if inside:
        return max(0.0, dmin)
    return -min(seg_dist(G, hp[i], hp[(i+1) % len(hp)]) for i in range(len(hp)))


def margin_of(pts, n_dir=360, n_rad=300, R=R_C):
    """在极坐标网格上求 min 带符号余量。返回 (margin, argmin_G, argmin_subset_size)。"""
    P = np.array([(x, y) for x, y in pts])
    ang = np.linspace(0, 2*math.pi, n_dir, endpoint=False)
    rad = np.linspace(R_AREA/n_rad, R_AREA, n_rad)
    A, Rd = np.meshgrid(ang, rad)
    GX, GY = (Rd*np.cos(A)).ravel(), (Rd*np.sin(A)).ravel()
    best, bx, by, bk = None, 0.0, 0.0, 0
    for i in range(GX.size):
        gx, gy = GX[i], GY[i]
        d = np.hypot(P[:, 0]-gx, P[:, 1]-gy)
        sub = [tuple(P[j]) for j in np.nonzero(d <= R)[0]]
        m = signed_margin((gx, gy), sub)
        if best is None or m < best:
            best, bx, by, bk = m, gx, gy, len(sub)
    return best, (bx, by), bk


# ---------------- 逐方向精确核验 ----------------
def _circle_circle(s, t, r1, r2):
    """两圆交点(≤2 个)。"""
    d = math.hypot(t[0]-s[0], t[1]-s[1])
    if d < 1e-12 or d > r1+r2 or d < abs(r1-r2):
        return []
    a = (r1*r1 - r2*r2 + d*d)/(2*d)
    h2 = r1*r1 - a*a
    if h2 < 0:
        return []
    h = math.sqrt(max(0.0, h2))
    x0 = s[0] + a*(t[0]-s[0])/d
    y0 = s[1] + a*(t[1]-s[1])/d
    rx, ry = -(t[1]-s[1])*(h/d), (t[0]-s[0])*(h/d)
    return [(x0+rx, y0+ry), (x0-rx, y0-ry)]


def dir_violation(pts, th, R=R_C):
    """固定方向 th 下的精确判定: 返回该方向下最大"未覆盖张开度"(米, >0 即违例)。"""
    u = (math.cos(th), math.sin(th))
    up = (-math.sin(th), math.cos(th))
    hs = [x*u[0] + y*u[1] for x, y in pts]
    xs = [x*up[0] + y*up[1] for x, y in pts]
    ys = list(hs)
    crit = [-R_AREA, R_AREA]
    for h in hs:
        crit += [h, h-R, h+R]
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for p in _circle_circle(pts[i], pts[j], R, R):
                crit.append(p[0]*u[0] + p[1]*u[1])
    for p in pts:
        for q in _circle_circle(p, (0.0, 0.0), R, R_AREA):
            crit.append(q[0]*u[0] + q[1]*u[1])
    crit = sorted(c for c in crit if -R_AREA - 1e-9 <= c <= R_AREA + 1e-9)
    worst = 0.0
    for a, b in zip(crit, crit[1:]):
        if b - a < 1e-9:
            continue
        y = 0.5*(a+b)
        W2 = R_AREA*R_AREA - y*y
        if W2 <= 0:
            continue
        W = math.sqrt(W2)
        iv = []
        for k, h in enumerate(hs):
            if h < y:                                   # 仅"高于等于 G"的站会构成违例
                continue
            dh = h - y
            if abs(dh) > R:
                continue
            w = math.sqrt(max(0.0, R*R - dh*dh))
            lo, hi = max(-W, xs[k]-w), min(W, xs[k]+w)
            if hi > lo:
                iv.append((lo, hi))
        if not iv:
            worst = max(worst, 2*W)
            continue
        iv.sort()
        cur = -W
        for lo, hi in iv:
            if lo > cur:
                worst = max(worst, min(lo, W) - cur)
            cur = max(cur, hi)
        if cur < W:
            worst = max(worst, W - cur)
    return worst


def verify_exact(pts, n_dir=180, R=R_C):
    """逐方向精确核验; 返回 (是否无违例, 最大张开度, 最差方向角度)。"""
    worst, wth = 0.0, None
    for i in range(n_dir):
        th = 2*math.pi*i/n_dir
        v = dir_violation(pts, th, R)
        if v > worst:
            worst, wth = v, math.degrees(th)
    return (worst <= 0.0), worst, wth


# ---------------- 仿真配对 ----------------
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
    return sim_time(cli), cli.n_measure, cli.fail, chk["case_full_clear"]


def evaluate(pts, seed, n_case):
    r4.Problem4Robot.MESH_PTS_OVERRIDE = [tuple(p) for p in pts]
    T, meas, fail, miss = [], 0.0, 0.0, 0
    for k in range(n_case):
        t, m, f, ok = run_case(seed, k)
        T.append(t); meas += m; fail += f
        miss += 0 if ok else 1
    T = np.array(T)
    return dict(T=T, mean=float(T.mean()), meas=meas/n_case, fail=fail/n_case,
                miss=miss, n=n_case)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "margin"
    n_case = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = 6260
    r4.Problem4Robot.ORDER_OVERRIDE = None
    base = [tuple(p) for p in r4.Problem4Robot.MESH_DESIGN_PTS]
    print(f"充要判据: G ∈ conv(S ∩ B(G,{R_C:.0f}))  目标区 R={R_AREA:.0f} m; 站集起点 = V4 {len(base)} 点")

    if stage == "exact":
        for name, pts in (("V4(27 点)", base),
                          ("对照: 只留外环 8 点", [p for p in base
                                                   if math.hypot(*p) > 1400.0][:8])):
            ok, worst, th = verify_exact(pts, n_dir=180)
            print(f"  {name}: 逐方向精确核验 {'无违例(PASS)' if ok else '发现违例(FAIL)'}"
                  f"  最大未覆盖张开度 {worst:.1f} m"
                  + (f"  最差方向 {th:.1f} 度" if th is not None else ""), flush=True)
        return

    print("\n=== V4 连续余量(充要判据) ===")
    m, g, k = margin_of(base)
    print(f"  margin = {m:+.2f} m  最紧处 G=({g[0]:.0f},{g[1]:.0f})  该点子集 {k} 点"
          f"  -> {'判据成立' if m > 0 else '判据违反'}")

    if stage == "margin":
        return

    print("\n=== 贪心删站(删站使余量最大) + 时间配对 ===")
    cur = list(base)
    rows = []
    r = evaluate(cur, seed, n_case)
    rows.append((len(cur), m, r, list(cur)))
    print(f"  n={len(cur)}: margin {m:+.2f} m  T {r['mean']:.0f} s  检测/例 {r['meas']:.1f}")
    while len(cur) > 20:
        best = None
        for i in range(len(cur)):
            cand = cur[:i] + cur[i+1:]
            mc, _, _ = margin_of(cand, n_dir=120, n_rad=150)
            if best is None or mc > best[0]:
                best = (mc, cand, i)
        mc, cand, i = best
        if mc <= 0:
            print(f"  n={len(cur)-1}: 删除任何一站都会使余量转负(最不坏 {mc:+.2f} m) -> 停止")
            break
        mfull, g, k = margin_of(cand)
        cur = cand
        r = evaluate(cur, seed, n_case)
        rows.append((len(cur), mfull, r, list(cur)))
        print(f"  n={len(cur)}: margin {mfull:+.2f} m (删去原第 {i} 点)  T {r['mean']:.0f} s "
              f"({100*(r['mean']-rows[0][2]['mean'])/rows[0][2]['mean']:+.2f}%)  "
              f"检测/例 {r['meas']:.1f}  全清 {r['n']-r['miss']}/{r['n']}", flush=True)
    print("\n=== 汇总(充要判据余量 vs 站数 vs 时间) ===")
    b = rows[0]
    for n, mg, r, pts in rows:
        d = r["T"] - b[2]["T"]
        se = d.std(ddof=1)/math.sqrt(len(d))
        print("  n=%2d  margin %+7.2f m   T %7.0f s  (%+6.2f%%, CI [%+.0f,%+.0f])  检测/例 %5.1f"
              % (n, mg, r["mean"], 100*(r["mean"]-b[2]["mean"])/b[2]["mean"],
                 d.mean()-1.96*se, d.mean()+1.96*se, r["meas"]))
    print("\n(注: V4 的 27 点是用**充分**判据(盘内最大边≤1000)得到的; 本节用**充要**判据重估"
          "站数下界, 两者之差即保守性代价。)")


if __name__ == "__main__":
    main()
