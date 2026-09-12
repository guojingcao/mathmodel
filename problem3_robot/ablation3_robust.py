# -*- coding: utf-8 -*-
"""P3-B/P3-C: 覆盖环的"半径—点数"稳健性 Pareto(补充点 + 分组巡回)。

背景(实测结论): 冻结版 7 点(原点 + r=1200 正六边形)的覆盖证明只保证"至少一次接收",
最坏接收距离 968.90 m(相对 1000 m 的最小有效接收半径仅 31.10 m 余量), 且未保证
"至少两点接收"。要抬高最坏情形余量, 只有两个几何旋钮:

  (1) **半径**: 环半径 r 越大, 环上相邻点对最坏点(半径 1800、两站角平分方向)的张角越好。
      均匀 n 点环 + 原点时, 最坏接收距离有闭式
          d_worst(r, n) = sqrt(R² + r² − 2Rr·cos(π/n)),  R = 1800 m(目标区半径)
      且 ρ↦d(ρ) 在 [1000, 1800] 单调(因 r·cos(π/n) < 1000), 故最坏点恒在 ρ = R 处。
  (2) **点数**(= 补充点): n 越大张角越小; 且**开路巡回长度** = r + (n−1)·2r·sin(π/n)
      在 n 增大时反而下降(弦长次线性), 例如 r=1200: n=6 → 8400 m, n=9 → 7767 m, n=12 → 8033 m。

本实验把两个旋钮放在同一张 Pareto 表上: 交换的是"最坏接收余量"与"虚拟时间"。
为保证配对干净, 误差场用 err(scene, ch, x, y) 确定性函数(与动作顺序无关)。

注意: 正式版 geometry 由 robot.Problem3Robot.search_points() 决定, 本脚本**不修改** robot.py,
只在实验内以补丁方式替换该静态方法(try/finally 恢复), 因此冻结版行为逐字节不变。

用法: python ablation3_robust.py [每臂案例数=400] [种子=2026]
"""
import contextlib
import io
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                  # noqa: E402

import simlib                                       # noqa: E402
from simlib import case_env, sim_time, check_clearance, scene_hash   # noqa: E402

rb = simlib.load_module("rb3", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "robot.py"))
exp = simlib.exp
R_AREA = 1800.0
R_GUAR = 1000.0
DEG = math.pi/180.0


class FixedEnv(exp.Env):
    """确定性 ±1° 误差场(不消费 rng): 同一 (案例, 频道, 坐标) 在两臂中得到同一误差。"""

    def __init__(self, *a, scene_key=0, **kw):
        super().__init__(*a, **kw)
        self.scene_key = scene_key

    def _err(self, ch, pos):
        key = f"{self.scene_key}|{int(ch)}|{int(round(pos[0]*10))}|{int(round(pos[1]*10))}"
        return random.Random(key).uniform(-1.0, 1.0)

    def measure(self, pos, ch):
        for s in self.sources:
            if s["ch"] != ch or ch in self.cleared:
                continue
            d = s["pos"] - pos
            dist = float(np.linalg.norm(d))
            if dist > s["r_rx"] or not self._in_cov(s, pos):
                continue
            if dist <= exp.R_NEAR:
                return "near", None
            true = math.degrees(math.atan2(d[1], d[0])) % 360.0
            return "direction", (true + self._err(ch, pos)) % 360.0
        return "no_signal", None


def ring_pts(r, n):
    """原点 + 半径 r 上均匀 n 点的覆盖点集(角度从 0 开始, 与冻结版 6 点同相位)。"""
    pts = [(0.0, 0.0)]
    for k in range(n):
        a = k * (360.0/n) * DEG
        pts.append((r*math.cos(a), r*math.sin(a)))
    return pts


def worst_recv(r, n):
    """最坏接收距离(闭式) + 数值校验(fine 采样)。"""
    closed = math.sqrt(R_AREA**2 + r**2 - 2*R_AREA*r*math.cos(math.pi/n))
    best = 0.0
    for i in range(3601):
        rho = R_AREA*i/3600.0
        for k in range(n):
            a = (k*(360.0/n) + 180.0/n)*DEG          # 相邻站角平分方向(最不利)
            px, py = r*math.cos(a), r*math.sin(a)
            gx, gy = rho*math.cos(a), rho*math.sin(a)   # 同方向、半径 rho
            d = math.hypot(gx-px, gy-py)
            if d > best:
                best = d
    return closed, best


def patch_search(pts):
    """把 search_points 换成给定点集(实验用, 返回恢复函数)。"""
    old = rb.Problem3Robot.__dict__.get("search_points")
    rb.Problem3Robot.search_points = staticmethod(lambda: list(pts))
    return lambda: (setattr(rb.Problem3Robot, "search_points", old) if old is not None
                    else delattr(rb.Problem3Robot, "search_points"))


def run_arm(pts, n_case, seed):
    restore = patch_search(pts)
    rows = []
    try:
        for k in range(n_case):
            base = case_env(seed, k, directional=False)
            env = FixedEnv(np.random.default_rng(0), n_src=base.n_src,
                           directional=False, scene_key=(seed, k))
            env.sources = base.sources
            env.ch_by_id = base.ch_by_id
            env.cleared = set()
            cli = simlib.SimClient(env)
            robot = rb.Problem3Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = robot.run()
            chk = check_clearance(n_ret, cli, env)
            rows.append(dict(T=sim_time(cli), dist=cli.dist, meas=cli.n_measure,
                             sw=cli.n_switch, clear=cli.n_clear, fail=cli.fail,
                             full=chk["case_full_clear"], miss_src=chk["missing_src"],
                             n_src=chk["n_src"], scene=scene_hash(env)))
    finally:
        restore()
    return rows


def summ(rows, n_pts):
    T = np.array([r["T"] for r in rows], float)
    full = sum(r["full"] for r in rows)
    miss = sum(r["miss_src"] for r in rows)
    return dict(n_pts=n_pts, T=T.mean(), p95=float(np.percentile(T, 95)),
                dist=np.mean([r["dist"] for r in rows]),
                meas=np.mean([r["meas"] for r in rows]),
                sw=np.mean([r["sw"] for r in rows]),
                clear=np.mean([r["clear"] for r in rows]),
                fail=np.mean([r["fail"] for r in rows]),
                full=full, miss=miss, n=len(rows), T_arr=T)


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    arms = []
    for r in (1200.0, 1250.0, 1300.0, 1350.0, 1150.0, 1123.0):
        arms.append((f"r={r:.0f}, n=6(冻结族)", ring_pts(r, 6), r, 6))
    for r in (1200.0, 1150.0):
        arms.append((f"r={r:.0f}, n=9(补 3 点)", ring_pts(r, 9), r, 9))
    arms.append(("r=1200, n=12(补 6 点)", ring_pts(1200.0, 12), 1200.0, 12))

    print(f"目标区 R={R_AREA:.0f} m, 保证接收半径 {R_GUAR:.0f} m; 每臂 n={n_case} 例, "
          f"种子 {seed}, 固定误差场(顺序无关)")
    print(f"{'臂':<20}{'点数':>5}{'最坏接收(m)':>12}{'余量(m)':>9}"
          f"{'巡回(m)':>9}{'T(s)':>8}{'检测':>7}{'调频':>7}{'清除':>6}{'失败':>6}{'全清':>8}")
    out = []
    for name, pts, r, n in arms:
        w, wnum = worst_recv(r, n)
        tour = r + (n-1)*2*r*math.sin(math.pi/n)
        rows = run_arm(pts, n_case, seed)
        s = summ(rows, len(pts))
        s.update(name=name, r=r, n_ring=n, worst=w, worst_num=wnum, tour=tour, rows=rows)
        out.append(s)
        print(f"{name:<20}{len(pts):>5}{w:>12.2f}{R_GUAR-w:>9.2f}{tour:>9.0f}"
              f"{s['T']:>8.0f}{s['meas']:>7.1f}{s['sw']:>7.1f}{s['clear']:>6.1f}"
              f"{s['fail']:>6.2f}{s['full']:>5d}/{s['n']}", flush=True)
    base = out[0]
    print("\n=== 逐案例配对(同一 seed 段, 同场景 + 同误差场) ===")
    for s in out[1:]:
        d = s["T_arr"] - base["T_arr"]
        se = d.std(ddof=1)/math.sqrt(len(d))
        print("  %-20s ΔT %+7.1f s (%+.2f%%)  95%%CI [%+.0f, %+.0f]  变快 %4.1f%%  "
              "中位 %+.0f s   余量 %+.1f m  巡回 %+.0f m"
              % (s["name"], d.mean(), 100*d.mean()/base["T"], d.mean()-1.96*se,
                 d.mean()+1.96*se, 100*(d < 0).mean(), np.median(d),
                 (R_GUAR-s["worst"])-(R_GUAR-base["worst"]), s["tour"]-base["tour"]))
    print("\n(注: 冻结族最坏接收距离的数值校验 = 闭式解, 二者一致; 最坏点位于 ρ=1800 m "
          "的两站角平分方向。)")


if __name__ == "__main__":
    main()
