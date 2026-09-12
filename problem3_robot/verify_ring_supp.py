# -*- coding: utf-8 -*-
"""P3-C 候选(补充点环)的门禁核验: 覆盖审计 + 独立种子配对 + 病理集。

候选: 原点 + 半径 r=1150 m 上均匀 9 点(即在冻结六边形的基础上补 3 个分角点)。
  * 最坏接收距离 968.90 → **819.86 m**(相对 1000 m 的余量 31.10 → **180.14 m**);
  * 覆盖开路巡回 (r=1150,n=9): 1150 + 8·2·1150·sin20° = 1150 + 6 293 = 7 443 m,
    对照冻结族 (r=1200,n=6): 1200 + 5·1200 = 7 200 m(+243 m);
  * 冒烟配对(400 例, 固定误差场): **−247.9 s(−5.39 %), CI [−290, −206], 72.2 % 更快**。

本脚本用独立种子段与病理集重新核验, 并给出密集采样覆盖审计(仅作数值旁证,
闭式解 + 单调性论证才是证明主体)。

用法: python verify_ring_supp.py [配对案例数=1000] [配对种子=8188] [病理重复=200]
"""
import contextlib
import io
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import simlib                                           # noqa: E402
from simlib import (case_env, sim_time, check_clearance, scene_hash,
                    config_scope, frozen3, cfg_hash)     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rb = simlib.load_module("rb3", os.path.join(HERE, "robot.py"))
sc3 = simlib.load_module("sc3", os.path.join(HERE, "selfcheck.py"))
exp = simlib.exp
R_AREA, R_GUAR, DEG = 1800.0, 1000.0, math.pi/180.0

ARMS = [("A 冻结 7 点(r=1200,n=6)", 1200.0, 6),
        ("B 候选 10 点(r=1150,n=9)", 1150.0, 9)]


class FixedEnv(exp.Env):
    """确定性 ±1° 误差场: 同一 (案例, 频道, 坐标) 在两臂完全相同(与动作顺序无关)。"""

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


def pts_of(r, n):
    out = [(0.0, 0.0)]
    for k in range(n):
        a = k*(360.0/n)*DEG
        out.append((r*math.cos(a), r*math.sin(a)))
    return out


def worst_closed(r, n):
    return math.sqrt(R_AREA**2 + r**2 - 2*R_AREA*r*math.cos(math.pi/n))


def audit(pts, r, n):
    """覆盖审计: ① 闭式最坏接收距离; ② 径向-角向密集采样旁证。"""
    closed = worst_closed(r, n)
    worst = 0.0
    n_ang, n_rad = 1440, 900
    for ia in range(n_ang):
        a = 2*math.pi*ia/n_ang
        for ir in range(1, n_rad+1):
            rho = R_AREA*ir/n_rad
            gx, gy = rho*math.cos(a), rho*math.sin(a)
            d = min(math.hypot(gx-px, gy-py) for px, py in pts)
            if d > worst:
                worst = d
    return closed, worst, worst <= R_GUAR


def run_paired(pts, n_case, seed):
    rows = []
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
        rows.append(dict(T=sim_time(cli), dist=cli.dist, meas=cli.n_measure,
                         sw=cli.n_switch, clear=cli.n_clear, fail=cli.fail,
                         full=chk["case_full_clear"], miss_src=chk["missing_src"],
                         n_src=chk["n_src"], scene=scene_hash(env)))
    return rows


def run_patho(pts, n_rep, seed):
    """复用 selfcheck 的病理场景构造; 每类 n_rep 例, 记录漏清与时间。"""
    scen = [("随机·13源(对照)", "random", 13), ("上界16源", "random", 16),
            ("下界10源", "random", 10), ("最坏接收(全1000m)", "worst_rx", 13),
            ("全源贴近边界", "edge", 13), ("全源贴近原点", "center", 13),
            ("近简并双源", "degenerate", 13)]
    out = {}
    for label, kind, n in scen:
        rng = np.random.default_rng(seed)
        T, miss = [], 0
        for i in range(n_rep):
            specs = sc3.patho_sources(rng, kind, n)
            base = sc3.make_env(rng, specs)
            env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=False,
                           scene_key=(seed, kind, i))
            env.sources = base.sources
            env.ch_by_id = base.ch_by_id
            env.cleared = set()
            cli = simlib.SimClient(env)
            robot = rb.Problem3Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = robot.run()
            chk = check_clearance(n_ret, cli, env)
            T.append(sim_time(cli))
            if not chk["case_full_clear"]:
                miss += 1
        out[label] = dict(T=np.array(T), miss=miss, n=n_rep)
    return out


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 8188
    n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    cfg = frozen3()
    print(f"冻结配置 cfg_hash={cfg_hash(cfg)}  "
          f"臂: {[a[0] for a in ARMS]}  配对 n={n_case}(种子 {seed})  病理 {n_rep}/类")

    ptsA, ptsB = pts_of(1200.0, 6), pts_of(1150.0, 9)
    print("\n=== 覆盖审计 ===")
    old_sp = rb.Problem3Robot.__dict__.get("search_points")
    for name, pts, r, n in (ARMS[0][0], ptsA, 1200.0, 6), (ARMS[1][0], ptsB, 1150.0, 9):
        closed, sampled, ok = audit(pts, r, n)
        tour = r + (n-1)*2*r*math.sin(math.pi/n)
        print(f"  {name}: 闭式最坏接收 {closed:.2f} m  余量 {R_GUAR-closed:.2f} m  "
              f"密集采样 {sampled:.2f} m  通过={ok}  开路巡回 {tour:.0f} m")

    res = {}
    scope = config_scope((rb.Problem3Robot, cfg))
    scope.__enter__()
    try:
        for name, pts, r, n in (ARMS[0][0], ptsA, 1200.0, 6), (ARMS[1][0], ptsB, 1150.0, 9):
            rb.Problem3Robot.search_points = staticmethod(lambda P=list(pts): list(P))
            print(f"\n=== 配对核验: {name} ===")
            rows = run_paired(pts, n_case, seed)
            res[name] = rows
            T = np.array([x["T"] for x in rows], float)
            print("  均值 %7.1f s  P90 %7.0f  P95 %7.0f  移动 %7.0f m  检测 %5.1f  "
                  "调频 %5.1f  清除 %5.1f  失败 %4.2f  全清 %d/%d"
                  % (T.mean(), np.percentile(T, 90), np.percentile(T, 95),
                     np.mean([x["dist"] for x in rows]), np.mean([x["meas"] for x in rows]),
                     np.mean([x["sw"] for x in rows]), np.mean([x["clear"] for x in rows]),
                     np.mean([x["fail"] for x in rows]),
                     sum(x["full"] for x in rows), len(rows)))
            assert all(x["scene"] == res[ARMS[0][0]][i]["scene"] for i, x in enumerate(rows)), \
                "场景未配对!"
        a = np.array([x["T"] for x in res[ARMS[0][0]]], float)
        b = np.array([x["T"] for x in res[ARMS[1][0]]], float)
        d = b - a
        se = d.std(ddof=1)/math.sqrt(len(d))
        print(f"\n配对差(B - A): Δ = {d.mean():+.1f} s ({100*d.mean()/a.mean():+.2f} %)  "
              f"95% CI [{d.mean()-1.96*se:+.0f}, {d.mean()+1.96*se:+.0f}]  "
              f"变快 {100*(d < 0).mean():.1f} %  中位 {np.median(d):+.0f} s  "
              f"全清 B {sum(x['full'] for x in res[ARMS[1][0]])}/{len(b)}")
        print("  按源归一: A %.1f s/源, B %.1f s/源"
              % (a.sum()/sum(x["n_src"] for x in res[ARMS[0][0]]),
                 b.sum()/sum(x["n_src"] for x in res[ARMS[1][0]])))

        print(f"\n=== 病理集({n_rep} 例/类, 仅检查漏清与时序) ===")
        for name, pts, r, n in (ARMS[0][0], ptsA, 1200.0, 6), (ARMS[1][0], ptsB, 1150.0, 9):
            rb.Problem3Robot.search_points = staticmethod(lambda P=list(pts): list(P))
            po = run_patho(pts, n_rep, 99)
            tot_miss = sum(v["miss"] for v in po.values())
            print(f"  {name}: 合计漏清 {tot_miss} 例")
            for label, v in po.items():
                print("    %-20s 漏清 %2d  均值 %7.1f s  P95 %7.0f" %
                      (label, v["miss"], v["T"].mean(), np.percentile(v["T"], 95)))
    finally:
        if old_sp is not None:
            rb.Problem3Robot.search_points = old_sp
        else:
            delattr(rb.Problem3Robot, "search_points")
        scope.__exit__(None, None, None)


if __name__ == "__main__":
    main()
