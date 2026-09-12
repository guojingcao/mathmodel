# -*- coding: utf-8 -*-
"""robot.py 离线自检: 验证清除率 + 移动距离(衡量事件驱动调度收益)。"""
import importlib.util, sys, numpy as np, io, contextlib, os

# 统一计时口径: 必须用 simlib.sim_time()(实机标定 成功清除 4 s), 不要再写内联公式
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simlib import sim_time                     # noqa: E402

spec = importlib.util.spec_from_file_location("robotmod", r"D:\My_MathModeling_Project\problem3_robot\robot.py")
robotmod = importlib.util.module_from_spec(spec); sys.modules["robotmod"] = robotmod
spec.loader.exec_module(robotmod)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)

class MockClient:
    """同 SimClient 接口, 底层接仿真 Env, 并累计移动距离。"""
    def __init__(self, env):
        self.env = env
        self.position = (0.0, 0.0)
        self.channel = 1
        self.remaining_real = 1200
        self.log = []
        self.dist = 0.0
        self.n_measure = 0
        self.n_switch = 0; self.n_clear = 0; self.n_clear_ok = 0; self.fail = 0
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        self.position = (x, y)
    def enter(self): self.log.append(("/enter", {}))
    def measure(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel: self.n_switch += 1
        self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        # 题设: /clear 不换频, 也不改变测向机频道状态(附件1/附件2)
        self.n_clear += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r == "success": self.n_clear_ok += 1
        else: self.fail += 1
        return True, r
    def exit(self): self.log.append(("/exit", {}))

import math

def patho_sources(rng, kind, n):
    """构造极端/病理场景的全向源集合(随机采样很难碰到的几何)。"""
    ch = [int(c) for c in rng.choice(np.arange(1, robotmod.N_CH+1), size=n, replace=False)]
    out = []
    for i, c in enumerate(ch):
        if kind == "edge":                  # 全源贴近圆盘边界
            a = rng.uniform(0, 2*np.pi); r = rng.uniform(1750, 1800)
        elif kind == "center":              # 全源贴近原点
            a = rng.uniform(0, 2*np.pi); r = rng.uniform(0, 150)
        else:
            a = rng.uniform(0, 2*np.pi); r = robotmod.R_AREA*math.sqrt(rng.uniform())
        pos = (r*math.cos(a), r*math.sin(a))
        rx = 1000.0 if kind == "worst_rx" else float(rng.uniform(1000, 1500))
        out.append(dict(pos=pos, ch=c, r_rx=rx, pointing=None))
    if kind == "degenerate" and len(out) >= 2:      # 两源相距 <25m: 近简并
        p0 = out[0]["pos"]
        out[1]["pos"] = (p0[0] + rng.uniform(-20, 20), p0[1] + rng.uniform(-20, 20))
    return out


def make_env(rng, specs):
    """用显式源集合替换 Env 的随机源(不改 experiment.py)。"""
    env = exp.Env(rng, n_src=len(specs), directional=False)
    env.sources = specs
    env.ch_by_id = {s["ch"]: s for s in specs}
    env.cleared = set()
    return env


def run_patho(n_rep=200, seed=99):
    """极端场景压力测试: 每类重复 n_rep 次, 报告漏清与时间分布 + 漏清率置信上界。"""
    scen = [("随机·13源(对照)", "random", 13), ("上界16源", "random", 16),
            ("下界10源", "random", 10), ("最坏接收(全1000m)", "worst_rx", 13),
            ("全源贴近边界", "edge", 13), ("全源贴近原点", "center", 13),
            ("近简并双源", "degenerate", 13)]
    print(f"极端场景压力测试: 每类 {n_rep} 次")
    print("%-22s%9s%8s%9s%9s%11s%13s" % ("场景", "全清率", "漏清例", "平均(s)", "P95(s)",
                                         "最大(s)", "漏清率95%上界"))
    rng = np.random.default_rng(seed)
    bad = 0
    for label, kind, n in scen:
        crs = []; Ts = []; miss = 0
        for _ in range(n_rep):
            env = make_env(rng, patho_sources(rng, kind, n))
            cli = MockClient(env); robot = robotmod.Problem3Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                got = robot.run()
            crs.append(got/env.n_src); Ts.append(sim_time(cli))
            if got < env.n_src:
                miss += 1
        Ts = np.array(Ts)
        ub = 1 - 0.05**(1.0/n_rep) if miss == 0 else None
        bad += miss
        print("%-22s%8.1f%%%8d%9.0f%9.0f%11.0f%13s" % (
            label, np.mean(crs)*100, miss, Ts.mean(), np.percentile(Ts, 95), Ts.max(),
            f"{ub*100:.2f}%" if ub is not None else "见漏清例"))
    print(f"合计漏清 {bad} 例")
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="问题3 robot.py 离线自检")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--diag", action="store_true",
                    help="额外输出定位方式/Ω半径 与 逐次清除来源 的统计(诊断用)")
    ap.add_argument("--patho", action="store_true",
                    help="只跑极端场景压力测试(病理几何集)")
    ap.add_argument("--patho-rep", type=int, default=200, help="每类极端场景重复次数")
    args = ap.parse_args()
    if args.patho:
        raise SystemExit(0 if run_patho(args.patho_rep) == 0 else 1)
    rng = np.random.default_rng(2026)
    n_cases = args.cases
    cleared_all = 0
    ratios = []; dists = []; measures = []; TS = []
    loc_hist = []; clr_diag = []
    for t in range(n_cases):
        env = exp.Env(rng, directional=False)
        cli = MockClient(env)
        robot = robotmod.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = robot.run()
        ratios.append(n / env.n_src)
        dists.append(cli.dist)
        measures.append(cli.n_measure)
        TS.append(sim_time(cli))
        if args.diag:
            loc_hist.extend(getattr(cli, "locate_history", []))
            clr_diag.extend(getattr(cli, "clear_diag", []))
        if n == env.n_src:
            cleared_all += 1
    arr = np.array(ratios); ds = np.array(dists); ms = np.array(measures)
    print(f"事件驱动任务调度 自检 {n_cases} 案例(全向源):")
    print(f"  清除比例均值 = {arr.mean()*100:.3f}%  全清案例 = {cleared_all}/{n_cases}")
    print(f"  最差比例 = {arr.min()*100:.0f}%")
    print(f"  平均移动距离 = {ds.mean():.0f}m  (中位 {np.median(ds):.0f}m)")
    print(f"  平均检测次数 = {ms.mean():.0f}  (中位 {np.median(ms):.0f})")
    print(f"  平均总虚拟时间(统一口径: 移动/检测/换频/清除) = {np.mean(TS):.0f}s")

    if args.diag:
        def stat(v):
            if not v:
                return "无"
            v = np.array(v, dtype=float)
            return (f"n={len(v)} 中位={np.median(v):.1f} 均值={v.mean():.1f} "
                    f"P90={np.percentile(v,90):.1f} 最大={v.max():.1f}")
        n_mec = sum(1 for h in loc_hist if h.get("method") == "mec")
        n_ls = sum(1 for h in loc_hist if h.get("method") == "ls")
        n_fail = sum(1 for h in loc_hist if h.get("method") is None)
        print(f"\n[诊断] 定位调用 {len(loc_hist)} 次: MEC={n_mec} LS={n_ls} 不可定位={n_fail}")
        print("  Ω半径(m) 判定即清除(MEC<=20m): " +
              stat([h["omega_radius_m"] for h in loc_hist
                    if h.get("method") == "mec" and h.get("omega_radius_m") is not None]))
        print("  Ω半径(m) 改用最小二乘(LS):     " +
              stat([h["omega_radius_m"] for h in loc_hist
                    if h.get("method") == "ls" and h.get("omega_radius_m") is not None]))
        print("  LS 交会角(deg):                " +
              stat([h["cross_angle_deg"] for h in loc_hist
                    if h.get("method") == "ls" and h.get("cross_angle_deg") is not None]))
        print(f"\n[诊断] 清除尝试 {len(clr_diag)} 次, 按定位来源:")
        groups = {}
        for d in clr_diag:
            key = str(d.get("src", "?")).split(":")[0] + "|" + str(d.get("src", "?")).split(":")[-1]
            g = groups.setdefault(key, {"n": 0, "ok": 0, "omr": []})
            g["n"] += 1
            g["ok"] += 1 if d.get("result") == "success" else 0
            if d.get("omega_radius_m") is not None:
                g["omr"].append(d["omega_radius_m"])
        for k in sorted(groups):
            g = groups[k]
            omr = (f" Ω半径中位={np.median(g['omr']):.1f}m" if g["omr"] else "")
            print(f"  {k:28s} n={g['n']:5d} 成功={g['ok']:5d} "
                  f"({100.0*g['ok']/g['n']:5.1f}%){omr}")
