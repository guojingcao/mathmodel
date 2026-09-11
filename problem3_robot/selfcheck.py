# -*- coding: utf-8 -*-
"""robot.py 离线自检: 验证清除率 + 移动距离(衡量事件驱动调度收益)。"""
import importlib.util, sys, numpy as np, io, contextlib

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
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        self.position = (x, y)
    def enter(self): self.log.append(("/enter", {}))
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        return True, self.env.clear(np.array([x, y]), ch)
    def exit(self): self.log.append(("/exit", {}))

import math

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="问题3 robot.py 离线自检")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--diag", action="store_true",
                    help="额外输出定位方式/Ω半径 与 逐次清除来源 的统计(诊断用)")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    n_cases = args.cases
    cleared_all = 0
    ratios = []; dists = []; measures = []
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
    print(f"  平均估计总虚拟时间 = {ds.mean()/5 + ms.mean()*5:.0f}s")

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
