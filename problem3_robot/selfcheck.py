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
    rng = np.random.default_rng(2026)
    n_cases = 1000
    cleared_all = 0
    ratios = []; dists = []; measures = []
    for t in range(n_cases):
        env = exp.Env(rng, directional=False)
        cli = MockClient(env)
        robot = robotmod.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = robot.run()
        ratios.append(n / env.n_src)
        dists.append(cli.dist)
        measures.append(cli.n_measure)
        if n == env.n_src:
            cleared_all += 1
    arr = np.array(ratios); ds = np.array(dists); ms = np.array(measures)
    print(f"事件驱动任务调度 自检 {n_cases} 案例(全向源):")
    print(f"  清除比例均值 = {arr.mean()*100:.3f}%  全清案例 = {cleared_all}/{n_cases}")
    print(f"  最差比例 = {arr.min()*100:.0f}%")
    print(f"  平均移动距离 = {ds.mean():.0f}m  (中位 {np.median(ds):.0f}m)")
    print(f"  平均检测次数 = {ms.mean():.0f}  (中位 {np.median(ms):.0f})")
    print(f"  平均估计总虚拟时间 = {ds.mean()/5 + ms.mean()*5:.0f}s")
