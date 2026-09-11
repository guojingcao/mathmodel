# -*- coding: utf-8 -*-
"""受限顺路清除阈值扫描: 验证能捕获多少"严格分阶段"架构差距。"""
import importlib.util, sys, numpy as np, io, contextlib, math

spec = importlib.util.spec_from_file_location("robotmod", r"D:\My_MathModeling_Project\problem3_robot\robot.py")
robotmod = importlib.util.module_from_spec(spec); sys.modules["robotmod"] = robotmod
spec.loader.exec_module(robotmod)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.log = []; self.dist = 0.0; self.n_measure = 0
        self.dist_at_sweep_end = None
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1]); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        if self.n_measure == 7 * 20:
            self.dist_at_sweep_end = self.dist
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y); return True, self.env.clear(np.array([x, y]), ch)
    def exit(self): pass


def run(delta, n_cases=500, seed=2026):
    rng = np.random.default_rng(seed)
    robotmod.Problem3Robot.ON_WAY_DELTA = delta
    ratios = []; dists = []; measures = []
    for _ in range(n_cases):
        env = exp.Env(rng, directional=False)
        cli = MockClient(env); rb = robotmod.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = rb.run()
        ratios.append(n / env.n_src); dists.append(cli.dist); measures.append(cli.n_measure)
    return np.mean(ratios), np.mean(dists), np.mean(measures)


if __name__ == "__main__":
    print(f"{'δ(米)':<8}{'清除率':>9}{'平均移动距离':>14}{'检测次数':>10}{'估计总时间':>12}")
    for delta in [None, 200.0, 300.0, 500.0, 800.0, 1200.0]:
        cr, L, nm = run(delta)
        print(f"{str(delta):<8}{cr*100:>8.2f}%{L:>13.0f}m{nm:>10.0f}{L/5+nm*5:>11.0f}s")
