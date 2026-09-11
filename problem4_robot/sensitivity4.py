# -*- coding: utf-8 -*-
"""问题4 网格参数灵敏度消融: 网格边长 MESH_A × 外扩余量 MESH_MARGIN。
指标: 网格点数 / 圆盘覆盖率 / 三档定向比例下的清除率 / 移动距离 / 估计总时间。
"""
import importlib.util, sys, math, numpy as np, io, contextlib

spec = importlib.util.spec_from_file_location("r4", r"D:\My_MathModeling_Project\problem4_robot\robot4.py")
r4 = importlib.util.module_from_spec(spec); sys.modules["r4"] = r4; spec.loader.exec_module(r4)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp; spec2.loader.exec_module(exp)


def in_tri(p, a, b, c):
    def sg(x, y, z):
        return (x[0]-z[0])*(y[1]-z[1]) - (y[0]-z[0])*(x[1]-z[1])
    d1, d2, d3 = sg(p, a, b), sg(p, b, c), sg(p, c, a)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def coverage(pts, tris, N=6000, seed=11):
    rng = np.random.default_rng(seed); miss = 0
    for _ in range(N):
        r = r4.R_AREA*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        p = (r*math.cos(a), r*math.sin(a))
        if not any(in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]) for t in tris):
            miss += 1
    return 1 - miss/N


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0; self.fail = 0
        self.meta = {}
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1]); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
        return True, r
    def exit(self): pass


def run_mc(n, p_dir, seed=3026):
    rng = np.random.default_rng(seed)
    crs = []; Ls = []; ms = []
    for _ in range(n):
        env = exp.Env(rng, directional=True, p_dir=p_dir)
        cli = MockClient(env); rb = r4.Problem4Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            k = rb.run()
        crs.append(k/env.n_src); Ls.append(cli.dist); ms.append(cli.n_measure)
    return float(np.mean(crs)), float(np.mean(Ls)), float(np.mean(ms))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(f"{'a(m)':>6}{'余量(m)':>8}{'点数':>6}{'三角形':>7}{'覆盖':>8}"
          f"{'0%清':>8}{'50%清':>8}{'100%清':>8}{'移动m':>8}{'时间s':>8}")
    for a in (900.0, 950.0, 1000.0):
        for margin in (500.0, 800.0, 1000.0, 1200.0):
            r4.MESH_A = a; r4.MESH_MARGIN = margin
            pts = r4.tri_mesh(a=a, margin=margin)
            tris = r4.build_triangles(pts, a=a)
            cov = coverage(pts, tris)
            c0, _, _ = run_mc(n, 0.0)
            c5, L5, m5 = run_mc(n, 0.5)
            c1, _, _ = run_mc(n, 1.0)
            print(f"{a:>6.0f}{margin:>8.0f}{len(pts):>6}{len(tris):>7}{cov*100:>7.1f}%"
                  f"{c0*100:>7.1f}%{c5*100:>7.1f}%{c1*100:>7.1f}%{L5:>8.0f}{L5/5+m5*5:>8.0f}",
                  flush=True)
