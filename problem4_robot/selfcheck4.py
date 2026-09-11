# -*- coding: utf-8 -*-
"""问题4 自检: (a) 验证三角网格覆盖圆盘 + 证书距离条件; (b) 定向/混合源端到端清除率。"""
import importlib.util, sys, math, numpy as np, io, contextlib

spec = importlib.util.spec_from_file_location("r4", r"D:\My_MathModeling_Project\problem4_robot\robot4.py")
r4 = importlib.util.module_from_spec(spec); sys.modules["r4"] = r4
spec.loader.exec_module(r4)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)


def in_tri(p, a, b, c):
    def sgn(x, y, z):
        return (x[0]-z[0])*(y[1]-z[1]) - (y[0]-z[0])*(x[1]-z[1])
    d1, d2, d3 = sgn(p, a, b), sgn(p, b, c), sgn(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def verify_mesh():
    pts = r4.tri_mesh(margin=r4.MESH_MARGIN); tris = r4.build_triangles(pts)
    print(f"网格: {len(pts)} 点, {len(tris)} 三角形")
    # 最大边长
    mx = 0.0
    for t in tris:
        for i in range(3):
            for j in range(i+1, 3):
                mx = max(mx, math.hypot(pts[t[i]][0]-pts[t[j]][0], pts[t[i]][1]-pts[t[j]][1]))
    print(f"最大边长 = {mx:.1f} m (需 <= 1000)")
    # 覆盖性: 圆盘内随机采样, 检查是否落在某三角形内
    rng = np.random.default_rng(7); miss = 0; N = 20000
    for _ in range(N):
        r = r4.R_AREA*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        p = (r*math.cos(a), r*math.sin(a))
        ok = False
        for t in tris:
            if in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]):
                ok = True; break
        if not ok:
            miss += 1
    print(f"覆盖性: 圆盘内 {N} 个采样点, 未被三角形覆盖 {miss} 个 ({miss/N*100:.3f}%)")
    # 认证条件: 三角形内任一点到顶点距离 <= 最大边长(由凸性保证) => 打印最坏
    print(f"=> 任一点到三角形顶点距离 <= 最大边长 = {mx:.1f}m <= 1000m: {'满足' if mx<=1000.5 else '不满足'}")
    return len(pts), len(tris)


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0; self.n_clear = 0
        self.fail = 0
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1]); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y); self.n_clear += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
        return True, r
    def exit(self): pass


def run_mc(n_cases, p_dir):
    rng = np.random.default_rng(3026)
    ratios = []; dists = []; meas = []; fails = []
    for _ in range(n_cases):
        env = exp.Env(rng, directional=True, p_dir=p_dir)
        cli = MockClient(env); rb = r4.Problem4Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = rb.run()
        ratios.append(n/env.n_src); dists.append(cli.dist)
        meas.append(cli.n_measure); fails.append(cli.fail)
    return (np.mean(ratios), np.mean(dists), np.mean(meas), np.mean(fails))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if margin is not None:
        r4.MESH_MARGIN = margin
    print(f"(MESH_MARGIN={r4.MESH_MARGIN})")
    verify_mesh()
    print()
    for pd in [0.0, 0.5, 1.0]:
        cr, L, nm, fl = run_mc(n, pd)
        print(f"定向比例{pd*100:>3.0f}%: 清除率={cr*100:6.2f}%  移动={L:6.0f}m  "
              f"检测={nm:5.0f}  清除失败={fl:.2f}  估计总时间={L/5+nm*5:6.0f}s")
