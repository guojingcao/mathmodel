# -*- coding: utf-8 -*-
"""
架构上限审计 (architecture upper-bound audit)

对同一批随机案例同时记录:
  LB0  从扫描终点出发、访问全部【真实目标】的精确开放最短路径 (oracle clear)
  LB1  = 最短覆盖路径(7200m) + LB0                       (分阶段理想下界)
  LB2  从原点出发、联合访问【7个覆盖点 + 全部真实目标】的近似最短开放路径 (融合下界)
  ALG  当前在线算法实际移动距离
指标:
  Gap_online = (ALG - LB1)/LB1   当前调度+定位损失
  Gap_arch   = (LB1 - LB2)/LB1   "严格分阶段"架构损失
  G_info     = L_post_ALG - LB0  信息未知带来的额外距离
"""
import importlib.util, sys, math, numpy as np, io, contextlib

spec = importlib.util.spec_from_file_location("robotmod", r"D:\My_MathModeling_Project\problem3_robot\robot.py")
robotmod = importlib.util.module_from_spec(spec); sys.modules["robotmod"] = robotmod
spec.loader.exec_module(robotmod)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)

COVER_PATH = 6 * robotmod.HEX_R      # 7200 m


def d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def exact_open_path(pts, start, cap=14):
    """Held-Karp 精确开放路径(终点自由), 纯 Python 列表 DP。"""
    n = len(pts)
    if n == 0:
        return 0.0
    if n > cap:
        return None
    INF = float("inf")
    size = 1 << n
    D = [[d(pts[i], pts[j]) for j in range(n)] for i in range(n)]
    Ds = [d(pts[i], start) for i in range(n)]
    dp = [[INF]*n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = Ds[i]
    for mask in range(size):
        row = dp[mask]
        for i in range(n):
            base = row[i]
            if base == INF or not (mask >> i) & 1:
                continue
            Di = D[i]
            for j in range(n):
                if (mask >> j) & 1:
                    continue
                v = base + Di[j]
                nm = mask | (1 << j)
                if v < dp[nm][j]:
                    dp[nm][j] = v
    return min(dp[size-1])


def path_len(path):
    return sum(d(path[i], path[i+1]) for i in range(len(path)-1))


def nn_path(pts, start):
    order = []; unv = set(range(len(pts))); cur = start
    while unv:
        k = min(unv, key=lambda i: d(pts[i], cur))
        order.append(k); cur = pts[k]; unv.discard(k)
    return [tuple(start)] + [tuple(pts[k]) for k in order]


def two_opt(path):
    n = len(path) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i+1, n):
                old = d(path[i-1], path[i]) + d(path[j], path[j+1])
                new = d(path[i-1], path[j]) + d(path[i], path[j+1])
                if new < old - 1e-9:
                    path[i:j+1] = path[i:j+1][::-1]; improved = True
    return path


def heuristic_open_path(pts, start):
    path = two_opt(nn_path(pts, start))
    return path_len(path)


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.log = []; self.dist = 0.0; self.n_measure = 0
        self.dist_at_sweep_end = None
    def _move(self, x, y):
        self.dist += d((x, y), self.position); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        if self.n_measure == 7 * 20:
            self.dist_at_sweep_end = self.dist
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y); return True, self.env.clear(np.array([x, y]), ch)
    def exit(self): pass


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(2026)
    cover_pts = robotmod.Problem3Robot.search_points()
    rows = []
    for t in range(N):
        env = exp.Env(rng, directional=False)
        targets = [tuple(s['pos']) for s in env.sources]
        cli = MockClient(env); rb = robotmod.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n_clear = rb.run()
        L_alg = cli.dist
        L_post = L_alg - cli.dist_at_sweep_end if cli.dist_at_sweep_end else L_alg - COVER_PATH
        scan_end = cover_pts[-1]
        L_lb0 = exact_open_path(targets, scan_end)
        L_lb0_2opt = heuristic_open_path(targets, scan_end)   # 同点集的 2-opt 解(基准一)
        if L_lb0 is None:
            L_lb0 = L_lb0_2opt
        L_lb1 = COVER_PATH + L_lb0
        L_lb2 = heuristic_open_path(list(cover_pts) + targets, (0.0, 0.0))
        rows.append(dict(n_src=env.n_src, L_alg=L_alg, L_lb0=L_lb0, L_lb0_2opt=L_lb0_2opt,
                         L_lb1=L_lb1, L_lb2=L_lb2, L_post=L_post))
        print(f"  case {t+1}/{N}: n={env.n_src} ALG={L_alg:.0f} LB0={L_lb0:.0f} "
              f"LB1={L_lb1:.0f} LB2={L_lb2:.0f}", flush=True)
    K = rows[0].keys()
    A = {k: np.array([r[k] for r in rows], float) for k in K}
    gap_online = (A['L_alg'] - A['L_lb1']) / A['L_lb1']
    gap_arch = (A['L_lb1'] - A['L_lb2']) / A['L_lb1']
    g_info = A['L_post'] - A['L_lb0']
    print("\n=== 架构上限审计汇总 ===")
    print(f"源数均值          : {A['n_src'].mean():.1f}")
    print(f"ALG 实际          : {A['L_alg'].mean():.0f} m")
    print(f"LB0 oracle 清除   : {A['L_lb0'].mean():.0f} m")
    print(f"LB1 分阶段下界    : {A['L_lb1'].mean():.0f} m")
    print(f"LB2 融合下界(近似): {A['L_lb2'].mean():.0f} m")
    print(f"Gap_online        : {gap_online.mean()*100:.2f}%  (中位 {np.median(gap_online)*100:.2f}%)")
    print(f"Gap_arch          : {gap_arch.mean()*100:.2f}%  (中位 {np.median(gap_arch)*100:.2f}%)")
    print(f"G_info            : {g_info.mean():.0f} m")
    gap_2opt = (A['L_lb0_2opt'] - A['L_lb0']) / A['L_lb0']
    print(f"基准一: 2-opt vs 精确开放TSP (同点集): {gap_2opt.mean()*100:.2f}%  "
          f"(中位 {np.median(gap_2opt)*100:.2f}%, 最大 {gap_2opt.max()*100:.2f}%)")
    print(f"覆盖占比          : {COVER_PATH/A['L_alg'].mean()*100:.1f}%")
    print(f"后处理占比        : {A['L_post'].mean()/A['L_alg'].mean()*100:.1f}%")
