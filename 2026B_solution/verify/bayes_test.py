# -*- coding: utf-8 -*-
"""
严格贝叶斯概率地图 M3' 复测
= 网格 P(x,y) 信念 + 示向似然/no_signal负信息更新 + 滚动清除(删除已清除频道)
对比 M1(批量固定覆盖) / M2(+DOP) / M3'(严格贝叶斯)
"""
import importlib.util, sys, numpy as np, time
spec = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec); sys.modules["exp"] = exp; spec.loader.exec_module(exp)

from itertools import combinations
DEG = exp.DEG

def model_M3bayes(env, ring_r=1200.0, n_ring=8, grid_n=16):
    rb = exp.Robot(env)
    xs = np.linspace(-1800, 1800, grid_n)
    X, Y = np.meshgrid(xs, xs)
    cells = np.stack([X.ravel(), Y.ravel()], axis=1)
    cells = cells[np.linalg.norm(cells, axis=1) <= 1800]
    G = len(cells)
    p_exist = {c: 0.65 for c in range(1, 21)}      # P(该频道存在未清除源)
    loc = {c: np.ones(G)/G for c in range(1, 21)}  # P(位置|存在)
    active = set(range(1, 21))
    estimates = {}

    def update(c, p, res, deg):
        pv = np.array(p, float)
        dcell = np.linalg.norm(cells - pv, axis=1)
        if res == 'no_signal':
            # 保守负信息: 1000m 内确定无源(接收半径下限)
            mask = dcell > 1000.0
            loc[c] = loc[c] * mask
            cov = 1 - np.mean(mask)
            pe = p_exist[c]
            p_exist[c] = pe*(1-cov)/(pe*(1-cov) + (1-pe) + 1e-12)
        elif res == 'near':
            loc[c] = (dcell <= 5.0).astype(float); p_exist[c] = 1.0
        elif res == 'direction':
            a = deg*DEG
            ang = np.arctan2(cells[:,1]-pv[1], cells[:,0]-pv[0])
            diff = np.abs(ang - a); diff = np.minimum(diff, 2*np.pi-diff)
            loc[c] = loc[c] * ((diff <= 1.0*DEG) & (dcell <= 1500.0) & (dcell >= 5.0))
            p_exist[c] = 1.0
        s = loc[c].sum()
        if s > 0:
            loc[c] = loc[c]/s

    stations = exp._stations(ring_r, n_ring)
    for sp in stations:
        # 测量所有 active 频道(滚动: 已清除的跳过)
        for c in sorted(active):
            res, deg = rb.measure(sp, c)
            update(c, sp, res, deg)
        # 滚动清除: 已充分定位的频道立即清除
        for c in sorted(active):
            if p_exist[c] < 0.999:
                continue
            dirs = [(st, d) for st, d in rb.obs[c] if d != 'NEAR']
            if len(dirs) >= 2 and max(exp.crossing(a, b) for _, a in dirs for _, b in dirs) >= 35.0:
                est = exp.bearing_intersection([st for st, _ in dirs], [d for _, d in dirs])
                if est is not None and rb.homing_clear(c, est):
                    active.discard(c); estimates[c] = est
        # 贝叶斯排除: 已确定无源
        for c in sorted(active):
            if p_exist[c] < 0.05:
                active.discard(c)

    # 收尾: 剩余 active 频道 -> 补测定位 + TSP 清除
    for c in sorted(active):
        dirs = [(st, d) for st, d in rb.obs[c] if d != 'NEAR']
        if len(dirs) >= 2:
            est = exp.bearing_intersection([st for st, _ in dirs], [d for _, d in dirs])
            if est is not None: estimates[c] = est
        elif len(dirs) == 1:
            S1 = np.array(dirs[0][0], float); th = dirs[0][1]
            for dd in (300.0, 500.0):
                for sgn in (+1, -1):
                    a = (th + sgn*90)*DEG
                    q = S1 + dd*np.array([np.cos(a), np.sin(a)])
                    res, deg = rb.measure(q, c)
                    if res == 'direction':
                        est = exp.bearing_intersection([tuple(S1), tuple(q)], [th, deg])
                        if est is not None: estimates[c] = est; break
                    elif res == 'near':
                        estimates[c] = q.copy(); break
                if c in estimates: break
    for c, est in estimates.items():
        s = rb.env.ch_by_id.get(c)
        if s is not None: rb.est_err[c] = exp.dist(est, s['pos'])
    ordered = []; cur = rb.pos.copy(); unvisited = set(estimates.keys())
    while unvisited:
        c = min(unvisited, key=lambda k: exp.dist(cur, estimates[k]))
        ordered.append(c); cur = estimates[c]; unvisited.discard(c)
    for c in ordered:
        rb.homing_clear(c, estimates[c])
    return rb.metrics()

def mc(model, n_cases, seed=2026, directional=False, p_dir=0.5):
    rng = np.random.default_rng(seed)
    keys = ['CR','T','T_f','e','N_d','L','eta']
    acc = {k: [] for k in keys}
    for _ in range(n_cases):
        env = exp.Env(rng, directional=directional, p_dir=p_dir)
        m = model(env)
        for k in keys: acc[k].append(m[k])
    return {k: (float(np.mean(a)), float(np.std(a))) for k, a in acc.items()}

def row(m, r):
    return (f"{m:<10}CR={r['CR'][0]*100:>6.1f}%  T={r['T'][0]:>6.0f}s  e={r['e'][0]:>5.1f}m  "
            f"N_d={r['N_d'][0]:>4.0f}  L={r['L'][0]:>6.0f}m  η={r['eta'][0]*1000:>5.2f}/ks")

if __name__ == "__main__":
    print("=" * 90)
    print("严格贝叶斯 M3' 复测 (全向/问题3)")
    print("【初步 30 例】")
    t0 = time.time()
    r1 = mc(exp.model_M1, 30); r2 = mc(exp.model_M2, 30); r3 = mc(model_M3bayes, 30)
    print(row('M1 批量覆盖', r1)); print(row('M2 +DOP', r2)); print(row("M3' 严格贝叶斯", r3))
    print(f"  [初筛用时 {time.time()-t0:.0f}s]")
    print("\n【正式 500 例】")
    t0 = time.time()
    r1 = mc(exp.model_M1, 500); r2 = mc(exp.model_M2, 500); r3 = mc(model_M3bayes, 500)
    print(row('M1 批量覆盖', r1)); print(row('M2 +DOP', r2)); print(row("M3' 严格贝叶斯", r3))
    print(f"  [正式用时 {time.time()-t0:.0f}s]")
