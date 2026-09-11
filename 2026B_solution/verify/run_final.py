# -*- coding: utf-8 -*-
"""正式实验: 500例对比 + 消融 + 鲁棒性"""
import importlib.util, sys, numpy as np, time
spec = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec); sys.modules["exp"] = exp; spec.loader.exec_module(exp)

def row(m, r):
    return (f"{m:<4}{r['CR'][0]*100:>7.1f}%  T={r['T'][0]:>6.0f}s  T_f={r['T_f'][0]:>5.0f}s  "
            f"e={r['e'][0]:>5.1f}m  N_d={r['N_d'][0]:>4.0f}  L={r['L'][0]:>6.0f}m  "
            f"eta={r['eta'][0]*1000:>5.2f}/ks  σ(T)={r['T'][1]:.0f}")

print("=" * 96)
print("【正式对比】500 案例, 全向源(问题3)")
t0 = time.time()
for m in ['M0','M1','M2','M3','M4']:
    r = exp.monte_carlo(m, n_cases=500)
    print(row(m, r))
print(f"  [用时 {time.time()-t0:.0f}s]")

print("\n" + "=" * 96)
print("【鲁棒性A】干扰源数量敏感性 (M1, 全向, 200例)")
for ns in [10, 13, 16]:
    r = exp.monte_carlo('M1', n_cases=200, n_src=ns)
    print(f"  n_src={ns:>2}: CR={r['CR'][0]*100:.1f}%  T={r['T'][0]:.0f}s  e={r['e'][0]:.1f}m")

print("\n" + "=" * 96)
print("【鲁棒性B】定向比例敏感性 (200例, 多环布局)")
rings = [(500, 8), (1200, 8), (2200, 12)]
for pd in [0.0, 0.3, 0.5, 0.7, 1.0]:
    # 用 M1_multi(固定覆盖) 和 M1_multi+动态删除 对比
    rng = np.random.default_rng(2028)
    keys = ['CR','T','e','N_d']
    acc1 = {k:[] for k in keys}; acc3 = {k:[] for k in keys}
    for _ in range(200):
        env = exp.Env(rng, directional=True, p_dir=pd)
        m1 = exp.model_M1_multi(env, rings)
        for k in keys: acc1[k].append(m1[k])
        env2 = exp.Env(rng, directional=True, p_dir=pd)
        m3 = exp.model_M1_multi(env2, rings, dynamic=True)
        for k in keys: acc3[k].append(m3[k])
    c1 = np.mean(acc1['CR'])*100; c3 = np.mean(acc3['CR'])*100
    print(f"  定向比例{pd*100:>3.0f}%: 固定覆盖 CR={c1:.1f}% T={np.mean(acc1['T']):.0f}s | "
          f"动态删除 CR={c3:.1f}% T={np.mean(acc3['T']):.0f}s")

print("\n" + "=" * 96)
print("【鲁棒性C】接收半径范围敏感性 (M1, 全向, 200例)")
for (rmn, rmx) in [(1000, 1500), (1000, 1250), (1250, 1500)]:
    r = exp.monte_carlo('M1', n_cases=200, r_min=rmn, r_max=rmx)
    print(f"  R∈[{rmn},{rmx}]: CR={r['CR'][0]*100:.1f}%  T={r['T'][0]:.0f}s  e={r['e'][0]:.1f}m")
