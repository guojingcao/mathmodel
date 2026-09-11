# -*- coding: utf-8 -*-
"""诊断问题3 未清除的案例: 找出漏清的根因"""
import numpy as np
import importlib.util, sys

spec = importlib.util.spec_from_file_location("p3", r"D:\My_MathModeling_Project\2026B_solution\verify\problem3.py")
p3 = importlib.util.module_from_spec(spec)
sys.modules["p3"] = p3
spec.loader.exec_module(p3)

rng = np.random.default_rng(7)
found = 0
for trial in range(2000):
    env = p3.Env(rng)
    # 记录真值
    truth = {s['ch']: s for s in env.sources}
    res = p3.run_strategy(env, ring_radius=1200, n_ring=8, rng=rng)
    if res['cleared'] < res['n_src']:
        found += 1
        print(f"=== miss 案例 trial={trial}, n_src={res['n_src']}, cleared={res['cleared']}")
        for s in env.sources:
            if s['ch'] not in env.cleared:
                r = np.linalg.norm(s['pos'])
                ang = np.degrees(np.arctan2(s['pos'][1], s['pos'][0])) % 360
                print(f"  未清除: ch={s['ch']}, pos=({s['pos'][0]:.0f},{s['pos'][1]:.0f}), "
                      f"r={r:.0f}, ang={ang:.0f}°, r_rx={s['r_rx']:.0f}")
        if found >= 8:
            break
print("共找到 miss 案例:", found)
