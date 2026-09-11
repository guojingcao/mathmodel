# -*- coding: utf-8 -*-
"""诊断问题4 未清除案例"""
import numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p4", r"D:\My_MathModeling_Project\2026B_solution\verify\problem4.py")
p4 = importlib.util.module_from_spec(spec); sys.modules["p4"] = p4; spec.loader.exec_module(p4)

rng = np.random.default_rng(11)
found = 0
for trial in range(3000):
    env = p4.Env4(rng)
    res = p4.run_strategy(env, r_in=1200, n_in=8, r_out=2200, n_out=12, rng=rng)
    if res['cleared'] < res['n_src']:
        found += 1
        print(f"=== miss trial={trial} n_src={res['n_src']} cleared={res['cleared']} "
              f"n_dir={res['n_dir']}")
        for s in env.sources:
            if s['ch'] not in env.cleared:
                r = np.linalg.norm(s['pos']); ang = np.degrees(np.arctan2(s['pos'][1], s['pos'][0])) % 360
                pt = np.degrees(s['pointing']) if s['pointing'] is not None else None
                print(f"  ch={s['ch']} 全向/定向={'定向' if s['pointing'] is not None else '全向'} "
                      f"r={r:.0f} ang={ang:.0f}° 指向={pt:.0f}° r_rx={s['r_rx']:.0f}")
        if found >= 12:
            break
print("共找到 miss 案例:", found)
