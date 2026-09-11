# -*- coding: utf-8 -*-
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("p4", r"D:\My_MathModeling_Project\2026B_solution\verify\problem4.py")
p4 = importlib.util.module_from_spec(spec); sys.modules["p4"] = p4; spec.loader.exec_module(p4)

configs = [
    ("原点+R1200x8+R2200x12 (21站)", dict(r_in=1200, n_in=8, r_out=2200, n_out=12)),
    ("原点+R500x8+R1200x8+R2200x12 (29站)", dict(r_mid=500, n_mid=8, r_in=1200, n_in=8, r_out=2200, n_out=12)),
    ("原点+R500x8+R1000x8+R2000x12 (29站)", dict(r_mid=500, n_mid=8, r_in=1000, n_in=8, r_out=2000, n_out=12)),
]
for name, kw in configs:
    s = p4.monte_carlo(n_cases=500, **kw)
    print(f"{name}")
    print(f"   清除比例={s['clear']*100:.3f}%  最差={s['min_clear']*100:.0f}%  "
          f"平均定位清除时间={s['avg']:.0f}s  总时间={s['total']:.0f}s  未清完={s['miss']}/500")
