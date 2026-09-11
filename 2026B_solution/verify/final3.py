# -*- coding: utf-8 -*-
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("p3", r"D:\My_MathModeling_Project\2026B_solution\verify\problem3.py")
p3 = importlib.util.module_from_spec(spec); sys.modules["p3"] = p3; spec.loader.exec_module(p3)

for rr, nr in [(1200, 8), (1200, 6), (1300, 8)]:
    s = p3.monte_carlo(n_cases=1000, ring_radius=rr, n_ring=nr)
    print(f"R{rr}x{nr}: 清除比例={s['clear']*100:.3f}%  最差={s['min_clear']*100:.0f}%  "
          f"平均定位清除时间={s['avg']:.0f}s  总时间={s['total']:.0f}s  未清完案例={s['miss']}/1000")
