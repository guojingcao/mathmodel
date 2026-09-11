# -*- coding: utf-8 -*-
"""问题4 单案例深度追踪"""
import numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p4", r"D:\My_MathModeling_Project\2026B_solution\verify\problem4.py")
p4 = importlib.util.module_from_spec(spec); sys.modules["p4"] = p4; spec.loader.exec_module(p4)

DEG = np.pi/180.0

def trace(rng, seed):
    rng = np.random.default_rng(seed)
    env = p4.Env4(rng)
    res = p4.run_strategy(env, r_in=1200, n_in=8, r_out=2200, n_out=12, rng=rng)
    miss = [s for s in env.sources if s['ch'] not in env.cleared]
    if not miss:
        print(f"seed={seed}: 全部清除 (cleared={res['cleared']}/{res['n_src']})")
        return
    print(f"seed={seed}: cleared={res['cleared']}/{res['n_src']} n_dir={res['n_dir']}")
    for s in miss:
        r = np.linalg.norm(s['pos']); ang = np.degrees(np.arctan2(s['pos'][1], s['pos'][0])) % 360
        pt = np.degrees(s['pointing']) % 360
        print(f"  [漏] ch={s['ch']} r={r:.0f} ang={ang:.0f}° 指向={pt:.0f}° r_rx={s['r_rx']:.0f} 定向")
        # 哪些站能测到它
        stations = [(0.0,0.0)]
        for k in range(8):
            a=2*np.pi*k/8; stations.append((1200*np.cos(a),1200*np.sin(a)))
        for k in range(12):
            a=2*np.pi*k/12; stations.append((2200*np.cos(a),2200*np.sin(a)))
        seen = []
        for (x,y) in stations:
            d = s['pos'] - np.array([x,y]); dist = np.linalg.norm(d)
            dirs = np.arctan2(d[1], d[0])  # 检测点指向源
            diff = abs(dirs - s['pointing']); diff = min(diff, 2*np.pi-diff)
            incov = diff <= np.pi/2+1e-9
            inrx = dist <= s['r_rx']
            if incov and inrx:
                seen.append((x, y, dist, np.degrees(dirs)%360))
        print(f"    可测到它的站(覆盖内+接收内)数={len(seen)}:")
        for (x,y,dist,dirs) in seen[:8]:
            print(f"      站({x:.0f},{y:.0f}) 距源{dist:.0f}m 源方向{dirs:.0f}°")

for seed in [11, 12, 13]:
    trace(None, seed)
