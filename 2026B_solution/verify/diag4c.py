# -*- coding: utf-8 -*-
"""精确复现并追踪问题4 漏清案例 trial=11"""
import numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p4", r"D:\My_MathModeling_Project\2026B_solution\verify\problem4.py")
p4 = importlib.util.module_from_spec(spec); sys.modules["p4"] = p4; spec.loader.exec_module(p4)

DEG = np.pi/180.0
rng = np.random.default_rng(11)
target = 11
env = None
for t in range(target+1):
    env = p4.Env4(rng)
    p4.run_strategy(env, r_in=1200, n_in=8, r_out=2200, n_out=12, rng=rng)
print("(trial %d 执行完毕, 漏清源如下)" % target)

# 找出漏清源
miss = [s for s in env.sources if s['ch'] not in env.cleared]
print("漏清源:")
for s in miss:
    r = np.linalg.norm(s['pos']); ang = np.degrees(np.arctan2(s['pos'][1], s['pos'][0])) % 360
    pt = (np.degrees(s['pointing']) % 360) if s['pointing'] is not None else None
    print(f"  ch={s['ch']} 类型={'定向' if s['pointing'] is not None else '全向'} r={r:.0f} ang={ang:.0f}° 指向={pt} r_rx={s['r_rx']:.0f}")

# 手动追踪该源: 用同 rng 重新走一遍(重建测量误差序列不可行, 这里直接看几何)
s = miss[0]
stations = [(0.0,0.0)]
for k in range(8):
    a=2*np.pi*k/8; stations.append((1200*np.cos(a),1200*np.sin(a)))
for k in range(12):
    a=2*np.pi*k/12; stations.append((2200*np.cos(a),2200*np.sin(a)))
print(f"\n可测到源 ch={s['ch']} 的站:")
seen=[]
for (x,y) in stations:
    d = s['pos']-np.array([x,y]); dist=np.linalg.norm(d)
    dirs=np.arctan2(d[1],d[0])
    if s['pointing'] is not None:
        diff=abs(dirs-s['pointing']); diff=min(diff,2*np.pi-diff)
        incov = diff<=np.pi/2+1e-9
    else:
        incov = True
    if incov and dist<=s['r_rx']:
        seen.append((x,y,dist,np.degrees(dirs)%360))
        print(f"  站({x:.0f},{y:.0f}) 距{dist:.0f}m 源方向{np.degrees(dirs)%360:.1f}°")
print(f"共 {len(seen)} 站")

# 交会角与定位误差估计
if len(seen)>=2:
    (x1,y1,d1,a1)=seen[0]; (x2,y2,d2,a2)=seen[1]
    # 站->源 方位 = 源方向 + 180
    b1=(a1+180)%360; b2=(a2+180)%360
    g=p4.crossing(b1,b2)
    print(f"两站交会角={g:.1f}° (阈值35°), 是否直接交会定位:{g>=35}")
    # 无误差交会点=真源; 误差放大
    err = 0.01745*np.sqrt(d1**2+d2**2)/np.sin(g*DEG) if np.sin(g*DEG)>0.05 else 999
    print(f"定位误差估计≈{err:.0f}m (1σ), 最坏≈{2*err:.0f}m")
