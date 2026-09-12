# -*- coding: utf-8 -*-
"""n=8 环的"半径 -> 覆盖余量 -> 时间"三联权衡(用于选一个**有余量**的新候选)。

背景: ring_pareto_lens 发现 n=8、r=938(=覆盖最小半径 r_min)比当前候选快 6.88 %,
但 r=r_min 时最坏接收距离恰好 = 1000 m, **覆盖余量为 0**(任何建模偏差都会破坏保证),
不可部署。故需要扫 r > r_min, 给出 (余量, 时间) 的权衡, 再选一个有余量且仍显著更快的点。

用法: python ring8_margin_sweep.py [案例数=1000] [验证种子=4477]
"""
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

from ring_k_test import run_arm                          # noqa: E402

R_AREA, R_GUAR = 1800.0, 1000.0
HERE = os.path.dirname(os.path.abspath(__file__))


def worst_recv(r, n):
    return math.sqrt(R_AREA**2 + r**2 - 2*R_AREA*r*math.cos(math.pi/n))


def route_len(r, n):
    return r + (n-1)*2*r*math.sin(math.pi/n)


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 4477
    arms = [("当前候选 n=9 r=1150", 1150.0, 9)]
    for r in (938.1, 960.0, 985.0, 1005.0, 1030.0, 1060.0):
        arms.append((f"n=8 r={r:.0f}", r, 8))
    print(f"配对 {n_case} 例(种子 {seed}); 覆盖余量 = 1000 - 最坏接收距离")
    print(f"{'臂':<20}{'点':>4}{'最坏接收':>10}{'余量(m)':>9}{'开路':>8}{'T(s)':>8}"
          f"{'检测':>7}{'失败':>6}{'全清':>10}")
    res = {}
    for name, r, n in arms:
        rows = run_arm(r, n, n_case, seed)
        res[name] = rows
        w = worst_recv(r, n)
        print(f"{name:<20}{n+1:>4}{w:>10.1f}{R_GUAR-w:>9.1f}{route_len(r, n):>8.0f}"
              f"{np.mean([x['T'] for x in rows]):>8.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
    cur = np.array([x["T"] for x in res[arms[0][0]]], float)
    print(f"\n相对当前候选的配对差:")
    table = []
    for name, r, n in arms[1:]:
        a = np.array([x["T"] for x in res[name]], float)
        d = a - cur
        se = d.std(ddof=1)/math.sqrt(len(d))
        w = R_GUAR - worst_recv(r, n)
        table.append((name, r, w, d.mean(), d.mean()-1.96*se, d.mean()+1.96*se,
                      100*(d < 0).mean()))
        print(f"  {name:<20} 余量 {w:>6.1f} m  ΔT {d.mean():+7.1f} s "
              f"({100*d.mean()/cur.mean():+6.2f} %)  CI [{d.mean()-1.96*se:+6.0f},"
              f"{d.mean()+1.96*se:+6.0f}]  变快 {100*(d<0).mean():4.1f} %")
    print("\n建议(兼顾余量与时间): 取余量 >= 50 m 中 ΔT 最小者")
    cand = [x for x in table if x[2] >= 50.0]
    if cand:
        cand.sort(key=lambda x: x[3])
        b = cand[0]
        print(f"  -> {b[0]} (r={b[1]:.0f} m, 余量 {b[2]:.1f} m, ΔT {b[3]:+.1f} s = "
              f"{100*b[3]/cur.mean():+.2f} %, CI [{b[4]:+.0f},{b[5]:+.0f}], "
              f"变快 {b[6]:.1f} %)")
    out = os.path.join(HERE, "results", "ring8_margin_sweep.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# n=8 半径扫描: 逐案\n")
        fh.write("arm,r,margin_m,k,n_src,T,dist,meas,fail,full\n")
        for name, r, n in arms:
            for x in res[name]:
                fh.write("%s,%.1f,%.1f,%d,%d,%.1f,%.0f,%d,%d,%d\n" % (
                    name, r, R_GUAR-worst_recv(r, n), n+1, x["n_src"], x["T"],
                    x["dist"], x["meas"], x["fail"], x["full"]))
    print(f"逐案明细 -> {out}")


if __name__ == "__main__":
    main()
