# -*- coding: utf-8 -*-
"""问题三: 在**新透镜判据**下重扫"环半径 x 点数"帕累托(搜索集/验证集分离)。

动机: ring_k_test 发现 r 取"覆盖最优半径"的小环(n=8, r=938)比当前候选(n=9, r=1150)
快约 6.9 %; 旧结论(小环几何惩罚更大)是在旧补测规则下得出的。本脚本:
  阶段1(搜索, 种子 8821): 扫 覆盖最优半径 与 若干放大半径 x n=8..12 + 当前候选
  阶段2(验证, 新种子 4477): 把搜索集最优与当前候选做**独立种子配对**确认

用法: python ring_pareto_lens.py [每臂案例数=1000] [搜索种子=8821] [验证种子=4477]
"""
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

from ring_k_test import run_arm                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def r_min(n):
    """原点 + n 点均匀环覆盖圆盘所需的最小半径(解析根)。"""
    c = math.cos(math.pi/n)
    disc = (1800.0*c)**2 - (1800.0**2 - 1000.0**2)
    if disc < 0:
        return None
    return 1800.0*c - math.sqrt(disc)


def route_len(r, n):
    return r + (n-1)*2*r*math.sin(math.pi/n)


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    s_search = int(sys.argv[2]) if len(sys.argv) > 2 else 8821
    s_val = int(sys.argv[3]) if len(sys.argv) > 3 else 4477
    arms = [("current n=9 r=1150", 1150.0, 9)]
    for n in (8, 9, 10, 11, 12):
        rm = r_min(n)
        arms.append((f"n={n} r={rm:.0f}(覆盖最优)", rm, n))
        arms.append((f"n={n} r={rm*1.15:.0f}(放大 15%)", rm*1.15, n))
    print(f"阶段1 搜索(种子 {s_search}, 每臂 {n_case} 例; 透镜已开)")
    print(f"{'臂':<26}{'点':>4}{'r':>7}{'开路':>8}{'T(s)':>8}{'检测':>7}{'补测':>7}"
          f"{'归航':>7}{'失败':>6}{'全清':>9}")
    res = {}
    for name, r, n in arms:
        rows = run_arm(r, n, n_case, s_search)
        res[name] = rows
        T = np.mean([x["T"] for x in rows])
        print(f"{name:<26}{n+1:>4}{r:>7.0f}{route_len(r, n):>8.0f}{T:>8.0f}"
              f"{np.mean([x['meas'] for x in rows]):>7.1f}"
              f"{np.mean([x['supplement'] for x in rows]):>7.1f}"
              f"{np.mean([x['homing']+x['queue_homing']+x['on_way_homing'] for x in rows]):>7.1f}"
              f"{np.mean([x['fail'] for x in rows]):>6.2f}"
              f"{sum(x['full'] for x in rows):>6d}/{len(rows)}", flush=True)
    cur = np.array([x["T"] for x in res[arms[0][0]]], float)
    print(f"\n相对当前候选的配对差(搜索集):")
    scored = []
    for name, r, n in arms[1:]:
        a = np.array([x["T"] for x in res[name]], float)
        d = a - cur
        se = d.std(ddof=1)/math.sqrt(len(d))
        scored.append((d.mean(), name, r, n))
        print(f"  {name:<26} ΔT {d.mean():+7.1f} s ({100*d.mean()/cur.mean():+6.2f} %) "
              f"CI [{d.mean()-1.96*se:+6.0f},{d.mean()+1.96*se:+6.0f}]  变快 "
              f"{100*(d<0).mean():4.1f} %")
    scored.sort()
    best_mean, best_name, best_r, best_n = scored[0]
    print(f"\n搜索集最优: {best_name}(搜索集 ΔT {best_mean:+.1f} s)")

    print(f"\n阶段2 验证(种子 {s_val}, 独立于搜索集; 配对 {n_case} 例)")
    a_cur = run_arm(1150.0, 9, n_case, s_val)
    a_new = run_arm(best_r, best_n, n_case, s_val)
    b = np.array([x["T"] for x in a_cur], float)
    a = np.array([x["T"] for x in a_new], float)
    d = a - b
    se = d.std(ddof=1)/math.sqrt(len(d))
    print(f"  当前候选 n=9 r=1150: T {b.mean():.1f} s  P95 {np.percentile(b,95):.0f}  "
          f"移动 {np.mean([x['dist'] for x in a_cur]):.0f} m  检测 "
          f"{np.mean([x['meas'] for x in a_cur]):.1f}  失败 {np.mean([x['fail'] for x in a_cur]):.2f}"
          f"  全清 {sum(x['full'] for x in a_cur)}/{len(a_cur)}")
    print(f"  新候选 {best_name}: T {a.mean():.1f} s  P95 {np.percentile(a,95):.0f}  "
          f"移动 {np.mean([x['dist'] for x in a_new]):.0f} m  检测 "
          f"{np.mean([x['meas'] for x in a_new]):.1f}  失败 {np.mean([x['fail'] for x in a_new]):.2f}"
          f"  全清 {sum(x['full'] for x in a_new)}/{len(a_new)}")
    print(f"  配对: ΔT {d.mean():+.1f} s ({100*d.mean()/b.mean():+.2f} %)  "
          f"CI [{d.mean()-1.96*se:+.0f},{d.mean()+1.96*se:+.0f}]  变快 {100*(d<0).mean():.1f} %  "
          f"中位 {np.median(d):+.0f} s")
    print(f"  每源: 当前 {b.sum()/sum(x['n_src'] for x in a_cur):.1f} s/源  "
          f"新 {a.sum()/sum(x['n_src'] for x in a_new):.1f} s/源")
    out = os.path.join(HERE, "results", "ring_pareto_lens.txt")


if __name__ == "__main__":
    main()
