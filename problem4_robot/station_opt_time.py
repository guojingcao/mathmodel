# -*- coding: utf-8 -*-
"""P4-A: 站集 + 路线联合优化, 目标改为**完整虚拟时间**(不再用 13.7 次/站 的代理)。

动机(实测): 顺序方向的分析显示观测质量 R = T − L/5 确实随顺序变化(内环优先 R 低 1034 s、
检测 331.2 vs 374.7), 但同族最优顺序的巡回长 5.6 万 m, 兑换比约 6:1 不利。要同时拿到
"好的空间顺序"和"短巡回", 必须**联合设计站集**; 而站集质量只能由完整虚拟时间直接度量。

目标: J(S) = mean_case T + 200·失败清除/例 + 1e4·(非全清案例数)   (共用随机数与固定误差场,
      故各候选**逐场景配对**, 差异方差远小于场景方差)
硬约束: 连续覆盖证书(与 station_opt.certificate 同一判据): 凸包 ⊇ 圆盘、无孔、两两不重叠、
        盘内最大边 ≤ 960 m、支撑边余量 ≥ 10 m。
搜索: 模拟退火, 抖动坐标 / 删点(n≥23) / 加点(n≤28)。
验证: 独立种子段配对比较(优化集与验证集严格分离)。

用法: python station_opt_time.py [迭代数=120] [优化集案例数=12] [验证集案例数=150]
      [搜索种子=5501] [验证种子=7742] [墙钟上限秒=1800]
"""
import contextlib
import io
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import robot4 as r4                                     # noqa: E402
import selfcheck4 as sc4                                # noqa: E402
import station_opt as so                                # noqa: E402
from ablation4_mecfreeze import FixedEnv                # noqa: E402
from simlib import case_env, sim_time, check_clearance  # noqa: E402

P_DIR = 0.5
EDGE_LIM = 960.0
SUPPORT_LIM = 10.0
N_MIN, N_MAX = 23, 28
FAIL_PEN = 200.0
MISS_PEN = 1e4
R = r4.R_AREA


def run_case(seed, k):
    base = case_env(seed, k, directional=True, p_dir=P_DIR)
    env = FixedEnv(np.random.default_rng(0), n_src=base.n_src, directional=True,
                   p_dir=P_DIR, scene_key=(seed, k))
    env.sources = base.sources
    env.ch_by_id = base.ch_by_id
    env.cleared = set()
    cli = sc4.MockClient(env)
    rb = r4.Problem4Robot(cli)
    with contextlib.redirect_stdout(io.StringIO()):
        got = rb.run()
    chk = check_clearance(got, cli, env)
    return sim_time(cli), cli.n_measure, cli.fail, chk["case_full_clear"]


def evaluate(pts, seed, n_case):
    """完整虚拟时间评估(逐场景配对: 同一 seed 段 + 固定误差场)。"""
    r4.Problem4Robot.MESH_PTS_OVERRIDE = [tuple(p) for p in pts]
    T, meas, fail, miss = [], 0.0, 0.0, 0
    for k in range(n_case):
        t, m, f, ok = run_case(seed, k)
        T.append(t); meas += m; fail += f
        miss += 0 if ok else 1
    T = np.array(T)
    return dict(T=T, mean=float(T.mean()), p95=float(np.percentile(T, 95)),
                meas=meas/n_case, fail=fail/n_case, miss=miss, n=n_case,
                J=float(T.mean()) + FAIL_PEN*fail/n_case + MISS_PEN*miss)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    n_opt = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    s_opt = int(sys.argv[4]) if len(sys.argv) > 4 else 5501
    s_val = int(sys.argv[5]) if len(sys.argv) > 5 else 7742
    wall = float(sys.argv[6]) if len(sys.argv) > 6 else 1800.0
    rnd = random.Random(5501 + s_opt)
    r4.Problem4Robot.ORDER_OVERRIDE = None      # 顺序由站集自身的环带扫描+2-opt 决定

    base_pts = [tuple(p) for p in r4.Problem4Robot.MESH_DESIGN_PTS]
    t0 = time.time()
    print(f"起点: {len(base_pts)} 点(V4 非规则站集); 优化集 n={n_opt}(种子 {s_opt}), "
          f"验证集 n={n_val}(种子 {s_val}); 墙钟上限 {wall:.0f} s")
    b = evaluate(base_pts, s_opt, n_opt)
    per = (time.time()-t0)/n_opt
    print(f"基准(优化集): T 均值 {b['mean']:.0f} s  P95 {b['p95']:.0f}  检测/例 {b['meas']:.1f}  "
          f"失败/例 {b['fail']:.2f}  非全清 {b['miss']}  J {b['J']:.0f}   (单例 {per:.2f} s)")

    ok, info, viol = so.certificate(base_pts, max_edge_lim=EDGE_LIM, support_lim=SUPPORT_LIM)
    print(f"基准证书: {'通过' if ok else '不通过'} 支撑余量 {info['support_margin']:.2f} m "
          f"盘内最大边 {info['max_edge_disk']:.1f} m 面积缺口 {info['gap']:.2e} "
          f"重叠 {info['ov']:.2e}")

    cur, cur_J = base_pts, b["J"]
    best = (cur_J, list(cur), b, info, viol)
    n_try = n_sim = n_rej = 0
    T0, Tend = 150.0, 3.0
    for it in range(iters):
        if time.time() - t0 > wall:
            print(f"  (达到墙钟上限 {wall:.0f} s, 于第 {it} 轮停止)")
            break
        cand = [list(p) for p in cur]
        mv = rnd.random()
        if mv < 0.72 or len(cand) <= N_MIN:
            i = rnd.randrange(len(cand))
            cand[i][0] += rnd.gauss(0, 60)
            cand[i][1] += rnd.gauss(0, 60)
        elif mv < 0.86 and len(cand) > N_MIN:
            cand.pop(rnd.randrange(len(cand)))
        elif len(cand) < N_MAX:
            a = rnd.uniform(0, 2*math.pi); rr = R*math.sqrt(rnd.random())
            cand.append([rr*math.cos(a), rr*math.sin(a)])
        else:
            i = rnd.randrange(len(cand))
            cand[i][0] += rnd.gauss(0, 60)
            cand[i][1] += rnd.gauss(0, 60)
        cand = [tuple(p) for p in cand]
        n_try += 1
        ok2, info2, viol2 = so.certificate(cand, max_edge_lim=EDGE_LIM, support_lim=SUPPORT_LIM)
        if not ok2:
            n_rej += 1
            continue                                    # 证书不通过: 直接否决, 不花仿真时间
        n_sim += 1
        r = evaluate(cand, s_opt, n_opt)
        T = T0*(Tend/T0)**(it/max(1, iters-1))
        if r["J"] < cur_J or rnd.random() < math.exp(-(r["J"]-cur_J)/max(1e-9, T)):
            cur, cur_J = cand, r["J"]
            if r["J"] < best[0]:
                best = (r["J"], list(cand), r, info2, viol2)
                print(f"  [{it}] J={r['J']:.0f}  T {r['mean']:.0f} s "
                      f"({100*(r['mean']-b['mean'])/b['mean']:+.2f}%)  P95 {r['p95']:.0f}  "
                      f"检测/例 {r['meas']:.1f}  {len(cand)} 点  支撑余量 "
                      f"{info2['support_margin']:.1f} m  盘内最大边 {info2['max_edge_disk']:.0f} m",
                      flush=True)
    J, bpts, r, info, viol = best
    el = time.time() - t0
    print(f"\n搜索结束: {n_try} 候选, 证书否决 {n_rej}, 仿真评估 {n_sim}, 用时 {el:.0f} s "
          f"({el/max(1, n_sim):.1f} s/评估)")
    print(f"最优(优化集): {len(bpts)} 点  T {r['mean']:.0f} s (基准 {b['mean']:.0f}, "
          f"{100*(r['mean']-b['mean'])/b['mean']:+.2f}%)  检测/例 {r['meas']:.1f}  "
          f"J {J:.0f} (基准 {b['J']:.0f})")
    ok, info, viol = so.certificate(bpts, max_edge_lim=EDGE_LIM, support_lim=SUPPORT_LIM)
    print(f"最优证书: {'通过' if ok else '不通过'} 支撑余量 {info['support_margin']:.2f} m "
          f"盘内最大边 {info['max_edge_disk']:.1f} m 面积缺口 {info['gap']:.2e} "
          f"重叠 {info['ov']:.2e} 三角形 {info['n_tri']} 凸包顶点 {info['hull_v']}")

    print(f"\n=== 验证集(独立种子 {s_val}, n={n_val}, 逐场景配对) ===")
    A = evaluate(base_pts, s_val, n_val)
    B = evaluate(bpts, s_val, n_val)
    d = B["T"] - A["T"]
    se = d.std(ddof=1)/math.sqrt(len(d))
    for tag, x in (("基准 V4", A), ("优化站集", B)):
        print("  %-9s %2d 点  均值 %7.0f s  P95 %7.0f  检测/例 %5.1f  失败/例 %4.2f  "
              "非全清 %d/%d" % (tag, len(base_pts) if tag == "基准 V4" else len(bpts),
                                x["mean"], x["p95"], x["meas"], x["fail"], x["miss"], x["n"]))
    print("  配对: Δ = %+.1f s (%+.2f%%)  95%%CI [%+.0f, %+.0f]  变快比例 %.1f%%  中位 %+.0f s"
          % (d.mean(), 100*d.mean()/A["mean"], d.mean()-1.96*se, d.mean()+1.96*se,
             100*(d < 0).mean(), np.median(d)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "station_opt_time_best.txt")
    with open(out, "w", encoding="utf-8") as f:
        for x, y in bpts:
            f.write(f"{x:.3f},{y:.3f}\n")
    print(f"  最优站集已写入 {out}")
    r4.Problem4Robot.MESH_PTS_OVERRIDE = base_pts


if __name__ == "__main__":
    main()
