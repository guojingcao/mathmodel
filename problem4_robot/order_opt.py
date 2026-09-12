# -*- coding: utf-8 -*-
"""P4-B: 固定 27 站, 优化**访问顺序**以降低完整虚拟时间(前缀观测质量)。

依据(用户修正): 固定站集上路线已接近下界(18 701 m vs MST 18 112 m, 仅 3.26 %),
故单纯换顺序最多再省约 118 s; 但"顺序 → 观测成熟时刻(MEC 冻结序号) → 完整虚拟时间"
没有下界证明: 近似等长的顺序之间可能存在明显信息效率差异。

目标: J(π) = E[T] + λ·CVaR_0.95(T), 约束 L(π)/5 ≤ (1+γ)·L0/5 (L0=当前顺序 18 701 m)。
优化集与验证集**严格分离**(不同种子段), 防止优化器记住少数场景。

用法: python order_opt.py [搜索轮数=150] [优化集规模=30] [验证集规模=200]
"""
import contextlib
import io
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                  # noqa: E402

import robot4 as r4                                 # noqa: E402
import selfcheck4 as sc4                            # noqa: E402
from ablation4_mecfreeze import FixedEnv            # noqa: E402
from simlib import case_env, sim_time, check_clearance   # noqa: E402

L0 = 18701.0          # 当前(V4)顺序的巡回长度
GAMMA = 0.25          # 允许巡回增长比例(2-opt 最优解的任何扰动都会增程, 故需放宽)
LAM = 0.5             # CVaR 权重
P_DIR = 0.5


def tour_len_of(order, pts):
    L = math.hypot(pts[order[0]][0], pts[order[0]][1])
    for a, b in zip(order, order[1:]):
        L += math.hypot(pts[b][0]-pts[a][0], pts[b][1]-pts[a][1])
    return L


def run_case(order, seed, k, p_dir=P_DIR):
    """单案例: 指定访问顺序跑一局, 返回 (T, 检测数, 失败数, 全清)。"""
    base = case_env(seed, k, directional=True, p_dir=p_dir)
    env = FixedEnv(np.random.default_rng(0), n_src=base.n_src,
                   directional=True, p_dir=p_dir, scene_key=(seed, k))
    env.sources = base.sources
    env.ch_by_id = base.ch_by_id
    env.cleared = set()
    cli = sc4.MockClient(env)
    rb = r4.Problem4Robot(cli)
    with contextlib.redirect_stdout(io.StringIO()):
        got = rb.run()
    chk = check_clearance(got, cli, env)
    return sim_time(cli), cli.n_measure, cli.fail, chk["case_full_clear"]


def evaluate(order, pts, seed, n_case):
    T, meas, fail, full = [], 0.0, 0.0, 0
    for k in range(n_case):
        t, m, f, ok = run_case(order, seed, k)
        T.append(t); meas += m; fail += f; full += 1 if ok else 0
    T = np.array(T)
    cvar = float(T[T >= np.percentile(T, 95)].mean())      # CVaR_0.95(上尾均值)
    return dict(T=T, mean=float(T.mean()), p90=float(np.percentile(T, 90)),
                p95=float(np.percentile(T, 95)),
                cvar=cvar, meas=meas/n_case, fail=fail/n_case, full=full, n=n_case,
                J=float(T.mean() + LAM*cvar))


def perturb(order, rnd, n_move=1):
    """局部扰动: 段反转 / 交换 / 单点重定位(整体乱序会远离等长邻域, 故不用 shuffle)。"""
    cand = list(order)
    for _ in range(n_move):
        mv = rnd.random()
        if mv < 0.45:
            i, j = rnd.sample(range(len(cand)), 2); cand[i], cand[j] = cand[j], cand[i]
        elif mv < 0.8:
            i, j = sorted(rnd.sample(range(len(cand)), 2)); cand[i:j+1] = cand[i:j+1][::-1]
        else:
            i = rnd.randrange(len(cand)); p = cand.pop(i)
            cand.insert(rnd.randrange(len(cand)+1), p)
    return cand


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    n_opt = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    rnd = random.Random(11)
    pts = list(r4.Problem4Robot.MESH_DESIGN_PTS)
    rb0 = r4.Problem4Robot.__new__(r4.Problem4Robot)
    rb0.pts = pts
    rb0.tris = r4.build_triangles(pts, a=r4.MESH_A)
    rb0.cover_tris = r4.covering_triangles(rb0.tris, pts)
    base_order = [mi for mi, _, _ in r4.Problem4Robot._order_points(rb0)]
    L_base = tour_len_of(base_order, pts)
    print(f"站集 {len(pts)} 点; 基准顺序巡回 {L_base:.0f} m (约束 ≤ {L_base*(1+GAMMA):.0f} m)")
    opt_seed, val_seed = 9137, 3026          # 优化集/验证集严格分离
    def set_order(od):
        r4.Problem4Robot.ORDER_OVERRIDE = list(od) if od else None
    set_order(base_order)
    b = evaluate(base_order, pts, opt_seed, n_opt)
    print(f"基准(优化集 n={n_opt}): T 均值 {b['mean']:.0f} s  P95 {b['p95']:.0f}  "
          f"CVaR {b['cvar']:.0f}  J {b['J']:.0f}  检测/例 {b['meas']:.1f}  全清 {b['full']}/{b['n']}")
    cur, cur_J, cur_L = list(base_order), b["J"], L_base
    best = (cur_J, list(cur), cur_L, b)
    n_eval = n_skip = 0
    # 先随机采样: 判断"顺序"是否真的影响完整虚拟时间(若采样内 T 几乎不变, 则本方向为负结果)
    print("\n[顺序敏感性探测] 随机采样 40 个顺序(仅保留 L ≤ %.0f m):" % (L_base*(1+GAMMA)))
    samples = []
    for s in range(40):
        cand = perturb(base_order, rnd, n_move=rnd.randint(1, 3))
        L = tour_len_of(cand, pts)
        if L > L_base*(1+GAMMA):
            n_skip += 1
            continue
        set_order(cand)
        r = evaluate(cand, pts, opt_seed, n_opt)
        n_eval += 1
        samples.append((L, r["mean"], r["meas"], r["J"]))
    if samples:
        Ls = [x[0] for x in samples]; Ts = [x[1] for x in samples]
        Ms = [x[2] for x in samples]
        print("   有效样本 %d 个: L 范围 [%.0f, %.0f] m; T 范围 [%.0f, %.0f] s "
              "(极差 %.0f s); 检测/例范围 [%.1f, %.1f]; 与基准 T=%.0f 相比最好 %+.0f s"
              % (len(samples), min(Ls), max(Ls), min(Ts), max(Ts), max(Ts)-min(Ts),
                 min(Ms), max(Ms), b["mean"], min(Ts)-b["mean"]))
    print(f"   (因超长被跳过 {n_skip} 个)")
    for it in range(iters):
        cand = list(cur)
        mv = rnd.random()
        if mv < 0.45:                                   # 交换两点
            i, j = rnd.sample(range(len(cand)), 2); cand[i], cand[j] = cand[j], cand[i]
        elif mv < 0.8:                                  # 段反转
            i, j = sorted(rnd.sample(range(len(cand)), 2)); cand[i:j+1] = cand[i:j+1][::-1]
        else:                                           # 单点重定位
            i = rnd.randrange(len(cand)); p = cand.pop(i)
            cand.insert(rnd.randrange(len(cand)+1), p)
        L = tour_len_of(cand, pts)
        if L > L_base*(1+GAMMA):
            continue
        set_order(cand)
        r = evaluate(cand, pts, opt_seed, n_opt)
        if r["J"] < cur_J:
            cur, cur_J, cur_L = cand, r["J"], L
            if r["J"] < best[0]:
                best = (r["J"], list(cand), L, r)
                print(f"  [{it}] J={r['J']:.0f} (T {r['mean']:.0f} P95 {r['p95']:.0f} "
                      f"CVaR {r['cvar']:.0f}) L={L:.0f} m 检测/例 {r['meas']:.1f} "
                      f"全清 {r['full']}/{r['n']}", flush=True)
    J, od, L, r = best
    print(f"\n优化后(优化集): J={J:.0f}  T {r['mean']:.0f} s (基准 {b['mean']:.0f}, "
          f"{100*(r['mean']-b['mean'])/b['mean']:+.2f}%)  L={L:.0f} m "
          f"({100*(L-L_base)/L_base:+.2f}%)  检测/例 {r['meas']:.1f}")
    print("\n=== 验证集(独立种子段, 严格配对) ===")
    set_order(base_order); A = evaluate(base_order, pts, val_seed, n_val)
    set_order(od);          B = evaluate(od, pts, val_seed, n_val)
    d = B["T"] - A["T"]
    se = d.std(ddof=1)/math.sqrt(len(d))
    for tag, x in (("基准顺序", A), ("优化顺序", B)):
        print("  %-10s 均值 %7.0f s  P90 %7.0f  P95 %7.0f  CVaR %7.0f  检测/例 %5.1f  "
              "失败/例 %4.2f  全清 %d/%d" % (tag, x["mean"], x["p90"], x["p95"], x["cvar"],
                                             x["meas"], x["fail"], x["full"], x["n"]))
    print("  配对: Δ = %+.1f s (%.2f%%)  95%%CI [%.0f,%.0f]  变快比例 %.1f%%  中位 %+.0f s"
          % (d.mean(), 100*d.mean()/A["mean"], d.mean()-1.96*se, d.mean()+1.96*se,
             100*(d < 0).mean(), np.median(d)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "order_opt_best.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(str(i) for i in od))
    print(f"  最优顺序(网格编号)已写入 {out}")
    set_order(None)


if __name__ == "__main__":
    main()
