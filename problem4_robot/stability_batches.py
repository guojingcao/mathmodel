# -*- coding: utf-8 -*-
"""问题四"12 回模拟实验"稳定性检验(最终配置, 固定误差场)。

每回独立抽取 n 个案例(种子按回次偏移 => 各回相互独立), 报告每回的均值/标准差/全清率,
以及 12 回之间的稳定性(CV、极差), 用于判定算法在**场景波动**下的稳定性。

最终配置: 27 点最小覆盖设计 + MEC 就绪冻结 + 顺路 LS 失败后暂缓归航
          (BEARING_ORDER=acq, NEIGHBOR_MODE=rings 为实测维持的原行为)

用法: python stability_batches.py [每回案例数=200] [回数=12]
"""
import contextlib
import io
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                  # noqa: E402

import robot4 as r4                                 # noqa: E402
import selfcheck4 as sc4                            # noqa: E402
from ablation4_mecfreeze import FixedEnv            # noqa: E402
from simlib import case_env, sim_time, check_clearance   # noqa: E402


def one_batch(batch, n, seed_base, p_dir):
    rows = []
    for k in range(n):
        seed = seed_base + 1000*batch                  # 每回独立种子段
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
        rows.append((sim_time(cli), chk["case_full_clear"], chk["n_src"],
                     cli.n_measure, cli.dist, cli.fail))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    batches = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    p_dir = 0.5
    print(f"问题四 · {batches} 回模拟实验 · 每回 {n} 例 · 定向比例 {100*p_dir:.0f}%")
    print(f"配置: 27 点设计 + MEC_FREEZE={r4.Problem4Robot.MEC_FREEZE} + "
          f"DEFER_ONWAY_HOMING={r4.Problem4Robot.DEFER_ONWAY_HOMING} + "
          f"BEARING_ORDER={r4.Problem4Robot.BEARING_ORDER} + "
          f"NEIGHBOR_MODE={r4.Problem4Robot.NEIGHBOR_MODE}\n")
    print("%-6s%11s%10s%9s%9s%9s%9s%9s" % (
        "回次", "均值(s)", "中位(s)", "标准差", "CV", "全清率", "检测/例", "移动/例"))
    means = []
    for b in range(batches):
        rows = one_batch(b, n, 4100, p_dir)
        T = [r[0] for r in rows]
        means.append(st.mean(T))
        print("%-6d%11.1f%10.1f%9.1f%8.2f%%%8.1f%%%9.1f%9.0f" % (
            b+1, st.mean(T), st.median(T), st.pstdev(T),
            100*st.pstdev(T)/st.mean(T),
            100*sum(1 for r in rows if r[1])/len(rows),
            st.mean([r[3] for r in rows]), st.mean([r[4] for r in rows])))
    print("\n=== 回间稳定性 ===")
    m = st.mean(means); sd = st.pstdev(means)
    print("  12 回均值: 均值 %.1f s  标准差 %.1f s  CV %.2f%%  范围 [%.0f, %.0f]"
          % (m, sd, 100*sd/m, min(means), max(means)))
    print("  注: 回间 CV 衡量的是『整体水平』的稳定性; 每回内部 CV 反映单局波动(场景差异)。")


if __name__ == "__main__":
    main()
