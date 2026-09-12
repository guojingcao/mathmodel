# -*- coding: utf-8 -*-
"""问题四 MEC 就绪冻结的**固定误差场**配对实验。

为什么要固定误差场: 原仿真里 ±1° 噪声从共享 rng 抽取, 抽到第几个数取决于动作顺序;
两臂动作次数不同 -> 消费的随机数不同 -> 同一案例下"噪声"其实不同, 配对不干净。
本脚本把误差改成**确定性函数 err(scene, ch, x, y)**(与动作顺序无关), 于是同一案例
在两臂中面对完全相同的测量误差场, 配对差才归因于算法本身。

用法: python ablation4_mecfreeze.py [n, 默认 400]
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
from simlib import case_env, sim_time, check_clearance   # noqa: E402

EXP = sc4.exp


class FixedEnv(EXP.Env):
    """把 ±1° 噪声替换为 err(scene, ch, x, y) 的确定性误差场(不消费 rng)。"""

    def __init__(self, *a, scene_key=0, **kw):
        super().__init__(*a, **kw)
        self.scene_key = scene_key

    def _err(self, ch, pos):
        # 种子必须是标量: 用字符串拼接(scene|ch|x|y), 保证与动作顺序无关
        key = f"{self.scene_key}|{int(ch)}|{int(round(pos[0]*10))}|{int(round(pos[1]*10))}"
        return random.Random(key).uniform(-1.0, 1.0)

    def measure(self, pos, ch):
        for s in self.sources:
            if s["ch"] != ch or ch in self.cleared:
                continue
            d = s["pos"] - pos
            dist = float(np.linalg.norm(d))
            if dist > s["r_rx"] or not self._in_cov(s, pos):
                continue
            if dist <= EXP.R_NEAR:
                return "near", None
            true = math.degrees(math.atan2(d[1], d[0])) % 360.0
            return "direction", (true + self._err(ch, pos)) % 360.0
        return "no_signal", None


def run_arm(freeze, p_dir, n, seed):
    rows = []
    old = r4.Problem4Robot.MEC_FREEZE
    r4.Problem4Robot.MEC_FREEZE = freeze
    try:
        for k in range(n):
            # 场景由 case_env 派生(逐案例固定, 两臂一致); 误差场按 (seed,k,ch,坐标) 固定
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
            rows.append(dict(T=sim_time(cli), cr=chk["src_clear_ratio"],
                             miss=1-chk["case_full_clear"], full=chk["case_full_clear"],
                             n_src=chk["n_src"], dist=cli.dist, meas=cli.n_measure,
                             fail=cli.fail, scene=chk.get("scene_hash")))
    finally:
        r4.Problem4Robot.MEC_FREEZE = old
    return rows


def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    n = len(d)
    m = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd/math.sqrt(n) if n else 0.0
    return dict(mean=m, lo=m-1.96*se, hi=m+1.96*se, win=float((d < 0).mean()),
                med=float(np.median(d)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = 3026
    print(f"MEC 就绪冻结 · 固定误差场配对实验 (n={n}/档, seed={seed})")
    print("误差 err(scene, ch, x, y) 与动作顺序无关 => 两臂面对同一误差场\n")
    print("%-12s%-16s%9s%10s%12s%10s%11s%10s" % (
        "定向比例", "配置", "全清率", "平均(s)", "vs 冻结前", "95%CI", "变快比例", "检测/例"))
    for p_dir in (0.5, 1.0):
        res = {}
        for freeze in (False, True):
            res[freeze] = run_arm(freeze, p_dir, n, seed)
        base, new = res[False], res[True]
        p = paired([r["T"] for r in new], [r["T"] for r in base])
        for tag, rows in (("冻结前(正式版)", base), ("MEC 冻结", new)):
            line = "%-12s%-16s%8.1f%%%10.0f" % (
                f"{100*p_dir:.0f}%", tag, 100*np.mean([r["cr"] for r in rows]),
                np.mean([r["T"] for r in rows]))
            if tag.startswith("MEC"):
                line += "%+12.1f%10s%10.1f%%%11.1f" % (
                    p["mean"], f"[{p['lo']:.0f},{p['hi']:.0f}]", 100*p["win"],
                    np.mean([r["meas"] for r in rows]))
            else:
                line += "%12s%10s%10s%11.1f" % ("—", "—", "—",
                                                np.mean([r["meas"] for r in rows]))
            print(line)
        print("   漏清: 冻结前 %d 例, MEC 冻结 %d 例; 移动 %.0f -> %.0f m; "
              "清除失败 %.1f -> %.1f 次/例" % (
                  sum(r["miss"] for r in base), sum(r["miss"] for r in new),
                  np.mean([r["dist"] for r in base]), np.mean([r["dist"] for r in new]),
                  np.mean([r["fail"] for r in base]), np.mean([r["fail"] for r in new])))
    print("\n注: 本实验为**固定误差场的离线配对**; 采纳前仍需实机 A/B(官方模拟器在环)。")


if __name__ == "__main__":
    main()
