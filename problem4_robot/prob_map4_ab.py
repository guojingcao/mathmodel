# -*- coding: utf-8 -*-
"""问题四: 贝叶斯概率图(模块P)移植后的配对实验。

与问题三的差别(定向源):
  * 似然要把源的**朝向未知**边际化: 检测点落在发射半平面内的概率 = 1/2, 故
        P(接收 | 格 j, 站点 s) = P(R_c >= dist)·(1/2)   (指向未知)
    这正是问题四 `no_signal` 三义("无源/超距/盲区")的概率表达: P(no_signal) >= 1/2 恒成立。
  * 因此"零信息增益"判据比问题三更宽松(只有 dim 很大或后验已收缩时才判零)。

两个杠杆(默认关):
  PROB_SKIP_IG    已发现但未就绪的频道, 若"从该站点根本收不到"则不测(不影响任何证书);
  PROB_COUNT_CERT 已确认有源数达题设上界 16 -> 其余频道确定性判空(比三角形证书更强)。

用法: python prob_map4_ab.py [案例数=300] [种子=5331]
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402

import robot4 as r4                                     # noqa: E402
import selfcheck4 as sc4                                # noqa: E402
from ablation4_mecfreeze import FixedEnv                # noqa: E402
from simlib import case_env, sim_time, check_clearance   # noqa: E402

ARMS = [
    ("base(当前采纳)", dict(PROB_SKIP_IG=False, PROB_COUNT_CERT=False)),
    ("skip_ig", dict(PROB_SKIP_IG=True, PROB_COUNT_CERT=False)),
    ("count_cert", dict(PROB_SKIP_IG=False, PROB_COUNT_CERT=True)),
    ("skip_ig+cert", dict(PROB_SKIP_IG=True, PROB_COUNT_CERT=True)),
]
FLAGS = ("PROB_SKIP_IG", "PROB_COUNT_CERT", "PROB_SKIP_IG_EPS")


def run_arm(cfg, p_dir, n_case, seed):
    old = {k: getattr(r4.Problem4Robot, k, None) for k in FLAGS}
    for k, v in cfg.items():
        setattr(r4.Problem4Robot, k, v)
    rows = []
    try:
        for k in range(n_case):
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
            rows.append(dict(T=sim_time(cli), dist=cli.dist, meas=cli.n_measure,
                             sw=cli.n_switch, clr=cli.n_clear, fail=cli.fail,
                             full=chk["case_full_clear"], n_src=chk["n_src"],
                             cert=getattr(rb, "prob_cert_fired", 0) or 0,
                             skip=getattr(rb, "prob_skip_ig", 0) or 0))
    finally:
        for k, v in old.items():
            setattr(r4.Problem4Robot, k, v)
    return rows


def main():
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5331
    print(f"配对 {n_case} 例(种子 {seed}), 同场景 + 固定误差场; "
          f"环境 = 当前采纳配置(V4 27 站 + MEC 冻结 + 暂缓归航)")
    run_cache = {}
    for p_dir in (0.5, 1.0):
        print(f"\n########## 定向比例 {p_dir:.0%} ##########")
        print(f"{'臂':<18}{'T(s)':>8}{'移动':>9}{'检测':>7}{'换频':>7}{'失败':>6}"
              f"{'证书':>6}{'跳过':>6}{'全清':>10}")
        res = {}
        for name, cfg in ARMS:
            rows = run_arm(cfg, p_dir, n_case, seed)
            res[name] = rows
            run_cache[(p_dir, name)] = rows
            print(f"{name:<18}{np.mean([x['T'] for x in rows]):>8.0f}"
                  f"{np.mean([x['dist'] for x in rows]):>9.0f}"
                  f"{np.mean([x['meas'] for x in rows]):>7.1f}"
                  f"{np.mean([x['sw'] for x in rows]):>7.1f}"
                  f"{np.mean([x['fail'] for x in rows]):>6.2f}"
                  f"{np.mean([x['cert'] for x in rows]):>6.2f}"
                  f"{np.mean([x['skip'] for x in rows]):>6.1f}"
                  f"{sum(x['full'] for x in rows):>7d}/{len(rows)}", flush=True)
        b = np.array([x["T"] for x in res[ARMS[0][0]]], float)
        for name, _ in ARMS[1:]:
            a = np.array([x["T"] for x in res[name]], float)
            d = a - b
            se = d.std(ddof=1)/math.sqrt(len(d))
            dm = (np.mean([x["meas"] for x in res[name]])
                  - np.mean([x["meas"] for x in res[ARMS[0][0]]]))
            print(f"  {name:<18} ΔT {d.mean():+7.1f} s ({100*d.mean()/b.mean():+6.2f} %) "
                  f"CI [{d.mean()-1.96*se:+6.0f},{d.mean()+1.96*se:+6.0f}]  变快 "
                  f"{100*(d<0).mean():4.1f} %  Δ检测 {dm:+6.1f}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "prob_map4_ab_cases.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 问题四 模块P 配对实验逐案\n")
        fh.write("p_dir,arm,k,n_src,T,dist,meas,switch,clear,fail,cert,skip\n")
        for p_dir in (0.5, 1.0):
            for name, cfg in ARMS:
                for k, x in enumerate(run_cache[(p_dir, name)]):
                    fh.write("%.2f,%s,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%d\n" % (
                        p_dir, name, k, x["n_src"], x["T"], x["dist"], x["meas"],
                        x["sw"], x["clr"], x["fail"], int(x["cert"]), int(x["skip"])))
    print(f"\n逐案明细 -> {out}")


if __name__ == "__main__":
    main()
