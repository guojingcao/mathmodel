# -*- coding: utf-8 -*-
"""策略审计 (第二步: 只测量, 不改策略)

回答两个"要不要优化"的门控问题:
  P3: "纯垂直补测"到底有多少比例 (a)收不到信号, (b)补测后仍达不到定位门槛, (c)这些失败案例的后续恢复成本。
      —— 若目标行为几乎不存在, 就不实施补测点改进。
  P4: 保证扫描中的动作有多少是"强制"的 (未发现频道的发现/证书测量), 有多少是可省的
      (已认证频道仍被测量 / 可减少的换频)。
      —— 若已认证频道冗余≈0, 则"认证后停止测量"这条候选不成立, 不必另立模块。

只用客户端轨迹 + 环境真值, **不修改任何策略代码**。
"""
import sys
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import simlib
from simlib import (frozen3, frozen4_cls, frozen4_mod, case_env, SimClient,
                    sim_time, phase_time, check_clearance, config_scope, scene_hash)

rb = simlib.load_module("rb", ROOT / "problem3_robot" / "robot.py")
r4 = simlib.load_module("r4", ROOT / "problem4_robot" / "robot4.py")


# ---------------- P3: 补测审计 ----------------
def audit_p3(n=300, seed=2026, theta_gate=30.0):
    rows = []
    with config_scope((rb.Problem3Robot, frozen3())):
        for k in range(n):
            env = case_env(seed, k, directional=False)
            cli = SimClient(env); robot = rb.Problem3Robot(cli)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = robot.run()
            chk = check_clearance(n_ret, cli, env)
            # 补测阶段的逐次测量
            sup = [t for t in cli.trace if t[0] == "supplement" and t[1] == "measure"]
            # 该案例后续的恢复成本(队列归航/顺路归航 时间 + 失败清除)
            recov = phase_time(cli, "queue_homing") + phase_time(cli, "on_way_homing")
            # 每个频道的最终状态
            ch_state = dict(robot.state)
            rows.append(dict(
                n_src=chk["n_src"], full=chk["case_full_clear"], T=sim_time(cli),
                sup_n=len(sup),
                sup_nosig=sum(1 for t in sup if t[3] == "no_signal"),
                sup_ok=sum(1 for t in sup if t[3] in ("direction", "near")),
                # 补测后该频道是否仍未清除(近似"补测没能救回来")
                sup_ch_uncleared=sum(1 for ch in {t[2] for t in sup}
                                     if ch_state.get(ch) != "cleared"),
                recov=recov, fails=cli.fail,
                supp_t=phase_time(cli, "supplement"),
                homing_eps=sum(1 for t in cli.trace if t[0] in ("queue_homing", "on_way_homing")
                               and t[1] == "measure"),
                scene=scene_hash(env)))
    A = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "scene"}
    sup_n = A["sup_n"].sum()
    print("=" * 78)
    print(f"[P3 补测审计] n={n} 例 (均值源数 {A['n_src'].mean():.1f}, 全清率 "
          f"{A['full'].mean()*100:.1f}%)")
    print(f"  专用补测总次数            : {int(sup_n)}  ({sup_n/n:.2f} 次/例)")
    if sup_n:
        print(f"    其中收不到信号(no_signal): {int(A['sup_nosig'].sum())} "
              f"({A['sup_nosig'].sum()/sup_n*100:.1f}%)")
        print(f"    其中收到示向            : {int(A['sup_ok'].sum())} "
              f"({A['sup_ok'].sum()/sup_n*100:.1f}%)")
        print(f"    补测后该频道仍未清除    : {int(A['sup_ch_uncleared'].sum())} "
              f"({A['sup_ch_uncleared'].sum()/max(1,(A['sup_nosig']+A['sup_ok']).sum())*100:.1f}% "
              f"of 涉及频道)")
    print(f"  补测阶段时间              : {A['supp_t'].mean():.0f} s/例")
    print(f"  恢复阶段时间(归航)        : {A['recov'].mean():.1f} s/例 "
          f"(失败清除 {A['fails'].mean():.2f} 次/例)")
    print(f"  仅当补测失败才计入的恢复成本占比: "
          f"{A['recov'].mean()/max(1e-9, A['T'].mean())*100:.2f}%")
    return rows


# ---------------- P4: 扫描动作审计 ----------------
def audit_p4(n=200, seed=3026, ratios=(0.5, 1.0)):
    out = {}
    with config_scope((r4.Problem4Robot, frozen4_cls()), (r4, frozen4_mod())):
        for pd in ratios:
            rows = []
            for k in range(n):
                env = case_env(seed, k, directional=True, p_dir=pd)
                cli = SimClient(env); robot = r4.Problem4Robot(cli)
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    n_ret = robot.run()
                chk = check_clearance(n_ret, cli, env)
                pts = robot.pts
                # 覆盖三角形的顶点集合(用于判断"何时全部证伪")
                tris = robot.cover_tris
                meas = [t for t in cli.trace if t[1] == "measure"]
                # 每个频道: 在哪些顶点返回 no_signal(按动作先后顺序的顶点序)
                seen_ns = {}
                sig_ch = set()
                order_ns = {}
                for t in meas:
                    ch = t[2]
                    if t[3] == "no_signal":
                        order_ns.setdefault(ch, [])
                    else:
                        sig_ch.add(ch)
                # 用坐标 -> 顶点编号
                idx_of = {tuple(round(c, 6) for c in p): i for i, p in enumerate(pts)}
                for t in meas:
                    ch = t[2]
                    if t[3] == "no_signal":
                        key = (round(t[4], 6), round(t[5], 6))
                        i = idx_of.get(key)
                        if i is not None:
                            order_ns.setdefault(ch, []).append(i)
                # 每个频道的"认证时刻": 必须用**增量** no_signal 集合(不能预先用最终全集)
                cert_meas_index = {}
                ns_seen = {}
                for t_i, t in enumerate(meas):
                    ch = t[2]
                    if t[3] == "no_signal":
                        key = (round(t[4], 6), round(t[5], 6))
                        i = idx_of.get(key)
                        if i is not None:
                            ns_seen.setdefault(ch, set()).add(i)
                    if ch not in cert_meas_index:
                        ns = ns_seen.get(ch, set())
                        if ns and len(ns) >= 3 and all(all(v in ns for v in tri) for tri in tris):
                            cert_meas_index[ch] = t_i
                # 已认证之后仍被测量的次数(按动作顺序比较)
                after_cert = 0
                for t_i, t in enumerate(meas):
                    ch = t[2]
                    ci = cert_meas_index.get(ch)
                    if ci is not None and t_i > ci:
                        after_cert += 1
                # 换频: 实际 vs 最少(同站内 #distinct-1 + 站间衔接)
                seq = [(t[2], (round(t[4], 6), round(t[5], 6))) for t in meas]
                actual_sw = sum(1 for a, b in zip(seq, seq[1:]) if a[0] != b[0])
                stations = []
                for ch, pos in seq:
                    if not stations or stations[-1][0] != pos:
                        stations.append((pos, [ch]))
                    else:
                        stations[-1][1].append(ch)
                minimal = 0
                for pos, chs in stations:
                    distinct = set(chs)
                    if len(distinct) > 1:
                        minimal += len(distinct) - 1
                for (p1, c1), (p2, c2) in zip(stations, stations[1:]):
                    if c1[-1] != c2[0]:
                        minimal += 1
                rows.append(dict(
                    n_src=chk["n_src"], full=chk["case_full_clear"], T=sim_time(cli),
                    meas=len(meas),
                    meas_never_signal=sum(1 for t in meas if t[2] not in sig_ch),
                    after_cert=after_cert,
                    actual_sw=actual_sw, minimal_sw=minimal,
                    n_pts=len(pts), scene=scene_hash(env)))
            A = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "scene"}
            out[pd] = A
            print("=" * 78)
            print(f"[P4 扫描审计] 定向比例 {pd*100:.0f}%, n={n} (均值源数 {A['n_src'].mean():.1f}, "
                  f"全清率 {A['full'].mean()*100:.1f}%, 网格 {int(A['n_pts'][0])} 点)")
            print(f"  检测总次数                      : {A['meas'].mean():.0f} 次/例")
            print(f"    未发现频道的发现/证书测量占比  : "
                  f"{A['meas_never_signal'].sum()/A['meas'].sum()*100:.1f}%  "
                  f"({A['meas_never_signal'].mean():.0f} 次/例)")
            print(f"    已定位频道的定位测量占比      : "
                  f"{(A['meas'].sum()-A['meas_never_signal'].sum())/A['meas'].sum()*100:.1f}%")
            print(f"  已认证后仍被测量的次数          : {A['after_cert'].mean():.2f} 次/例 "
                  f"(若≈0 则'认证后停止测量'已实现, 不必另立模块)")
            print(f"  换频次数 实际/最少              : {A['actual_sw'].mean():.0f} / "
                  f"{A['minimal_sw'].mean():.0f}  (可省 {A['actual_sw'].mean()-A['minimal_sw'].mean():.0f} 次/例"
                  f" = {(A['actual_sw'].mean()-A['minimal_sw'].mean())*1.0:.0f} s)")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if which in ("p3", "both"):
        audit_p3(n)
    if which in ("p4", "both"):
        audit_p4(n)
