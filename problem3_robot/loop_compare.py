# -*- coding: utf-8 -*-
"""问题三在环两臂对照分析(从日志的 __summary__ 行读取, 只统计有数据的局)。

用法: python loop_compare.py <臂A的glob> <臂B的glob> [臂A名] [臂B名]
例:   python loop_compare.py "logs/p3_log_*ring1150n9_a*.jsonl" "logs/p3_log_*ring6pt_c*.jsonl"
"""
import glob
import json
import math
import os
import sys

import numpy as np


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        s = None
        for line in open(f, encoding="utf-8"):
            if line.startswith('{"__summary__"'):
                s = json.loads(line)["__summary__"]
        if s is None:
            continue
        if s.get("total_actions", 0) <= 0 or s.get("final_virtual_time_s", 0.0) <= 0:
            rows.append(dict(file=os.path.basename(f), empty=True))
            continue
        rows.append(dict(
            file=os.path.basename(f), empty=False,
            T=float(s["final_virtual_time_s"]),
            dist=float(s.get("movement_distance_m", 0.0)),
            meas=int(s.get("measure_count", 0)),
            clr=int(s.get("clear_success_count", 0)),
            fail=int(s.get("clear_failure_count", 0)),
            actions=int(s.get("total_actions", 0)),
            rej=int(s.get("rejected_count", 0)),
            n_src=int(s.get("cleared_count", s.get("clear_success_count", 0))),
            status=s.get("exit_status"),
            unresolved=s.get("unresolved_channels", s.get("unresolved_kind")),
            ring=(s.get("config", {}) or {}).get("ring_r"),
            ringn=(s.get("config", {}) or {}).get("ring_n"),
            phases=dict(s.get("phase_stats", {}) or {}),
        ))
    return rows


def stat(rows, key):
    v = np.array([r[key] for r in rows if not r["empty"]], float)
    return v


def describe(name, rows):
    good = [r for r in rows if not r["empty"]]
    T = stat(good, "T"); D = stat(good, "dist")
    M = stat(good, "meas"); C = stat(good, "clr"); FA = stat(good, "fail")
    print(f"\n=== {name} ===")
    print(f"  有效局 {len(good)}/{len(rows)} (空局 {len(rows)-len(good)}: "
          f"{[r['file'].split('_')[-1] for r in rows if r['empty']]})")
    print(f"  配置: ring_r={good[0]['ring']} ring_n={good[0]['ringn']}")
    print(f"  虚拟时间: 均值 {T.mean():.1f} s  中位 {np.median(T):.1f}  "
          f"标准差 {T.std(ddof=1):.1f}  CV {100*T.std(ddof=1)/T.mean():.2f} %  "
          f"范围 [{T.min():.0f}, {T.max():.0f}]")
    print(f"  移动: {D.mean():.0f} m    检测: {M.mean():.1f} 次    成功清除: {C.mean():.2f} 次"
          f"    失败清除: {FA.mean():.2f} 次")
    print(f"  每源(Σ T / Σ 清除数): {T.sum()/C.sum():.1f} s/源")
    print(f"  全清局: {sum(1 for r in good if r['clr'] == r['n_src'])}/{len(good)}"
          f"   未解决标记: {sorted(set(str(r['unresolved']) for r in good))}")
    print(f"  状态: {sorted(set(str(r['status']) for r in good))}")
    return good


def main():
    pa = sys.argv[1] if len(sys.argv) > 1 else "problem3_robot/logs/p3_log_*ring1150n9_a*.jsonl"
    pb = sys.argv[2] if len(sys.argv) > 2 else "problem3_robot/logs/p3_log_*ring6pt_c*.jsonl"
    na = sys.argv[3] if len(sys.argv) > 3 else "A 采纳(1150m x 9 点)"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B 回退(1200m x 6 点)"
    A = describe(na, load(pa))
    B = describe(nb, load(pb))
    a = stat(A, "T"); b = stat(B, "T")
    print("\n=== 两臂对照(在环, 场景不配对; 仅区间与均值对照) ===")
    se = math.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    t = (b.mean()-a.mean())/se if se > 0 else float("nan")
    print(f"  ΔT(B-A) = {b.mean()-a.mean():+.1f} s ({100*(b.mean()-a.mean())/a.mean():+.2f} %)"
          f"  均值差 95% CI [{b.mean()-a.mean()-1.96*se:+.0f}, {b.mean()-a.mean()+1.96*se:+.0f}]"
          f"   Welch t = {t:.2f}")
    print(f"  移动 Δ = {stat(B,'dist').mean()-stat(A,'dist').mean():+.0f} m"
          f"   检测 Δ = {stat(B,'meas').mean()-stat(A,'meas').mean():+.1f} 次"
          f"   失败清除 Δ = {stat(B,'fail').mean()-stat(A,'fail').mean():+.2f} 次")
    print(f"  区间: A [{a.min():.0f}, {a.max():.0f}]  B [{b.min():.0f}, {b.max():.0f}]"
          f"  中位 A {np.median(a):.0f} / B {np.median(b):.0f}")

    # ---- 源数校正: 在环两臂场景不配对, 且源数分布不同 -> 分源数对照 + ANCOVA ----
    print("\n=== 源数校正(关键: 两臂源数分布不同, 总时间直接相减会有混杂) ===")
    NsA = np.array([r["n_src"] for r in A], float)
    NsB = np.array([r["n_src"] for r in B], float)
    print(f"  源数分布: A 均值 {NsA.mean():.2f} 范围 [{NsA.min():.0f},{NsA.max():.0f}] | "
          f"B 均值 {NsB.mean():.2f} 范围 [{NsB.min():.0f},{NsB.max():.0f}]")
    print(f"  {'源数':>5}{'A 局数':>8}{'A 均值(s)':>12}{'B 局数':>8}{'B 均值(s)':>12}{'Δ(B-A)':>10}")
    for n in sorted(set(NsA.tolist()) | set(NsB.tolist())):
        ta = a[NsA == n]; tb = b[NsB == n]
        da = "-" if ta.size == 0 else "%+.0f" % (tb.mean()-ta.mean()) if tb.size else "-"
        print(f"  {int(n):>5}{ta.size:>8}{(ta.mean() if ta.size else float('nan')):>12.0f}"
              f"{tb.size:>8}{(tb.mean() if tb.size else float('nan')):>12.0f}{da:>10}")
    # ANCOVA: T = α + β·N + γ·arm(B=1), 报告 γ 与 95% CI
    N = np.concatenate([NsA, NsB])
    y = np.concatenate([a, b])
    arm = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    X = np.column_stack([np.ones(len(N)), N, arm])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_g = math.sqrt(cov[2, 2])
    print(f"  ANCOVA: T = {beta[0]:.0f} + {beta[1]:.1f}·N  {'+' if beta[2]>=0 else '-'} "
          f"{abs(beta[2]):.0f}·[arm=B]   残差标准差 {math.sqrt(s2):.0f} s")
    print(f"    臂效应 γ = {beta[2]:+.1f} s (在**相同源数**下 B 相对 A), "
          f"95% CI [{beta[2]-1.96*se_g:+.0f}, {beta[2]+1.96*se_g:+.0f}], "
          f"t = {beta[2]/se_g:.2f}  -> {'显著' if abs(beta[2]/se_g) > 2 else '不显著'}")
    perA = a.sum()/NsA.sum(); perB = b.sum()/NsB.sum()
    print(f"  按源归一: A {perA:.1f} s/源 | B {perB:.1f} s/源 ({100*(perB-perA)/perA:+.2f} %) "
          f"—— 该口径受源数分布影响, 与上面的 ANCOVA 结论需一起读")

    # ---- 阶段分解(覆盖段成本近乎场景无关, 最能说明机制) ----
    print("\n=== 阶段分解(虚拟时间均值, 局内缺失记 0) ===")

    def phases(rows):
        keys = set()
        for r in rows:
            keys |= set(r["phases"].keys())
        acc = {}
        for k in sorted(keys):
            tot, cnt = 0.0, 0
            for r in rows:
                d = r["phases"].get(k) or {}
                if isinstance(d, dict):
                    tot += float(d.get("virtual_time_s", 0.0)); cnt += 1
            acc[k] = (tot/cnt if cnt else float("nan"), cnt)
        return acc
    pa, pb = phases(A), phases(B)
    print(f"  {'阶段':<16}{'A 均值(s)':>11}{'B 均值(s)':>11}{'Δ(B-A)':>10}{'A 占比':>9}{'B 占比':>9}")
    ta = sum(v[0] for v in pa.values() if v[0] == v[0])
    tb = sum(v[0] for v in pb.values() if v[0] == v[0])
    for k in sorted(set(pa) | set(pb)):
        va = pa.get(k, (float("nan"), 0))[0]
        vb = pb.get(k, (float("nan"), 0))[0]
        print(f"  {k:<16}{va:>11.0f}{vb:>11.0f}{vb-va:>+10.0f}"
              f"{100*va/ta:>8.1f}%{100*vb/tb:>8.1f}%")
    print(f"  {'合计':<16}{ta:>11.0f}{tb:>11.0f}{tb-ta:>+10.0f}")

    def ancova_phase(key):
        ya = np.array([float((r["phases"].get(key) or {}).get("virtual_time_s", 0.0)) for r in A])
        yb = np.array([float((r["phases"].get(key) or {}).get("virtual_time_s", 0.0)) for r in B])
        yy = np.concatenate([ya, yb])
        X = np.column_stack([np.ones(len(yy)), N, np.concatenate(
            [np.zeros(len(ya)), np.ones(len(yb))])])
        bb, *_ = np.linalg.lstsq(X, yy, rcond=None)
        rr = yy - X @ bb
        dd = len(yy) - X.shape[1]
        cv = (rr @ rr/dd) * np.linalg.inv(X.T @ X)
        s = math.sqrt(cv[2, 2])
        return bb[2], s
    print("\n=== 阶段级 ANCOVA(在相同源数下的臂效应; 覆盖段与补测段最能说明机制) ===")
    for k in ("coverage", "supplement", "on_way", "queue_clear", "homing"):
        g, s = ancova_phase(k)
        print(f"  {k:<14} 臂效应(B 相对 A) {g:+8.1f} s   95% CI [{g-1.96*s:+7.1f}, {g+1.96*s:+7.1f}]"
              f"   t = {g/s:5.2f}  -> {'显著' if abs(g/s) > 2 else '不显著'}")


if __name__ == "__main__":
    main()
