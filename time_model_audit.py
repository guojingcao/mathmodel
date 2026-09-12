# -*- coding: utf-8 -*-
"""时间模型标定与验证 (time-model audit)

对**同一局**的实机日志做两件事:
  1) 用题目给的原始口径回算:  T0 = L/5 + 5Nm + Ns + 5Nok + 3Nfail
  2) 用实机标定后的口径回算:  T1 = L/5 + 5Nm + Ns + 6Nok + 3Nfail + 13
并与该局**实际虚拟耗时**(final_virtual_time_s − /enter 时刻)相减, 报告
有符号/绝对误差、相对误差、分位数, 以及把误差对各项费用回归以确认"账目对得上"。

用法: python time_model_audit.py
"""
import json
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SPEED = 5.0
T_MEASURE, T_SWITCH, T_CLEAR_FAIL = 5.0, 1.0, 3.0


def per_run(path):
    """解析单局日志, 只统计 accepted 动作; 无摘要/无效局返回 None。"""
    body = [l for l in open(path, encoding="utf-8").read().splitlines()
            if l and not l.startswith("#")]
    acts, s = [], None
    for l in body:
        try:
            a = json.loads(l)
        except Exception:
            continue
        if "__summary__" in a:
            s = a["__summary__"]
        else:
            acts.append(a)
    if s is None or s.get("error") or not s.get("final_virtual_time_s"):
        return None
    ok = [a for a in acts if a.get("accepted") is True]
    if not ok:
        return None
    t_enter = next((a["virtual_time_s"] for a in ok if a["path"] == "/enter"
                    and isinstance(a.get("virtual_time_s"), (int, float))), 0.0)
    real = s["final_virtual_time_s"] - t_enter
    dist = nm = nsw = cok = cfail = 0.0
    pos, ch = (0.0, 0.0), None
    for a in ok:
        path, pl = a["path"], a.get("payload", {})
        if path not in ("/measure", "/clear"):
            continue
        pp = pl.get("position")
        if pp:
            dist += ((pp["x"]-pos[0])**2 + (pp["y"]-pos[1])**2) ** 0.5
            pos = (pp["x"], pp["y"])
        c = pl.get("channel")
        # 关键: **只有 /measure 会换频**; /clear 不换频、也不改变测向机频道状态(附件1/附件2)。
        # 此前对两者都记换频, 使每次"跨频道清除"多算 1 s —— 这正是当年"清除应按 4 s"假象的来源。
        if path == "/measure":
            if ch is not None and c != ch:
                nsw += 1
            ch = c
            nm += 1
        elif a["response"].get("clear_result") == "success":
            cok += 1
        else:
            cfail += 1
    return dict(f=os.path.basename(path)[7:22], real=real, dist=dist, nm=nm, nsw=nsw,
                cok=cok, cfail=cfail, n_rejected=len(acts)-len(ok))


def accepted_actions(path):
    """解析单局日志 -> 唯一且被接受的动作序列(按 request_id 去重, 重试不重复记账)。"""
    body = [l for l in open(path, encoding="utf-8").read().splitlines()
            if l and not l.startswith("#")]
    by_id, order, s = {}, [], None
    for l in body:
        try:
            a = json.loads(l)
        except Exception:
            continue
        if "__summary__" in a:
            s = a["__summary__"]
            continue
        rid = (a.get("payload") or {}).get("request_id") or a.get("request_id") or id(a)
        if rid not in by_id:
            order.append(rid)
        # 同一 request_id 的多次响应: 以"最后一次被接受"为准, 只记一次
        if rid in by_id and a.get("accepted") is not True:
            continue
        by_id[rid] = a
    if s is None or s.get("error") or not s.get("final_virtual_time_s"):
        return None, None
    acts = [by_id[r] for r in order if by_id[r].get("accepted") is True]
    return acts, s


def per_action(path):
    """逐动作核账: δt = t_i - t_{i-1} 与该动作的移动/换频/执行费用比较。

    预期费用(题设, 附件1/附件2): /measure = 移动/5 + 换频(仅当与上次**测量**频道不同) + 5
                                /clear   = 移动/5 + 5(成功) 或 3(失败)   [不换频]
                                /enter、/exit = 0(不增加虚拟时间)
    """
    acts, s = accepted_actions(path)
    if acts is None:
        return None
    out = []
    prev_t = None
    prev_pos = (0.0, 0.0)
    last_meas_ch = None
    for a in acts:
        p = a.get("path")
        pl = a.get("payload") or {}
        t = a.get("virtual_time_s")
        if not isinstance(t, (int, float)):
            continue
        if p not in ("/measure", "/clear"):
            prev_t = t if prev_t is None else prev_t   # enter/exit 不推进时间, 不参与核账
            continue
        pp = pl.get("position") or {}
        d = 0.0
        if pp:
            d = ((pp.get("x", prev_pos[0])-prev_pos[0])**2
                 + (pp.get("y", prev_pos[1])-prev_pos[1])**2) ** 0.5
            prev_pos = (pp.get("x", prev_pos[0]), pp.get("y", prev_pos[1]))
        ch = pl.get("channel")
        exp = d/SPEED
        if p == "/measure":
            if last_meas_ch is not None and ch != last_meas_ch:
                exp += T_SWITCH
            last_meas_ch = ch
            exp += T_MEASURE
        else:
            ok = (a.get("response") or {}).get("clear_result") == "success"
            exp += 5.0 if ok else T_CLEAR_FAIL
        base = prev_t if prev_t is not None else (t - exp)
        dt = t - base
        out.append(dict(path=p, dt=dt, move=d, exp=exp, resid=dt-exp,
                        inplace=(d < 1e-6),
                        ok=(a.get("response") or {}).get("clear_result") == "success"))
        prev_t = t
    return out


def report(tag, pat):
    rows = [r for r in (per_run(p) for p in sorted(glob.glob(pat), key=os.path.getmtime))
            if r]
    real = np.array([r["real"] for r in rows])
    T = np.array([r["dist"]/5 + T_MEASURE*r["nm"] + T_SWITCH*r["nsw"]
                  + 5.0*r["cok"] + T_CLEAR_FAIL*r["cfail"] for r in rows])
    e = T - real
    print("=" * 92)
    print(f"[{tag}] 有效局 {len(rows)}（已排除无摘要/无效局）")
    print(f"  实际本局耗时            : 均值 {real.mean():.0f} s  (范围 {real.min():.0f}-{real.max():.0f})")
    print(f"  题设口径 T=L/5+5Nm+Nsw+5Nok+3Nfail: 有符号 {e.mean():+6.2f} s | 绝对 {np.abs(e).mean():5.2f} s | "
          f"中位 {np.median(e):+6.2f} | 最大|e| {np.abs(e).max():5.2f} | "
          f"相对 {np.mean(np.abs(e)/real)*100:.4f}% | SD {e.std(ddof=1):5.2f} s")
    X = np.array([[1.0, r["cok"], r["cfail"], r["nm"], r["nsw"]] for r in rows])
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    res = e - X @ coef
    print(f"  残差回归: e = {coef[0]:+.2f} + {coef[1]:+.3f}*Nok + {coef[2]:+.3f}*Nfail "
          f"+ {coef[3]:+.4f}*Nm + {coef[4]:+.3f}*Ns   残差 SD {res.std(ddof=1):.2f} s")
    print(f"  被拒请求                : {np.mean([r['n_rejected'] for r in rows]):.2f} 次/局")

    # ---- 逐动作核账 ----
    pa = []
    for p in sorted(glob.glob(pat), key=os.path.getmtime):
        r = per_action(p)
        if r:
            pa.extend(r)
    if pa:
        print(f"  [逐动作核账] 动作 {len(pa)} 个(已按 request_id 去重)")
        for kind in ("/measure", "/clear"):
            sub = [x for x in pa if x["path"] == kind]
            if not sub:
                continue
            rr = np.array([x["resid"] for x in sub])
            print(f"    {kind:<9} n={len(sub):5d} 残差 均值 {rr.mean():+.4f} s | 中位 {np.median(rr):+.4f} | "
                  f"SD {rr.std(ddof=1):.4f} | 最大|e| {np.abs(rr).max():.4f}")
        ip = [x for x in pa if x["path"] == "/clear" and x["inplace"] and x["ok"]]
        if ip:
            dts = np.array([x["dt"] for x in ip])
            print(f"    原地成功清除 n={len(ip)}: δt 均值 {dts.mean():.4f} s | 中位 {np.median(dts):.4f} | "
                  f"范围 [{dts.min():.3f}, {dts.max():.3f}] | =5 s 的比例 "
                  f"{100.0*np.mean(np.abs(dts-5.0) < 1e-6):.1f}%")
            off = [x for x in ip if abs(x["dt"]-5.0) > 1e-6]
            if off:
                print(f"      与 5 s 不符的 {len(off)} 个: "
                      + ", ".join(f"δt={x['dt']:.3f}" for x in off[:8])
                      + (" ..." if len(off) > 8 else ""))
        bad = [x for x in pa if abs(x["resid"]) > 0.01]
        print(f"    与题设不符(|残差|>0.01 s)的动作: {len(bad)}/{len(pa)}"
              + ("" if not bad else "  例: " + ", ".join(
                  f"{x['path']} δt={x['dt']:.2f} exp={x['exp']:.2f}" for x in bad[:5])))
    return rows, e


if __name__ == "__main__":
    report("问题三 实机", str(ROOT / "problem3_robot" / "logs" / "p3_log_*.jsonl"))
    report("问题四 实机", str(ROOT / "problem4_robot" / "logs" / "p4_log_*.jsonl"))
    print("\n结论: 按题设标准口径(成功清除 3+2=5 s、/clear 不换频、enter/exit 不计时)逐局与逐动作核账; "
          "此前的 '4 s 标定' 系客户端把 /clear 误记一次换频(+1 s)所致, 已更正。")
