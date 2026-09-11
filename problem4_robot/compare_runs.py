# -*- coding: utf-8 -*-
"""问题4 实机日志对照工具(δ 阈值 A/B 风险复核)

用法:
    python compare_runs.py logs/p4_log_*.jsonl                 # 列出每局并按 δ 分组汇总
    python compare_runs.py --pair 300 500 logs/p4_log_*_d*.jsonl  # 指定两组做对照

按审查口径输出 6 项检查:
  1) 清除成功数与 unresolved_list(任何漏清直接否决该阈值)
  2) certificate_index_ok 是否为真
  3) mesh_scan 固定成本单独列出
  4) 扫描后清除阶段(顺路/补测/归航/队列)的时间与移动
  5) 顺路清除成功/失败、归航 episode 数、失败清除请求数
  6) 每个实际源的清除阶段时间(不只比总时间)

注意: 不同实机随机场景不是严格配对, 本工具只做工程风险复核;
      1~2 局无法估计"整局 P90", 故只给局内归航 episode 开销的 P90 并明确标注。
"""
import glob
import json
import os
import sys

CLEAR_PHASES = ("on_way", "on_way_homing", "supplement", "homing",
                "queue_clear", "queue_homing", "verify", "verify_homing")


def load_run(path):
    """读一条 JSONL: 返回 (汇总 dict 或 None, 动作列表)。"""
    lines = open(path, encoding="utf-8").read().splitlines()
    body = [l for l in lines if l and not l.startswith("#")]
    acts, summ = [], None
    for l in body:
        try:
            d = json.loads(l)
        except Exception:
            continue
        if isinstance(d, dict) and "__summary__" in d:
            summ = d["__summary__"]
        else:
            acts.append(d)
    return summ, acts


def pct(v):
    return "—" if v is None else f"{v*100:.1f}%"


def p90(v):
    if not v:
        return None
    v = sorted(v)
    k = min(len(v) - 1, int(round(0.9 * (len(v) - 1))))
    return v[k]


def phases_of(summ):
    """返回 (mesh_scan, 顺路, 扫描后清除) 三组阶段成本; 旧日志无 phase 字段则返回 None。"""
    ps = summ.get("phase_stats")
    if not ps:
        return None
    scan = dict(ps.get("mesh_scan", {}))
    onway = {}
    after = {}
    for k, v in ps.items():
        if k == "mesh_scan":
            continue
        tgt = onway if k.startswith("on_way") else after
        for kk, vv in v.items():
            tgt[kk] = tgt.get(kk, 0) + vv
    return scan, onway, after


def row(path):
    summ, acts = load_run(path)
    name = os.path.basename(path)
    if summ is None:
        return {"file": name, "bad": f"无摘要(动作 {len(acts)} 条), 本局无效"}
    cfg = summ.get("config", {})
    meta = summ.get("robot", {}) or {}
    hs = summ.get("homing_stats", {}) or {}
    ph = phases_of(summ)
    # 顺路清除成功/失败: 从 clear_diag 统计(修复后日志才有)
    cd = summ.get("clear_diag") or []
    onway_ok = sum(1 for d in cd if str(d.get("src", "")).startswith("on_way")
                   and d.get("result") == "success")
    onway_fail = sum(1 for d in cd if str(d.get("src", "")).startswith("on_way")
                     and d.get("result") != "success")
    clear_t = None
    per_src = None
    if ph:
        onway, after = ph[1], ph[2]
        clear_t = (onway.get("virtual_time_s", 0.0) + after.get("virtual_time_s", 0.0))
        n_src = meta.get("cleared_count") or summ.get("clear_success_count") or 0
        per_src = (clear_t / n_src) if n_src else None
    return {
        "file": name,
        "delta": cfg.get("on_way_delta"),
        "vt": summ.get("final_virtual_time_s"),
        "moves": summ.get("movement_distance_m"),
        "meas": summ.get("measure_count"),
        "clear_ok": summ.get("clear_success_count"),
        "clear_try": summ.get("clear_attempt_count"),
        "clear_fail": summ.get("clear_failure_count"),
        "cleared_count": meta.get("cleared_count"),
        "unresolved": meta.get("unresolved_list"),
        "cert_ok": meta.get("certificate_index_ok"),
        "scan": ph[0] if ph else None,
        "on_way": ph[1] if ph else None,
        "after": ph[2] if ph else None,
        "clear_phase_t": clear_t,
        "per_src_t": per_src,
        "onway_ok": onway_ok if cd else None,
        "onway_fail": onway_fail if cd else None,
        "episodes": hs.get("episodes"),
        "hard": (hs.get("per_hard_source") or {}).get("n"),
        "ep_time_p90": p90([e.get("episode_time_s", 0.0)
                            for e in (summ.get("homing_diag") or [])
                            if e.get("cleared_by")]),
        "neighbor_rate": hs.get("neighbor_success_rate"),
        "cleared_by": hs.get("cleared_by"),
        "error": summ.get("error"),
    }


def show(rows):
    valid = [r for r in rows if "bad" not in r]
    print(f"共 {len(rows)} 局, 有效 {len(valid)} 局\n")
    for r in rows:
        if "bad" in r:
            print(f"[{r['file']}] {r['bad']}")
            continue
        print(f"[{r['file']}] δ={r['delta']}  虚拟时刻={r['vt']:.0f}s  "
              f"移动={r['moves']:.0f}m  检测={r['meas']}")
        print(f"  ①清除 {r['clear_ok']}/{r['clear_try']} 次(失败 {r['clear_fail']})  "
              f"实际源={r['cleared_count']}  未解决={r['unresolved']}"
              + ("  ← 漏清, 否决" if (r["unresolved"] or r["error"]) else ""))
        print(f"  ②证书索引 = {r['cert_ok']}"
              + ("" if r["cert_ok"] else "  ← 异常, 结论不可用"))
        if r["scan"]:
            print(f"  ③mesh_scan 固定成本: 移动 {r['scan'].get('movement_distance_m'):.0f}m, "
                  f"检测 {r['scan'].get('measure_count')}, 时间 {r['scan'].get('virtual_time_s'):.0f}s")
            print(f"  ④扫描后清除阶段: 顺路 {r['on_way'].get('movement_distance_m'):.0f}m/"
                  f"{r['on_way'].get('clear_success_count')}清, "
                  f"补测+归航+队列 {r['after'].get('movement_distance_m'):.0f}m/"
                  f"{r['after'].get('clear_success_count')}清, "
                  f"合计时间 {r['clear_phase_t']:.0f}s")
            print(f"  ⑥每个实际源清除阶段时间 = {r['per_src_t']:.0f}s "
                  f"(清除阶段 {r['clear_phase_t']:.0f}s / {r['cleared_count']} 源)")
        if r["onway_ok"] is not None:
            print(f"  ⑤顺路清除 成功 {r['onway_ok']} / 失败 {r['onway_fail']}; "
                  f"归航 episode {r['episodes']} 个(困难源 {r['hard']}), "
                  f"失败清除请求 {r['clear_fail']}")
            print(f"     归航结果: {r['cleared_by']}; 邻域成功率={pct(r['neighbor_rate'])}; "
                  f"局内 episode 时间 P90 = "
                  f"{r['ep_time_p90'] if r['ep_time_p90'] is not None else '—'}s "
                  f"(注: 这是局内 episode 的 P90, 不是多局总时间 P90)")
        print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pair = None
    if "--pair" in sys.argv:
        i = sys.argv.index("--pair")
        pair = (float(sys.argv[i+1]), float(sys.argv[i+2]))
        args = [a for a in args if a not in (sys.argv[i+1], sys.argv[i+2])]
    files = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?") else [a])
    files = [f for f in files if f.endswith(".jsonl")]
    if not files:
        files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "logs", "*jsonl")))
    rows = [row(f) for f in files]
    show(rows)
    valid = [r for r in rows if "bad" not in r and r.get("delta") is not None]
    if pair:
        valid = [r for r in valid if r["delta"] in pair]
    groups = {}
    for r in valid:
        groups.setdefault(r["delta"], []).append(r)
    if len(groups) > 1:
        print("=" * 78)
        print("分组对照(同 δ 多局均值; 非严格配对, 仅工程风险复核)")
        print("%-8s%5s%11s%11s%12s%12s%11s%10s" % (
            "δ", "局数", "虚拟时刻", "移动(m)", "清除阶段(s)", "每源(s)",
            "失败清除", "归航ep"))
        for d in sorted(groups):
            g = groups[d]
            def avg(k):
                v = [x[k] for x in g if x.get(k) is not None]
                return sum(v)/len(v) if v else None
            print("%-8s%5d%11s%11s%12s%12s%11s%10s" % (
                d, len(g),
                f"{avg('vt'):.0f}" if avg('vt') is not None else "—",
                f"{avg('moves'):.0f}" if avg('moves') is not None else "—",
                f"{avg('clear_phase_t'):.0f}" if avg('clear_phase_t') is not None else "—",
                f"{avg('per_src_t'):.0f}" if avg('per_src_t') is not None else "—",
                f"{avg('clear_fail'):.1f}" if avg('clear_fail') is not None else "—",
                f"{avg('episodes'):.1f}" if avg('episodes') is not None else "—"))
        print("\n判定规则: 只有较大 δ 在全清、零未解决、证书索引为真、且两局均未增加"
              "归航风险并稳定降低扫描后时间时, 才考虑改默认值; 否则保持 δ=300。")


if __name__ == "__main__":
    main()
