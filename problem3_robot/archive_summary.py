"""问题三实机日志归档汇总(用于论文数字核对)。

用法:
    python archive_summary.py                 # 扫描 logs/ 下全部 jsonl
    python archive_summary.py --keys          # 只打印首个 summary 的字段结构
    python archive_summary.py logs/a.jsonl ...# 指定文件

口径说明:
  * 只统计含 `__summary__` 的有效日志; 缺失/损坏日志单独列出(不计入均值)。
  * `全清` 判定: deep_clear(环境真值) 与 resolved_status 双条件;
    并列出 exit_status 非 completed 与 unresolved 非空的局, 任何一局异常都会显式打印。
  * 每源时间给出两种口径: 先按局求商再平均(mean-of-ratios) 与 总和之比(ratio-of-sums)。
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _deep(d):
    if not isinstance(d, dict):
        return None
    for k in ("deep_clear", "scene_clear", "all_clear", "true_clear"):
        if k in d:
            return d[k]
    for v in d.values():
        r = _deep(v)
        if r is not None:
            return r
    return None


def _pick(d, *names, default=None):
    """在嵌套 dict 中按 key 名查找(广度优先), 返回首个命中。"""
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d:
            return d[n]
    for v in d.values():
        if isinstance(v, dict):
            r = _pick(v, *names, default=None)
            if r is not None:
                return r
    return default


def load(path):
    s = None
    lines = 0
    for line in open(path, encoding="utf-8"):
        lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and "__summary__" in o:
            s = o["__summary__"]
    return s, lines


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keys_only = "--keys" in sys.argv
    files = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?") else [a])
    if not files:
        files = sorted(glob.glob(os.path.join(HERE, "logs", "*.jsonl")))
    if keys_only:
        for f in files:
            s, _ = load(f)
            if s:
                print(f)
                print(json.dumps(s, ensure_ascii=False, indent=2)[:4000])
                return
        print("未找到有效 summary")
        return

    rows, bad = [], []
    for f in sorted(files):
        s, lines = load(f)
        if not s:
            bad.append((os.path.basename(f), lines, "无 summary"))
            continue
        vt = _pick(s, "final_virtual_time_s", "virtual_time_s")
        cleared = _pick(s, "cleared_count")
        nsrc = _pick(s, "source_count", "n_source", "source_num", "sources")
        if nsrc is None:
            truth = _pick(s, "truth")
            nsrc = len(truth) if isinstance(truth, (list, dict)) else None
        exit_status = _pick(s, "exit_status", default="?")
        unres = _pick(s, "unresolved_list", "unresolved_channels", default=[])
        if isinstance(unres, int):
            unres = [] if unres == 0 else [unres]
        if isinstance(unres, dict):
            unres = [c for c, v in unres.items() if v not in ("cleared", "excluded")]
        deep = _deep(s)
        mv = _pick(s, "movement_distance_m")
        meas = _pick(s, "total_measure", "measure_count")
        phases = _pick(s, "phases", "phase_stats", default={}) or {}
        rows.append(dict(file=os.path.basename(f), family=(
            os.path.basename(f).split("_log_")[0] if "_log_" in os.path.basename(f) else "other"),
            vt=vt, cleared=cleared, nsrc=nsrc,
            exit_status=exit_status, unresolved=list(unres or []),
            deep=deep, mv=mv, meas=meas, phases=phases))

    print(f"共 {len(rows) + len(bad)} 个日志文件; 有效 {len(rows)}, 无效 {len(bad)}")
    for b in bad:
        print(f"  [无效] {b[0]} ({b[1]} 行): {b[2]}")
    print("-" * 108)
    print("%-44s%8s%7s%8s%9s%9s%9s%8s%6s" % (
        "文件", "vt(s)", "清除", "真值", "深清", "移动(m)", "检测", "退出", "未决"))
    for r in rows:
        print("%-44s%8s%7s%8s%9s%9s%9s%8s%6d" % (
            r["file"], f"{r['vt']:.0f}" if isinstance(r["vt"], (int, float)) else "—",
            r["cleared"], r["nsrc"], r["deep"],
            f"{r['mv']:.0f}" if isinstance(r["mv"], (int, float)) else "—",
            r["meas"], r["exit_status"], len(r["unresolved"])))

    if not rows:
        return
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    print("=" * 108)
    print("日志族分布: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(fams.items())))
    if len(fams) > 1:
        print("  注: 只有 p3_log_* 是问题三当前格式; robot_log_* 为早期格式(无 cleared/移动字段),"
              " robot4_log_* 为问题四日志(误放于本目录), 均不计入下方统计。")
    core = fams.get("p3", [])
    if not core:
        core = rows
    rows_core = core
    vt = [r["vt"] for r in rows_core if isinstance(r["vt"], (int, float))]
    print("-" * 108)
    print(f"[统计口径] 族=p3_log_*, n={len(rows_core)}")
    print(f"虚拟时间: n={len(vt)} 均值 {st.mean(vt):.1f}s 中位 {st.median(vt):.0f}s "
          f"标准差 {st.pstdev(vt):.1f}s CV {100*st.pstdev(vt)/st.mean(vt):.2f}% "
          f"范围 [{min(vt):.0f}, {max(vt):.0f}]")
    uncleared = [r for r in rows_core if r["unresolved"]]
    print(f"有未决频道的局: " + (", ".join(r["file"] for r in uncleared) or "无(0 局)"))
    nz = [r for r in rows_core if r["cleared"]]
    print(f"cleared_count>0 的局: {len(nz)}/{len(rows_core)}; "
          f"区间 [{min(r['cleared'] for r in nz)}, {max(r['cleared'] for r in nz)}]")
    print(f"exit_status=completed 的局: "
          f"{sum(1 for r in rows_core if r['exit_status'] == 'completed')}/{len(rows_core)}"
          f" (其余为早期日志, 无该字段)")
    per_ratio = [r["vt"] / r["cleared"] for r in rows_core
                 if isinstance(r["vt"], (int, float)) and r["cleared"]]
    if per_ratio:
        print(f"每源时间(先按局求商再平均): {st.mean(per_ratio):.1f} s/源 (n={len(per_ratio)})")
    num = sum(r["vt"] for r in rows_core if isinstance(r["vt"], (int, float)))
    den = sum(r["cleared"] for r in rows_core if r["cleared"])
    if den:
        print(f"每源时间(总和之比 ΣT/ΣN): {num/den:.1f} s/源 (ΣT={num:.0f}s, ΣN={den})")
    mv = [r["mv"] for r in rows_core if isinstance(r["mv"], (int, float))]
    if mv:
        print(f"移动距离: 均值 {st.mean(mv):.0f} m, 范围 [{min(mv):.0f}, {max(mv):.0f}] m")
    xy = [(r["cleared"], r["vt"]) for r in rows_core
          if r["cleared"] and isinstance(r["vt"], (int, float))]
    if len(xy) > 2:
        n = len(xy)
        mx = st.mean(x for x, _ in xy)
        my = st.mean(y for _, y in xy)
        sxx = sum((x - mx) ** 2 for x, _ in xy)
        sxy = sum((x - mx) * (y - my) for x, y in xy)
        syy = sum((y - my) ** 2 for _, y in xy)
        b = sxy / sxx
        a = my - b * mx
        r2 = (sxy ** 2) / (sxx * syy)
        print(f"时间-源数线性回归(n={n}): T = {a:.0f} + {b:.1f}*N, R^2 = {r2:.3f}")
        print("  注: 源数不能单独解释总时间(批次/场景差异显著), 论文按源归一使用。")
    ph_agg = {}
    for r in rows_core:
        for k, v in (r["phases"] or {}).items():
            if isinstance(v, dict) and isinstance(v.get("virtual_time_s"), (int, float)):
                ph_agg.setdefault(k, []).append(v["virtual_time_s"])
    if ph_agg:
        print("分阶段平均时间(s): " + ", ".join(
            f"{k}={st.mean(v):.0f}" for k, v in sorted(ph_agg.items())))
    print("注: 本脚本只做归档核对, 不修改任何数据; 异常局一律显式列出。")


if __name__ == "__main__":
    main()
