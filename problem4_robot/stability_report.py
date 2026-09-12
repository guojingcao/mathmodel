# -*- coding: utf-8 -*-
"""问题四 官方模拟器在环实验的稳定性报告。

用法: python stability_report.py "logs/p4_log_*_s??.jsonl"

输出: 逐局明细 + 稳定性统计(均值/中位/标准差/CV/范围) + 两种每源口径 +
      总时间对源数的回归(用于把"场景方差"与"算法方差"分开) + 分阶段稳定性。
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    s = None
    for ln in open(path, encoding="utf-8"):
        if "__summary__" in ln:
            try:
                s = json.loads(ln)["__summary__"]
            except json.JSONDecodeError:
                pass
    return s


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "logs", "*.jsonl")
    files = sorted(glob.glob(pat))
    rows = []
    for f in files:
        s = load(f)
        if not s or s.get("error"):
            continue
        ph = s.get("phase_stats") or {}
        rows.append(dict(
            f=os.path.basename(f), vt=s["final_virtual_time_s"],
            n=s["robot"]["cleared_count"], meas=s["measure_count"],
            mv=s["movement_distance_m"], unres=len(s["robot"].get("unresolved_list") or []),
            tri=s.get("mesh_triangles") or 0,
            ms=(ph.get("mesh_scan") or {}).get("virtual_time_s", 0.0),
            after=(ph.get("queue_clear") or {}).get("virtual_time_s", 0.0),
            onway=(ph.get("on_way") or {}).get("virtual_time_s", 0.0),
            fail=s.get("clear_failure_count", 0), ok=s.get("clear_success_count", 0)))
    if not rows:
        print("无有效日志")
        return
    print(f"有效局 {len(rows)}")
    print("%-8s%10s%6s%8s%10s%9s%9s%8s%6s" % (
        "局", "vt(s)", "源", "检测", "移动(m)", "网格(s)", "每源(s)", "失败", "未决"))
    for r in rows:
        print("%-8s%10.1f%6d%8d%10.0f%9.0f%9.1f%8d%6d" % (
            r["f"][-9:-5], r["vt"], r["n"], r["meas"], r["mv"], r["ms"],
            r["vt"]/r["n"], r["fail"], r["unres"]))
    vt = [r["vt"] for r in rows]
    ns = [r["n"] for r in rows]
    per = [r["vt"]/r["n"] for r in rows]
    print("\n=== 稳定性统计 ===")
    for name, v in (("总虚拟时间(s)", vt), ("每源时间(局内求商再平均, s/源)", per),
                    ("源数", ns), ("检测次数", [r["meas"] for r in rows]),
                    ("移动(m)", [r["mv"] for r in rows]),
                    ("网格扫描阶段(s)", [r["ms"] for r in rows])):
        m = st.mean(v); sd = st.pstdev(v)
        print("  %-28s 均值 %9.1f  中位 %9.1f  标准差 %8.1f  CV %6.2f%%  范围 [%.0f, %.0f]"
              % (name, m, st.median(v), sd, 100*sd/m, min(v), max(v)))
    print("  每源时间(ΣT/ΣN)            %.1f s/源" % (sum(vt)/sum(ns)))
    print("  全清率                     %d/%d = %.1f%%" % (
        sum(1 for r in rows if r["unres"] == 0), len(rows),
        100.0*sum(1 for r in rows if r["unres"] == 0)/len(rows)))
    print("  零未解决/零清除失败        %d/%d 局, %d/%d 局" % (
        sum(1 for r in rows if r["unres"] == 0), len(rows),
        sum(1 for r in rows if r["fail"] == 0), len(rows)))
    # 总时间对源数的回归: 分离"场景方差"与"算法方差"
    n = len(rows)
    mx = st.mean(ns); my = st.mean(vt)
    sxx = sum((x-mx)**2 for x in ns)
    sxy = sum((x-mx)*(y-my) for x, y in zip(ns, vt))
    syy = sum((y-my)**2 for y in vt)
    if sxx > 0 and syy > 0:
        b = sxy/sxx; a = my - b*mx; r2 = (sxy**2)/(sxx*syy)
        resid = [y - (a + b*x) for x, y in zip(ns, vt)]
        print("  回归: T = %.0f + %.1f*N    R^2 = %.3f   残差标准差 %.0f s "
              "(残差 CV %.1f%% -> 扣除源数后的算法波动)" % (
                  a, b, r2, st.pstdev(resid), 100*st.pstdev(resid)/my))
    print("\n注: 官方模拟器每局随机生成场景, 源数不可控; 故跨局比较以每源口径为准。")


if __name__ == "__main__":
    main()
