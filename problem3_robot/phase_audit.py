"""问题三分阶段口径审计: 逐局核对 Σ(阶段虚拟时间) 与 final_virtual_time_s 是否闭合。

用法:
    python phase_audit.py                  # 审计 logs/ 下全部 p3_log_*.jsonl
    python phase_audit.py --all            # 连同早期格式/异构日志一起列出
    python phase_audit.py <file> ...

判定:
  * 闭合  : |Σ阶段 - final_vt| <= 0.05 s   → 阶段边界无重叠/无遗漏
  * 不闭合: 打印残差与逐阶段明细, 用于定位归属缺陷

输出:
  * 逐局残差表
  * 阶段占比(按"局内归一"再平均, 因此天然合计 100%, 可安全用于分阶段图)
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.05


def load(path):
    s = None
    for line in open(path, encoding="utf-8"):
        if "__summary__" in line:
            try:
                s = json.loads(line)["__summary__"]
            except json.JSONDecodeError:
                pass
    return s


def phases_of(s):
    ps = s.get("phase_stats") or s.get("phases") or {}
    out = {}
    for k, v in ps.items():
        if isinstance(v, dict):
            out[k] = float(v.get("virtual_time_s") or 0.0)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    files = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?") else [a])
    if not files:
        files = sorted(glob.glob(os.path.join(HERE, "logs", "p3_log_*.jsonl")))

    rows = []
    for f in files:
        s = load(f)
        name = os.path.basename(f)
        if not s:
            rows.append((name, None, None, None, None, "无 summary"))
            continue
        ps = phases_of(s)
        if not ps:
            rows.append((name, s.get("final_virtual_time_s"), None, None, None,
                         "无 phase_stats(早期格式)"))
            continue
        tot = float(s.get("final_virtual_time_s") or 0.0)
        sp = sum(ps.values())
        rows.append((name, tot, sp, sp - tot, ps,
                     "闭合" if abs(sp - tot) <= TOL else "不闭合"))

    print(f"审计 {len(rows)} 个日志文件 (容差 {TOL} s)")
    print("-" * 104)
    print("%-42s%10s%10s%10s%8s  %s" % ("文件", "final_vt", "Σ阶段", "残差", "阶段数", "判定"))
    ok = bad = na = 0
    for name, tot, sp, res, ps, verdict in rows:
        if sp is None:
            na += 1
            print("%-42s%10s%10s%10s%8s  %s" % (
                name, f"{tot:.0f}" if isinstance(tot, (int, float)) else "—",
                "—", "—", "—", verdict))
            continue
        if verdict == "闭合":
            ok += 1
        else:
            bad += 1
        print("%-42s%10.1f%10.1f%10.1f%8d  %s" % (name, tot, sp, res, len(ps), verdict))
        if verdict == "不闭合":
            for k in sorted(ps, key=lambda x: -ps[x]):
                print(f"      {k:<24}{ps[k]:10.1f}")

    print("=" * 104)
    print(f"闭合 {ok} 局, 不闭合 {bad} 局, 无阶段数据 {na} 局")
    closed = [r for r in rows if r[3] is not None and abs(r[3]) <= TOL]
    if closed:
        res_abs = [abs(r[3]) for r in closed]
        print(f"闭合局残差: 均值 {st.mean(res_abs):.4f} s, 最大 {max(res_abs):.4f} s "
              f"(相对最大 {100*max(res_abs)/st.mean([r[1] for r in closed]):.4f} %)")

    # 阶段占比: 先按"局内占比", 再对**全部有阶段数据的局**求平均(缺失阶段记 0),
    # 这样每个阶段的均值都是同一子集(n 相同)上的平均, 合计恒为 100%。
    runs = [(n, t, ps) for (n, t, sp, res, ps, v) in rows if ps and t]
    if runs:
        names = sorted({k for _, _, ps in runs for k in ps})
        n_all = len(runs)
        tmean = st.mean(t for _, t, _ in runs)
        shares = {k: [ps.get(k, 0.0) / t for _, t, ps in runs] for k in names}
        tot_share = sum(st.mean(v) for v in shares.values())
        print("-" * 104)
        print(f"阶段占比 (局内归一后对全部 {n_all} 局平均, 缺失记 0; 合计 {tot_share:.1%})")
        for k in sorted(names, key=lambda x: -st.mean(shares[x])):
            v = shares[k]
            have = sum(1 for _, _, ps in runs if k in ps)
            print(f"  {k:<22}{st.mean(v):8.2%}   折算 {st.mean(v)*tmean:7.0f} s "
                  f"(该阶段出现在 {have}/{n_all} 局)")
        print(f"  {'总虚拟时间均值':<20}{'':8}   折算 {tmean:7.0f} s")
        print(f"  校验: 阶段占比合计 = {tot_share:.1%} (应为 100%); "
              f"各阶段折算时间之和 = {sum(st.mean(v)*tmean for v in shares.values()):.0f} s "
              f"vs 总均值 {tmean:.0f} s")
    print("注: 阶段占比必须在**同一子集**上平均(缺失记 0), 否则"
          "\"各阶段在各自子集上求平均再相加\"会超过 100%。")


if __name__ == "__main__":
    main()
