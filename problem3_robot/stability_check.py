# -*- coding: utf-8 -*-
"""部署稳定性测试检查器(仅 A 臂连跑, 不与 B 臂做统计比较)。

只检查工程健壮性五项:
  1) 是否全清(每局清除数 == 源数);
  2) 是否出现 HTTP 拒绝/异常(rejected_count、error 字段、空局);
  3) 是否有未解决频道;
  4) 最大虚拟时间是否越过既有 4897 s 上界;
  5) 日志/退出摘要/配置是否完整(有 __summary__、动作数 > 0、ring 配置与预期一致)。

用法: python stability_check.py "logs/p3_log_*stab_a*.jsonl" [ring_r 预期] [ring_n 预期] [时间上界]
"""
import glob
import json
import os
import sys


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "problem3_robot/logs/p3_log_*stab_a*.jsonl"
    exp_r = float(sys.argv[2]) if len(sys.argv) > 2 else 1150.0
    exp_n = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    t_ub = float(sys.argv[4]) if len(sys.argv) > 4 else 4897.0
    files = sorted(glob.glob(pat))
    print(f"部署稳定性测试: {len(files)} 个日志 ({pat})")
    print(f"预期配置 ring_r={exp_r} ring_n={exp_n}; 时间上界参照 {t_ub:.0f} s")
    print(f"{'局':>4}{'源数':>6}{'清除':>6}{'虚拟时间(s)':>13}{'动作':>7}{'拒绝':>6}"
          f"{'失败清除':>9}{'全清':>6}{'摘要':>6}{'配置':>8}")
    n_ok = n_full = n_empty = n_reject = n_unres = 0
    tmax = 0.0
    problems = []
    for i, f in enumerate(files, 1):
        tag = os.path.basename(f)
        s = None
        for line in open(f, encoding="utf-8"):
            if line.startswith('{"__summary__"'):
                s = json.loads(line)["__summary__"]
        if s is None:
            print(f"{i:>4}  !! 无 __summary__(日志不完整)  {tag}")
            problems.append(f"{tag}: 缺 __summary__")
            continue
        acts = int(s.get("total_actions", 0))
        T = float(s.get("final_virtual_time_s", 0.0))
        rej = int(s.get("rejected_count", 0))
        clr = int(s.get("clear_success_count", 0))
        fail = int(s.get("clear_failure_count", 0))
        nsrc = int(s.get("cleared_count", clr)) or clr
        cfg = s.get("config", {}) or {}
        err = s.get("error")
        unres = s.get("unresolved_channels", s.get("unresolved_kind"))
        cfg_ok = (abs(float(cfg.get("ring_r", -1)) - exp_r) < 1e-9
                  and int(cfg.get("ring_n", -1)) == exp_n)
        if acts <= 0 or T <= 0:
            print(f"{i:>4}  !! 空局(未产生有效数据)  {tag}")
            n_empty += 1
            problems.append(f"{tag}: 空局")
            continue
        full = (clr == nsrc)
        n_full += 1 if full else 0
        n_ok += 1
        n_reject += 1 if rej > 0 else 0
        if unres not in (None, "", [], {}):
            n_unres += 1
        if err:
            problems.append(f"{tag}: error={err}")
        if rej > 0:
            problems.append(f"{tag}: 拒绝 {rej} 次")
        tmax = max(tmax, T)
        print(f"{i:>4}{nsrc:>6}{clr:>6}{T:>13.1f}{acts:>7}{rej:>6}{fail:>9}"
              f"{('是' if full else '否'):>6}{('OK' if s else '缺'):>6}"
              f"{('OK' if cfg_ok else '异常'):>8}")
        if not cfg_ok:
            problems.append(f"{tag}: 配置异常 ring_r={cfg.get('ring_r')} ring_n={cfg.get('ring_n')}")
    print(f"\n汇总: 有效局 {n_ok}/{len(files)}  全清 {n_full}  "
          f"空局/不完整 {n_empty}  出现拒绝的局 {n_reject}  有未解决标记的局 {n_unres}")
    print(f"最大虚拟时间 {tmax:.1f} s  (上界 {t_ub:.0f} s: "
          f"{'未越过' if tmax <= t_ub else '**已越过**'})")
    print("问题清单:", "无" if not problems else "")
    for p in problems:
        print("   -", p)
    print("\n(本测试仅检查工程健壮性, 不与 B 臂做统计比较。)")


if __name__ == "__main__":
    main()
