# -*- coding: utf-8 -*-
"""问题4 外部改进模块消融实验(统一口径: 共享库 simlib)

模块:
  1+4 负信息 -> 联合可行域/指向状态   USE_NEG_INFO
  3   改进 PSO 精化位置               USE_PSO
  5   清除后多方向复核                 DO_VERIFY
  N   归航邻域试探圈                  NEIGHBOR_RINGS
  S   补测距离上限                    SUPP_MAX_DIST
  D   顺路清除阈值                    ON_WAY_DELTA
  M   网格几何(边长/外扩/旋转/平移)    MESH_*
每臂 = 完整冻结配置 + 本臂覆盖项(simlib.config_scope, 退出必恢复); 计时/清除核验统一。
"""
import sys, math, time, contextlib, io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simlib
from simlib import (FROZEN4_CLS, FROZEN4_MOD, frozen4_cls, frozen4_mod, case_env,
                    scene_hash, SimClient, sim_time, ledger_time, check_clearance,
                    summarize, paired_stat, bootstrap_ci, config_scope, cfg_hash)

r4 = simlib.load_module("r4", Path(__file__).resolve().parent / "robot4.py")
exp = simlib.exp




def run_cfg(neg, pso, verify, n=8, ratios=(0.5, 1.0), seed=3026):
    r4.Problem4Robot.USE_NEG_INFO = neg
    r4.Problem4Robot.USE_PSO = pso
    r4.Problem4Robot.DO_VERIFY = verify
    out = {}
    for pd in ratios:
        crs = []; Ls = []; ms = []; fs = []; Ts = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = SimClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = rb.run()
            chk = check_clearance(n_ret, cli, env)
            crs.append(chk["src_clear_ratio"]); Ls.append(cli.dist); ms.append(cli.n_measure); fs.append(cli.fail)
            Ts.append(cli.dist/5 + cli.n_measure*5)
        out[pd] = (np.mean(crs), np.mean(Ls), np.mean(ms), float(np.mean(fs)), np.mean(Ts))
    return out




def run_neighbor(rings, n=30, seed=3026, ratios=(0.5, 1.0)):
    """邻域试探消融: 同一批随机案例(同 seed)下跑一种 NEIGHBOR_RINGS 配置。

    逐案例产出: 清除率/是否漏清、归航 episode 明细(homing_diag)、清除明细(clear_diag)、
    总移动/检测/清除/换频/时间。
    """
    r4.Problem4Robot.NEIGHBOR_RINGS = tuple(rings)
    out = {}
    for pd in ratios:
        rows = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = SimClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = rb.run()
            chk = check_clearance(n_ret, cli, env)
            T = sim_time(cli)
            rows.append(dict(
                cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                full=chk["case_full_clear"], miss_src=chk["missing_src"],
                consistent=chk["consistent"], n_src=chk["n_src"], scene=scene_hash(env),
                dist=cli.dist, meas=cli.n_measure, clear=cli.n_clear,
                fail=cli.fail, ok=cli.n_clear_ok, sw=cli.n_switch, T=T,
                homing=list(getattr(cli, "homing_diag", [])),
                cdiag=list(getattr(cli, "clear_diag", []) or
                           getattr(rb.c, "clear_diag", []))))
        out[pd] = rows
    return out


def _ep_stats(rows, key):
    """从逐案例的 episode 明细汇总邻域/换示向指标(邻域按圈半径分别统计)。"""
    eps = [e for r in rows for e in r["homing"]]
    cd = [d for r in rows for d in r["cdiag"]]
    ring = {}
    for d in cd:
        s = str(d.get("src"))
        if "邻域" not in s or d.get("result") == "rejected":
            continue
        tag = s.split("|")[-1]                     # 例: 邻域8m / 邻域15m
        a = ring.setdefault(tag, [0, 0])
        a[0] += 1
        a[1] += 1 if d.get("result") == "success" else 0
    ring_att = sum(v[0] for v in ring.values())
    ring_ok = sum(v[1] for v in ring.values())
    by = {}
    for e in eps:
        if e.get("cleared_by"):
            by[e["cleared_by"]] = by.get(e["cleared_by"], 0) + 1
    ring_src = sum(v for k, v in by.items() if str(k).startswith("邻域"))
    multi = [e for e in eps if e.get("bearings_tried", 0) > 1]
    later = [e for e in multi if (e.get("cleared_at_bearing") or 0) > 0]
    hard = [e for e in eps if e.get("cleared_by")]
    n_cases = len(rows)
    return dict(
        episodes=len(eps), cleared_by=by, ring=ring,
        ring_att=ring_att, ring_ok=ring_ok,
        ring_rate=(ring_ok/ring_att if ring_att else None),
        ring_cond=(ring_src/len(eps) if eps else None),
        multi=len(multi), later=len(later),
        later_rate=(len(later)/len(multi) if multi else None),
        ep_moves=(np.mean([e.get("episode_moves_m", 0.0) for e in hard]) if hard else 0.0),
        ep_clears=(np.mean([(e.get("cost") or {}).get("n_clear") or 0 for e in hard])
                   if hard else 0.0),
        ep_time=(np.mean([e.get("episode_time_s", 0.0) for e in hard]) if hard else 0.0),
        ep_per_case=len(eps)/n_cases, hard_per_case=len(hard)/n_cases)


def paired_neighbor(n=100, seed=3026):
    """邻域试探四臂消融: 只做既有流程的开关组合, 不设计新算法。默认策略不变。"""
    arms = [("当前: 8m+15m 两圈各6点", (8.0, 15.0)),
            ("仅 8m 圈", (8.0,)),
            ("仅 15m 圈", (15.0,)),
            ("取消邻域试探(失败即换下一条示向)", ())]
    res = {}
    for name, rings in arms:
        t0 = time.time()
        res[name] = run_neighbor(rings, n, seed)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]
    print(f"\n[邻域试探消融] n={n} 案例/档, 同 seed 同场景配对")
    for pd in (0.5, 1.0):
        print(f"\n=== 定向比例 {pd*100:.0f}% ===")
        hdr = "%-34s%8s%8s%9s%10s%11s%13s%9s" % (
            "配置", "清除率", "漏清例", "移动(m)", "时间(s)", "P90(s)",
            "Δ时间(s)", "Δ时间95%CI")
        print(hdr); print("-" * len(hdr))
        for name, _ in arms:
            rows = res[name][pd]
            cr = np.mean([r["cr"] for r in rows])
            T = np.array([r["T"] for r in rows])
            s = "%-34s%7.1f%%%8d%9.0f%10.0f%11.0f" % (
                name, cr*100, sum(r["miss"] for r in rows),
                np.mean([r["dist"] for r in rows]), T.mean(), np.percentile(T, 90))
            if name == arms[0][0]:
                s += "%13s%9s" % ("—", "—")
            else:
                d = T - np.array([r["T"] for r in base[pd]])
                se = d.std(ddof=1)/math.sqrt(len(d))
                s += "%13.0f%9s" % (d.mean(), f"[{d.mean()-1.96*se:.0f},{d.mean()+1.96*se:.0f}]")
            print(s)
        print("%-34s%8s%8s%9s%10s%11s" % ("", "episode", "邻域尝试", "邻域成功",
                                          "邻域成功率", "条件成功率"))
        for name, _ in arms:
            e = _ep_stats(res[name][pd], name)
            print("%-34s%8d%8d%10d%11s%12s" % (
                name, e["episodes"], e["ring_att"], e["ring_ok"],
                f"{e['ring_rate']*100:.1f}%" if e["ring_rate"] is not None else "—",
                f"{e['ring_cond']*100:.1f}%" if e["ring_cond"] is not None else "—"))
            print("%-34s%s" % ("", "  按圈: " + ("; ".join(
                f"{k} 尝试{v[0]} 成功{v[1]} ({100.0*v[1]/v[0]:.1f}%)"
                for k, v in sorted(e["ring"].items())) if e["ring"] else "无邻域尝试")))
        print("%-34s%10s%10s%10s%10s" % ("", "换示向例", "补回例", "补回比例", "清除来源"))
        for name, _ in arms:
            e = _ep_stats(res[name][pd], name)
            print("%-34s%10d%10d%10s%10s" % (
                name, e["multi"], e["later"],
                f"{e['later_rate']*100:.1f}%" if e["later_rate"] is not None else "—",
                ",".join(f"{k}:{v}" for k, v in sorted(e["cleared_by"].items())) or "—"))
        print("%-34s%11s%11s%11s%11s" % ("", "困难源/例", "额外移动", "额外清除", "额外时间"))
        for name, _ in arms:
            e = _ep_stats(res[name][pd], name)
            print("%-34s%11.2f%11.0f%11.2f%11.0f" % (
                name, e["hard_per_case"], e["ep_moves"], e["ep_clears"], e["ep_time"]))
    r4.Problem4Robot.NEIGHBOR_RINGS = (8.0, 15.0)   # 复位默认(正式策略不变)
    return res


def run_supp_cap(cap, n=400, seed=3026, ratios=(0.5, 1.0)):
    """补测距离上限消融: 同一批随机案例(同 seed)下跑一个上限档。

    逐案例记录: 清除率/是否漏清、总时间(含换频与清除)、总移动、检测、清除,
    补测最近点距离(决策与执行)、归航 episode、被上限跳过的频道及其后续补回情况。
    """
    r4.Problem4Robot.SUPP_MAX_DIST = cap
    out = {}
    for pd in ratios:
        rows = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = SimClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = rb.run()
            chk = check_clearance(n_ret, cli, env)
            T = sim_time(cli)
            sup = list(getattr(cli, "supp_diag", []))
            hom = list(getattr(cli, "homing_diag", []))
            sk = set(rb.supp_skipped)
            rows.append(dict(
                cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                full=chk["case_full_clear"], miss_src=chk["missing_src"],
                consistent=chk["consistent"], n_src=chk["n_src"], scene=scene_hash(env),
                dist=cli.dist, meas=cli.n_measure, T=T, fail=cli.fail,
                sup_dec=sum(1 for d in sup if d["action"] == "measure"),
                sup_exec=sum(1 for d in sup if d["action"] == "exec"),
                sup_skip=sum(1 for d in sup if d["action"] == "skip"),
                sup_max_m=max([d["nearest_m"] for d in sup
                               if d["action"] in ("measure", "exec")] or [0.0]),
                sup_mean_m=(np.mean([d["nearest_m"] for d in sup
                                     if d["action"] in ("measure", "exec")])
                            if any(d["action"] in ("measure", "exec") for d in sup) else 0.0),
                hom_ep=len(hom), hom_moves=sum(e.get("episode_moves_m", 0.0) for e in hom),
                sk_ch=len(sk),
                sk_ok=sum(1 for ch in sk if rb.state[ch] == "cleared"),
            ))
        out[pd] = rows
    return out


def paired_supp_cap(n=400, seed=3026):
    """补测距离上限四臂配对实验(既有流程内的调度阈值, 默认 None 不变)。"""
    arms = [("当前: 无上限", None), ("上限 1500 m", 1500.0),
            ("上限 2500 m", 2500.0), ("上限 3500 m", 3500.0)]
    res = {}
    for name, cap in arms:
        t0 = time.time()
        res[name] = run_supp_cap(cap, n, seed)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]
    print(f"\n[补测距离上限消融] n={n} 案例/档, 同 seed 同场景配对")
    print("硬约束: 任何档只要出现更多漏清或未解决频道, 即使平均时间下降也不采用")
    for pd in (0.5, 1.0):
        print(f"\n=== 定向比例 {pd*100:.0f}% ===")
        bo = np.array([r["T"] for r in base[pd]])
        thr = float(np.percentile(bo, 95))
        hdr = "%-15s%8s%7s%8s%8s%8s%8s%9s%9s%11s%10s%8s%9s%9s" % (
            "配置", "全清率", "漏清例", "平均(s)", "P90", "P95", "P99", "最大",
            "长尾例数", "Δ时间95%CI", "补测距离", "跳过", "归航移动", "补回率")
        print(hdr); print("-" * len(hdr))
        for name, _ in arms:
            rows = res[name][pd]
            T = np.array([r["T"] for r in rows])
            cr = np.mean([r["cr"] for r in rows])
            miss = sum(r["miss"] for r in rows)
            supd = [r["sup_mean_m"] for r in rows if r["sup_exec"] or r["sup_dec"]]
            sk = sum(r["sk_ch"] for r in rows)
            skok = sum(r["sk_ok"] for r in rows)
            tail = int((T > thr).sum())          # 超过基线 P95 的案例数(长尾计数)
            s = "%-15s%7.1f%%%7d%8.0f%8.0f%8.0f%8.0f%9.0f%9d" % (
                name, cr*100, miss, T.mean(), np.percentile(T, 90),
                np.percentile(T, 95), np.percentile(T, 99), T.max(), tail)
            if name == arms[0][0]:
                s += "%11s" % "—"
            else:
                d = T - bo
                se = d.std(ddof=1)/math.sqrt(len(d))
                s += "%11s" % f"[{d.mean()-1.96*se:.0f},{d.mean()+1.96*se:.0f}]"
            s += "%10s%8.2f%9.0f%9s" % (
                f"{np.mean(supd):.0f}m" if supd else "—",
                np.mean([r["sup_skip"] for r in rows]),
                np.mean([r["hom_moves"] for r in rows]),
                f"{100.0*skok/sk:.0f}%" if sk else "—")
            print(s)
        print(f"  (长尾例数 = 超过基线 P95 = {thr:.0f}s 的案例数; 基线 P95 以上共 "
              f"{int((bo > thr).sum())} 例)")
    r4.Problem4Robot.SUPP_MAX_DIST = None    # 复位默认(正式策略不变)
    return res


def run_mesh(mesh, n=400, seed=3026, ratios=(0.5, 1.0)):
    """网格几何消融: mesh = (a, margin, theta, offx, offy) 或 None(=冻结默认)。

    同一批场景(按案例编号派生)下跑一种网格, 逐案例记录时间/移动/检测/清除与证书状态。
    """
    if mesh is None:
        r4.MESH_A, r4.MESH_MARGIN, r4.MESH_THETA, r4.MESH_OFFSET = 900.0, 800.0, 0.0, (0.0, 0.0)
    else:
        a, mg, th, ox, oy = mesh
        r4.MESH_A, r4.MESH_MARGIN, r4.MESH_THETA, r4.MESH_OFFSET = a, mg, th, (ox, oy)
    out = {}
    for pd in ratios:
        rows = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = SimClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = rb.run()
            chk = check_clearance(n_ret, cli, env)
            T = sim_time(cli)
            rows.append(dict(cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
                full=chk["case_full_clear"], miss_src=chk["missing_src"],
                consistent=chk["consistent"], n_src=chk["n_src"], scene=scene_hash(env),
                             T=T, dist=cli.dist, meas=cli.n_measure, fail=cli.fail,
                             n_pts=len(rb.pts), n_tris=len(rb.tris),
                             cert_ok=bool(getattr(rb, "mesh_stats", None) is not None)))
        out[pd] = rows
    return out


def paired_mesh(n=400, seed=3026,
                cand=(920.0, 700.0, 20.0, 460.0, 398.0)):
    """网格几何配对实验: 冻结网格 vs 候选网格(旋转+平移), 硬约束=全清率100%。"""
    arms = [("冻结 900/800/θ0/off0", None), (f"候选 {cand[0]:.0f}/{cand[1]:.0f}/"
                                          f"θ{cand[2]:.0f}/off({cand[3]:.0f},{cand[4]:.0f})", cand)]
    res = {}
    for name, mesh in arms:
        t0 = time.time()
        res[name] = run_mesh(mesh, n, seed)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]
    print(f"\n[网格几何配对] n={n}/臂/定向比例, 同场景(按案例编号派生)")
    for pd in (0.5, 1.0):
        print(f"\n=== 定向比例 {pd*100:.0f}% ===")
        print("%-34s%8s%6s%9s%10s%13s%9s%9s%9s" % (
            "网格", "全清率", "漏清", "平均(s)", "Δ时间(s)", "Δ95%CI", "P90(s)",
            "最大(s)", "检测/例"))
        for name, _ in arms:
            rows = res[name][pd]
            T = np.array([r["T"] for r in rows])
            p = _pair_stat(T, np.array([r["T"] for r in base[pd]]))
            print("%-34s%7.1f%%%6d%9.0f%+10.1f%13s%9.0f%9.0f%9.1f" % (
                name, np.mean([r["cr"] for r in rows])*100,
                sum(r["miss"] for r in rows), T.mean(), p[0],
                f"[{p[2][0]:.0f},{p[2][1]:.0f}]", np.percentile(T, 90), T.max(),
                np.mean([r["meas"] for r in rows])))
        print("  移动: " + "  ".join(
            f"{name.split()[0]} {np.mean([r['dist'] for r in res[name][pd]]):.0f}m" for name, _ in arms))
        print("  网格: " + "  ".join(
            f"{name.split()[0]} {res[name][pd][0]['n_pts']}点/{res[name][pd][0]['n_tris']}三角"
            for name, _ in arms))
    r4.MESH_A, r4.MESH_MARGIN, r4.MESH_THETA, r4.MESH_OFFSET = 900.0, 800.0, 0.0, (0.0, 0.0)
    return res


def run_onway(delta, n=30, seed=3026, ratios=(0.5, 1.0)):
    """同一批随机案例(同 seed)下跑一个顺路清除阈值, 返回逐案例明细用于配对比较。

    时间模型(与题设一致): 移动/5 + 检测*5 + 换频*1 + 成功清除*5 + 失败清除*3。
    """
    r4.ON_WAY_DELTA = delta            # 模块级常量, run() 在调用时读取
    out = {}
    for pd in ratios:
        rows = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = SimClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = rb.run()
            chk = check_clearance(n_ret, cli, env)
            T = sim_time(cli)
            homing = sum(1 for d in getattr(cli, "clear_diag", [])
                         if "homing" in str(d.get("phase")) or "归航" in str(d.get("src")))
            rows.append((chk["src_clear_ratio"], cli.dist, cli.n_measure, cli.fail, T, homing))
        out[pd] = np.array(rows)
    return out


def _pair_stat(arm, base):
    """配对差值的均值/标准误/95%置信区间/改善案例占比。"""
    d = np.asarray(arm, dtype=float) - np.asarray(base, dtype=float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return m, se, (m - 1.96*se, m + 1.96*se), float((d < 0).mean()), d


def paired_onway(n=30, seed=3026, arms=(200.0, 300.0, 500.0, 800.0, None)):
    """第5项配对实验: 修复证书索引后重新检验顺路清除阈值(默认值暂不改)。"""
    res = {}
    for d in arms:
        t0 = time.time()
        res[d] = run_onway(d, n, seed)
        print(f"  已跑 δ={d}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[300.0]
    print(f"\n[配对实验] 顺路清除阈值  n={n} 案例(同 seed, 同场景; 时间模型含换频与清除)")
    for pd in (0.5, 1.0):
        hdr = "%-12s%9s%9s%10s%9s%8s%13s%9s%15s%9s" % (
            f"δ(m) {pd*100:.0f}%定向", "全清率", "移动(m)", "时间(s)", "P90(s)",
            "失败/归航", "Δ时间(s)", "Δ移动(m)", "Δ时间95%CI", "变快比例")
        print("\n" + hdr); print("-" * len(hdr))
        for d in arms:
            r = res[d][pd]
            cr = float(np.mean(r[:, 0]))
            s = "%-12s%8.1f%%%9.0f%10.0f%9.0f%8.1f" % (
                str(d) + ("(当前)" if d == 300.0 else ""), cr*100,
                r[:, 1].mean(), r[:, 4].mean(), np.percentile(r[:, 4], 90),
                r[:, 3].mean() + r[:, 5].mean())
            if d == 300.0:
                s += "%13s%9s%15s%9s" % ("—", "—", "—", "—")
            else:
                mt, se, ci, frac, _ = _pair_stat(r[:, 4], base[pd][:, 4])
                mm = (r[:, 1] - base[pd][:, 1]).mean()
                s += "%13.0f%9.0f%15s%8.0f%%" % (
                    mt, mm, f"[{ci[0]:.0f},{ci[1]:.0f}]", frac*100)
            print(s)
    r4.ON_WAY_DELTA = 300.0        # 复位到默认值(默认策略不变)
    return res


def _main4():
    """命令行入口。每个实验臂都在 simlib.config_scope 内运行, 退出必恢复冻结配置。"""
    if len(sys.argv) > 1 and sys.argv[1] == "--mesh-paired":
        paired_mesh(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--supp-cap":
        paired_supp_cap(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--neighbor":
        paired_neighbor(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--onway-paired":
        paired_onway(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
        return
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfgs = [
        ("基准(全关)",        False, False, False),
        ("+1/4 负信息",       True,  False, False),
        ("+3 PSO",            False, True,  False),
        ("+5 多方向复核",     False, False, True),
        ("+1/4+5 负信息+复核", True,  False, True),
    ]
    hdr = "%-20s%16s%16s" % ("配置", "50%定向", "100%定向")
    print(hdr); print("-" * len(hdr))
    print("%-20s%16s%16s" % ("", "清除率/移动/时间", "清除率/移动/时间"))
    for name, a, b, cc in cfgs:
        t0 = time.time()
        r = run_cfg(a, b, cc, N)
        line = "%-20s" % name
        for pd in (0.5, 1.0):
            cr, L, m, f, T = r[pd]
            line += "%6.1f%%/%5.0f/%5.0fs" % (cr*100, L, T)
        print(line + "  [%.0fs]" % (time.time()-t0), flush=True)


if __name__ == "__main__":
    try:
        _main4()
    finally:
        simlib.apply_cfg(r4.Problem4Robot, FROZEN4_CLS)
        simlib.apply_cfg(r4, FROZEN4_MOD)

