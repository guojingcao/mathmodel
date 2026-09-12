# -*- coding: utf-8 -*-
"""问题3 外部改进模块消融实验

模块:
  A 贝叶斯概率图排序   ORDER_BY_PROB
  B DOP(交会角)预筛   DOP_PRESCREEN
  C 滚动优化(受限顺路清除)  ON_WAY_DELTA  (0/None = 关闭)
  E LS 试探清除信任门限 LS_CLEAR_GATE
  O 机会性顺带观测     OPP_MEASURE
  R 补测点复用         SUPP_REUSE

统一口径(审查后):
  * 每个实验臂 = **完整冻结配置 + 本臂覆盖项**(simlib.config_scope, 退出必恢复)
  * 计时只用 simlib.sim_time(); 阶段时间用 simlib.phase_time()(含换频与清除成本)
  * 清除率用真值核验(simlib.check_clearance): 案例全清率 与 平均源清除比例 分开报告
  * 场景按 (seed, 案例编号) 派生并落盘 scene_hash, 可独立验证配对
  * 统计对空集/单例安全(simlib.summarize / paired_stat), 稀疏模块给 bootstrap 区间
"""
import sys, math, time, contextlib, io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simlib
from simlib import (FROZEN3, FROZEN4_CLS, FROZEN4_MOD, frozen3, case_env, scene_hash,
                    SimClient, sim_time, ledger_time, phase_time, check_clearance,
                    summarize, paired_stat, bootstrap_ci, config_scope, cfg_hash)

rb = simlib.load_module("rb", Path(__file__).resolve().parent / "robot.py")
exp = simlib.exp






def run_opp(opp_on, max_per_point=2, target="spec", n=400, seed=2026, paired=False):
    """模块O(机会性顺带观测)消融: 同一批随机案例(同 seed)下跑一种配置。

    时间模型(与题设一致): 移动/5 + 检测*5 + 换频*1 + 成功清除*5 + 失败清除*3。
    同时记录补测阶段时间(用于在"关闭版"结果上预先定义困难子集, 避免选择偏差)。
    """
    cfg = frozen3(OPP_MEASURE=opp_on, OPP_MAX_PER_POINT=max_per_point,
                  OPP_TARGET=target, SUPP_REUSE=False)
    rows = []
    scope = config_scope((rb.Problem3Robot, cfg))
    scope.__enter__()
    for k in range(n):
        env = case_env(seed, k, directional=False)
        cli = SimClient(env); robot = rb.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n_ret = robot.run()
        chk = check_clearance(n_ret, cli, env)
        T = sim_time(cli)
        opp = list(getattr(cli, "opp_diag", []))
        ph = getattr(cli, "ph", {})
        sup_t = ph.get("supplement", [0.0, 0, 0])
        rows.append(dict(
            cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
            full=chk["case_full_clear"], miss_src=chk["missing_src"],
            consistent=chk["consistent"], n_src=chk["n_src"],
            scene=scene_hash(env),
            T=T, dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch,
            clear=cli.n_clear, fail=cli.fail,
            supp_t=phase_time(cli, "supplement"),                 # 补测阶段时间(选困难子集用)
            homing_t=phase_time(cli, "queue_homing"),
            opp_n=len(opp),
            opp_sig=sum(1 for d in opp if d.get("result") in ("direction", "near")),
            opp_cert=sum(1 for d in opp if d.get("became_certified")),
            opp_avoid=sum(1 for d in opp if d.get("avoided_supplement")),
            opp_cost=sum(d.get("time_cost") or 0.0 for d in opp),
        ))
    return rows


def _paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1) if n > 1 else 0.0
    se = sd/math.sqrt(n) if n else 0.0
    return dict(mean=m, se=se, lo=m-1.96*se, hi=m+1.96*se,
                med=float(np.median(d)), p10=float(np.percentile(d, 10)),
                p90=float(np.percentile(d, 90)), win=float((d < 0).mean()), d=d)


def paired_opp(n=400, seed=2026):
    """模块O 配对实验: O0关闭 / 每点<=1 / <=2 / <=3 / 收紧口径<=2。"""
    arms = [("O0 关闭(冻结基线)", False, 2, "spec"),
            ("O1 每点<=1", True, 1, "spec"),
            ("O2 每点<=2", True, 2, "spec"),
            ("O3 每点<=3", True, 3, "spec"),
            ("O2u 收紧口径<=2", True, 2, "uncert")]
    res = {}
    for name, on, mx, tg in arms:
        t0 = time.time()
        res[name] = run_opp(on, mx, tg, n, seed)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]
    hard_idx = [i for i, r in enumerate(base) if r["supp_t"] > 600.0]   # 困难子集: 关闭版定义
    print(f"\n[模块O 配对实验] n={n} 案例/档, 同 seed 同场景; "
          f"困难子集(关闭版补测阶段>600s) {len(hard_idx)} 例")
    print("硬约束: 任何档出现漏清即否决; 不重复测量同一频道同一坐标")
    for label, idx in (("全部案例", None), (f"困难子集({len(hard_idx)}例)", hard_idx)):
        print(f"\n=== {label} ===")
        print("%-18s%8s%6s%9s%9s%9s%10s%12s%11s%10s" % (
            "配置", "全清率", "漏清", "平均(s)", "vs基线", "95%CI", "变快比例",
            "中位差(s)", "P10/P90差", "补测(s)"))
        for name, _on, _mx, _tg in arms:
            rows = res[name]
            sel = range(len(rows)) if idx is None else idx
            T = np.array([rows[i]["T"] for i in sel])
            cr = np.mean([rows[i]["cr"] for i in sel])
            miss = sum(rows[i]["miss"] for i in sel)
            supt = np.mean([rows[i]["supp_t"] for i in sel])
            if name == arms[0][0]:
                print("%-18s%7.1f%%%6d%9.0f%9s%9s%10s%12s%11s%10.0f" % (
                    name, cr*100, miss, T.mean(), "—", "—", "—", "—", "—", supt))
                continue
            p = _paired([rows[i]["T"] for i in sel], [base[i]["T"] for i in sel])
            print("%-18s%7.1f%%%6d%9.0f%+9.1f%9s%9.0f%%%12.0f%11s%10.0f" % (
                name, cr*100, miss, T.mean(), p["mean"],
                f"[{p['lo']:.0f},{p['hi']:.0f}]", p["win"]*100, p["med"],
                f"{p['p10']:.0f}/{p['p90']:.0f}", supt))
        print("%-18s%8s%6s%9s%9s%9s%10s%11s%11s%9s" % (
            "", "机会观测", "有信号", "认证转化", "避免补测", "额外成本s", "额外检测", "额外换频",
            "移动Δ(m)", "失败清除"))
        for name, _on, _mx, _tg in arms:
            rows = res[name]
            sel = range(len(rows)) if idx is None else idx
            a = lambda k: sum(rows[i][k] for i in sel)/len(list(sel))
            b = lambda k: sum(base[i][k] for i in sel)/len(list(sel))
            print("%-18s%8.1f%6.1f%10.1f%8.1f%9.0f%10.1f%9.1f%11.0f%9.2f" % (
                name, a("opp_n"), a("opp_sig"), a("opp_cert"), a("opp_avoid"),
                a("opp_cost"), a("meas")-b("meas"), a("sw")-b("sw"),
                a("dist")-b("dist"), a("fail")-b("fail")))
    rb.Problem3Robot.OPP_MEASURE = False          # 复位默认(正式策略不变)
    rb.Problem3Robot.OPP_TARGET = "spec"
    return res


def run_reuse(reuse=False, gate=2000.0, saving=100.0, opp=False, opp_max=1,
              delete_mode="cert", n=300, seed=2026, paired=False):
    """模块R(补测点复用)配置运行: 逐案例返回时间与机制明细。"""
    cfg = frozen3(SUPP_REUSE=reuse, SUPP_REUSE_ROUTE_GATE_M=gate,
                  SUPP_REUSE_MIN_SAVING_M=saving, SUPP_REUSE_DELETE_MODE=delete_mode,
                  OPP_MEASURE=opp, OPP_MAX_PER_POINT=opp_max, OPP_TARGET="spec")
    scope = config_scope((rb.Problem3Robot, cfg))
    scope.__enter__()
    rows = []
    for k in range(n):
        env = case_env(seed, k, directional=False)
        cli = SimClient(env); robot = rb.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n_ret = robot.run()
        chk = check_clearance(n_ret, cli, env)
        T = sim_time(cli)
        rd = list(getattr(cli, "reuse_diag", []))
        at = [x for x in rd if x.get("kind") != "gate"]
        rem = [x for x in at if x.get("removed")]
        ph = getattr(cli, "ph", {})
        od = list(getattr(cli, "opp_diag", []))
        rows.append(dict(
            cr=chk["src_clear_ratio"], miss=1-chk["case_full_clear"],
            full=chk["case_full_clear"], miss_src=chk["missing_src"],
            consistent=chk["consistent"], n_src=chk["n_src"], scene=scene_hash(env),
            T=T, dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch, fail=cli.fail,
            supp_t=phase_time(cli, "supplement"),
            gate=sum(1 for x in rd if x.get("kind") == "gate" and x.get("armed")),
            attempts=len(at), signals=sum(1 for x in at
                                          if x.get("result") in ("direction", "near")),
            removed=len(rem),
            save_pred=sum(x.get("saving_m") or 0.0 for x in rem),
            save_actual=sum(x.get("route_saving_actual_m") or 0.0 for x in rem),
            extra_t=sum(x.get("time_cost") or 0.0 for x in at),
            opp_attempts=len(od),
        ))
    scope.__exit__(None, None, None)
    return rows


def _split_stats(rows, base, idx=None):
    sel = list(range(len(rows))) if idx is None else idx
    T = np.array([rows[i]["T"] for i in sel])
    d = T - np.array([base[i]["T"] for i in sel])
    se = d.std(ddof=1)/math.sqrt(len(d)) if len(d) > 1 else 0.0
    return dict(cr=np.mean([rows[i]["cr"] for i in sel]),
                miss=sum(rows[i]["miss"] for i in sel), T=T.mean(),
                dm=d.mean(), lo=d.mean()-1.96*se, hi=d.mean()+1.96*se,
                win=float((d < 0).mean()), med=float(np.median(d)),
                p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
                attempts=sum(rows[i]["attempts"] for i in sel)/len(sel),
                signals=sum(rows[i]["signals"] for i in sel)/len(sel),
                removed=sum(rows[i]["removed"] for i in sel)/len(sel),
                save_a=sum(rows[i]["save_actual"] for i in sel)/len(sel),
                extra=sum(rows[i]["extra_t"] for i in sel)/len(sel),
                fail=np.mean([rows[i]["fail"] for i in sel]),
                supp_t=np.mean([rows[i]["supp_t"] for i in sel]))


def train_scan_reuse(n=300, seed=111):
    """训练集: 扫描全局门控 D0 与局部删除收益门控 S0(不用实机日志选阈值)。"""
    print(f"[训练集] 扫描 D0 x S0  (n={n}/格, seed={seed}; 训练集与验证集种子不同)")
    print("%-10s%10s%11s%11s%9s%10s%11s%11s" % ("D0(m)", "S0(m)", "平均(s)", "vs基线(s)",
                                                "变快比", "删除任务", "省路(m)", "额外成本s"))
    base = run_reuse(False, None, 0.0, False, n=n, seed=seed)
    T0 = np.array([r["T"] for r in base])
    best = None
    for D0 in (1500.0, 2000.0, 2500.0, 3000.0, 3500.0):
        for S0 in (0.0, 50.0, 100.0, 200.0, 300.0):
            rows = run_reuse(True, D0, S0, False, n=n, seed=seed)
            s = _split_stats(rows, base)
            print("%-10.0f%10.0f%11.0f%+11.1f%9.0f%%%10.2f%11.0f%11.1f" % (
                D0, S0, s["T"], s["dm"], s["win"]*100, s["removed"], s["save_a"], s["extra"]))
            if best is None or s["dm"] < best[0]:
                best = (s["dm"], D0, S0, s)
    print(f"\n训练集最优: D0={best[1]:.0f} m, S0={best[2]:.0f} m  (Δ={best[0]:+.1f} s, "
          f"删除 {best[3]['removed']:.2f} 个/例, 额外成本 {best[3]['extra']:.1f} s/例)")
    rb.Problem3Robot.SUPP_REUSE = False
    return best[1], best[2]


def validate_reuse(d0, s0, n=800, seed=20260711):
    """验证集(独立种子): C0 基线 / C1 仅全局门控 / C2 双门控(仅认证删除) /
    C2L 双门控但认证或 LS 即删(激进删除) / C3 C2+清除点自然触发观测。"""
    print(f"\n[验证集] D0={d0:.0f} m, S0={s0:.0f} m, n={n}/臂, seed={seed}(独立)")
    arms = [("C0 基线(全关)", dict(reuse=False, gate=None, saving=0.0, opp=False)),
            ("C1 仅全局门控", dict(reuse=True, gate=d0, saving=0.0, opp=False)),
            ("C2 双门控(仅认证删)", dict(reuse=True, gate=d0, saving=s0, opp=False)),
            ("C2L 双门控(认证或LS删)", dict(reuse=True, gate=d0, saving=s0, opp=False,
                                       delete_mode="cert_or_ls")),
            ("C3 C2+清除点观测", dict(reuse=True, gate=d0, saving=s0, opp=True, opp_max=1))]
    res = {}
    for name, kw in arms:
        t0 = time.time()
        res[name] = run_reuse(n=n, seed=seed, **kw)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]
    hard = [i for i, r in enumerate(base) if r["supp_t"] > 600.0]
    easy = [i for i, r in enumerate(base) if r["supp_t"] <= 600.0]
    for label, idx in (("全部案例", None), (f"困难子集({len(hard)}例)", hard),
                       (f"容易子集({len(easy)}例)", easy)):
        print(f"\n=== {label} ===")
        print("%-22s%8s%6s%9s%10s%13s%8s%9s%10s" % (
            "配置", "全清率", "漏清", "平均(s)", "Δ时间(s)", "Δ95%CI", "变快比",
            "中位差", "P10/P90差"))
        for name, _ in arms:
            s = _split_stats(res[name], base, idx)
            print("%-22s%7.1f%%%6d%9.0f%+10.1f%13s%7.0f%%%9.0f%10s" % (
                name, s["cr"]*100, s["miss"], s["T"], s["dm"],
                f"[{s['lo']:.0f},{s['hi']:.0f}]", s["win"]*100, s["med"],
                f"{s['p10']:.0f}/{s['p90']:.0f}"))
        print("%-22s%10s%10s%10s%10s%11s%11s%9s%9s" % (
            "", "复用尝试", "有信号", "删除任务", "省路(m)", "额外成本s", "补测阶段s",
            "η_remove", "失败清除"))
        for name, _ in arms:
            s = _split_stats(res[name], base, idx)
            eta = (s["removed"]/s["attempts"]) if s["attempts"] else 0.0
            print("%-22s%10.2f%10.2f%10.2f%10.0f%11.1f%11.0f%9s%9.2f" % (
                name, s["attempts"], s["signals"], s["removed"], s["save_a"],
                s["extra"], s["supp_t"], f"{eta*100:.0f}%" if s["attempts"] else "—",
                s["fail"]))
    for name, _ in arms[1:]:
        s = _split_stats(res[name], base)
        s2 = _split_stats(res[name], base, hard)
        print(f"\nη_T {name}: 全案例 {(-s['dm']/s['attempts']) if s['attempts'] else 0:+.1f} s/次 "
              f"; 困难子集 {(-s2['dm']/s2['attempts']) if s2['attempts'] else 0:+.1f} s/次")
    rb.Problem3Robot.SUPP_REUSE = False
    rb.Problem3Robot.OPP_MEASURE = False
    return res


def run_cfg(order_prob, dop, on_way, n=300, seed=2026, ls_gate=None, paired=False):
    """模块 A/B/C/E 消融: 每臂 = 完整冻结配置 + 本臂覆盖项(含显式关闭 O/R)。"""
    cfg = frozen3(ORDER_BY_PROB=order_prob, DOP_PRESCREEN=dop,
                  ON_WAY_DELTA=on_way, LS_CLEAR_GATE=ls_gate)
    crs = []; Ls = []; ms = []; fs = []; Ts = []; per_case = []
    with config_scope((rb.Problem3Robot, cfg)):
        for k in range(n):
            env = case_env(seed, k, directional=False)   # 同编号 = 完全相同场景(配对)
            cli = SimClient(env); robot = rb.Problem3Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                n_ret = robot.run()
            chk = check_clearance(n_ret, cli, env)
            T = sim_time(cli)
            crs.append(chk["src_clear_ratio"]); Ls.append(cli.dist)
            ms.append(cli.n_measure); fs.append(cli.fail)
            Ts.append(T)
            if paired:
                per_case.append((chk["src_clear_ratio"], cli.dist, cli.n_measure,
                                 cli.fail, T, chk["case_full_clear"]))
    out = dict(cr=np.mean(crs), L=np.mean(Ls), n=float(np.mean(ms)), f=float(np.mean(fs)),
               T=float(np.mean(Ts)), T90=float(np.percentile(Ts, 90)),
               cfg_hash=cfg_hash(cfg))
    if paired:
        out["per_case"] = np.array(per_case)
    return out


def paired_ls_gate(n=300, seed=2026):
    """第5项配对实验: 是否收紧最小二乘试探清除的信任门限。

    同一批随机案例下对比"当前策略(LS 点直接盲清除)"与"Ω 半径门限"各档,
    配对差值 = 门限档 − 当前档(<0 表示门限更省)。
    """
    arms = [("当前(LS 直接盲清除)", None), ("门限 Ω<=40m", 40.0), ("门限 Ω<=30m", 30.0),
            ("门限 Ω<=25m", 25.0), ("门限 Ω<=20m(=仅MEC)", 20.0)]
    res = {}
    for name, gate in arms:
        t0 = time.time()
        res[name] = run_cfg(False, False, 300.0, n, seed, ls_gate=gate, paired=True)
        print(f"  已跑 {name}  [{time.time()-t0:.0f}s]", flush=True)
    base = res[arms[0][0]]["per_case"]
    hdr = "%-22s%9s%9s%9s%10s%12s%12s%10s" % (
        "方案", "清除率", "移动(m)", "检测", "失败ms", "Δ时间(s)", "Δ移动(m)", "Δ失败")
    print("\n[配对实验] 最小二乘试探清除门限  n=%d 案例(全清为硬约束)" % n)
    print(hdr); print("-" * len(hdr))
    for name, _ in arms:
        r = res[name]
        if name == arms[0][0]:
            print("%-22s%8.2f%%%9.0f%9.0f%10.2f%12s%12s%10s" % (
                name, r['cr']*100, r['L'], r['n'], r['f'], "—", "—", "—"))
            continue
        d = np.array(r["per_case"]) - base
        print("%-22s%8.2f%%%9.0f%9.0f%10.2f%12.1f%12.1f%10.2f" % (
            name, r['cr']*100, r['L'], r['n'], r['f'], d[:, 4].mean(), d[:, 1].mean(),
            d[:, 3].mean()))
    return res


def order_consistency_test(n=60, seed=4242):
    """审查建议的测试: 同一臂在(独立跑 / 前置其他实验 / 打乱顺序)三种情况下逐案例必须一致。"""
    def one(cfg, tag):
        rows = run_reuse(cfg.get("reuse", False), cfg.get("gate", 1500.0),
                         cfg.get("saving", 0.0), cfg.get("opp", False),
                         n=n, seed=seed, delete_mode=cfg.get("delete", "cert_or_ls"))
        return [(r["scene"], round(r["T"], 6), r["full"], r["consistent"]) for r in rows]
    a = one({"reuse": True}, "独立: 只跑 R")
    one({"reuse": False, "opp": True}, "前置: 先跑 O")
    b = one({"reuse": True}, "顺序: 先 O 再 R")
    one({"reuse": False}, "前置: 先跑基线")
    c = one({"reuse": True}, "顺序: 基线后跑 R")
    same_ab = (a == b); same_ac = (a == c)
    print(f"\n[顺序一致性] 独立跑 vs 前置O后跑: {same_ab}; 独立跑 vs 基线后跑: {same_ac} "
          f"(n={n}, seed={seed})")
    if not (same_ab and same_ac):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  首处不一致 case{i}: {x} vs {y}"); break
    return same_ab and same_ac


def _main():
    if len(sys.argv) > 1 and sys.argv[1] == "--order-test":
        ok = order_consistency_test(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
        sys.exit(0 if ok else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "--reuse":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        d0, s0 = train_scan_reuse(N)
        validate_reuse(d0, s0, max(400, N))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--opp":
        paired_opp(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--ls-gate":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        paired_ls_gate(N)
        sys.exit(0)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfgs = [
        ("基准(全关)",              False, False, None),
        ("+C 滚动(顺路清除)",       False, False, 300.0),
        ("+C+A 概率图排序",         True,  False, 300.0),
        ("+C+B DOP预筛",            False, True,  300.0),
        ("+C+A+B 全开",             True,  True,  300.0),
    ]
    hdr = "%-24s%9s%11s%9s%9s%11s%11s" % ("配置", "清除率", "移动(m)", "检测", "失败", "时间(s)", "P90(s)")
    print(hdr); print("-" * len(hdr))
    for name, a, b, c in cfgs:
        t0 = time.time()
        r = run_cfg(a, b, c, N)
        print("%-24s%8.2f%%%11.0f%9.0f%9.2f%11.0f%11.0f  [%.0fs]" % (
            name, r['cr']*100, r['L'], r['n'], r['f'], r['T'], r['T90'], time.time()-t0), flush=True)


if __name__ == "__main__":
    try:
        _main()
    finally:
        # 无条件复位到冻结配置, 防止异常时类属性残留污染后续实验
        simlib.apply_cfg(rb.Problem3Robot, FROZEN3)
