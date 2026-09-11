# -*- coding: utf-8 -*-
"""问题4 外部改进模块消融实验

模块:
  1+4 负信息 -> 联合可行域/指向状态   USE_NEG_INFO
  3   改进 PSO 精化位置               USE_PSO
  5   清除后多方向复核                 DO_VERIFY
"""
import importlib.util, sys, math, numpy as np, io, contextlib, time

spec = importlib.util.spec_from_file_location("r4", r"D:\My_MathModeling_Project\problem4_robot\robot4.py")
r4 = importlib.util.module_from_spec(spec); sys.modules["r4"] = r4; spec.loader.exec_module(r4)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp; spec2.loader.exec_module(exp)


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0; self.fail = 0
        self.n_switch = 0; self.n_clear_ok = 0; self.n_clear = 0; self.meta = {}
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1]); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel: self.n_switch += 1
        self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel: self.n_switch += 1
        self.channel = ch; self.n_clear += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
        else: self.n_clear_ok += 1
        return True, r
    def exit(self): pass


def run_cfg(neg, pso, verify, n=8, ratios=(0.5, 1.0), seed=3026):
    r4.Problem4Robot.USE_NEG_INFO = neg
    r4.Problem4Robot.USE_PSO = pso
    r4.Problem4Robot.DO_VERIFY = verify
    out = {}
    for pd in ratios:
        crs = []; Ls = []; ms = []; fs = []; Ts = []
        for ci in range(n):
            env = case_env(seed, ci, directional=True, p_dir=pd)
            cli = MockClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                k = rb.run()
            crs.append(k/env.n_src); Ls.append(cli.dist); ms.append(cli.n_measure); fs.append(cli.fail)
            Ts.append(cli.dist/5 + cli.n_measure*5)
        out[pd] = (np.mean(crs), np.mean(Ls), np.mean(ms), float(np.mean(fs)), np.mean(Ts))
    return out


def case_env(seed, k, **kw):
    """按(seed, 案例编号)独立派生场景随机源。

    exp.Env 用同一个 rng 既生成场景、又在每次 measure 抽 ±1° 噪声; 若各臂共用一个 rng,
    臂间测量次数不同就会错开随机流 -> 同一编号在不同臂下是不同场景, 配对失效。
    """
    return exp.Env(np.random.default_rng([int(seed), int(k)]), **kw)


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
            cli = MockClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                k = rb.run()
            T = (cli.dist/5 + cli.n_measure*5 + cli.n_switch*1
                 + cli.n_clear_ok*5 + cli.fail*3)
            rows.append(dict(
                cr=k/env.n_src, miss=1 if k < env.n_src else 0, n_src=env.n_src,
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
            cli = MockClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                k = rb.run()
            T = (cli.dist/5 + cli.n_measure*5 + cli.n_switch*1
                 + cli.n_clear_ok*5 + cli.fail*3)
            sup = list(getattr(cli, "supp_diag", []))
            hom = list(getattr(cli, "homing_diag", []))
            sk = set(rb.supp_skipped)
            rows.append(dict(
                cr=k/env.n_src, miss=1 if k < env.n_src else 0, n_src=env.n_src,
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
            cli = MockClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                k = rb.run()
            T = (cli.dist/5 + cli.n_measure*5 + cli.n_switch*1
                 + cli.n_clear_ok*5 + cli.fail*3)
            homing = sum(1 for d in getattr(cli, "clear_diag", [])
                         if "homing" in str(d.get("phase")) or "归航" in str(d.get("src")))
            rows.append((k/env.n_src, cli.dist, cli.n_measure, cli.fail, T, homing))
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--supp-cap":
        paired_supp_cap(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--neighbor":
        paired_neighbor(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--onway-paired":
        paired_onway(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
        sys.exit(0)
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
        s = "%-20s" % name
        for pd in (0.5, 1.0):
            cr, L, m, f, T = r[pd]
            s += "%6.1f%%/%5.0f/%5.0fs" % (cr*100, L, T)
        print(s + "  [%.0fs]" % (time.time()-t0), flush=True)
