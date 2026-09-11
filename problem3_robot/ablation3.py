# -*- coding: utf-8 -*-
"""问题3 外部改进模块消融实验

模块:
  A 贝叶斯概率图排序   ORDER_BY_PROB
  B DOP(交会角)预筛   DOP_PRESCREEN
  C 滚动优化(受限顺路清除)  ON_WAY_DELTA  (0/None = 关闭)
  D 自适应网格 + Monte Carlo 验证  -> 本脚本本身就是 D 的一部分(1000 案例统计)

指标: 清除率 / 平均移动距离 / 平均检测次数 / 估计总虚拟时间 / 90分位时间
"""
import importlib.util, sys, math, numpy as np, io, contextlib, time

spec = importlib.util.spec_from_file_location("rb", r"D:\My_MathModeling_Project\problem3_robot\robot.py")
rb = importlib.util.module_from_spec(spec); sys.modules["rb"] = rb
spec.loader.exec_module(rb)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0
        self.fail = 0; self.meta = {}
        self.n_clear = 0; self.n_clear_ok = 0; self.n_switch = 0
        self.phase = "init"; self.ph = {}      # 阶段 -> [移动, 检测, 清除]
    def _p(self):
        return self.ph.setdefault(getattr(self, "phase", "init"), [0.0, 0, 0])
    def _move(self, x, y):
        d = math.hypot(x-self.position[0], y-self.position[1])
        self.dist += d; self._p()[0] += d; self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel: self.n_switch += 1
        self.channel = ch; self.n_measure += 1; self._p()[1] += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel: self.n_switch += 1
        self.channel = ch; self.n_clear += 1; self._p()[2] += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
        else: self.n_clear_ok += 1
        return True, r
    def exit(self): pass


def run_opp(opp_on, max_per_point=2, target="spec", n=400, seed=2026, paired=False):
    """模块O(机会性顺带观测)消融: 同一批随机案例(同 seed)下跑一种配置。

    时间模型(与题设一致): 移动/5 + 检测*5 + 换频*1 + 成功清除*5 + 失败清除*3。
    同时记录补测阶段时间(用于在"关闭版"结果上预先定义困难子集, 避免选择偏差)。
    """
    rb.Problem3Robot.OPP_MEASURE = opp_on
    rb.Problem3Robot.OPP_MAX_PER_POINT = max_per_point
    rb.Problem3Robot.OPP_TARGET = target
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        env = exp.Env(rng, directional=False)
        cli = MockClient(env); robot = rb.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            k = robot.run()
        T = (cli.dist/5 + cli.n_measure*5 + cli.n_switch*1
             + cli.n_clear_ok*5 + cli.fail*3)
        opp = list(getattr(cli, "opp_diag", []))
        ph = getattr(cli, "ph", {})
        sup_t = ph.get("supplement", [0.0, 0, 0])
        rows.append(dict(
            cr=k/env.n_src, miss=1 if k < env.n_src else 0, n_src=env.n_src,
            T=T, dist=cli.dist, meas=cli.n_measure, sw=cli.n_switch,
            clear=cli.n_clear, fail=cli.fail,
            supp_t=sup_t[0]/5 + sup_t[1]*5,                 # 补测阶段时间(选困难子集用)
            homing_t=(ph.get("queue_homing", [0.0, 0, 0])[0]/5
                      + ph.get("queue_homing", [0.0, 0, 0])[1]*5),
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


def run_cfg(order_prob, dop, on_way, n=300, seed=2026, ls_gate=None, paired=False):
    rb.Problem3Robot.ORDER_BY_PROB = order_prob
    rb.Problem3Robot.DOP_PRESCREEN = dop
    rb.Problem3Robot.ON_WAY_DELTA = on_way
    rb.Problem3Robot.LS_CLEAR_GATE = ls_gate
    rng = np.random.default_rng(seed)
    crs = []; Ls = []; ms = []; fs = []; Ts = []; per_case = []
    for _ in range(n):
        env = exp.Env(rng, directional=False)      # 同一 seed -> 各配置面对完全相同的案例(配对)
        cli = MockClient(env); robot = rb.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            k = robot.run()
        T = cli.dist/5 + cli.n_measure*5
        crs.append(k/env.n_src); Ls.append(cli.dist); ms.append(cli.n_measure); fs.append(cli.fail)
        Ts.append(T)
        if paired:
            per_case.append((k/env.n_src, cli.dist, cli.n_measure, cli.fail, T))
    out = dict(cr=np.mean(crs), L=np.mean(Ls), n=float(np.mean(ms)), f=float(np.mean(fs)),
               T=float(np.mean(Ts)), T90=float(np.percentile(Ts, 90)))
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


if __name__ == "__main__":
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
