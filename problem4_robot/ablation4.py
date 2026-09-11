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
        self.n_switch = 0; self.n_clear_ok = 0; self.meta = {}
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
        self.channel = ch
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
        rng = np.random.default_rng(seed)
        crs = []; Ls = []; ms = []; fs = []; Ts = []
        for _ in range(n):
            env = exp.Env(rng, directional=True, p_dir=pd)
            cli = MockClient(env); rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                k = rb.run()
            crs.append(k/env.n_src); Ls.append(cli.dist); ms.append(cli.n_measure); fs.append(cli.fail)
            Ts.append(cli.dist/5 + cli.n_measure*5)
        out[pd] = (np.mean(crs), np.mean(Ls), np.mean(ms), float(np.mean(fs)), np.mean(Ts))
    return out


def run_onway(delta, n=30, seed=3026, ratios=(0.5, 1.0)):
    """同一批随机案例(同 seed)下跑一个顺路清除阈值, 返回逐案例明细用于配对比较。

    时间模型(与题设一致): 移动/5 + 检测*5 + 换频*1 + 成功清除*5 + 失败清除*3。
    """
    r4.ON_WAY_DELTA = delta            # 模块级常量, run() 在调用时读取
    out = {}
    for pd in ratios:
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            env = exp.Env(rng, directional=True, p_dir=pd)
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
