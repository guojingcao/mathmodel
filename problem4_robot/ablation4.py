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
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0; self.fail = 0; self.meta = {}
    def _move(self, x, y):
        self.dist += math.hypot(x-self.position[0], y-self.position[1]); self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y)
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
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


if __name__ == "__main__":
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
