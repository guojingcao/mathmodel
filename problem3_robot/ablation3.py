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


def run_cfg(order_prob, dop, on_way, n=300, seed=2026):
    rb.Problem3Robot.ORDER_BY_PROB = order_prob
    rb.Problem3Robot.DOP_PRESCREEN = dop
    rb.Problem3Robot.ON_WAY_DELTA = on_way
    rng = np.random.default_rng(seed)
    crs = []; Ls = []; ms = []; fs = []; Ts = []
    for _ in range(n):
        env = exp.Env(rng, directional=False)
        cli = MockClient(env); robot = rb.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            k = robot.run()
        crs.append(k/env.n_src); Ls.append(cli.dist); ms.append(cli.n_measure); fs.append(cli.fail)
        Ts.append(cli.dist/5 + cli.n_measure*5)
    return dict(cr=np.mean(crs), L=np.mean(Ls), n=float(np.mean(ms)), f=float(np.mean(fs)),
                T=float(np.mean(Ts)), T90=float(np.percentile(Ts, 90)))


if __name__ == "__main__":
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
