# -*- coding: utf-8 -*-
"""问题4 自检: (a) 验证三角网格覆盖圆盘 + 证书距离条件; (b) 定向/混合源端到端清除率。"""
import importlib.util, sys, math, numpy as np, io, contextlib

spec = importlib.util.spec_from_file_location("r4", r"D:\My_MathModeling_Project\problem4_robot\robot4.py")
r4 = importlib.util.module_from_spec(spec); sys.modules["r4"] = r4
spec.loader.exec_module(r4)
spec2 = importlib.util.spec_from_file_location("exp", r"D:\My_MathModeling_Project\2026B_solution\verify\experiment.py")
exp = importlib.util.module_from_spec(spec2); sys.modules["exp"] = exp
spec2.loader.exec_module(exp)


def in_tri(p, a, b, c):
    def sgn(x, y, z):
        return (x[0]-z[0])*(y[1]-z[1]) - (y[0]-z[0])*(x[1]-z[1])
    d1, d2, d3 = sgn(p, a, b), sgn(p, b, c), sgn(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def verify_mesh():
    pts = r4.tri_mesh(margin=r4.MESH_MARGIN); tris = r4.build_triangles(pts)
    print(f"网格: {len(pts)} 点, {len(tris)} 三角形")
    # 最大边长
    mx = 0.0
    for t in tris:
        for i in range(3):
            for j in range(i+1, 3):
                mx = max(mx, math.hypot(pts[t[i]][0]-pts[t[j]][0], pts[t[i]][1]-pts[t[j]][1]))
    print(f"最大边长 = {mx:.1f} m (需 <= 1000)")
    # 覆盖性: 圆盘内随机采样, 检查是否落在某三角形内
    rng = np.random.default_rng(7); miss = 0; N = 20000
    for _ in range(N):
        r = r4.R_AREA*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        p = (r*math.cos(a), r*math.sin(a))
        ok = False
        for t in tris:
            if in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]):
                ok = True; break
        if not ok:
            miss += 1
    print(f"覆盖性: 圆盘内 {N} 个采样点, 未被三角形覆盖 {miss} 个 ({miss/N*100:.3f}%)")
    # 认证条件: 三角形内任一点到顶点距离 <= 最大边长(由凸性保证) => 打印最坏
    print(f"=> 任一点到三角形顶点距离 <= 最大边长 = {mx:.1f}m <= 1000m: {'满足' if mx<=1000.5 else '不满足'}")
    return len(pts), len(tris)


def verify_path_index():
    """第一优先级验证: 访问序列必须携带真实网格编号, 每点恰好一次, 锚点不重复计入。"""
    rb = r4.Problem4Robot(None)
    seq = rb._order_points()
    ids = [mi for mi, _, _ in seq]
    perm_ok = sorted(ids) == list(range(len(rb.pts)))
    coord_bad = [(mi, x, y) for mi, x, y in seq
                 if (round(x, 6), round(y, 6)) != rb.pts[mi]]
    dup = [i for i in set(ids) if ids.count(i) > 1]
    print(f"路径索引: {len(seq)} 项 / 网格 {len(rb.pts)} 点; 恰访问一次 = {perm_ok}; "
          f"坐标与编号不符 = {len(coord_bad)}; 重复编号 = {dup or '无'}")
    print(f"  第0项: 网格编号 {seq[0][0]} 坐标 ({seq[0][1]:.1f},{seq[0][2]:.1f})"
          f"  => 锚点与网格点合一, 原点不重复扫描")
    return perm_ok and not coord_bad and not dup


def verify_certificate(n_cases=60, p_dir=0.5, seed=4026):
    """第一优先级验证: 证书不得把"真实存在且可收到"的频道判为不存在。"""
    rng = np.random.default_rng(seed)
    miss = 0; wrong = 0
    for case in range(n_cases):
        env = exp.Env(rng, directional=True, p_dir=p_dir)
        cli = MockClient(env); robot = r4.Problem4Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = robot.run()
        if n < env.n_src:
            miss += 1
            print(f"  漏清: 第{case+1}例 清除 {n}/{env.n_src}")
        real = {int(s["ch"]) for s in env.sources}
        for ch in range(1, r4.N_CH+1):
            if robot.state[ch] == "excluded" and ch in real:
                wrong += 1
                print(f"  证书反例: 第{case+1}例 频道{ch} 真实存在却被排除")
    print(f"证书复核 {n_cases} 例(seed {seed}, 定向比例{p_dir*100:.0f}%): "
          f"漏清 {miss} 例, 误排除 {wrong} 次")
    return miss == 0 and wrong == 0


class MockClient:
    def __init__(self, env):
        self.env = env; self.position = (0.0, 0.0); self.channel = 1
        self.remaining_real = 1200; self.dist = 0.0; self.n_measure = 0; self.n_clear = 0
        self.fail = 0
        self.phase = "init"; self.ph = {}          # 阶段 -> [移动, 检测, 清除]
    def _p(self):
        return self.ph.setdefault(getattr(self, "phase", "init"), [0.0, 0, 0])
    def _move(self, x, y):
        d = math.hypot(x-self.position[0], y-self.position[1])
        self.dist += d; self._p()[0] += d; self.position = (x, y)
    def enter(self): pass
    def measure(self, x, y, ch):
        self._move(x, y); self.channel = ch; self.n_measure += 1; self._p()[1] += 1
        r, svd = self.env.measure(np.array([x, y]), ch); return True, r, svd
    def clear(self, x, y, ch):
        self._move(x, y); self.n_clear += 1; self._p()[2] += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r != 'success': self.fail += 1
        return True, r
    def exit(self): pass


def run_mc(n_cases, p_dir, diag=False):
    rng = np.random.default_rng(3026)
    ratios = []; dists = []; meas = []; fails = []
    ph = {}; lh = []; cd = []
    for _ in range(n_cases):
        env = exp.Env(rng, directional=True, p_dir=p_dir)
        cli = MockClient(env); rb = r4.Problem4Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n = rb.run()
        ratios.append(n/env.n_src); dists.append(cli.dist)
        meas.append(cli.n_measure); fails.append(cli.fail)
        if diag:
            for k, v in cli.ph.items():
                a = ph.setdefault(k, [0.0, 0, 0])
                a[0] += v[0]; a[1] += v[1]; a[2] += v[2]
            lh.extend(getattr(cli, "locate_history", []))
            cd.extend(getattr(cli, "clear_diag", []))
    out = dict(cr=np.mean(ratios), L=np.mean(dists), n=np.mean(meas), f=np.mean(fails))
    if diag:
        out.update(ph=ph, locate=lh, clear=cd, n_cases=n_cases)
    return out


def show_diag(tag, r):
    """打印分阶段归因 + 定位方式/Ω半径 + 清除来源成功率。"""
    n = r["n_cases"]
    print(f"\n[诊断{tag}] 分阶段归因(每例均值):")
    print("  %-16s%11s%9s%9s%8s" % ("阶段", "移动(m)", "检测", "清除", "时间(s)"))
    for k in sorted(r["ph"], key=lambda x: -r["ph"][x][0]):
        mv, nm, nc = r["ph"][k]
        print("  %-16s%11.0f%9.1f%9.1f%8.0f" % (
            k, mv/n, nm/n, nc/n, (mv/5 + nm*5 + nc*3)/n))
    def stat(v):
        if not v:
            return "无"
        v = np.array(v, float)
        return f"n={len(v)} 中位={np.median(v):.1f} 均值={v.mean():.1f} P90={np.percentile(v,90):.1f}"
    lh = r["locate"]
    print(f"[诊断{tag}] 定位调用 {len(lh)} 次: "
          f"MEC={sum(1 for h in lh if h.get('method')=='mec')} "
          f"LS={sum(1 for h in lh if h.get('method')=='ls')} "
          f"不可定位={sum(1 for h in lh if h.get('method') is None)}")
    print("  Ω半径(m) MEC 档: " + stat([h["omega_radius_m"] for h in lh
                                        if h.get("method") == "mec"
                                        and h.get("omega_radius_m") is not None]))
    print("  Ω半径(m) LS  档: " + stat([h["omega_radius_m"] for h in lh
                                        if h.get("method") == "ls"
                                        and h.get("omega_radius_m") is not None]))
    print(f"[诊断{tag}] 清除尝试 {len(r['clear'])} 次, 按来源:")
    g = {}
    for d in r["clear"]:
        key = str(d.get("src", "?")).replace("|", "/")
        a = g.setdefault(key, [0, 0, []])
        a[0] += 1; a[1] += 1 if d.get("result") == "success" else 0
        if d.get("omega_radius_m") is not None:
            a[2].append(d["omega_radius_m"])
    for k in sorted(g, key=lambda x: -g[x][0]):
        a = g[k]
        om = f" Ω半径中位={np.median(a[2]):.1f}m" if a[2] else ""
        print("  %-26s n=%5d 成功=%5d (%5.1f%%)%s" % (
            k, a[0], a[1], 100.0*a[1]/a[0], om))


if __name__ == "__main__":
    import sys as _sys
    args = [a for a in _sys.argv[1:] if not a.startswith("--")]
    if "--cert" in _sys.argv:
        # 只做"索引映射 + 证书正确性"复核(第一优先级回归)
        ok1 = verify_path_index()
        ok2 = verify_certificate(int(args[0]) if args else 60)
        print(f"结论: 索引映射 {'通过' if ok1 else '不通过'}, "
              f"证书 {'通过' if ok2 else '不通过'}")
        _sys.exit(0 if (ok1 and ok2) else 1)
    n = int(args[0]) if args else 30
    margin = float(args[1]) if len(args) > 1 else None
    diag = "--diag" in _sys.argv
    if margin is not None:
        r4.MESH_MARGIN = margin
    print(f"(MESH_MARGIN={r4.MESH_MARGIN})")
    verify_mesh()
    verify_path_index()
    print()
    for pd in [0.0, 0.5, 1.0]:
        r = run_mc(n, pd, diag=diag)
        print(f"定向比例{pd*100:>3.0f}%: 清除率={r['cr']*100:6.2f}%  移动={r['L']:6.0f}m  "
              f"检测={r['n']:5.0f}  清除失败={r['f']:.2f}  估计总时间={r['L']/5+r['n']*5:6.0f}s")
        if diag:
            show_diag(f" {pd*100:.0f}%", r)
