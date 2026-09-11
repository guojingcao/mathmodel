"""问题四病理场景"残留源"根因定位。

用法:
    python patho_diag.py                       # 诊断 edge_out(边界外指) 类, 200 例
    python patho_diag.py --kind edge_out 200
    python patho_diag.py --kind center_away 200

做法:
  用 selfcheck4.patho_stream 以**与 run_patho 完全相同的 rng 顺序**重放全部场景,
  对非目标类只调用 make_env(保持随机流一致)、不跑机器人; 对目标类跑机器人并记录:
    * 机器人自身日志(含证书判决/排除/未发现/恢复过程)
    * 每次 measure/clear 的坐标、结果, 以及到真实源的距离
    * 未清除频道最终状态(state)、示向数、网格顶点覆盖与"顶点→源"几何关系
    * 若该源所在覆盖三角形的三顶点都被测过且都返回 no_signal → 证书被错误签发(检出缺陷)
  从而区分两类根因: ①检出失败(证书错杀) ②清除失败(已发现但清不掉)。
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                      # noqa: E402

import selfcheck4 as sc4                # noqa: E402

r4 = sc4.r4


class RecClient(sc4.MockClient):
    """记录全部 measure/clear 明细的 MockClient。"""

    def __init__(self, env):
        super().__init__(env)
        self.meas = []
        self.clr = []

    def measure(self, x, y, ch):
        ok, r, svd = super().measure(x, y, ch)
        self.meas.append((float(x), float(y), int(ch), r))
        return ok, r, svd

    def clear(self, x, y, ch):
        ok, r = super().clear(x, y, ch)
        self.clr.append((float(x), float(y), int(ch), r))
        return ok, r


def in_tri(p, a, b, c):
    def cross(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
    d1 = cross(a, b, p); d2 = cross(b, c, p); d3 = cross(c, a, p)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def ang_in_cone(pos, point, vertex, half=90.0):
    """vertex 是否位于 pos 处、指向 point 的 ±half 扇区内。"""
    if point is None:
        return True
    a = math.degrees(math.atan2(vertex[1]-pos[1], vertex[0]-pos[0])) % 360.0
    b = math.degrees(point) % 360.0
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d <= half


def diag_case(rb, cli, env, buf, label, rep):
    n_src = env.n_src
    miss = []
    for ch in range(1, r4.N_CH+1):
        if any(s["ch"] == ch for s in env.sources) and rb.state.get(ch) != "cleared":
            miss.append(ch)
    print("=" * 100)
    print(f"[漏清案例] 场景={label} 重复#{rep+1}  清除 {rb.cleared_count}/{n_src}  "
          f"exit_status={getattr(rb, 'exit_status', '?')}  "
          f"未解决类型={getattr(rb, 'unresolved_kind', {})}")
    print(f"  检测 {cli.n_measure} 次, 清除尝试 {cli.n_clear} 次(成功 {cli.n_clear_ok}, "
          f"失败 {cli.fail}), 移动 {cli.dist:.0f} m, "
          f"T={cli.dist/5 + cli.n_measure*5 + cli.n_switch + cli.n_clear_ok*5 + cli.fail*3:.0f} s")
    pts = np.asarray(rb.pts, float)
    for ch in miss:
        src = next(s for s in env.sources if s["ch"] == ch)
        pos = np.asarray(src["pos"], float)
        pt = src.get("pointing")
        rx = float(src.get("r_rx", 0.0))
        st = rb.state.get(ch)
        nbr = len(getattr(rb, "bearings", {}).get(ch, []) or [])
        dmin = float(np.min(np.hypot(pts[:, 0]-pos[0], pts[:, 1]-pos[1])))
        near = np.hypot(pts[:, 0]-pos[0], pts[:, 1]-pos[1])
        inrx = near <= rx
        incone = np.array([ang_in_cone(pos, pt, q) for q in pts])
        usable = int(np.sum(inrx & incone))
        print(f"\n  ▸ 频道 {ch}: 状态={st}, 示向数={nbr}, 源位置=({pos[0]:.1f},{pos[1]:.1f}), "
              f"r=|{np.hypot(*pos):.1f}| m")
        print(f"    指向={'None(全向)' if pt is None else f'{math.degrees(pt)%360:.1f}°'}, "
              f"接收半径={rx:.0f} m")
        print(f"    网格顶点: 最近顶点距源 {dmin:.1f} m; 落在接收圈内 {int(inrx.sum())} 个, "
              f"其中同时落在指向扇区内(**可测到**) {usable} 个")
        # 实际测过的顶点
        ms = [(x, y, r) for (x, y, c, r) in cli.meas if c == ch]
        hit = [(x, y, r) for (x, y, r) in ms if r != "no_signal"]
        print(f"    对该频道检测 {len(ms)} 次, 其中非 no_signal {len(hit)} 次"
              + (f" → {hit[:4]}" if hit else ""))
        if ms:
            dm = [math.hypot(x-pos[0], y-pos[1]) for x, y, _ in ms]
            print(f"    检测点距源: 最小 {min(dm):.0f} m, 最大 {max(dm):.0f} m")
        # 证书: 源所在覆盖三角形
        tri_hit = []
        for t in rb.cover_tris:
            a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
            if in_tri(pos, a, b, c):
                tri_hit.append(t)
        for t in tri_hit:
            res = []
            for vi in t:
                vx, vy = pts[vi]
                got = [r for (x, y, cc, r) in cli.meas
                       if cc == ch and abs(x-vx) < 1e-6 and abs(y-vy) < 1e-6]
                res.append(got[-1] if got else "未测")
            allns = all(r == "no_signal" for r in res)
            print(f"    所在覆盖三角形 {tuple(int(i) for i in t)} 顶点结果={res}"
                  + ("   <<< 三顶点全 no_signal: **证书错误签发(漏检)**"
                     if allns else ""))
        # 清除尝试
        cs = [(x, y, r) for (x, y, c, r) in cli.clr if c == ch]
        for x, y, r in cs:
            d = math.hypot(x-pos[0], y-pos[1])
            print(f"    清除尝试 @({x:.1f},{y:.1f}) 距源 {d:.1f} m → {r}"
                  + ("  (在 20 m 内却失败?)" if d <= 20 and r != "success" else ""))
        if not cs:
            print("    清除尝试: **0 次**(从未发起)")
    # 机器人日志中与该频道相关的行
    lines = buf.getvalue().splitlines()
    key = [ln for ln in lines if any(f"频道 {ch}" in ln or f"频道{ch}" in ln
                                     or f"[{ch}]" in ln for ch in miss)]
    if key:
        print("\n  机器人与该频道相关的日志:")
        for ln in key[-25:]:
            print("    | " + ln.strip()[:160])
    print("=" * 100)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = "edge_out"
    if "--kind" in sys.argv:
        kind = sys.argv[sys.argv.index("--kind")+1]
    n_rep = int(args[-1]) if args and args[-1].isdigit() else 200
    seed = 99
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed")+1])
    print(f"目标场景={kind}, 每类 {n_rep} 例, seed={seed}")
    print("注意: 必须**跑完全部场景**——env.measure 会从共享 rng 取数(±1° 噪声/清除随机),"
          "跳过任何一例都会使随机流与 run_patho 分叉。非目标类只跑不分析。")
    n_run = n_miss = 0
    for label, k, n, specs, rng, rep in sc4.patho_stream(seed, n_rep):
        env = sc4.make_env(rng, specs)         # 每例都必须调用(保持随机流一致)
        cli = RecClient(env)
        rb = r4.Problem4Robot(cli)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = rb.run()
        if k != kind:
            continue
        n_run += 1
        if got < env.n_src:
            n_miss += 1
            diag_case(rb, cli, env, buf, label, rep)
    print(f"\n目标场景共跑 {n_run} 例, 其中残留源 {n_miss} 例")


if __name__ == "__main__":
    main()
