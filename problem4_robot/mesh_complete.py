"""补齐问题四网格的"覆盖空洞"(证书可靠性前提)并验证代价。

背景: 证书推理 "所有三角形被证伪 => 频道不存在" 成立的前提是
      **圆盘内每一点都落在某个三角形内**。当前 27 点网格在 R≈1800 内侧
      留有薄空洞(~0.03% 面积, 集中在 1780-1800 环), 落在空洞内且朝外辐射的
      源既测不到(无网格点在其 ±90° 扇区内)又会被错误"排除", 从而必然漏清。

用法:
    python mesh_complete.py --find            # 贪心求最小补点数与位置
    python mesh_complete.py --verify <n_rep>  # 用补点后的网格跑病理集, 报漏清与耗时变化
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                      # noqa: E402

import robot4 as r4                     # noqa: E402
import selfcheck4 as sc4                # noqa: E402

R = r4.R_AREA
EXTRA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "mesh_extra_points.txt")


def in_tri(p, a, b, c):
    def cr(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
    d1 = cr(a, b, p); d2 = cr(b, c, p); d3 = cr(c, a, p)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def sample_disk(n_uni=60000, n_edge=60000, seed=5):
    """均匀 + 边界环加权采样: 边界环是空洞所在, 必须加密。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_uni):
        r = R*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    for _ in range(n_edge):
        r = rng.uniform(1600, R); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    return out


def sample_disk_dense(n_uni=60000, n_mid=60000, n_ext=80000, seed=17):
    """更严的采样: 均匀 + [1600,1800] + 极致边界 [1780,1800](残洞所在)。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_uni):
        r = R*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    for _ in range(n_mid):
        r = rng.uniform(1600, R); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    for _ in range(n_ext):
        r = rng.uniform(1780, R); a = rng.uniform(0, 2*math.pi)
        out.append((r*math.cos(a), r*math.sin(a)))
    return out


def find_more(max_add=8, verbose=True):
    """从**当前配置网格**(含已补齐点)出发, 把残余空洞补到 0。

    候选 = 残缺点本身 + 其径向/角向微调(残缺点常紧贴既有顶点, 原样加入会被去重过滤)。
    评分 = 真实重建三角剖分后剩余未覆盖数(不是距离代理), 避免重复点/退化三角形。
    """
    pts = [tuple(p) for p in r4.Problem4Robot(None).pts]
    full = sample_disk_dense()
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    bad = uncovered(pts, tris, full)
    if verbose:
        print(f"起点: {len(pts)} 点 / {len(tris)} 三角形; 密集采样 {len(full)} 点中未覆盖 "
              f"{len(bad)} ({100.0*len(bad)/len(full):.5f}%)")
    added = []
    while bad and len(added) < max_add:
        cands = []
        for q in bad[::max(1, len(bad)//40)]:
            rr = math.hypot(*q)
            th = math.atan2(q[1], q[0])
            if rr < 1e-9:
                continue
            for dr in (0.0, 2.0, 6.0, 15.0, -6.0):
                for dth in (0.0, 0.0008, -0.0008, 0.0025, -0.0025):
                    r2, t2 = rr + dr, th + dth
                    if r2 <= 0:
                        continue
                    c = (round(r2*math.cos(t2), 1), round(r2*math.sin(t2), 1))
                    if min(math.hypot(c[0]-p[0], c[1]-p[1]) for p in pts) > 0.5:
                        cands.append(c)
        best, best_bad = None, len(bad)
        for cand in cands:
            pts2 = pts + [cand]
            tris2 = r4.build_triangles(pts2, a=r4.MESH_A)
            nbad = 0
            for q in bad:
                if not any(in_tri(q, pts2[t[0]], pts2[t[1]], pts2[t[2]]) for t in tris2):
                    nbad += 1
            if nbad < best_bad:
                best_bad, best = nbad, cand
        if best is None:
            if verbose:
                print(f"  无候选可继续降低(剩余 {len(bad)}), 停止")
            break
        added.append(best)
        pts = pts + [best]
        tris = r4.build_triangles(pts, a=r4.MESH_A)
        bad = uncovered(pts, tris, full)
        if verbose:
            print(f"  补点 {len(added)}: ({best[0]:8.1f},{best[1]:8.1f}) r={math.hypot(*best):7.1f} "
                  f"-> 剩余未覆盖 {len(bad)} ({100.0*len(bad)/len(full):.5f}%)")
    # 校验
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    mx = max(math.hypot(pts[t[i]][0]-pts[t[j]][0], pts[t[i]][1]-pts[t[j]][1])
             for t in tris for i in range(3) for j in range(i+1, 3))
    mind = min(math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
               for i in range(len(pts)) for j in range(i+1, len(pts)))
    if verbose:
        print(f"结果: {len(pts)} 点 / {len(tris)} 三角形; 最大边 {mx:.0f} m; "
              f"最近点距 {mind:.2f} m; 未覆盖 {len(bad)}")
    with open(EXTRA_FILE + ".all", "w", encoding="utf-8") as f:
        base_n = len(r4.tri_mesh(r4.MESH_A, r4.MESH_MARGIN, r4.MESH_THETA, r4.MESH_OFFSET))
        for p in pts[base_n:]:
            f.write(f"{p[0]},{p[1]}\n")
    print(f"全部补齐点已写入 {EXTRA_FILE}.all (共 {len(pts)-base_n} 个)")
    return pts, added


def uncovered(pts, tris, samples):
    bad = []
    for p in samples:
        ok = False
        for t in tris:
            if in_tri(p, pts[t[0]], pts[t[1]], pts[t[2]]):
                ok = True; break
        if not ok:
            bad.append(p)
    return bad


def find_extra(max_add=12):
    """贪心补点: 每个候选点都**真实重建三角剖分**后计分, 避免重复点/退化三角形。"""
    base = r4.tri_mesh(r4.MESH_A, r4.MESH_MARGIN, r4.MESH_THETA, r4.MESH_OFFSET)
    full = sample_disk()
    pts = list(base)
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    bad = uncovered(pts, tris, full)
    print(f"初始: {len(pts)} 点 / {len(tris)} 三角形; 采样 {len(full)} 点中未覆盖 "
          f"{len(bad)} ({100.0*len(bad)/len(full):.4f}%)")
    extra = []
    while bad and len(extra) < max_add:
        cands = [c for c in bad[::max(1, len(bad)//60)]
                 if min(math.hypot(c[0]-p[0], c[1]-p[1]) for p in pts) > 1.0]
        best = None
        best_bad = len(bad)
        for cand in cands:
            pts2 = pts + [cand]
            tris2 = r4.build_triangles(pts2, a=r4.MESH_A)
            nbad = 0
            for q in bad:                      # 只需重测当前未覆盖的点
                if not any(in_tri(q, pts2[t[0]], pts2[t[1]], pts2[t[2]]) for t in tris2):
                    nbad += 1
            if nbad < best_bad:
                best_bad, best = nbad, cand
        if best is None:
            print("  无候选能继续降低未覆盖数, 停止")
            break
        extra.append((round(best[0], 1), round(best[1], 1)))
        pts = list(base) + list(extra)
        tris = r4.build_triangles(pts, a=r4.MESH_A)
        bad = uncovered(pts, tris, full)
        print(f"  补点 {len(extra)}: ({best[0]:7.1f},{best[1]:7.1f}) "
              f"r={math.hypot(*best):7.1f} -> 剩余未覆盖 {len(bad)} "
              f"({100.0*len(bad)/len(full):.4f}%)")
    pts = list(base) + list(extra)
    tris = r4.build_triangles(pts, a=r4.MESH_A)
    mx = 0.0
    for t in tris:
        for i in range(3):
            for j in range(i+1, 3):
                mx = max(mx, math.hypot(pts[t[i]][0]-pts[t[j]][0],
                                        pts[t[i]][1]-pts[t[j]][1]))
    mind = min(math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
               for i in range(len(pts)) for j in range(i+1, len(pts)))
    print(f"结果: 补 {len(extra)} 点 -> {len(pts)} 点 / {len(tris)} 三角形; "
          f"最大边 {mx:.0f} m (需<=1000); 最近点距 {mind:.1f} m (需>0, 无重复点)")
    with open(EXTRA_FILE, "w", encoding="utf-8") as f:
        for x, y in extra:
            f.write(f"{x},{y}\n")
    print(f"补点已写入 {EXTRA_FILE}")
    return extra


def load_extra():
    if not os.path.exists(EXTRA_FILE):
        return []
    out = []
    for ln in open(EXTRA_FILE, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            x, y = ln.split(",")
            out.append((float(x), float(y)))
    return out


def patch_mesh(extra):
    orig = r4.tri_mesh

    def patched(*a, **k):
        return list(orig(*a, **k)) + list(extra)

    r4.tri_mesh = patched


def verify(n_rep=200):
    extra = load_extra()
    print(f"补点 {len(extra)} 个: {extra}")
    for tag, use_extra in (("原生网格", False), ("补点网格", True)):
        if use_extra:
            patch_mesh(extra)
        else:
            r4.tri_mesh = _ORIG
        rb = r4.Problem4Robot(None)
        samples = sample_disk()
        bad = uncovered(rb.pts, rb.tris, samples)
        print(f"\n[{tag}] {len(rb.pts)} 点 / {len(rb.tris)} 三角形; "
              f"采样未覆盖 {len(bad)} ({100.0*len(bad)/len(samples):.4f}%)")
        tot_miss = 0
        per = {}
        for label, kind, n, specs, rng, rep in sc4.patho_stream(99, n_rep):
            env = sc4.make_env(rng, specs)
            cli = sc4.MockClient(env)
            rb = r4.Problem4Robot(cli)
            with contextlib.redirect_stdout(io.StringIO()):
                got = rb.run()
            T = cli.dist/5 + cli.n_measure*5 + cli.n_switch + cli.n_clear_ok*5 + cli.fail*3
            a = per.setdefault(label, [0, 0, 0.0])
            a[0] += 1
            a[1] += 1 if got < env.n_src else 0
            a[2] += T
            if got < env.n_src:
                tot_miss += 1
        print(f"[{tag}] 总漏清 {tot_miss} 例")
        for label, kind, n in sc4.PATHO_SCEN:
            a = per[label]
            print(f"    {label:<24} n={a[0]:3d} 漏清={a[1]} 平均T={a[2]/a[0]:7.0f}s")


_ORIG = r4.tri_mesh

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--find" in sys.argv:
        find_extra()
    elif "--verify" in sys.argv:
        verify(int(args[0]) if args else 200)
    else:
        print(__doc__)
