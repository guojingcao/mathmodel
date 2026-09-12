# -*- coding: utf-8 -*-
"""候选站集(非规则 27 点)的离线门禁: 连续证书 + 覆盖审计 + 方向判据 + 病理集 + 证书回归。

部署口径: MESH_PTS_OVERRIDE = 候选坐标; MESH_A = 970(使 build_triangles 的 1.03 容差
          = 999.1 m 覆盖候选的最大边 959.4 m)。
注意: selfcheck4 用 importlib 另加载了一份 robot4, 必须**同时**在其模块对象上设属性,
      否则门禁会在旧设计上跑(本会话已踩过该坑)。

用法: python gates_candidate.py [patho_rep=200]
"""
import importlib.util
import io
import contextlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robot4 as r4                                 # noqa: E402
import mesh_complete as mc                          # noqa: E402

BEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "results", "station_opt_best.txt")
CAND_A = 970.0


def load_best():
    pts = []
    for ln in open(BEST, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            x, y = ln.split(",")
            pts.append((float(x), float(y)))
    return pts


def deploy(pts):
    """把候选口径同时写到两个 robot4 模块对象上。"""
    r4.MESH_A = CAND_A
    r4.Problem4Robot.MESH_PTS_OVERRIDE = list(pts)
    r4.Problem4Robot.MESH_EXTRA_PTS = []
    spec = importlib.util.spec_from_file_location(
        "r4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot4.py"))
    # selfcheck4 内部自行加载; 这里用同一路径再加载一次并显式覆盖其属性
    import selfcheck4 as sc4
    sc4.r4.MESH_A = CAND_A
    sc4.r4.Problem4Robot.MESH_PTS_OVERRIDE = list(pts)
    sc4.r4.Problem4Robot.MESH_EXTRA_PTS = []
    return sc4


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    pts = load_best()
    print(f"候选站集: {len(pts)} 点 (MESH_A={CAND_A})")
    sc4 = deploy(pts)
    rb = r4.Problem4Robot(None)
    print(f"部署核验(robot4 侧): {len(rb.pts)} 点 / {len(rb.tris)} 三角 / "
          f"需证伪 {len(rb.cover_tris)}; MESH_A={r4.MESH_A}")
    rb2 = sc4.r4.Problem4Robot(None)
    print(f"部署核验(selfcheck 侧): {len(rb2.pts)} 点 / {len(rb2.tris)} 三角; "
          f"MESH_A={sc4.r4.MESH_A}")
    results = {}
    # ---- ① 连续证书(与优化器内同判据, 独立复算) ----
    import hull_proof as hp
    hp.R = r4.R_AREA
    print("\n[① 连续证书] 见 hull_proof 输出:")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = hp.main()
    out = buf.getvalue()
    for ln in out.splitlines():
        if any(k in ln for k in ("(1)", "(1a)", "(1b)", "(3)", "(4)", "总判定")):
            print("   " + ln.strip())
    results["连续证书"] = (rc == 0)
    # ---- ② 覆盖审计(采样回归) + ③ 方向判据 ----
    print("\n[② 覆盖审计 / ③ 方向判据]")
    smp = mc.sample_disk(20000, 20000, seed=13)
    bad = len(mc.uncovered(rb.pts, rb.tris, smp))     # 返回的是未覆盖点列表
    print(f"   密集采样 {len(smp)} 点: 未覆盖 {bad} ({100.0*bad/len(smp):.5f}%)")
    results["覆盖审计"] = (bad == 0)
    rb3 = r4.Problem4Robot(None)
    rng = __import__("numpy").random.default_rng(5)
    R = r4.R_AREA
    ok = 0; N = 20000
    tq = 0
    for _ in range(N):
        r = R*math.sqrt(rng.uniform()); a = rng.uniform(0, 2*math.pi)
        S = (r*math.cos(a), r*math.sin(a))
        Q = [p for p in rb3.pts if math.hypot(p[0]-S[0], p[1]-S[1]) <= 1000.0]
        if len(Q) >= 3:
            hit = any(mc.in_tri(S, Q[i], Q[j], Q[k])
                      for i in range(len(Q)) for j in range(i+1, len(Q))
                      for k in range(j+1, len(Q)))
            ok += 1 if hit else 0
        else:
            tq += 1
    print(f"   S∈conv Q(S) 抽查: {ok}/{N} = {100.0*ok/N:.4f}%; |Q(S)|<3 的点 {tq}")
    results["方向判据"] = (ok == N and tq == 0)
    # ---- ④ 病理集 2000 例 ----
    print(f"\n[④ 病理集 10 类 ×{rep}]")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad_cnt = sc4.run_patho(rep)
    for ln in buf.getvalue().splitlines()[-3:]:
        print("   " + ln.strip())
    results["病理集零漏清"] = (bad_cnt == 0)
    # ---- ⑤ 证书回归 ----
    print("\n[⑤ 证书回归]")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o1 = sc4.verify_path_index(); o3 = sc4.verify_summary_schema()
        o4 = sc4.verify_determinism(); o2 = sc4.verify_certificate(60)
    for ln in buf.getvalue().splitlines():
        if any(k in ln for k in ("路径索引", "摘要构造", "固定种子", "证书复核")):
            print("   " + ln.strip())
    results["证书回归"] = bool(o1 and o2 and o3 and o4)
    print("\n=== 门禁汇总 ===")
    for k, v in results.items():
        print(f"   {k}: {'PASS' if v else 'FAIL'}")
    print("总判定: " + ("**全部通过** —— 可进入在环 12 局复核"
                     if all(results.values()) else "**存在 FAIL, 不得采纳**"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
