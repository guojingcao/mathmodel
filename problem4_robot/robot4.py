# -*- coding: utf-8 -*-
"""
问题4 机器人程序 —— 全向 + 定向干扰源的自动定位与清除

模型要点(对应《汇总版》问题四 + 审计结论):
  1. 方向鲁棒覆盖: 三角网格(边长 a<=1000m)覆盖目标圆盘并向圆外延伸 a/2,
     使边界处朝外辐射的定向源也落在某个"含外侧顶点"的三角形内。
  2. 逐频道【三角形覆盖证书】: 若某三角形三个顶点对该频道均返回 no_signal,
     则该三角形内不可能存在该频道干扰源(顶点凸包 + 1000m 保证接收),
     记为"已证伪"; 当覆盖圆盘的全部三角形都被证伪时, 该频道可退役(不存在)。
  3. 位置可行域: 每次 direction -> ±1° 楔形, 联合可行域 = 半平面交。
     no_signal 不用于排除位置(可能是定向盲区), 只用于三角证书。
  4. 定向鲁棒补测: 补测点可能在定向源盲区, 依次尝试多个方位, 直到测得示向。
  5. 清除: 最小覆盖圆 <=20m 保证清除; near 直接清; 失败归航; 批量 2-opt;
     受限顺路清除(ΔL<=δ)。
"""
import json, math, sys, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ==================== 配置 ====================
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_ARENA_ID = "default"

DEG = math.pi / 180.0
EPS = 1.0 * DEG
R_AREA = 1800.0
R_CLEAR = 20.0
R_NEAR = 5.0
R_GUARANTEE = 1000.0     # 有效接收半径下限(1000m 内必可收到, 三角证书用)
N_CH = 20
MESH_A = 900.0           # 三角网格边长(<=1000 保证接收; 实测 900 最快且全覆盖)
MESH_MARGIN = 800.0      # 网格向圆外延伸量(实测 800 起才 100% 覆盖圆盘)
ON_WAY_DELTA = 300.0     # 受限顺路清除阈值


# ==================== 模拟器 HTTP 客户端 ====================
class SimClient:
    def __init__(self, base_url, robot_id, arena_id="default"):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.arena_id = arena_id
        self._seq = 0
        self.channel = 1
        self.position = (0.0, 0.0)
        self.actions = []
        self.virtual_time = 0.0
        self.meta = {}          # 机器人回填: 配置/网格/逐频道结果, 写入 JSON 汇总

    def _new_req_id(self, tag):
        self._seq += 1
        return f"{tag}-{self._seq}"

    def _record(self, path, payload, resp):
        vt = resp.get("virtual_time_s")
        if resp.get("accepted") is True and isinstance(vt, (int, float)):
            self.virtual_time = float(vt)
        self.actions.append({"seq": self._seq, "path": path, "payload": payload,
                             "accepted": resp.get("accepted"), "virtual_time_s": vt,
                             "response": resp})

    def post(self, path, payload, timeout=8):
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        resp = None
        for attempt in range(6):
            req = Request(self.base_url + path, data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except HTTPError as e:
                try:
                    resp = json.loads(e.read().decode("utf-8"))
                except Exception:
                    resp = {"accepted": False, "http_error": e.code}
                break
            except (URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(min(0.5 * (2 ** attempt), 3.0))
        if resp is None:
            raise RuntimeError(f"网络请求失败(已重试): {path} {last_err}")
        self._record(path, payload, resp)
        return resp

    def dump_log(self, filepath):
        import datetime
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# robot4.py 本地行为日志(问题4)\n")
            f.write(f"# team_no={self.robot_id} base_url={self.base_url}\n")
            f.write(f"# generated_at={datetime.datetime.now().isoformat()}\n")
            for a in self.actions:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.write(json.dumps({"__summary__": self.build_summary()},
                               ensure_ascii=False) + "\n")
        return filepath

    def build_summary(self):
        """结构化 JSON 汇总(配置/网格/接口计数/逐频道结果)。"""
        moves = 0.0
        prev = (0.0, 0.0)
        for a in self.actions:
            pos = a.get("payload", {}).get("position")
            if pos and a.get("path") in ("/measure", "/clear"):
                moves += math.hypot(pos["x"]-prev[0], pos["y"]-prev[1])
                prev = (pos["x"], pos["y"])
        clears = [a for a in self.actions if a["path"] == "/clear"]
        return {
            "problem": 4,
            "team_no": self.robot_id,
            "base_url": self.base_url,
            "config": {"mesh_a": MESH_A, "mesh_margin": MESH_MARGIN,
                       "on_way_delta": ON_WAY_DELTA, "r_clear": R_CLEAR},
            "final_virtual_time_s": self.virtual_time,
            "total_actions": len(self.actions),
            "measure_count": sum(1 for a in self.actions if a["path"] == "/measure"),
            "clear_attempt_count": len(clears),
            "clear_success_count": sum(1 for a in clears
                                       if a["response"].get("clear_result") == "success"),
            "clear_failure_count": sum(1 for a in clears
                                       if a["response"].get("clear_result") != "success"),
            "movement_distance_m": round(moves, 1),
            "robot": self.meta,
        }

    def _base(self, rid):
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": rid}

    def enter(self):
        r = self.post("/enter", self._base(self._new_req_id("enter")))
        if r.get("accepted") is not True:
            raise RuntimeError(f"/enter 失败: {r}")
        self.remaining_real = r.get("remaining_real_duration_s", 1200)
        return r

    def measure(self, x, y, channel):
        p = self._base(self._new_req_id("measure"))
        p["position"] = {"x": x, "y": y}; p["channel"] = channel
        r = self.post("/measure", p)
        if r.get("accepted") is not True:
            return False, "rejected", None
        self.position = (x, y); self.channel = channel
        return True, r.get("measure_result"), r.get("svd_deg")

    def clear(self, x, y, channel):
        p = self._base(self._new_req_id("clear"))
        p["position"] = {"x": x, "y": y}; p["channel"] = channel
        r = self.post("/clear", p)
        if r.get("accepted") is not True:
            return False, "rejected"
        self.position = (x, y)
        return True, r.get("clear_result")

    def exit(self):
        return self.post("/exit", self._base(self._new_req_id("exit")))


# ==================== 几何 ====================
def wedge_halfplanes(S, theta_deg):
    th = theta_deg * DEG
    n1 = (math.cos(th + EPS - math.pi/2), math.sin(th + EPS - math.pi/2))
    n2 = (math.cos(th - EPS + math.pi/2), math.sin(th - EPS + math.pi/2))
    return [(n1, n1[0]*S[0] + n1[1]*S[1]), (n2, n2[0]*S[0] + n2[1]*S[1])]


def clip_polygon(poly, n, c):
    if len(poly) < 3:
        return []
    out = []
    m = len(poly)
    for i in range(m):
        A = poly[i]; B = poly[(i+1) % m]
        dA = n[0]*A[0] + n[1]*A[1] - c
        dB = n[0]*B[0] + n[1]*B[1] - c
        Ain = dA >= -1e-9; Bin = dB >= -1e-9
        if Ain:
            out.append(A)
        if Ain != Bin:
            t = dA / (dA - dB)
            out.append((A[0] + t*(B[0]-A[0]), A[1] + t*(B[1]-A[1])))
    return out


def disk_polygon(R=R_AREA, m=72):
    return [(R*math.cos(2*math.pi*k/m), R*math.sin(2*math.pi*k/m)) for k in range(m)]


def feasible_region(bearings):
    poly = disk_polygon()
    for (S, th) in bearings:
        for n, c in wedge_halfplanes(S, th):
            poly = clip_polygon(poly, n, c)
            if len(poly) < 3:
                return poly
    return poly


def minimal_enclosing_circle(points):
    pts = list(points); n = len(pts)
    if n == 0:
        return None, float("inf")
    if n == 1:
        return pts[0], 0.0
    def cov(c, r):
        return all(math.hypot(p[0]-c[0], p[1]-c[1]) <= r + 1e-9 for p in pts)
    br, bc = float("inf"), None
    for i in range(n):
        for j in range(i+1, n):
            c = ((pts[i][0]+pts[j][0])/2, (pts[i][1]+pts[j][1])/2)
            r = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])/2
            if cov(c, r) and r < br:
                br, bc = r, c
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = pts[i]; bx, by = pts[j]; cx, cy = pts[k]
                d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by))/d
                uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax))/d
                c = (ux, uy); r = math.hypot(ax-ux, ay-uy)
                if cov(c, r) and r < br:
                    br, bc = r, c
    return bc, br


def bearing_intersection(stations, bearings_deg):
    A00 = A01 = A11 = b0 = b1 = 0.0
    for (x, y), th in zip(stations, bearings_deg):
        a = th * DEG; dx, dy = math.cos(a), math.sin(a)
        P00, P01, P11 = 1-dx*dx, -dx*dy, 1-dy*dy
        A00 += P00; A01 += P01; A11 += P11
        b0 += P00*x + P01*y; b1 += P01*x + P11*y
    det = A00*A11 - A01*A01
    if abs(det) < 1e-12:
        return None
    return ((A11*b0 - A01*b1)/det, (A00*b1 - A01*b0)/det)


def crossing_angle(t1, t2):
    d = abs(t1 - t2) % 180.0
    return min(d, 180.0 - d)


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ==================== 三角网格 + 三角形 ====================
def tri_mesh(a=MESH_A, margin=MESH_MARGIN):
    """三角网格点(覆盖半径1800圆盘并向圆外延伸 margin)。"""
    Rmax = R_AREA + margin
    dy = a * math.sqrt(3) / 2.0
    nr = int(Rmax / dy) + 2
    pts = []
    for j in range(-nr, nr+1):
        y = j * dy
        xoff = (a/2.0) if (j % 2) else 0.0
        nx = int(Rmax / a) + 2
        for i in range(-nx, nx+1):
            x = i*a + xoff
            if math.hypot(x, y) <= Rmax + 1e-6:
                pts.append((round(x, 6), round(y, 6)))
    # 去重
    seen = set(); out = []
    for p in pts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def build_triangles(pts, a=MESH_A):
    """由网格点构造边长 <= a 的三角形(唯一化)。"""
    n = len(pts)
    tol = a * 1.03
    tris = set()
    for i in range(n):
        nbr = []
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
            if d <= tol:
                nbr.append((math.atan2(pts[j][1]-pts[i][1], pts[j][0]-pts[i][0]), j, d))
        nbr.sort()
        m = len(nbr)
        for k in range(m):
            j1, d1 = nbr[k][1], nbr[k][2]
            j2, d2 = nbr[(k+1) % m][1], nbr[(k+1) % m][2]
            d12 = math.hypot(pts[j1][0]-pts[j2][0], pts[j1][1]-pts[j2][1])
            if max(d1, d2, d12) <= tol:
                tris.add(tuple(sorted((i, j1, j2))))
    return [list(t) for t in sorted(tris)]


def covering_triangles(tris, pts):
    """与目标圆盘相交(近似: 形心在 1800+a/2 内)的三角形 -> 需被证伪的集合。"""
    keep = []
    for t in tris:
        cx = sum(pts[i][0] for i in t)/3.0
        cy = sum(pts[i][1] for i in t)/3.0
        if math.hypot(cx, cy) <= R_AREA + MESH_A*0.6:
            keep.append(t)
    return keep


# ==================== 机器人 ====================
class Problem4Robot:
    # ===== 外部改进模块开关(默认关, 用于消融实验) =====
    USE_NEG_INFO = False     # 模块1+4: 用 no_signal 负信息收缩联合可行域 + 指向状态
    USE_PSO = False          # 模块3: 定位阶段用改进 PSO 精化位置
    DO_VERIFY = False        # 模块5: 清除后对"已排除"频道做多方向复核

    def __init__(self, client):
        self.c = client
        self.state = {ch: None for ch in range(1, N_CH+1)}   # None/found/excluded/cleared
        self.bearings = {ch: [] for ch in range(1, N_CH+1)}
        self.near_pos = {ch: None for ch in range(1, N_CH+1)}
        self.ns_at = {ch: set() for ch in range(1, N_CH+1)}   # no_signal 的点索引
        self.ns_at_pos = {ch: [] for ch in range(1, N_CH+1)}  # no_signal 的点坐标(负信息用)
        self.cleared_count = 0
        self.onway_failed = set()   # 顺路清除失败过的频道: 不再顺路重试, 留给扫描后批量清除
        self.pts = tri_mesh(a=MESH_A, margin=MESH_MARGIN)
        self.tris = build_triangles(self.pts, a=MESH_A)
        self.cover_tris = covering_triangles(self.tris, self.pts)

    def log(self, *a):
        print("[robot4]", *a, flush=True)

    # ---- 覆盖率预估(离线) ----
    def mesh_stats(self):
        d = 0.0
        for i in range(1, len(self.pts)):
            d += math.hypot(self.pts[i][0]-self.pts[i-1][0], self.pts[i][1]-self.pts[i-1][1])
        return len(self.pts), len(self.tris), len(self.cover_tris)

    # ---- 路径: 最近邻 + 2-opt ----
    @staticmethod
    def _two_opt(path):
        def dd(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])
        n = len(path) - 1
        imp = True
        while imp:
            imp = False
            for i in range(1, n):
                for j in range(i+1, n):
                    old = dd(path[i-1], path[i]) + dd(path[j], path[j+1])
                    new = dd(path[i-1], path[j]) + dd(path[i], path[j+1])
                    if new < old - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]; imp = True
        return path

    def _order_points(self):
        pts = self.pts
        order = []; unv = set(range(len(pts))); cur = (0.0, 0.0)
        while unv:
            k = min(unv, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
            order.append(k); cur = pts[k]; unv.discard(k)
        path = self._two_opt([(0.0, 0.0)] + [pts[k] for k in order])
        return path

    # ---- 快速定位(不移动) ----
    def _locate_quick(self, ch):
        dirs = list(self.bearings[ch])
        if len(dirs) >= 2:
            poly = feasible_region(dirs)
            if len(poly) >= 3:
                center, radius = minimal_enclosing_circle(poly)
                if radius <= R_CLEAR:
                    return center
            bi = bj = 0; ba = -1
            for i in range(len(dirs)):
                for j in range(i+1, len(dirs)):
                    a = crossing_angle(dirs[i][1], dirs[j][1])
                    if a > ba:
                        ba, bi, bj = a, i, j
            if ba >= 30.0:
                est = bearing_intersection([dirs[bi][0], dirs[bj][0]],
                                           [dirs[bi][1], dirs[bj][1]])
                if est is not None and self.USE_PSO:
                    # 模块3: 用改进 PSO 在连续非凸空间精化位置
                    est = self._pso_refine(ch, dirs, est)
                return est
        return None

    # ---- 已发现频道的冗余测量过滤: 当前点交会角有明显改善才值得测 ----
    def _worth_measuring(self, ch, qx, qy):
        dirs = list(self.bearings[ch])
        if len(dirs) < 2:
            return True
        est = bearing_intersection([p for p, _ in dirs], [t for _, t in dirs])
        if est is None:
            return True
        ex, ey = est
        thQ = math.degrees(math.atan2(ey - qy, ex - qx)) % 360
        for _, ti in dirs:
            if crossing_angle(thQ, ti) >= 30.0:
                return True
        return False

    # ---- 模块1+4: 负信息 -> 联合可行域(位置×指向)与指向状态 ----
    def _pointing_arcs(self, ch, S):
        """给定位置估计 S, 返回 (可行指向集合, 参考指向)。

        约束(全部为 180° 圆弧):
          direction at P  : 指向须覆盖 P  -> 指向 ∈ Arc(S->P)
          no_signal at P  : 若 |S-P| <= 1000(1000m 内必可收), 则指向不得覆盖 P
          等价地: 用 72 个 5° 分箱的布尔数组表示指向的可行集合。
        """
        import math as _m
        bins = 72
        feas = [True] * bins
        ref = None
        for (P, th) in self.bearings[ch]:
            a = _m.degrees(_m.atan2(P[1]-S[1], P[0]-S[0])) % 360   # S->P 方向
            for k in range(bins):
                dk = k * (360.0 / bins)
                diff = abs((dk - a + 180) % 360 - 180)
                if diff > 90.0:            # 该指向不覆盖 P
                    feas[k] = False
            ref = a if ref is None else ref
        for P in self.ns_at_pos[ch]:
            if _m.hypot(P[0]-S[0], P[1]-S[1]) <= R_GUARANTEE:
                a = _m.degrees(_m.atan2(P[1]-S[1], P[0]-S[0])) % 360
                for k in range(bins):
                    dk = k * (360.0 / bins)
                    diff = abs((dk - a + 180) % 360 - 180)
                    if diff <= 90.0:       # 该指向覆盖了 P, 但 P 无信号 -> 排除
                        feas[k] = False
        return feas, ref

    def _neg_info_filter(self, ch, cands, S):
        """用负信息给候选补测点排序: 优先选"大概率落在源覆盖半平面内"的点。"""
        feas, _ = self._pointing_arcs(ch, S)
        ok = [k for k in range(len(feas)) if feas[k]]
        if not ok:
            return cands                      # 无可行指向 -> 不做过滤
        import math as _m
        def score(q):
            a = _m.degrees(_m.atan2(q[1]-S[1], q[0]-S[0])) % 360
            hit = 0
            for k in ok:
                dk = k * 5.0
                if abs((a - dk + 180) % 360 - 180) <= 90.0:
                    hit += 1
            return -hit
        return sorted(cands, key=score)

    # ---- 模块3: 改进 PSO 精化位置(连续非凸) ----
    @staticmethod
    def _pso_refine(ch, dirs, init, iters=40, n_p=24, seed=7):
        """以 (两站交会残差² + 位置与可行域质心的距离惩罚) 为目标做 PSO。"""
        import random
        rnd = random.Random(seed)
        def cost(p):
            s = 0.0
            for (P, th) in dirs:
                a = math.degrees(math.atan2(p[1]-P[1], p[0]-P[0])) % 360
                d = abs(a - th) % 360; d = min(d, 360-d)
                s += d*d
            s += 1e-4 * math.hypot(p[0]-init[0], p[1]-init[1])
            return s
        P = [[init[0] + rnd.uniform(-200, 200), init[1] + rnd.uniform(-200, 200)] for _ in range(n_p)]
        V = [[rnd.uniform(-30, 30), rnd.uniform(-30, 30)] for _ in range(n_p)]
        pb = [p[:] for p in P]; pbv = [cost(p) for p in P]
        gi = min(range(n_p), key=lambda i: pbv[i]); gb = pb[gi][:]; gv = pbv[gi]
        for _ in range(iters):
            for i in range(n_p):
                for d in (0, 1):
                    V[i][d] = (0.5*V[i][d] + 1.2*rnd.random()*(pb[i][d]-P[i][d])
                               + 1.2*rnd.random()*(gb[d]-P[i][d]))
                    V[i][d] = max(-200, min(200, V[i][d]))
                    P[i][d] = max(-2000, min(2000, P[i][d] + V[i][d]))
                f = cost(P[i])
                if f < pbv[i]:
                    pbv[i] = f; pb[i] = P[i][:]
                    if f < gv:
                        gv = f; gb = P[i][:]
        return (gb[0], gb[1])

    # ---- 定向鲁棒补测: 多方位依次尝试 ----
    def _supplement_points(self, ch, toward):
        """为一个待补测频道生成多个方位的候选补测点(应对定向盲区)。"""
        dirs = list(self.bearings[ch])
        if not dirs:
            return []
        (x0, y0), th0 = dirs[0]
        cands = []
        for off in (90.0, -90.0, 135.0, -135.0):
            for d in (400.0, 700.0):
                a = (th0 + off) * DEG
                q = (x0 + d*math.cos(a), y0 + d*math.sin(a))
                if math.hypot(q[0], q[1]) <= 2000000:
                    cands.append(q)
        # 按离 toward 的远近排序(近的优先)
        cands.sort(key=lambda q: math.hypot(q[0]-toward[0], q[1]-toward[1]))
        return cands

    # ---- 二分归航(指定检测点与示向) ----
    def _binary_homing(self, ch, P, th0):
        x0, y0 = P
        a = th0 * DEG; ux, uy = math.cos(a), math.sin(a)
        lo, hi = 0.0, 1500.0
        for _ in range(22):
            mid = (lo + hi)/2.0
            px, py = x0 + mid*ux, y0 + mid*uy
            ok, res, svd = self.c.measure(px, py, ch)
            if not ok:
                break
            if res == "near":
                return (px, py)
            if res == "direction":
                if angle_diff(svd, th0) > 90.0:
                    hi = mid
                else:
                    lo = mid
            else:
                hi = mid
            if hi - lo < 4.0:
                return (x0 + mid*ux, y0 + mid*uy)
        return (x0 + (lo+hi)/2.0*ux, y0 + (lo+hi)/2.0*uy)

    # ---- 就近精定位(清除失败兜底) ----
    def _homing_clear(self, ch, x, y):
        # 1) 先在给定点直接试
        okc, rc = self.c.clear(x, y, ch)
        if okc and rc == "success":
            self.state[ch] = "cleared"; self.cleared_count += 1
            return
        # 2) 对每条已有示向依次做二分归航(边界源可能只有个别方位稳健)
        for (P, th) in list(self.bearings[ch]):
            bt = self._binary_homing(ch, P, th)
            if bt is None:
                continue
            okc, rc = self.c.clear(bt[0], bt[1], ch)
            if okc and rc == "success":
                self.state[ch] = "cleared"; self.cleared_count += 1
                return
            for rad in (8.0, 15.0):
                for k in range(6):
                    a = k * 60 * DEG
                    q = (bt[0] + rad*math.cos(a), bt[1] + rad*math.sin(a))
                    ok2, rc2 = self.c.clear(q[0], q[1], ch)
                    if ok2 and rc2 == "success":
                        self.state[ch] = "cleared"; self.cleared_count += 1
                        return

    # ---- 主流程 ----
    def run(self):
        c = self.c
        self.log("调用 /enter ...")
        c.enter()
        npts, ntri, ncov = self.mesh_stats()
        self.log(f"进入成功. 三角网格: {npts} 点 / {ntri} 三角形 / 需证伪 {ncov} 个")

        path = self._order_points()
        self.log(f"覆盖路径点数 {len(path)}")

        # 保证层: 依序访问网格点, 蛇形扫描
        for i, (px, py) in enumerate(path):
            order = list(range(1, N_CH+1)) if i % 2 == 0 else list(range(N_CH, 0, -1))
            for ch in order:
                if self.state[ch] in ("excluded", "cleared"):
                    continue
                # 几何冗余过滤: 已发现频道只在交会角有改善时测(否则跳过)
                if self.state[ch] == "found" and not self._worth_measuring(ch, px, py):
                    continue
                ok, res, svd = c.measure(px, py, ch)
                if not ok:
                    continue
                if res == "direction":
                    if self.state[ch] is None:
                        self.state[ch] = "found"
                    self.bearings[ch].append(((px, py), svd))
                elif res == "near":
                    self.state[ch] = "found"
                    self.near_pos[ch] = (px, py)
                elif res == "no_signal":
                    self.ns_at[ch].add(i)
                    self.ns_at_pos[ch].append((px, py))
            # 三角形证书: 顶点均 no_signal 的三角形被证伪
            for ch in range(1, N_CH+1):
                if self.state[ch] in ("excluded", "cleared"):
                    continue
                if self.state[ch] is None:
                    if all(all(v in self.ns_at[ch] for v in t) for t in self.cover_tris):
                        self.state[ch] = "excluded"
            # 受限顺路清除
            if ON_WAY_DELTA is not None and i + 1 < len(path):
                nxt = path[i+1]
                for c2 in range(1, N_CH+1):
                    if self.state[c2] != "found" or c2 in self.onway_failed:
                        continue
                    Q = self.near_pos[c2] if self.near_pos[c2] else self._locate_quick(c2)
                    if Q is None:
                        continue
                    cur = c.position
                    dL = (math.hypot(Q[0]-cur[0], Q[1]-cur[1])
                          + math.hypot(Q[0]-nxt[0], Q[1]-nxt[1])
                          - math.hypot(cur[0]-nxt[0], cur[1]-nxt[1]))
                    if dL <= ON_WAY_DELTA:
                        ok, res = c.clear(Q[0], Q[1], c2)
                        if ok and res == "success":
                            self.state[c2] = "cleared"; self.cleared_count += 1
                            self.log(f"顺路清除: 频道 {c2} [{self.cleared_count}]")
                        else:
                            # 顺路失败: 记入冷却, 不再在后续网格点反复顺路重试
                            self.onway_failed.add(c2)
                            self._homing_clear(c2, Q[0], Q[1])
                            if self.state[c2] == "cleared":
                                self.onway_failed.discard(c2)

        self.log(f"扫描完成: 已发现 {sum(1 for s in self.state.values() if s=='found')}, "
                 f"已排除 {sum(1 for s in self.state.values() if s=='excluded')}")

        # 状态层: 快速定位分类
        clear_tasks = []; supp_tasks = []
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found":
                continue
            if self.near_pos[ch] is not None:
                clear_tasks.append((ch, self.near_pos[ch][0], self.near_pos[ch][1])); continue
            pt = self._locate_quick(ch)
            if pt is not None:
                clear_tasks.append((ch, pt[0], pt[1]))
            else:
                cands = self._supplement_points(ch, c.position)
                if self.USE_NEG_INFO and cands:
                    # 模块1+4: 用负信息推断指向, 优先选"大概率在覆盖内"的补测点
                    S0 = self._locate_quick(ch)
                    if S0 is None:
                        b = self.bearings[ch][0]
                        S0 = (b[0][0] + 500*math.cos(b[1]*DEG), b[0][1] + 500*math.sin(b[1]*DEG))
                    cands = self._neg_info_filter(ch, cands, S0)
                if cands:
                    supp_tasks.append((ch, cands[0][0], cands[0][1], cands))

        # 批量补测(方位鲁棒: 第一个方位失败则换下一个)
        if supp_tasks:
            self.log(f"补测任务 {len(supp_tasks)} 个(方位鲁棒)")
            spts = [(x, y) for _, x, y, _ in supp_tasks]
            spos = {(x, y): (ch, cands) for ch, x, y, cands in supp_tasks}
            order = self._optimal_open_path(spts, tuple(c.position))
            for idx in order:
                x, y = spts[idx]
                ch, cands = spos[(x, y)]
                got = False
                for (qx, qy) in cands:
                    ok, res, svd = c.measure(qx, qy, ch)
                    if ok and res == "near":
                        clear_tasks.append((ch, qx, qy)); got = True; break
                    if ok and res == "direction":
                        self.bearings[ch].append(((qx, qy), svd)); got = True
                        break
                pt = self._locate_quick(ch)
                if pt is not None:
                    clear_tasks.append((ch, pt[0], pt[1]))
                else:
                    # 补测失败(全在盲区)也必须兜底: 沿首示向二分归航
                    bt = self._binary_homing(ch, self.bearings[ch][0][0], self.bearings[ch][0][1])
                    if bt is not None:
                        clear_tasks.append((ch, bt[0], bt[1]))

        # 清除(2-opt)
        self.log(f"清除任务队列 {len(clear_tasks)} 个")
        pts = [(x, y) for _, x, y in clear_tasks]
        pos2ch = {(x, y): ch for ch, x, y in clear_tasks}
        if pts:
            order = []; unv = set(range(len(pts))); cur = c.position
            while unv:
                k = min(unv, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
                order.append(pts[k]); cur = pts[k]; unv.discard(k)
            pathc = self._two_opt([tuple(c.position)] + order)
            for (x, y) in pathc[1:]:
                ch = pos2ch[(x, y)]
                ok, res = c.clear(x, y, ch)
                if ok and res == "success":
                    self.state[ch] = "cleared"; self.cleared_count += 1
                    self.log(f"清除成功: 频道 {ch} [{self.cleared_count}]")
                else:
                    self.log(f"清除未发现: 频道 {ch}, 就近精定位")
                    self._homing_clear(ch, x, y)

        # 回填结构化元信息(写入 JSON 汇总)
        c.meta = {
            "mesh_points": len(self.pts),
            "mesh_triangles": len(self.tris),
            "cover_triangles": len(self.cover_tris),
            "cleared_count": self.cleared_count,
            "found_channels": sum(1 for s in self.state.values() if s == "found"),
            "excluded_channels": sum(1 for s in self.state.values() if s == "excluded"),
            "unresolved_channels": sum(1 for s in self.state.values()
                                      if s not in ("cleared", "excluded")),
            "channels": {str(ch): {"state": self.state[ch],
                                   "bearings": len(self.bearings[ch]),
                                   "near": self.near_pos[ch] is not None,
                                   "no_signal_points": len(self.ns_at[ch])}
                         for ch in range(1, N_CH+1)},
        }
        # 模块5: 清除后多方向复核 —— 走一遍圆外见证点, 在那逐频道复核"已排除"结论
        if self.DO_VERIFY:
            excl = [ch for ch in range(1, N_CH+1) if self.state[ch] == "excluded"]
            if excl:
                witnesses = [((R_AREA + 300)*math.cos(a*DEG), (R_AREA + 300)*math.sin(a*DEG))
                             for a in (45.0, 135.0, 225.0, 315.0)]
                self.log(f"多方向复核: {len(excl)} 个已排除频道, 见证点 {len(witnesses)} 个")
                for (wx, wy) in witnesses:
                    for ch in excl:
                        if self.state[ch] != "excluded":
                            continue
                        ok, res, svd = c.measure(wx, wy, ch)
                        if ok and res == "near":
                            self.state[ch] = "found"; self.near_pos[ch] = (wx, wy)
                            okc, rc = c.clear(wx, wy, ch)
                            if okc and rc == "success":
                                self.state[ch] = "cleared"; self.cleared_count += 1
                                self.log(f"复核后清除成功: 频道 {ch} [{self.cleared_count}]")
                        elif ok and res == "direction":
                            self.state[ch] = "found"
                            self.bearings[ch].append(((wx, wy), svd))
                            self.log(f"复核发现频道 {ch} 有信号, 转定位清除")
                            Q = self._locate_quick(ch)
                            if Q is None:
                                Q = self._binary_homing(ch, (wx, wy), svd)
                            if Q is not None:
                                okc, rc = c.clear(Q[0], Q[1], ch)
                                if okc and rc == "success":
                                    self.state[ch] = "cleared"; self.cleared_count += 1
                                    self.log(f"复核后清除成功: 频道 {ch} [{self.cleared_count}]")
                                else:
                                    self._homing_clear(ch, Q[0], Q[1])

        self.log("调用 /exit ...")
        c.exit()
        self.log(f"结束, 清除 {self.cleared_count} 个干扰源")
        return self.cleared_count

    @staticmethod
    def _optimal_open_path(points, start):
        n = len(points)
        if n <= 1:
            return list(range(n))
        def dd(a, b):
            return math.hypot(points[a][0]-points[b][0], points[a][1]-points[b][1])
        if n <= 9:
            import itertools
            best, bl = None, float("inf")
            for perm in itertools.permutations(range(n)):
                L = math.hypot(points[perm[0]][0]-start[0], points[perm[0]][1]-start[1])
                for k in range(1, n):
                    L += dd(perm[k-1], perm[k])
                if L < bl:
                    bl, best = L, perm
            return list(best)
        order = []; unv = set(range(n)); cur = start
        while unv:
            k = min(unv, key=lambda i: math.hypot(points[i][0]-cur[0], points[i][1]-cur[1]))
            order.append(k); cur = points[k]; unv.discard(k)
        return order


def main(argv=None):
    import argparse, os, datetime
    ap = argparse.ArgumentParser(description="问题4 全向+定向干扰源机器人")
    ap.add_argument("--robot-id", dest="robot_id", default=os.environ.get("ROBOT_ID"))
    ap.add_argument("--base-url", dest="base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--arena-id", dest="arena_id", default=DEFAULT_ARENA_ID)
    ap.add_argument("--log-file", dest="log_file", default=None)
    ap.add_argument("--mesh-stats", action="store_true", help="只打印网格统计后退出")
    args = ap.parse_args(argv)

    if args.mesh_stats:
        rb = Problem4Robot.__new__(Problem4Robot)
        rb.pts = tri_mesh(margin=MESH_MARGIN); rb.tris = build_triangles(rb.pts)
        rb.cover_tris = covering_triangles(rb.tris, rb.pts)
        print(f"网格点数={len(rb.pts)} 三角形数={len(rb.tris)} 需证伪={len(rb.cover_tris)}")
        return

    if not args.robot_id:
        print("错误: 未提供参赛队号。用法: python robot4.py --robot-id 你的队号", file=sys.stderr)
        sys.exit(2)
    # 默认日志写到本程序目录下的 logs/ (与问题3 一致)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(
        log_dir, "p4_log_%s.jsonl" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    client = SimClient(args.base_url, args.robot_id, args.arena_id)
    robot = Problem4Robot(client)
    try:
        cleared = robot.run()
        s = client.build_summary()
        print(f"\n[汇总] 清除干扰源 {cleared} 个, 虚拟时刻 {s['final_virtual_time_s']:.1f}s, "
              f"移动 {s['movement_distance_m']:.0f}m, 检测 {s['measure_count']} 次, "
              f"清除 {s['clear_attempt_count']} 次(成功 {s['clear_success_count']})")
    finally:
        # 单个 JSONL 文件: 逐条动作 + 末行 __summary__ 结构化汇总(不再另写第二份)
        p = client.dump_log(log_file)
        print(f"[日志] 已写入 {p}")


if __name__ == "__main__":
    main()
