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
BEARING_ERR_DEG = 1.0     # 题设测向误差界(±1°), 用于"横向误差上界 d̂·sin1°"的示向排序
EPS = 1.0 * DEG
R_AREA = 1800.0
R_CLEAR = 20.0
R_NEAR = 5.0
R_GUARANTEE = 1000.0     # 有效接收半径下限(1000m 内必可收到, 三角证书用)
N_CH = 20
MESH_A = 970.0           # 三角网格边长(<=1000 保证接收; 实测 920 最优)
MESH_MARGIN = 700.0   # 仅在无 MESH_PTS_OVERRIDE 时使用      # 网格向圆外延伸量(实测 700 起才 100% 覆盖圆盘)
# 以下两项经配对实验选定(3 seed × 2 定向比例, n=400/臂, 全部显著 -920~-1059 s = -8.8~-9.9%):
#   27 点 / 37 三角形 / 最大边 920 m / 空洞 0.000%(8000 点) / 定向可发现率 100%(4000 点)
#   固定巡回 27659 -> 24528 m; 检测/例 -66~-75; P90 全面改善; 全清率 100% 不变
# 网格几何的额外自由度(默认值 = 冻结版: 无旋转无平移)。旋转/平移不改变任何算法逻辑,
# 只改变 31 个顶点的位置, 从而改变"固定扫描巡回"的长度与证书检测次数。
MESH_THETA = 20.0        # 旋转角(度)
MESH_OFFSET = (460.0, 398.0)  # 平移(米)
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
        self.error = None       # 异常说明(如接口未开放), 写入 JSON 汇总
        self.phase = "init"     # 当前算法阶段(机器人回填, 用于分阶段统计)
        self.locate_history = []  # 机器人回填: 每次定位的 方式/Ω半径/交会角
        self.clear_diag = []      # 机器人回填: 每次清除的 定位来源/Ω半径/结果
        # 计数(供归航 episode 开销归因; 与离线 MockClient 同名同义)
        self.dist = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_ok = 0
        self.n_switch = 0
        self.homing_diag = []     # 机器人回填: 每次归航 episode 的开销与结果
        self.supp_diag = []       # 机器人回填: 每次补测决策/执行(距离与动作)

    def _new_req_id(self, tag):
        self._seq += 1
        return f"{tag}-{self._seq}"

    def _record(self, path, payload, resp):
        vt = resp.get("virtual_time_s")
        if resp.get("accepted") is True and isinstance(vt, (int, float)):
            self.virtual_time = float(vt)
        self.actions.append({"seq": self._seq, "path": path, "payload": payload,
                             "accepted": resp.get("accepted"), "virtual_time_s": vt,
                             "phase": getattr(self, "phase", None),
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
        """结构化 JSON 汇总(配置/网格/接口计数/分阶段/定位诊断/逐频道结果)。"""
        moves = 0.0
        prev = (0.0, 0.0)
        phases = {}
        prev_vt = None

        def ph(name):
            return phases.setdefault(name or "unknown", {
                "movement_distance_m": 0.0, "measure_count": 0,
                "clear_attempt_count": 0, "clear_success_count": 0,
                "virtual_time_s": 0.0})

        for a in self.actions:
            pos = a.get("payload", {}).get("position")
            p = ph(a.get("phase"))
            if pos and a.get("path") in ("/measure", "/clear"):
                leg = math.hypot(pos["x"]-prev[0], pos["y"]-prev[1])
                moves += leg
                p["movement_distance_m"] += leg
                prev = (pos["x"], pos["y"])
            if a.get("path") == "/measure":
                if a.get("accepted") is True:
                    p["measure_count"] += 1
            elif a.get("path") == "/clear":
                if a.get("accepted") is True:
                    p["clear_attempt_count"] += 1
                    if a["response"].get("clear_result") == "success":
                        p["clear_success_count"] += 1
            vt = a.get("virtual_time_s")     # 虚拟时间按串行时间轴分段归属
            if isinstance(vt, (int, float)):
                if prev_vt is not None and vt >= prev_vt:
                    p["virtual_time_s"] += vt - prev_vt
                prev_vt = vt
        for p in phases.values():
            p["movement_distance_m"] = round(p["movement_distance_m"], 1)
            p["virtual_time_s"] = round(p["virtual_time_s"], 1)

        clears = [a for a in self.actions if a["path"] == "/clear"]
        s = {
            "problem": 4,
            "team_no": self.robot_id,
            "base_url": self.base_url,
            "config": {"mesh_a": MESH_A, "mesh_margin": MESH_MARGIN,
                       "mesh_theta": MESH_THETA, "mesh_offset": list(MESH_OFFSET),
                       "on_way_delta": ON_WAY_DELTA, "r_clear": R_CLEAR,
                       "r_guarantee": R_GUARANTEE,
                       # 开关是类属性(不是模块常量), 必须经类名读取
                       "use_neg_info": getattr(Problem4Robot, "USE_NEG_INFO", None),
                       "use_pso": getattr(Problem4Robot, "USE_PSO", None),
                       "do_verify": getattr(Problem4Robot, "DO_VERIFY", None),
                       "neighbor_rings": list(getattr(Problem4Robot,
                                                      "NEIGHBOR_RINGS", ())),
                       "supp_max_dist": getattr(Problem4Robot, "SUPP_MAX_DIST", None)},
            "final_virtual_time_s": self.virtual_time,
            "total_actions": len(self.actions),
            "rejected_count": sum(1 for a in self.actions if a.get("accepted") is not True),
            "measure_count": sum(1 for a in self.actions
                                 if a["path"] == "/measure" and a.get("accepted") is True),
            "clear_attempt_count": sum(1 for a in clears if a.get("accepted") is True),
            "clear_success_count": sum(1 for a in clears if a.get("accepted") is True
                                       and a["response"].get("clear_result") == "success"),
            "clear_failure_count": sum(1 for a in clears if a.get("accepted") is True
                                       and a["response"].get("clear_result") != "success"),
            "clear_rejected_count": sum(1 for a in clears if a.get("accepted") is not True),
            "movement_distance_m": round(moves, 1),
            "phase_stats": phases,                  # 分阶段移动距离/动作数/虚拟时间
            "locate_stats": self._locate_stats(),   # 定位方式与 Ω 半径统计
            "clear_diag": self.clear_diag,          # 逐次清除的定位来源+Ω 半径(诊断)
            "homing_stats": self._homing_stats(),    # 归航 episode 汇总(邻域试探消融口径)
            "homing_diag": self.homing_diag,         # 归航 episode 明细
            "supp_stats": self._supp_stats(),        # 补测决策汇总(距离上限口径)
            "supp_diag": self.supp_diag,             # 补测决策明细
            "robot": self.meta,
        }
        if self.error:
            s["error"] = self.error
        return s

    def _locate_stats(self):
        """汇总定位诊断: MEC(Ω 半径<=20m) 与最小二乘(交会角) 各占多少、半径分布。"""
        hist = self.locate_history
        mec = [h["omega_radius_m"] for h in hist
               if h.get("method") == "mec" and h.get("omega_radius_m") is not None]
        ls = [h for h in hist if h.get("method") == "ls"]
        over = [h["omega_radius_m"] for h in ls if h.get("omega_radius_m") is not None]

        def stat(v):
            if not v:
                return None
            v = sorted(v)
            return {"n": len(v), "min": round(v[0], 1), "median": round(v[len(v)//2], 1),
                    "max": round(v[-1], 1)}
        return {
            "calls": len(hist),
            "method_counts": {"mec": sum(1 for h in hist if h.get("method") == "mec"),
                              "ls": len(ls),
                              "fail": sum(1 for h in hist if h.get("method") is None)},
            "omega_radius_m_when_mec": stat(mec),
            "omega_radius_m_when_ls": stat(over),
            "ls_cross_angle_deg": stat([h["cross_angle_deg"] for h in ls
                                        if h.get("cross_angle_deg") is not None]),
        }

    def _homing_stats(self):
        """归航 episode 汇总: 邻域清除成功率/条件成功率、换示向补回比例、每困难源开销。"""
        eps = self.homing_diag
        if not eps:
            return {"episodes": 0}
        cleared = [e for e in eps if e.get("cleared_by")]
        by = {}
        for e in cleared:
            by[e["cleared_by"]] = by.get(e["cleared_by"], 0) + 1
        ring_att = sum(1 for d in self.clear_diag
                       if "邻域" in str(d.get("src")) and d.get("result") != "rejected")
        ring_ok = sum(1 for d in self.clear_diag
                      if "邻域" in str(d.get("src")) and d.get("result") == "success")
        tried_multi = [e for e in eps if e.get("bearings_tried", 0) > 1]
        later = [e for e in tried_multi if e.get("cleared_at_bearing", 0) > 0]
        hard = [e for e in eps if e.get("cleared_by")]
        n = len(hard) or 1
        moved = [e.get("episode_moves_m", 0.0) for e in hard]
        clears = [(e.get("cost") or {}).get("n_clear") or 0 for e in hard]
        times = [e.get("episode_time_s", 0.0) for e in hard]

        def avg(v):
            return round(sum(v)/n, 1) if v else None
        return {
            "episodes": len(eps),
            "cleared_episodes": len(cleared),
            "cleared_by": by,                       # 原位/归航点/邻域/后续示向
            "neighbor_attempts": ring_att,
            "neighbor_success": ring_ok,
            "neighbor_success_rate": round(ring_ok/ring_att, 4) if ring_att else None,
            "neighbor_conditional_rate": round(by.get("邻域", 0)/len(eps), 4),
            "multi_bearing_episodes": len(tried_multi),
            "recovered_by_later_bearing": len(later),
            "later_bearing_recovery_rate": (round(len(later)/len(tried_multi), 4)
                                            if tried_multi else None),
            "per_hard_source": {"n": len(hard), "moves_m": avg(moved),
                                "clears": avg(clears), "time_s": avg(times)},
        }

    def _supp_stats(self):
        """补测决策汇总: 最近补测点距离分布、被上限跳过的次数与后续补回率。"""
        d = self.supp_diag
        if not d:
            return {"decisions": 0}
        def stat(v):
            if not v:
                return None
            v = sorted(v)
            return {"n": len(v), "median": round(v[len(v)//2], 1), "max": round(v[-1], 1)}
        m = self.meta or {}
        sk_ch = m.get("supp_skipped_channels") or []
        return {
            "cap_m": getattr(Problem4Robot, "SUPP_MAX_DIST", None),
            "decisions": len(d),
            "skipped_by_cap": sum(1 for x in d if x["action"] == "skip"),
            "executed": sum(1 for x in d if x["action"] == "exec"),
            "nearest_m_when_measure": stat([x["nearest_m"] for x in d
                                            if x["action"] == "measure"]),
            "nearest_m_when_exec": stat([x["nearest_m"] for x in d
                                         if x["action"] == "exec"]),
            "nearest_m_when_skip": stat([x["nearest_m"] for x in d
                                         if x["action"] == "skip"]),
            "skipped_channels": len(sk_ch),
            "skipped_recovered": m.get("supp_skipped_recovered"),
            "skipped_recovered_rate": (round(m.get("supp_skipped_recovered", 0)/len(sk_ch), 4)
                                       if sk_ch else None),
        }

    def _base(self, rid):
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": rid}

    def enter(self, wait_s=90):
        """进入目标区域。接口未开放时连接会被直接拒绝, 故自动等待重试。"""
        rid = self._new_req_id("enter")
        payload = self._base(rid)
        t0 = time.time()
        r = None
        while True:
            try:
                r = self.post("/enter", payload, timeout=5)
                break
            except RuntimeError:
                if time.time() - t0 > wait_s:
                    raise RuntimeError(
                        "无法连接模拟器接口(连接被拒绝)。请确认:\n"
                        "  1) 模拟器已启动并已在线登录;\n"
                        "  2) 已在模拟器中点击开始\"问题4测试\";\n"
                        "  3) 5 秒倒计时已结束、界面提示机器狗接口已就绪;\n"
                        "  4) --robot-id 与当前登录的参赛队号一致;\n"
                        "  5) 端口与模拟器设置一致(默认 2026)。")
                if int(time.time() - t0) // 15 > int((time.time() - t0 - 0.1) // 15):
                    print(f"  [等待] 模拟器接口未就绪({int(time.time()-t0)}s), 正在重试 ...",
                          flush=True)
                time.sleep(3)
        if r.get("accepted") is not True:
            raise RuntimeError(f"/enter 被拒绝: {r}")
        self.remaining_real = r.get("remaining_real_duration_s", 1200)
        return r

    def measure(self, x, y, channel):
        p = self._base(self._new_req_id("measure"))
        p["position"] = {"x": x, "y": y}; p["channel"] = channel
        r = self.post("/measure", p)
        if r.get("accepted") is not True:
            return False, "rejected", None
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        if channel != self.channel:
            self.n_switch += 1
        self.n_measure += 1
        self.position = (x, y); self.channel = channel
        return True, r.get("measure_result"), r.get("svd_deg")

    def clear(self, x, y, channel):
        p = self._base(self._new_req_id("clear"))
        p["position"] = {"x": x, "y": y}; p["channel"] = channel
        r = self.post("/clear", p)
        if r.get("accepted") is not True:
            return False, "rejected"
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        # 题设: /clear 不换频, 也不改变测向机频道状态(附件1/附件2)
        self.n_clear += 1
        if r.get("clear_result") == "success":
            self.n_clear_ok += 1
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
    """目标圆盘的外接 72 边形(与问题一/问题三口径一致: Rc = R/cos(pi/m) 不缩小真实圆域)。"""
    Rc = R / math.cos(math.pi / m)
    return [(Rc*math.cos(2*math.pi*k/m), Rc*math.sin(2*math.pi*k/m)) for k in range(m)]


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
def tri_mesh(a=MESH_A, margin=MESH_MARGIN, theta_deg=None, offset=None):
    """三角网格点(覆盖半径1800圆盘并向圆外延伸 margin), 可选旋转/平移。

    theta_deg / offset 为 None 时取模块常量 MESH_THETA / MESH_OFFSET(默认 0 / (0,0),
    即与冻结版完全一致)。旋转与平移不改变证书逻辑与覆盖判定方式, 只改变顶点位置。
    """
    theta_deg = MESH_THETA if theta_deg is None else theta_deg
    offset = MESH_OFFSET if offset is None else offset
    th = math.radians(theta_deg); ct, st = math.cos(th), math.sin(th)
    Rmax = R_AREA + margin
    dy = a * math.sqrt(3) / 2.0
    nr = int(Rmax / dy) + 2
    pts = []
    for j in range(-nr, nr+1):
        y0 = j * dy + offset[1]
        xoff = (a/2.0) if (j % 2) else 0.0
        nx = int(Rmax / a) + 2
        for i in range(-nx, nx+1):
            x0 = i*a + xoff + offset[0]
            x = x0*ct - y0*st
            y = x0*st + y0*ct
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
    # 归航失败后的邻域试探圈: (8,15) = 既有流程; (8,) / (15,) = 仅保留一圈; () = 取消试探
    NEIGHBOR_RINGS = (8.0, 15.0)
    # 补测距离上限(米): None = 无上限(既有流程); 数值 = 最近补测点超过该距离就跳过补测,
    # 直接走已有的"沿首示向二分归航"(只加调度阈值, 不改覆盖/定位模型)。
    SUPP_MAX_DIST = None
    # ===== 网格: 最小覆盖顶点集(覆盖设计, 由 mesh_design.py 搜索得到) =====
    # 27 点三角格点块(a=910, θ=30°, 外扩半径 2412, 平移(460,0))。
    # 关键: **完整格点块**的三角剖分并集 = 其凸包 ⊇ 圆盘, 故不存在"最外环薄空洞"——
    # 从构造上根除过去"稀疏外环 + 事后补点"引入的证书失效问题; 且点数更少、巡回更短:
    #   扫描成本 ≈ 巡回/5 + 20 频道×5 s×点数 = 24 120/5 + 2 700 = 7 524 s
    #   (旧 31 点方案 24 319/5 + 3 100 = 7 964 s) -> −440 s
    # 置 None 即回退到 tri_mesh + MESH_EXTRA_PTS 路径。
    MESH_DESIGN_PTS = [
        (-1548.505, -1171.594), (-1810.648, -283.272), (-1838.208, 435.093),
        (-1366.795, 1254.801), (-772.343, -1714.099), (-1053.633, -848.967),
        (-1037.264, -32.657), (-1060.406, 883.632), (-911.526, 1613.717),
        (66.165, -1861.716), (-357.775, -1391.629), (-463.223, -474.349),
        (-250.053, 450.618), (-258.892, 1388.985), (-296.147, 1808.476),
        (516.348, -1777.483), (406.711, -877.507), (333.103, -6.504),
        (401.193, 927.350), (568.695, 1815.845), (1146.574, -1440.378),
        (1090.117, -492.846), (1175.054, 419.428), (1221.611, 1377.741),
        (1797.953, -738.632), (1837.899, -122.731), (1730.516, 805.452),
    ]
    # 旧设计(格点块, 27 点): 保留作回退对照; 若需回退把下面两行换成它并把 MESH_A 改为 920
    MESH_DESIGN_PTS_V1_LATTICE = [
        (-1904.249, -1365.0), (-1904.249, -455.0), (-1904.249, 455.0),
        (-1904.249, 1365.0), (-1116.166, -1820.0), (-1116.166, -910.0),
        (-1116.166, 0.0), (-1116.166, 910.0), (-1116.166, 1820.0),
        (-328.083, -2275.0), (-328.083, -1365.0), (-328.083, -455.0),
        (-328.083, 455.0), (-328.083, 1365.0), (-328.083, 2275.0),
        (460.0, -1820.0), (460.0, -910.0), (460.0, 0.0),
        (460.0, 910.0), (460.0, 1820.0), (1248.083, -1365.0),
        (1248.083, -455.0), (1248.083, 455.0), (1248.083, 1365.0),
        (2036.166, -910.0), (2036.166, 0.0), (2036.166, 910.0),
    ]
    MESH_PTS_OVERRIDE = MESH_DESIGN_PTS
    # 覆盖补齐点: 被 MESH_PTS_OVERRIDE 取代(保留供回退路径使用)
    MESH_EXTRA_PTS = []
    # ---- #2 证书核(提前停止测量): 已实现并**实测否决**(默认关)。
    #      27 点最小覆盖设计太紧: 贪心集合覆盖得到的"能覆盖圆盘的最小子集"= 全部 37 个三角形
    #      (每个三角形都独立承担覆盖某段边界), 故无提前停止余量; 配 core-first 排序后病理集
    #      10 类均值 9 869 s vs 关闭时 9 849 s(反而 +0.2%), 低于 0.5% 门槛 -> 维持关闭。
    #      True 时只对证伪该子集即可排除频道(证书语义不变: 子集同样覆盖圆盘)。
    CERT_CORE = False
    # ---- #3 巡回求解器: "2opt" = 2-opt(默认); "or3opt" = 2-opt + Or-opt 段重定位。
    #      已实现并**实测否决**(默认关): 覆盖巡回 24 120 m 时 Or-opt 找不到任何改进
    #      (该巡回已近"面积/边长+直径"下界), 清除巡回 n=10~16 时 2-opt 已足够 ->
    #      病理集 10 类均值差异 +0.2%(默认反而慢), 低于门槛 -> 维持 "2opt"。----
    TSP_MODE = "2opt"
    # ---- MEC 就绪冻结: 一旦 Ω_c 的最小覆盖圆半径 <= R_CLEAR, 真源必在该圆内
    #      (G_c ∈ Ω_c ⊆ B(z_c, r_c), r_c <= 20 m), 清除点已被**确定性认证**; 此后继续测量
    #      最多只是进一步缩小区域, 不会改变"该点可保证清除"的结论 —— 故进入 ready 状态、
    #      不再测量该频道。不做绕路: 仍按原 δ 顺路规则, 否则并入扫描后队列;
    #      清除失败仍由归航与恢复闭环兜底。默认关(需配对实验验证后启用)。----
    #      MEC_FREEZE 默认 **开启**(经固定误差场配对实验采纳:
    #      50%定向 −590.3 s/CI[−611,−570]/100% 案例变快, 100%定向 −500.1 s/CI[−517,−483];
    #      移动量与清除失败次数不变; 病理集 2000 例 0 漏清且 10 类全部变快)。
    MEC_FREEZE = True
    # ---- 顺路 LS 试清失败后**暂缓归航**(默认关, 待配对实验验证) ----
    # 动机(12 局失效结构): 首次 LS 试清成功率 92.3%, 但 21 次失败中 18 次来自失败后的恢复链
    # (8/15 m 邻域试探占 16 次), 3 个困难源的归航恢复共 1241.6 s; 真正昂贵的是"沿错误示向的
    # 长距离归航 + 附加检测", 而不是那次只花 3 s 的失败清除。
    # 开启后: 顺路失败 -> 只标记本轮顺路失败 -> 继续既定网格扫描(移动本来就要做) ->
    # 后续"免费"获得的示向使 Ω_c 继续收缩(Ω_c←Ω_c∩W) -> 扫描后队列重新定位清除;
    # 仍失败才走原有归航+邻域兜底。不取消归航, 只是把它推迟到信息更充分时。
    #      DEFER_ONWAY_HOMING 默认 **开启**(经固定误差场配对实验 + 病理门禁采纳:
    #      50%定向 −64.4 s/CI[−77,−51]、100%定向 −88.5 s/CI[−106,−71]; 清除失败/例 1.2→0.6、
    #      2.1→0.9; 最大时间 −478/−186 s; 病理集 2000 例 0 漏清)。
    DEFER_ONWAY_HOMING = True
    # ---- 归航示向顺序: "acq" = 按获取顺序(原行为); "err" = 按(横向误差上界 d̂·sin1°, 移动代价)
    #      字典序最小优先。多示向时先拿到的未必最可靠(边界源尤其如此)。默认先保持 acq,
    #      经固定误差场配对实验验证后再决定是否切换。----
    BEARING_ORDER = "acq"
    # ---- 邻域试探模式(归航点失败后的兜底): "rings"=8/15 m 两圈各 6 点(原行为);
    #      "normal2"=沿示向法线两侧各 1 点(偏移取横向误差上界 d̂·sin1°, 限幅 6~26 m);
    #      "none"=不做邻域, 直接换下一条示向。NORMAL2_STOP=True 时"两侧2点"失败即结束本次归航。
    #      依据: 12 局在环日志中 21 次失败里 16 次来自 12 点圆周试探(76.2%)。----
    NEIGHBOR_MODE = "rings"
    NORMAL2_STOP = False
    # ---- 访问顺序覆盖(P4-B 前缀观测质量实验): None = 内置 NN+2-opt;
    #      给定网格编号的排列则按该顺序访问(必须是全网格点的排列)。----
    ORDER_OVERRIDE = None
    _core_cache = None
    _core_cache_key = None

    def __init__(self, client):
        self.c = client
        self.state = {ch: None for ch in range(1, N_CH+1)}   # None/found/excluded/cleared
        self.bearings = {ch: [] for ch in range(1, N_CH+1)}
        self.near_pos = {ch: None for ch in range(1, N_CH+1)}
        self.ns_at = {ch: set() for ch in range(1, N_CH+1)}   # no_signal 的点索引
        self.ns_at_pos = {ch: [] for ch in range(1, N_CH+1)}  # no_signal 的点坐标(负信息用)
        self.cleared_count = 0
        self.exit_status = "completed"
        self.unresolved_kind = {"found_uncleared": [], "uncertified": []}
        self.recovery = dict(rounds=0, measures=0, clears=0, unresolved=[])
        self.onway_failed = set()   # 顺路清除失败过的频道: 不再顺路重试, 留给扫描后批量清除
        self.locate_diag = {}       # 频道 -> 最近一次定位诊断(方式/Ω半径/交会角)
        self.supp_skipped = set()   # 因补测距离上限被跳过、改走二分归航的频道
        self.ready_pos = {ch: None for ch in range(1, N_CH+1)}   # MEC 冻结后的认证清除点
        self.n_defer = 0            # 顺路失败后暂缓归航的次数(诊断)
        ov = getattr(Problem4Robot, "MESH_PTS_OVERRIDE", None)
        if ov:
            self.pts = [tuple(p) for p in ov]          # 最小覆盖设计(格点块, 无空洞)
        else:
            self.pts = tri_mesh(a=MESH_A, margin=MESH_MARGIN) + [
                tuple(p) for p in getattr(Problem4Robot, "MESH_EXTRA_PTS", []) or []]
        self.tris = build_triangles(self.pts, a=MESH_A)
        self.cover_tris = covering_triangles(self.tris, self.pts)
        # #2: 证书集合 = 能覆盖圆盘的**最小三角形子集**(若开启), 否则全部覆盖三角形
        if getattr(Problem4Robot, "CERT_CORE", False):
            key = tuple(self.pts)
            if Problem4Robot._core_cache_key != key:
                Problem4Robot._core_cache = self._build_cert_core()
                Problem4Robot._core_cache_key = key
            self.cert_tris = Problem4Robot._core_cache or self.cover_tris
        else:
            self.cert_tris = self.cover_tris

    def _build_cert_core(self):
        """贪心集合覆盖: 求"并集覆盖圆盘"的最小三角形子集(仅用于提前停止, 不改证书语义)。

        确定性(固定种子 + 固定遍历序), 且按点集缓存(arms 换网格时按点集键失效)。
        """
        def in_tri(p, a, b, c):
            def cr(o, u, v):
                return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])
            d1, d2, d3 = cr(a, b, p), cr(b, c, p), cr(c, a, p)
            return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and
                        ((d1 > 0) or (d2 > 0) or (d3 > 0)))
        # 确定性黄金角螺旋采样(不依赖 RNG, 保证证书核可复现)
        ga = math.pi*(3.0 - math.sqrt(5.0))
        smp = []
        for i in range(2000):
            r = R_AREA*math.sqrt((i + 0.5)/2000.0)
            a2 = i*ga
            smp.append((r*math.cos(a2), r*math.sin(a2)))
        for i in range(2000):
            r = 1600.0 + (R_AREA - 1600.0)*((i + 0.5)/2000.0)
            a2 = (i*ga*1.7) % (2*math.pi)
            smp.append((r*math.cos(a2), r*math.sin(a2)))
        todo = set(range(len(smp)))
        core = []
        while todo and len(core) < len(self.cover_tris):
            best, best_cnt = None, 0
            for t in self.cover_tris:
                A, B, C = self.pts[t[0]], self.pts[t[1]], self.pts[t[2]]
                cnt = 0
                for i in todo:
                    if in_tri(smp[i], A, B, C):
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt, best = cnt, t
            if best is None or best_cnt == 0:
                break
            core.append(best)
            A, B, C = self.pts[best[0]], self.pts[best[1]], self.pts[best[2]]
            todo = {i for i in todo if not in_tri(smp[i], A, B, C)}
        return core

    def log(self, *a):
        print("[robot4]", *a, flush=True)

    # ---- 阶段标签 + 诊断记录(只记录, 不改变决策) ----
    def _phase(self, name):
        try:
            self.c.phase = name
        except Exception:
            pass

    @staticmethod
    def _diag_list(client, name):
        v = getattr(client, name, None)
        if v is None:
            v = []
            try:
                setattr(client, name, v)
            except Exception:
                pass
        return v

    def _note_locate(self, rec):
        self.locate_diag[rec["ch"]] = rec
        self._diag_list(self.c, "locate_history").append(rec)

    def _note_clear(self, ch, x, y, src, omega_r, cross_ang, result):
        """记录一次清除尝试的定位来源与当时 Ω 半径(供诊断与配对实验)。"""
        self._diag_list(self.c, "clear_diag").append({
            "ch": ch, "point": [round(x, 1), round(y, 1)], "src": src,
            "omega_radius_m": omega_r, "cross_angle_deg": cross_ang,
            "phase": getattr(self.c, "phase", None), "result": result,
        })

    def _note_supp(self, ch, dist_m, n_cands, action):
        """记录一次补测决策/执行: 最近补测点距离、候选数、动作(measure/skip/exec)。"""
        self._diag_list(self.c, "supp_diag").append({
            "ch": ch, "nearest_m": round(dist_m, 1), "n_candidates": n_cands,
            "action": action, "cap_m": self.SUPP_MAX_DIST,
            "phase": getattr(self.c, "phase", None),
        })

    # ---- 覆盖率预估(离线) ----
    def mesh_stats(self):
        d = 0.0
        for i in range(1, len(self.pts)):
            d += math.hypot(self.pts[i][0]-self.pts[i-1][0], self.pts[i][1]-self.pts[i-1][1])
        return len(self.pts), len(self.tris), len(self.cover_tris)

    # ---- 路径: 最近邻 + 2-opt(任务级, 携带网格编号; 含开放路径尾段反转) ----
    @staticmethod
    def _two_opt_tasks(tasks, start):
        """对 [(key, x, y), ...] 做开放路径 2-opt。start 只作锚点, 不属于待访问集合。

        j == n 的尾段反转必须纳入: 它把边 (v_{i-1},v_i) 换成 (v_{i-1},v_n),
        否则开放路径永远无法倒转末段(实测 165/1000 场景还能再省)。
        """
        path = [(-1, start[0], start[1])] + [tuple(t) for t in tasks]
        n = len(path) - 1

        def dd(a, b):
            return math.hypot(a[1]-b[1], a[2]-b[2])
        imp = True
        while imp:
            imp = False
            for i in range(1, n):
                for j in range(i+1, n+1):
                    if j < n:
                        old = dd(path[i-1], path[i]) + dd(path[j], path[j+1])
                        new = dd(path[i-1], path[j]) + dd(path[i], path[j+1])
                    else:                       # 尾段反转
                        old = dd(path[i-1], path[i])
                        new = dd(path[i-1], path[n])
                    if new < old - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]; imp = True
        if getattr(Problem4Robot, "TSP_MODE", "2opt") == "or3opt":
            # #3 Or-opt: 把长度 1~3 的连续段整段重定位到更省的位置(2-opt 无法做到)
            improved = True
            while improved:
                improved = False
                for seg in (1, 2, 3):
                    for i in range(1, n+2-seg):
                        j = i + seg - 1
                        if j > n:
                            continue
                        seg_pts = path[i:j+1]
                        post = path[j+1] if j+1 <= n else None
                        rm = dd(path[i-1], seg_pts[0]) + (dd(seg_pts[-1], post) if post else 0.0)
                        keep = dd(path[i-1], post) if post else 0.0
                        g_rm = rm - keep
                        if g_rm <= 1e-9:
                            continue
                        for k in range(0, n+1):
                            if i-1 <= k <= j:
                                continue
                            nxt = path[k+1] if k+1 <= n else None
                            add = dd(path[k], seg_pts[0]) + (dd(seg_pts[-1], nxt) if nxt else 0.0)
                            old_e = dd(path[k], nxt) if nxt else 0.0
                            if add - old_e < g_rm - 1e-9:
                                newp = path[:]
                                del newp[i:j+1]
                                ins = k+1 if k < i else k+1-seg
                                newp[ins:ins] = seg_pts
                                path = newp
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return path[1:]

    def _order_points(self):
        """网格点访问序列 [(mesh_index, x, y), ...]。

        每点恰好访问一次(锚点原点不重复作为待访问点); 序列元素携带**真实网格编号**,
        三角形证书必须用该编号而非访问次序。P4-B 实验可用 ORDER_OVERRIDE 指定顺序。
        """
        ov = getattr(Problem4Robot, "ORDER_OVERRIDE", None)
        if ov is not None:
            ids = list(ov)
            assert sorted(ids) == list(range(len(self.pts))), "ORDER_OVERRIDE 必须是全点排列"
            return [(i, self.pts[i][0], self.pts[i][1]) for i in ids]
        pts = self.pts
        start = (0.0, 0.0)
        core_idx = set()
        for t in (getattr(self, "cert_tris", None) or []):
            core_idx.update(t)
        if (getattr(Problem4Robot, "CERT_CORE", False) and core_idx
                and len(core_idx) < len(pts)):
            # #2: 证书核顶点先访问(其三角形已覆盖圆盘 => 无源频道可提前证伪、提前停止测量),
            #     两组各自做 NN+2-opt/Or-opt, 拼接后仍是"每点恰好一次"的置换。
            def nn_order(idx, cur):
                out = []
                unv = set(idx)
                while unv:
                    k = min(unv, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
                    out.append(k); cur = pts[k]; unv.discard(k)
                return out
            ord_core = nn_order(core_idx, start)
            seq_core = self._two_opt_tasks([(k, pts[k][0], pts[k][1]) for k in ord_core], start)
            last = (seq_core[-1][1], seq_core[-1][2]) if seq_core else start
            ord_rest = nn_order(set(range(len(pts))) - core_idx, last)
            seq_rest = self._two_opt_tasks([(k, pts[k][0], pts[k][1]) for k in ord_rest], last)
            return seq_core + seq_rest
        order = []; unv = set(range(len(pts))); cur = start
        while unv:
            k = min(unv, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
            order.append(k); cur = pts[k]; unv.discard(k)
        tasks = [(k, pts[k][0], pts[k][1]) for k in order]
        return self._two_opt_tasks(tasks, start)

    # ---- 快速定位(不移动); 同时记录定位方式与 Ω(可行域最小覆盖圆)半径 ----
    def _locate_quick(self, ch, tag=""):
        """方式 'mec': Ω 最小覆盖圆半径 <= R_CLEAR; 'ls': 退化时改用最大交会角两示向交会。"""
        dirs = list(self.bearings[ch])
        rec = {"ch": ch, "tag": tag, "n_dirs": len(dirs), "method": None,
               "omega_radius_m": None, "cross_angle_deg": None, "point": None}
        est = None
        if len(dirs) >= 2:
            poly = feasible_region(dirs)
            if len(poly) >= 3:
                center, radius = minimal_enclosing_circle(poly)
                rec["omega_radius_m"] = round(radius, 1)
                if radius <= R_CLEAR:
                    rec["method"] = "mec"
                    est = center
            if est is None:
                bi = bj = 0; ba = -1
                for i in range(len(dirs)):
                    for j in range(i+1, len(dirs)):
                        a = crossing_angle(dirs[i][1], dirs[j][1])
                        if a > ba:
                            ba, bi, bj = a, i, j
                rec["cross_angle_deg"] = round(ba, 1)
                if ba >= 30.0:
                    rec["method"] = "ls"
                    est = bearing_intersection([dirs[bi][0], dirs[bj][0]],
                                               [dirs[bi][1], dirs[bj][1]])
                    if est is not None and self.USE_PSO:
                        # 模块3: 用改进 PSO 在连续非凸空间精化位置
                        est = self._pso_refine(ch, dirs, est)
        if est is not None:
            rec["point"] = [round(est[0], 1), round(est[1], 1)]
        self._note_locate(rec)
        if tag:
            self.log(f"定位[{tag}] 频道 {ch}: 方式={rec['method']} "
                     f"Ω半径={rec['omega_radius_m']}m 交会角={rec['cross_angle_deg']}° "
                     f"示向数={rec['n_dirs']}")
        return est

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
                self.near_pos[ch] = (px, py)
                return (px, py)
            if res == "direction":
                if svd is not None and all(math.hypot(px-q[0], py-q[1]) > 1.0
                                           for q, _ in self.bearings[ch]):
                    self.bearings[ch].append(((px, py), svd))   # 回灌有效示向
                if angle_diff(svd, th0) > 90.0:
                    hi = mid
                else:
                    lo = mid
            else:
                hi = mid
            if hi - lo < 4.0:
                return (x0 + mid*ux, y0 + mid*uy)
        return (x0 + (lo+hi)/2.0*ux, y0 + (lo+hi)/2.0*uy)

    # ---- 就近精定位(清除失败兜底)。返回 True 仅当模拟器确实返回 success ----
    def _homing_clear(self, ch, x, y, phase="homing", src="homing",
                      omega_r=None, cross_ang=None, tried=False):
        """tried=True 表示调用者已在 (x,y) 清除过一次并失败, 不再原地重复请求。

        邻域试探圈半径由类属性 NEIGHBOR_RINGS 给定(默认 (8,15) 两圈各 6 点);
        NEIGHBOR_RINGS = () 表示取消邻域试探, 归航点失败后直接换下一条示向。
        每进入一次归航记一条 episode 诊断(来源/试过几条示向/由谁清除/开销)。
        """
        self._phase(phase)
        snap0 = self._snapshot()
        ep = {"ch": ch, "src": src, "phase": phase, "tried_first": bool(tried),
              "n_bearings": len(self.bearings[ch]), "bearings_tried": 0,
              "rings": list(self.NEIGHBOR_RINGS), "cleared_by": None,
              "cleared_at_bearing": None}
        self._diag_list(self.c, "homing_diag").append(ep)

        def done(how, idx):
            ep["cleared_by"] = how
            ep["cleared_at_bearing"] = idx
            self._close_episode(ep, snap0)
            return True
        if not tried:
            okc, rc = self.c.clear(x, y, ch)
            self._note_clear(ch, x, y, src, omega_r, cross_ang, rc if okc else "rejected")
            if okc and rc == "success":
                self.state[ch] = "cleared"; self.cleared_count += 1
                return done("原位", -1)
        # 对每条已有示向依次做二分归航(边界源可能只有个别方位稳健)
        bears = list(self.bearings[ch])
        if getattr(Problem4Robot, "BEARING_ORDER", "acq") == "err" and len(bears) > 1:
            # #2 示向排序: 角度误差 ±1° 在距离 d̂ 处造成横向误差上界 d̂·sin1°; 故优先选
            # (横向误差, 移动代价) 字典序最小的示向, 而不是"先拿到的那条"。
            # 依据(12 局在环日志): s08 按获取顺序第一条示向归航失败, 换第二条才成功,
            # 单个源因此产生 14 次失败、归航移动 2573 m、耗时 649.6 s。
            est = self._locate_quick(ch, tag="示向排序")
            if est is not None:
                cx, cy = self.c.position
                bears.sort(key=lambda it: (
                    math.hypot(est[0]-it[0][0], est[1]-it[0][1])*math.sin(BEARING_ERR_DEG*DEG),
                    math.hypot(it[0][0]-cx, it[0][1]-cy)))
        for bi, (P, th) in enumerate(bears):
            bt = self._binary_homing(ch, P, th)
            if bt is None:
                continue
            ep["bearings_tried"] += 1
            if tried and math.hypot(bt[0]-x, bt[1]-y) < 1e-6:
                continue                      # 归航点与失败点重合: 跳过重复请求
            okc, rc = self.c.clear(bt[0], bt[1], ch)
            self._note_clear(ch, bt[0], bt[1], src+"|归航", omega_r, cross_ang,
                             rc if okc else "rejected")
            if okc and rc == "success":
                self.state[ch] = "cleared"; self.cleared_count += 1
                return done("归航点" if bi == 0 else "后续示向", bi)
            # 邻域试探: 三种模式(四臂比较用)
            mode = getattr(Problem4Robot, "NEIGHBOR_MODE", "rings")
            if mode == "none":
                continue                     # 不试邻域, 立即换下一条示向
            if mode == "normal2":
                # 角度误差的主要分量是"横向"而非各向同性: 沿示向法线两侧各试一点,
                # 偏移量取横向误差上界 d̂·sin1°(限幅 6~26 m; 最远 1500 m 时约 26.2 m)
                ux, uy = math.cos(th*DEG), math.sin(th*DEG)
                nx, ny = -uy, ux
                dhat = math.hypot(bt[0]-P[0], bt[1]-P[1])
                off = min(26.0, max(6.0, dhat*math.sin(BEARING_ERR_DEG*DEG)))
                for sgn in (1.0, -1.0):
                    q = (bt[0] + sgn*off*nx, bt[1] + sgn*off*ny)
                    ok2, rc2 = self.c.clear(q[0], q[1], ch)
                    self._note_clear(ch, q[0], q[1], f"{src}|法线{off:.0f}m", omega_r,
                                     cross_ang, rc2 if ok2 else "rejected")
                    if ok2 and rc2 == "success":
                        self.state[ch] = "cleared"; self.cleared_count += 1
                        return done(f"法线{off:.0f}m", bi)
                if getattr(Problem4Robot, "NORMAL2_STOP", False):
                    break                    # 两侧2点失败即结束本次归航(留待后续恢复链)
                continue                     # 否则换下一条示向
            for rad in self.NEIGHBOR_RINGS:
                for k in range(6):
                    a = k * 60 * DEG
                    q = (bt[0] + rad*math.cos(a), bt[1] + rad*math.sin(a))
                    ok2, rc2 = self.c.clear(q[0], q[1], ch)
                    self._note_clear(ch, q[0], q[1], f"{src}|邻域{rad:.0f}m", omega_r,
                                     cross_ang, rc2 if ok2 else "rejected")
                    if ok2 and rc2 == "success":
                        self.state[ch] = "cleared"; self.cleared_count += 1
                        return done(f"邻域{rad:.0f}m", bi)
        self._close_episode(ep, snap0)
        return False

    # ---- 归航 episode 开销(供消融统计; 真实/离线客户端均可) ----
    _SNAP_KEYS = ("dist", "n_measure", "n_clear", "n_clear_ok", "n_switch")

    def _snapshot(self):
        return {k: getattr(self.c, k, None) for k in self._SNAP_KEYS}

    def _close_episode(self, ep, snap0):
        snap1 = self._snapshot()
        ep["cost"] = {k: (snap1[k] - snap0[k])
                      if isinstance(snap0.get(k), (int, float))
                      and isinstance(snap1.get(k), (int, float)) else None
                      for k in self._SNAP_KEYS}
        c = ep["cost"]
        if c.get("dist") is not None:
            ep["episode_time_s"] = round(
                c["dist"]/5.0 + (c.get("n_measure") or 0)*5.0
                + (c.get("n_switch") or 0)*1.0 + (c.get("n_clear_ok") or 0)*5.0
                + max((c.get("n_clear") or 0) - (c.get("n_clear_ok") or 0), 0)*3.0, 1)
            ep["episode_moves_m"] = round(c["dist"], 1)

    # ---- 主流程 ----
    def run(self):
        c = self.c
        self.log("调用 /enter ...")
        c.enter()
        npts, ntri, ncov = self.mesh_stats()
        self.log(f"进入成功. 三角网格: {npts} 点 / {ntri} 三角形 / 需证伪 {ncov} 个")

        seq = self._order_points()
        # 覆盖路径自检: 31 个网格点必须恰好各访问一次(锚点不重复计入)
        _seen = sorted(mi for mi, _, _ in seq)
        path_ok = (_seen == list(range(len(self.pts))))
        self.log(f"覆盖路径 {len(seq)} 个网格点(网格共 {len(self.pts)} 个): "
                 f"{'每点恰好一次' if path_ok else '!! 编号重复/缺失 ' + str(_seen)}")

        # 保证层: 依序访问网格点, 蛇形扫描
        self._phase("mesh_scan")
        for i, (mi, px, py) in enumerate(seq):
            order = list(range(1, N_CH+1)) if i % 2 == 0 else list(range(N_CH, 0, -1))
            for ch in order:
                if self.state[ch] in ("excluded", "cleared", "ready"):
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
                    self.ns_at[ch].add(mi)          # 真实网格编号(不是访问序号)
                    self.ns_at_pos[ch].append((px, py))
            # 三角形证书: 顶点均 no_signal 的三角形被证伪
            for ch in range(1, N_CH+1):
                if self.state[ch] in ("excluded", "cleared"):
                    continue
                if self.state[ch] is None:
                    if all(all(v in self.ns_at[ch] for v in t) for t in self.cert_tris):
                        self.state[ch] = "excluded"
            # MEC 就绪冻结: 一旦认证通过即固定清除点并停止该频道的后续测量(不强制绕路)
            if getattr(Problem4Robot, "MEC_FREEZE", False):
                for ch in range(1, N_CH+1):
                    if self.state[ch] != "found" or self.ready_pos[ch] is not None:
                        continue
                    if self.near_pos[ch] is not None:
                        self.ready_pos[ch] = self.near_pos[ch]
                        self.state[ch] = "ready"
                        continue
                    pt = self._locate_quick(ch, tag="MEC冻结")
                    if (pt is not None
                            and self.locate_diag.get(ch, {}).get("method") == "mec"):
                        self.ready_pos[ch] = (pt[0], pt[1])
                        self.state[ch] = "ready"
            # 受限顺路清除
            if ON_WAY_DELTA is not None and i + 1 < len(seq):
                nxt = (seq[i+1][1], seq[i+1][2])
                self._phase("on_way")
                for c2 in range(1, N_CH+1):
                    if self.state[c2] not in ("found", "ready") or c2 in self.onway_failed:
                        continue
                    if self.state[c2] == "ready":
                        Q, qsrc, qr, qa = self.ready_pos[c2], "mec_frozen", None, None
                    elif self.near_pos[c2]:
                        Q, qsrc, qr, qa = self.near_pos[c2], "near", None, None
                    else:
                        Q = self._locate_quick(c2, tag=f"顺路@{i}")
                        dg = self.locate_diag.get(c2, {})
                        qsrc = dg.get("method") or "none"
                        qr, qa = dg.get("omega_radius_m"), dg.get("cross_angle_deg")
                        if Q is None:
                            continue
                    cur = c.position
                    dL = (math.hypot(Q[0]-cur[0], Q[1]-cur[1])
                          + math.hypot(Q[0]-nxt[0], Q[1]-nxt[1])
                          - math.hypot(cur[0]-nxt[0], cur[1]-nxt[1]))
                    if dL <= ON_WAY_DELTA:
                        ok, res = c.clear(Q[0], Q[1], c2)
                        self._note_clear(c2, Q[0], Q[1], "on_way:"+qsrc, qr, qa,
                                         res if ok else "rejected")
                        if ok and res == "success":
                            self.state[c2] = "cleared"; self.cleared_count += 1
                            self.log(f"顺路清除: 频道 {c2}  来源={qsrc} [{self.cleared_count}]")
                        else:
                            # 顺路失败: 记入冷却, 不再在后续网格点反复顺路重试
                            self.onway_failed.add(c2)
                            if getattr(Problem4Robot, "DEFER_ONWAY_HOMING", False):
                                # 暂缓归航: 继续扫描, 让后续免费示向先改善定位(见类属性说明)
                                self.n_defer += 1
                                self.log(f"顺路失败暂缓归航: 频道 {c2} (留待扫描后重新定位)")
                            else:
                                self._homing_clear(c2, Q[0], Q[1], phase="on_way_homing",
                                                   src="on_way:"+qsrc, omega_r=qr,
                                                   cross_ang=qa, tried=True)
                                self._phase("on_way")   # 恢复阶段标签
                                if self.state[c2] == "cleared":
                                    self.onway_failed.discard(c2)
                self._phase("mesh_scan")

        self.log(f"扫描完成: 已发现 {sum(1 for s in self.state.values() if s=='found')}, "
                 f"已排除 {sum(1 for s in self.state.values() if s=='excluded')}")

        # 状态层: 快速定位分类 (任务元组: ch, x, y, src, Ω半径, 交会角)
        clear_tasks = []; supp_tasks = []
        for ch in range(1, N_CH+1):
            if self.state[ch] not in ("found", "ready"):
                continue
            if self.state[ch] == "ready" and self.ready_pos[ch] is not None:
                # MEC 冻结频道: 直接使用已认证的清除点, 无需再定位/补测
                clear_tasks.append((ch, self.ready_pos[ch][0], self.ready_pos[ch][1],
                                    "mec_frozen", R_CLEAR, self.locate_diag.get(ch, {}).get("cross_angle_deg")))
                continue
            if self.near_pos[ch] is not None:
                clear_tasks.append((ch, self.near_pos[ch][0], self.near_pos[ch][1],
                                    "near", None, None))
                continue
            pt = self._locate_quick(ch, tag="预筛")
            dg = self.locate_diag.get(ch, {})
            if pt is not None:
                clear_tasks.append((ch, pt[0], pt[1], dg.get("method") or "none",
                                    dg.get("omega_radius_m"), dg.get("cross_angle_deg")))
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
                    d0 = math.hypot(cands[0][0]-c.position[0], cands[0][1]-c.position[1])
                    cap = self.SUPP_MAX_DIST
                    if cap is not None and d0 > cap:
                        # 远距离补测: 跳过, 直接走已有的沿首示向二分归航(不新增算法)
                        self._note_supp(ch, d0, len(cands), "skip")
                        self.supp_skipped.add(ch)
                        self._phase("homing")
                        bt = self._binary_homing(ch, self.bearings[ch][0][0],
                                                 self.bearings[ch][0][1])
                        if bt is not None:
                            clear_tasks.append((ch, bt[0], bt[1], "binary:cap", None, None))
                        self._phase("mesh_scan")
                    else:
                        self._note_supp(ch, d0, len(cands), "measure")
                        supp_tasks.append((ch, cands[0][0], cands[0][1], cands))

        # 批量补测(方位鲁棒: 第一个方位失败则换下一个)。任务按索引排序, 不按坐标查表
        if supp_tasks:
            self.log(f"补测任务 {len(supp_tasks)} 个(方位鲁棒)")
            self._phase("supplement")
            spts = [(x, y) for _, x, y, _ in supp_tasks]
            order = self._optimal_open_path(spts, tuple(c.position))
            for idx in order:
                ch, _, _, cands = supp_tasks[idx]
                for k, (qx, qy) in enumerate(cands):
                    if k == 0:
                        self._note_supp(ch, math.hypot(qx-c.position[0], qy-c.position[1]),
                                        len(cands), "exec")
                    ok, res, svd = c.measure(qx, qy, ch)
                    if ok and res == "near":
                        clear_tasks.append((ch, qx, qy, "near", None, None)); break
                    if ok and res == "direction":
                        self.bearings[ch].append(((qx, qy), svd)); break
                pt = self._locate_quick(ch, tag="补测后")
                dg = self.locate_diag.get(ch, {})
                if pt is not None:
                    clear_tasks.append((ch, pt[0], pt[1], dg.get("method") or "none",
                                        dg.get("omega_radius_m"),
                                        dg.get("cross_angle_deg")))
                else:
                    # 补测失败(全在盲区)也必须兜底: 沿首示向二分归航
                    self._phase("homing")
                    bt = self._binary_homing(ch, self.bearings[ch][0][0], self.bearings[ch][0][1])
                    if bt is not None:
                        clear_tasks.append((ch, bt[0], bt[1], "binary", None, None))
                    self._phase("supplement")

        # 清除(任务级最近邻 + 2-opt, 任务始终携带频道编号)
        self.log(f"清除任务队列 {len(clear_tasks)} 个")
        self._phase("queue_clear")
        ordered = []; unv = set(range(len(clear_tasks))); cur = c.position
        while unv:
            k = min(unv, key=lambda i: math.hypot(clear_tasks[i][1]-cur[0],
                                                  clear_tasks[i][2]-cur[1]))
            ordered.append(clear_tasks[k])
            cur = (clear_tasks[k][1], clear_tasks[k][2]); unv.discard(k)
        for (ch, x, y, src, omr, cra) in self._two_opt_tasks(ordered, c.position):
            ok, res = c.clear(x, y, ch)
            self._note_clear(ch, x, y, "queue:"+str(src), omr, cra, res if ok else "rejected")
            if ok and res == "success":
                self.state[ch] = "cleared"; self.cleared_count += 1
                self.log(f"清除成功: 频道 {ch} 来源={src} Ω半径={omr}m [{self.cleared_count}]")
            else:
                self.log(f"清除未发现: 频道 {ch} 来源={src} Ω半径={omr}m, 就近精定位")
                self._homing_clear(ch, x, y, phase="queue_homing", src="queue:"+str(src),
                                   omega_r=omr, cross_ang=cra, tried=True)
                self._phase("queue_clear")

        # 恢复闭环(正确性)
        self._phase("recovery")
        self._recovery()
        # 退出前校验
        unresolved = [ch for ch in range(1, N_CH+1)
                      if self.state[ch] not in ("cleared", "excluded")]
        # 语义区分: "发现过信号但没清除"= 真漏清(非正常完成, 返回非零码);
        #           "始终无任何证据"(state 仍为 None)= 空白证书未补完, 对"清除全部源"无影响。
        found_uncleared = [ch for ch in range(1, N_CH+1)
                           if self.state[ch] not in ("cleared", "excluded")
                           and (self.bearings[ch] or self.near_pos[ch] is not None)]
        uncertified = [ch for ch in range(1, N_CH+1) if self.state[ch] is None]
        self.unresolved_kind = {"found_uncleared": found_uncleared, "uncertified": uncertified}
        self.exit_status = "completed" if not found_uncleared else "incomplete"
        if unresolved:
            self.log(f"警告: 仍有未解决频道 {unresolved}(可能未找到/未清除), "
                     f"将在退出摘要中记录")
        else:
            self.log("20 个频道均已了结(已清除或已排除)")

        # 回填结构化元信息(写入 JSON 汇总)
        c.meta = {
            "mesh_points": len(self.pts),
            "mesh_triangles": len(self.tris),
            "cover_triangles": len(self.cover_tris),
            "path_points": len(seq),
            "certificate_index_ok": bool(path_ok),
            "cleared_count": self.cleared_count,
            "found_channels": sum(1 for s in self.state.values()
                                  if s in ("found", "ready")),
            "excluded_channels": sum(1 for s in self.state.values() if s == "excluded"),
            "unresolved_channels": len(unresolved),
            "unresolved_list": unresolved,
            "exit_status": self.exit_status,
            "unresolved_kind": self.unresolved_kind,
            "recovery": self.recovery,
            "supp_skipped_channels": sorted(self.supp_skipped),
            "supp_skipped_recovered": sum(1 for ch in self.supp_skipped
                                          if self.state[ch] == "cleared"),
            "channels": {str(ch): {"state": self.state[ch],
                                   "bearings": len(self.bearings[ch]),
                                   "near": self.near_pos[ch] is not None,
                                   "no_signal_points": len(self.ns_at[ch]),
                                   "locate": self.locate_diag.get(ch)}
                         for ch in range(1, N_CH+1)},
        }
        # 模块5: 清除后多方向复核 —— 走一遍圆外见证点, 在那逐频道复核"已排除"结论
        if self.DO_VERIFY:
            self._phase("verify")
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
                                    self._homing_clear(ch, Q[0], Q[1], phase="verify_homing",
                                                       src="verify", tried=True)

        self._phase("exit")
        self.log("调用 /exit ...")
        c.exit()
        self.log(f"结束, 清除 {self.cleared_count} 个干扰源")
        return self.cleared_count

    # ---- 恢复闭环(正确性): 未解决频道有限预算恢复 ----
    RECOV_MAX_ROUNDS = 2
    RECOV_MAX_MEASURES = 40
    RECOV_MAX_CLEARS = 20

    def _recovery(self):
        c = self.c
        seq = self._order_points()
        n_meas = n_clear = rounds = 0

        def unresolved():
            return [ch for ch in range(1, N_CH+1)
                    if self.state[ch] not in ("cleared", "excluded")]

        while (unresolved() and rounds < self.RECOV_MAX_ROUNDS
               and n_meas < self.RECOV_MAX_MEASURES and n_clear < self.RECOV_MAX_CLEARS):
            rounds += 1
            self._phase("recovery")
            for ch in list(unresolved()):
                if self.state[ch] in ("cleared", "excluded"):
                    continue
                if self.state[ch] is None:
                    # 补齐缺失的网格观测(被拒绝的请求不算有效观测; 不得据此推进证书)
                    for mi, px, py in seq:
                        if n_meas >= self.RECOV_MAX_MEASURES:
                            break
                        if mi in self.ns_at[ch]:
                            continue
                        ok, res, svd = c.measure(px, py, ch)
                        n_meas += 1
                        if not ok:
                            continue
                        if res == "direction":
                            self.state[ch] = "found"
                            self.bearings[ch].append(((px, py), svd))
                            break
                        if res == "near":
                            self.state[ch] = "found"
                            self.near_pos[ch] = (px, py)
                            break
                        self.ns_at[ch].add(mi)
                        self.ns_at_pos[ch].append((px, py))
                    if (self.state[ch] is None
                            and all(all(v in self.ns_at[ch] for v in t)
                                    for t in self.cert_tris)):
                        self.state[ch] = "excluded"
                    continue
                pt = self._locate_quick(ch, tag="恢复")
                if pt is None:
                    cands = self._supplement_points(ch, c.position)
                    for (qx, qy) in (cands or []):
                        if n_meas >= self.RECOV_MAX_MEASURES:
                            break
                        ok, res, svd = c.measure(qx, qy, ch)
                        n_meas += 1
                        if ok and res == "near":
                            self.near_pos[ch] = (qx, qy)
                            break
                        if ok and res == "direction":
                            self.bearings[ch].append(((qx, qy), svd))
                            break
                    pt = self._locate_quick(ch, tag="恢复补测")
                if pt is None or n_clear >= self.RECOV_MAX_CLEARS:
                    continue
                ok, res = c.clear(pt[0], pt[1], ch)
                n_clear += 1
                dg = self.locate_diag.get(ch, {})
                self._note_clear(ch, pt[0], pt[1], "recovery:" + str(dg.get("method")),
                                 dg.get("omega_radius_m"), dg.get("cross_angle_deg"),
                                 res if ok else "rejected")
                if ok and res == "success":
                    self.state[ch] = "cleared"
                    self.cleared_count += 1
                    self.log(f"恢复清除: 频道 {ch} @ ({pt[0]:.0f},{pt[1]:.0f}) "
                             f"[{self.cleared_count}]")
                else:
                    self._homing_clear(ch, pt[0], pt[1], phase="recovery_homing",
                                       src="recovery", tried=True)
                    self._phase("recovery")
        self.recovery = dict(rounds=rounds, measures=n_meas, clears=n_clear,
                             unresolved=unresolved())
        self.log(f"恢复闭环: {rounds} 轮, 额外检测 {n_meas} 次, 额外清除 {n_clear} 次, "
                 f"未解决 {self.recovery['unresolved']}")

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
    global ON_WAY_DELTA, MESH_A, MESH_MARGIN, MESH_THETA, MESH_OFFSET  # 必须在首次引用前声明
    ap = argparse.ArgumentParser(description="问题4 全向+定向干扰源机器人")
    ap.add_argument("--robot-id", dest="robot_id", default=os.environ.get("ROBOT_ID"))
    ap.add_argument("--base-url", dest="base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--arena-id", dest="arena_id", default=DEFAULT_ARENA_ID)
    ap.add_argument("--log-file", dest="log_file", default=None)
    def _env_delta():
        """δ 也可用环境变量 ON_WAY_DELTA 提供(给固定启动脚本用, 不改命令行)。"""
        v = os.environ.get("ON_WAY_DELTA")
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            print(f"错误: 环境变量 ON_WAY_DELTA={v!r} 不是数字", file=sys.stderr)
            sys.exit(2)
    ap.add_argument("--on-way-delta", dest="on_way_delta", type=float, default=_env_delta(),
                    help=f"顺路清除阈值(默认 {ON_WAY_DELTA}; 传 0 或负数=关闭顺路清除; "
                         f"也可用环境变量 ON_WAY_DELTA)")
    ap.add_argument("--tag", dest="tag", default=os.environ.get("LOG_TAG", ""),
                    help="日志文件名后缀标记(如 d300/d500, 便于 A/B 对照; 也可用 LOG_TAG)")
    ap.add_argument("--mesh", dest="mesh", default=None,
                    help="临时覆盖网格几何 'a,margin,theta,offx,offy'(如实机 A/B; "
                         "旧网格为 '900,800,0,0,0'; 不传则用默认 920,700,20,460,398)")
    ap.add_argument("--mesh-stats", action="store_true", help="只打印网格统计后退出")
    ap.add_argument("--no-mesh-extra", dest="no_mesh_extra", action="store_true",
                    help="关闭网格覆盖补齐点(仅用于 A/B 对照; 默认开启, 关闭即退回原网格)")
    ap.add_argument("--no-mesh-override", dest="no_mesh_override", action="store_true",
                    help="关闭最小覆盖设计点集(MESH_PTS_OVERRIDE), 回到 tri_mesh+补齐点; "
                         "配合 --mesh 可复现旧网格(如实机 A/B 的旧臂)")
    ap.add_argument("--no-mec-freeze", dest="no_mec_freeze", action="store_true",
                    help="关闭 MEC 就绪冻结(仅用于 A/B 对照; 默认开启)")
    ap.add_argument("--mesh-pts", dest="mesh_pts", default=None,
                    help="从文件载入站集坐标(每行 x,y), 用于在环验证优化候选; 默认不载入")
    ap.add_argument("--mesh-a", dest="mesh_a", type=float, default=None,
                    help="临时覆盖 MESH_A(邻接容差 1.03*MESH_A); 默认不覆盖")
    args = ap.parse_args(argv)

    if args.mesh_a is not None:
        MESH_A = args.mesh_a
        print(f"[config] MESH_A={MESH_A} (临时覆盖)", flush=True)
    if args.mesh_pts:
        pts = []
        for ln in open(args.mesh_pts, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                x, y = ln.split(",")
                pts.append((float(x), float(y)))
        Problem4Robot.MESH_PTS_OVERRIDE = pts
        Problem4Robot.MESH_EXTRA_PTS = []
        print(f"[config] MESH_PTS_OVERRIDE 载入 {len(pts)} 点自 {args.mesh_pts}", flush=True)

    if args.no_mec_freeze:
        Problem4Robot.MEC_FREEZE = False
        print("[config] MEC_FREEZE=False (A/B 对照: 不冻结)", flush=True)

    if args.no_mesh_override:
        Problem4Robot.MESH_PTS_OVERRIDE = None
        print("[config] MESH_PTS_OVERRIDE=None (回退 tri_mesh 网格)", flush=True)
    if args.no_mesh_extra:
        Problem4Robot.MESH_EXTRA_PTS = []
        print("[config] MESH_EXTRA_PTS=[] (A/B 对照: 不含覆盖补齐点)", flush=True)

    # 顺路清除阈值: 默认不变; 仅当显式传参时覆盖(用于 δ A/B 风险复核)
    if args.on_way_delta is not None:
        ON_WAY_DELTA = None if args.on_way_delta <= 0 else args.on_way_delta
    print(f"[config] on_way_delta={ON_WAY_DELTA}", flush=True)

    if args.mesh_stats:
        # 必须走真实构造路径, 否则漏掉 MESH_EXTRA_PTS 与 --mesh 覆盖(曾恒报 27 点)
        rb = Problem4Robot(None)
        print(f"网格点数={len(rb.pts)} 三角形数={len(rb.tris)} 需证伪={len(rb.cover_tris)} "
              f"(含补齐点 {len(getattr(Problem4Robot, 'MESH_EXTRA_PTS', []) or [])} 个)")
        return

    if args.mesh:
        v = [float(x) for x in args.mesh.split(",")]
        MESH_A, MESH_MARGIN, MESH_THETA, MESH_OFFSET = v[0], v[1], v[2], (v[3], v[4])
    print(f"[config] mesh a={MESH_A} margin={MESH_MARGIN} theta={MESH_THETA} "
          f"offset={MESH_OFFSET}", flush=True)

    if not args.robot_id:
        print("错误: 未提供参赛队号。用法: python robot4.py --robot-id 你的队号", file=sys.stderr)
        sys.exit(2)
    # 默认日志写到本程序目录下的 logs/ (与问题3 一致)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(
        log_dir, "p4_log_%s%s.jsonl" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                        ("_" + args.tag) if args.tag else ""))
    client = SimClient(args.base_url, args.robot_id, args.arena_id)
    robot = Problem4Robot(client)
    exit_code = 0
    try:
        cleared = robot.run()
        s = client.build_summary()
        print(f"\n[汇总] 清除干扰源 {cleared} 个, 虚拟时刻 {s['final_virtual_time_s']:.1f}s, "
              f"移动 {s['movement_distance_m']:.0f}m, 检测 {s['measure_count']} 次, "
              f"清除 {s['clear_attempt_count']} 次(成功 {s['clear_success_count']})")
    except Exception as e:
        exit_code = 3
        print(f"\n[错误] {e}", file=sys.stderr)
        print("[提示] 测试未开始或已结束, 本次未产生有效数据。", file=sys.stderr)
    finally:
        # 单个 JSONL 文件: 逐条动作 + 末行 __summary__ 结构化汇总(不再另写第二份)
        if exit_code != 0 and not client.actions:
            client.error = "未连接上模拟器接口(测试未开始或已结束), 本局无有效数据"
        p = client.dump_log(log_file)
        print(f"[日志] 已写入 {p}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
