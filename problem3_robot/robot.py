# -*- coding: utf-8 -*-
"""
问题3 机器人程序 —— 无线电干扰源自动定位与清除
(基于《汇总版》模型设计)

模型要点:
  1. 保证覆盖: 原点 + 半径 a=1200m 圆周上 6 个正六边形顶点, 共 7 个搜索点,
     覆盖半径1800m圆域(任意点到最近搜索点 <= 968.58m < 1000m, 保证全向源被发现)。
  2. 蛇形扫描: 相邻搜索点用相反频道顺序(1->20 / 20->1), 避免跨点频道切换。
  3. 位置可行域: 每次示向度 -> 一个 ±1° 楔形(两个半平面); 联合可行域 = 半平面交(凸多边形)。
  4. 清除判据: 可行域最小覆盖圆半径 <= 20m 时, 在圆心清除(保证成功);
     检测返回 near(距离<=5m)时, 立即在当前点清除。
  5. 补测: 可行域仍未 <= 20m 时, 沿首次示向的垂直方向做补充检测, 压缩可行域。
  6. 终止: 20 个频道全部为"已排除"(7点均无信号)或"已清除", 才主动 /exit。

运行前:
  1. 在模拟器中在线登录并启动"问题3演练测试", 等待接口就绪(倒计时结束)。
  2. 命令行传入参赛队号:  python robot.py --robot-id 你的队号
     (也可用环境变量 ROBOT_ID, 或 --base-url 指定接口地址/端口)
"""

import json
import math
import time
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from itertools import combinations

# ==================== 配置(默认值) ====================
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_ARENA_ID = "default"

DEG = math.pi / 180.0
EPS = 1.0 * DEG          # 示向度误差 ±1°
R_AREA = 1800.0          # 目标区域半径
R_CLEAR = 20.0           # 清除半径
R_NEAR = 5.0             # 近距阈值
R_GUARANTEE = 1000.0     # 全向源最小有效接收半径(保证覆盖用)
HEX_R = 1200.0           # 外围六边形半径(覆盖安全余量约 31.42m)
N_CH = 20                # 频道数


# ==================== 模拟器 HTTP 客户端 ====================
class SimClient:
    """串行 HTTP+JSON 客户端, 处理重试幂等与 accepted 检查。"""

    def __init__(self, base_url, robot_id, arena_id="default"):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.arena_id = arena_id
        self._seq = 0
        self.channel = 1            # 测向机当前频道(模拟器自动维护, 本地仅记录)
        self.position = (0.0, 0.0)  # 本地记录的当前位置
        self.actions = []           # 本地行为日志: 每动作一条
        self.virtual_time = 0.0     # 最近一次 accepted 响应的虚拟时刻
        self.meta = {}              # 机器人回填: 配置/覆盖点/逐频道结果(写入 JSON 汇总)
        self.error = None           # 异常说明(如接口未开放), 写入 JSON 汇总
        self.phase = "init"         # 当前算法阶段(机器人回填, 用于分阶段统计)
        self.locate_history = []    # 机器人回填: 每次定位的 方式/Ω半径/交会角
        self.clear_diag = []        # 机器人回填: 每次清除的 定位来源/Ω半径/结果
        self.opp_diag = []          # 机器人回填: 模块O 逐次机会观测明细
        # 计数(供机会观测动作成本归因; 与离线 MockClient 同名同义)
        self.dist = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_ok = 0
        self.n_switch = 0

    def _new_req_id(self, tag):
        self._seq += 1
        return f"{tag}-{self._seq}"

    def _record(self, path, payload, resp):
        vt = resp.get("virtual_time_s")
        if resp.get("accepted") is True and isinstance(vt, (int, float)):
            self.virtual_time = float(vt)
        self.actions.append({
            "seq": self._seq, "path": path, "payload": payload,
            "accepted": resp.get("accepted"), "virtual_time_s": vt,
            "phase": getattr(self, "phase", None),   # 阶段标签(分阶段移动距离用)
            "response": resp,
        })

    def post(self, path, payload, timeout=8):
        """发送 POST, 网络失败复用同一 payload/request_id 重试, 返回 dict。"""
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
                # 结构错误(400)/409 等: 不盲目重试
                try:
                    resp = json.loads(e.read().decode("utf-8"))
                except Exception:
                    resp = {"accepted": False, "http_error": e.code}
                break
            except (URLError, TimeoutError, ConnectionError) as e:
                # 网络中断: 复用原 request_id 重试
                last_err = e
                time.sleep(min(0.5 * (2 ** attempt), 3.0))
        if resp is None:
            raise RuntimeError(f"网络请求失败(已重试): {path} {last_err}")
        self._record(path, payload, resp)
        return resp

    def dump_log(self, filepath):
        """把本地行为日志写为 JSON Lines 文件, 并在末尾附汇总。"""
        import datetime, os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# robot.py 本地行为日志(问题3)\n")
            f.write(f"# team_no={self.robot_id} base_url={self.base_url}\n")
            f.write(f"# generated_at={datetime.datetime.now().isoformat()}\n")
            for a in self.actions:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.write(json.dumps({"__summary__": self.build_summary()},
                               ensure_ascii=False) + "\n")
        return filepath

    def build_summary(self):
        """结构化 JSON 汇总(与问题4 同 schema: 配置/接口计数/移动距离/分阶段/逐频道结果)。"""
        moves = 0.0
        prev = (0.0, 0.0)
        phases = {}          # 阶段 -> {movement_m, measure, clear, clear_success, virtual_time_s}
        prev_vt = None

        def ph(name):
            return phases.setdefault(name or "unknown", {
                "movement_distance_m": 0.0, "measure_count": 0,
                "clear_attempt_count": 0, "clear_success_count": 0,
                "virtual_time_s": 0.0})

        for a in self.actions:
            name = a.get("phase")
            pos = a.get("payload", {}).get("position")
            p = ph(name)
            if pos and a.get("path") in ("/measure", "/clear"):
                leg = math.hypot(pos["x"]-prev[0], pos["y"]-prev[1])
                moves += leg
                p["movement_distance_m"] += leg
                prev = (pos["x"], pos["y"])
            if a.get("path") == "/measure":
                p["measure_count"] += 1
            elif a.get("path") == "/clear":
                p["clear_attempt_count"] += 1
                if a["response"].get("clear_result") == "success":
                    p["clear_success_count"] += 1
            # 虚拟时间按串行时间轴分段归属到各阶段
            vt = a.get("virtual_time_s")
            if isinstance(vt, (int, float)):
                if prev_vt is not None and vt >= prev_vt:
                    p["virtual_time_s"] += vt - prev_vt
                prev_vt = vt
        for p in phases.values():
            p["movement_distance_m"] = round(p["movement_distance_m"], 1)
            p["virtual_time_s"] = round(p["virtual_time_s"], 1)

        clears = [a for a in self.actions if a["path"] == "/clear"]
        s = {
            "problem": 3,
            "team_no": self.robot_id,
            "base_url": self.base_url,
            "config": {"hex_r": HEX_R, "n_cover_points": 7,
                       "on_way_delta": getattr(Problem3Robot, "ON_WAY_DELTA", None),
                       "r_clear": R_CLEAR, "r_guarantee": R_GUARANTEE,
                       "order_by_prob": getattr(Problem3Robot, "ORDER_BY_PROB", None),
                       "dop_prescreen": getattr(Problem3Robot, "DOP_PRESCREEN", None)},
            "final_virtual_time_s": self.virtual_time,
            "total_actions": len(self.actions),
            "measure_count": sum(1 for a in self.actions if a["path"] == "/measure"),
            "clear_attempt_count": len(clears),
            "clear_success_count": sum(1 for a in clears
                                       if a["response"].get("clear_result") == "success"),
            "clear_failure_count": sum(1 for a in clears
                                       if a["response"].get("clear_result") != "success"),
            "movement_distance_m": round(moves, 1),
            "phase_stats": phases,                      # 分阶段移动距离/动作数/虚拟时间
            "locate_stats": self._locate_stats(),       # 定位方式与 Ω 半径统计
            "clear_diag": self.clear_diag,              # 逐次清除的定位来源+Ω 半径(诊断)
            "opp_stats": self._opp_stats(),             # 模块O 机会观测汇总
            "opp_diag": self.opp_diag,                  # 模块O 逐次机会观测明细
            "robot": self.meta,
        }
        if self.error:
            s["error"] = self.error
        return s

    def _opp_stats(self):
        """模块O 汇总: 触发次数、命中率、认证转化率、补测规避与动作成本。"""
        d = self.opp_diag
        if not d:
            return {"enabled": getattr(Problem3Robot, "OPP_MEASURE", None), "attempts": 0}
        def stat(v):
            v = [x for x in v if x is not None]
            if not v:
                return None
            v = sorted(v)
            return {"n": len(v), "median": round(v[len(v)//2], 1), "max": round(v[-1], 1)}
        sig = [x for x in d if x.get("result") in ("direction", "near")]
        cert = [x for x in d if x.get("became_certified")]
        return {
            "enabled": getattr(Problem3Robot, "OPP_MEASURE", None),
            "max_per_point": getattr(Problem3Robot, "OPP_MAX_PER_POINT", None),
            "attempts": len(d),
            "signal": len(sig),
            "no_signal": sum(1 for x in d if x.get("result") == "no_signal"),
            "rejected": sum(1 for x in d if x.get("result") == "rejected"),
            "became_certified": len(cert),
            "avoided_supplement": sum(1 for x in d if x.get("avoided_supplement")),
            "signal_rate": round(len(sig)/len(d), 4),
            "cert_rate": round(len(cert)/len(d), 4),
            "mec_before": stat([x.get("mec_before") for x in d]),
            "mec_after": stat([x.get("mec_after") for x in d]),
            "cross_pred_deg": stat([x.get("cross_pred_deg") for x in d]),
            "min_sep_m": stat([x.get("min_sep_m") for x in d]),
            "switch_count": stat([x.get("switch_count") for x in d]),
            "time_cost_s": stat([x.get("time_cost") for x in d]),
            "time_cost_total_s": round(sum(x.get("time_cost") or 0.0 for x in d), 1),
            "by_site": {k: sum(1 for x in d if x.get("site") == k)
                        for k in sorted({x.get("site") for x in d if x.get("site")})},
        }

    def _locate_stats(self):
        """汇总定位诊断: MEC(Ω 半径<=20m) 与最小二乘(交会角) 各占多少、半径分布。"""
        hist = self.locate_history
        mec = [h["omega_radius_m"] for h in hist
               if h.get("method") == "mec" and h.get("omega_radius_m") is not None]
        ls = [h for h in hist if h.get("method") == "ls"]
        over = [h["omega_radius_m"] for h in hist
                if h.get("method") == "ls" and h.get("omega_radius_m") is not None]

        def stat(v):
            if not v:
                return None
            v = sorted(v)
            return {"n": len(v), "min": round(v[0], 1), "median": round(v[len(v)//2], 1),
                    "max": round(v[-1], 1)}
        return {
            "calls": len(hist),
            "method_counts": {
                "mec": sum(1 for h in hist if h.get("method") == "mec"),
                "ls": len(ls),
                "fail": sum(1 for h in hist if h.get("method") is None),
            },
            "omega_radius_m_when_mec": stat(mec),        # 判定即清除的 Ω 半径(<=20m)
            "omega_radius_m_when_ls": stat(over),        # 改用最小二乘时的 Ω 半径(>20m 或退化)
            "ls_cross_angle_deg": stat([h["cross_angle_deg"] for h in ls
                                        if h.get("cross_angle_deg") is not None]),
        }

    def _base(self, req_id):
        return {"arena_id": self.arena_id, "robot_id": self.robot_id,
                "request_id": req_id}

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
                        "  2) 已在模拟器中点击开始\"问题3测试\";\n"
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
        """移动+切换+检测一次, 返回 (accepted, measure_result, svd_deg)。"""
        req_id = self._new_req_id("measure")
        payload = self._base(req_id)
        payload["position"] = {"x": x, "y": y}
        payload["channel"] = channel
        r = self.post("/measure", payload)
        if r.get("accepted") is not True:
            return False, "rejected", None
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        if channel != self.channel:
            self.n_switch += 1
        self.n_measure += 1
        self.position = (x, y)
        self.channel = channel
        return True, r.get("measure_result"), r.get("svd_deg")

    def clear(self, x, y, channel):
        """在 (x,y) 尝试清除 channel, 返回 (accepted, clear_result)。"""
        req_id = self._new_req_id("clear")
        payload = self._base(req_id)
        payload["position"] = {"x": x, "y": y}
        payload["channel"] = channel
        r = self.post("/clear", payload)
        if r.get("accepted") is not True:
            return False, "rejected"
        self.dist += math.hypot(x-self.position[0], y-self.position[1])
        if channel != self.channel:
            self.n_switch += 1
        self.n_clear += 1
        if r.get("clear_result") == "success":
            self.n_clear_ok += 1
        self.position = (x, y)
        self.channel = channel
        return True, r.get("clear_result")

    def exit(self):
        return self.post("/exit", self._base(self._new_req_id("exit")))


# ==================== 几何: 半平面交 / 直径 / 最小覆盖圆 ====================
def wedge_halfplanes(S, theta_deg):
    """检测点 S、示向度 theta(deg) -> 两个指向楔形内部的半平面 (n, c): n·P >= c。"""
    th = theta_deg * DEG
    n1 = (math.cos(th + EPS - math.pi/2), math.sin(th + EPS - math.pi/2))
    n2 = (math.cos(th - EPS + math.pi/2), math.sin(th - EPS + math.pi/2))
    return [(n1, n1[0]*S[0] + n1[1]*S[1]),
            (n2, n2[0]*S[0] + n2[1]*S[1])]

def clip_polygon(poly, n, c):
    """用半平面 n·P >= c 裁剪凸多边形(顶点列表)。"""
    if len(poly) < 3:
        return []
    out = []
    m = len(poly)
    for i in range(m):
        A = poly[i]; B = poly[(i+1) % m]
        dA = n[0]*A[0] + n[1]*A[1] - c
        dB = n[0]*B[0] + n[1]*B[1] - c
        Ain = dA >= -1e-9
        Bin = dB >= -1e-9
        if Ain:
            out.append(A)
        if Ain != Bin:
            t = dA / (dA - dB)
            out.append((A[0] + t*(B[0]-A[0]), A[1] + t*(B[1]-A[1])))
    return out

def disk_polygon(R=R_AREA, m=72):
    """目标圆域的**保守外接**正 m 边形。

    内接多边形(顶点在半径 R 的圆周上)的边是弦, 会切掉边界附近的一条弓形,
    可能把真实干扰源位置错误排除, 与问题一"保守外包络"口径相反。故顶点取在
    半径 R/cos(pi/m) 上, 使多边形整体包含圆域(外接); 面积/半径误差 O(1/m^2)
    且方向保守(只会略微放大可行域, 不会漏掉真值)。
    """
    Rc = R / math.cos(math.pi / m)
    return [(Rc*math.cos(2*math.pi*k/m), Rc*math.sin(2*math.pi*k/m)) for k in range(m)]

def feasible_region(bearings):
    """bearings: [( (x,y), theta_deg ), ...] -> 可行域顶点(凸多边形)。"""
    poly = disk_polygon()
    for (S, th) in bearings:
        for n, c in wedge_halfplanes(S, th):
            poly = clip_polygon(poly, n, c)
            if len(poly) < 3:
                return poly
    return poly

def polygon_diameter(verts):
    best = 0.0
    for a, b in combinations(verts, 2):
        d = math.hypot(a[0]-b[0], a[1]-b[1])
        if d > best:
            best = d
    return best

def _wedge_vertices(P, t1, Q, t2, eps=1.0):
    """两点两示向(各 ±eps°)的 4 条边界线两两交点(用于最坏定位直径评价)。"""
    lines = []
    for S, t in ((P, t1), (Q, t2)):
        for e in (+eps, -eps):
            a = (t + e) * DEG
            lines.append(((math.cos(a), math.sin(a)), S))
    verts = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            d1, S1 = lines[i]; d2, S2 = lines[j]
            det = d1[0]*(-d2[1]) - (-d2[0])*d1[1]
            if abs(det) < 1e-12:
                continue
            rx, ry = S2[0]-S1[0], S2[1]-S1[1]
            u = (rx*(-d2[1]) - (-d2[0])*ry) / det
            verts.append((S1[0] + u*d1[0], S1[1] + u*d1[1]))
    return verts


def bearing_intersection(stations, bearings_deg):
    """多站示向度最小二乘交会(点估计)。"""
    A00 = A01 = A11 = b0 = b1 = 0.0
    for (x, y), th in zip(stations, bearings_deg):
        a = th * DEG
        dx, dy = math.cos(a), math.sin(a)
        # 投影矩阵 I - d d^T
        P00, P01, P11 = 1-dx*dx, -dx*dy, 1-dy*dy
        A00 += P00; A01 += P01; A11 += P11
        b0 += P00*x + P01*y
        b1 += P01*x + P11*y
    det = A00*A11 - A01*A01
    if abs(det) < 1e-12:
        return None
    return ((A11*b0 - A01*b1)/det, (A00*b1 - A01*b0)/det)

def crossing_angle(t1, t2):
    d = abs(t1 - t2) % 180.0
    return min(d, 180.0 - d)

def angle_diff(a, b):
    """方位差 0..180 (180 表示方向相反)。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)

def minimal_enclosing_circle(points):
    """枚举 2 点(直径)圆与 3 点外接圆, 求最小覆盖圆 (O(m^3), m 很小)。"""
    pts = [p for p in points]
    n = len(pts)
    if n == 0:
        return None, 0.0
    if n == 1:
        return pts[0], 0.0
    def covers(c, r):
        return all(math.hypot(p[0]-c[0], p[1]-c[1]) <= r + 1e-9 for p in pts)
    best_r, best_c = float("inf"), None
    for i, j in combinations(range(n), 2):
        c = ((pts[i][0]+pts[j][0])/2, (pts[i][1]+pts[j][1])/2)
        r = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])/2
        if covers(c, r) and r < best_r:
            best_r, best_c = r, c
    for i, j, k in combinations(range(n), 3):
        ax, ay = pts[i]; bx, by = pts[j]; cx, cy = pts[k]
        d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-12:
            continue
        ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by))/d
        uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax))/d
        c = (ux, uy)
        r = math.hypot(ax-ux, ay-uy)
        if covers(c, r) and r < best_r:
            best_r, best_c = r, c
    return best_c, best_r


# ==================== 策略 ====================
class Problem3Robot:
    # 受限顺路清除阈值(米): None=关闭; 数值=仅在插入增量 ΔL<=该值时才顺路清除。
    # 实测最优 δ≈300m(1000案例扫描 200/300/500/800/1200: 300 最优, 更大反而回归)。
    ON_WAY_DELTA = 300.0
    # ===== 外部改进模块开关(默认关, 用于消融实验) =====
    ORDER_BY_PROB = False    # 模块A: 贝叶斯概率图给覆盖点排访问顺序(替代固定六边形顺序)
    DOP_PRESCREEN = False    # 模块B: DOP(交会角)预筛候选补测点, 再按极小极大+路程择优
    # 模块E(诊断用配对实验开关): 最小二乘试探清除的信任门限(Ω 半径, 米)。
    #   None = 当前策略: Ω 半径超限或可行域退化时, 直接用 LS 交会点盲清除;
    #   数值 = 仅当 Ω 半径 <= 该值才允许盲清除, 否则改走补测/归航(用于配对实验)。
    LS_CLEAR_GATE = None
    # ===== 模块O(机会性顺带观测, 默认关): 既定清除停靠点兼作观测站 =====
    # 定位: 只复用"本来就要去的清除点", 不重构搜索/覆盖/清除调度; 不改任何既有默认决策。
    OPP_MEASURE = False        # 总开关(默认关 = 行为与冻结版完全一致)
    OPP_MAX_PER_POINT = 2      # 每个停靠点最多顺带观测几个困难频道
    OPP_MIN_CROSS_DEG = 45.0   # 预测有效交会角门槛
    OPP_MIN_SEP_M = 200.0      # 与历史观测点最小间距
    # 机会集合口径: "spec" = 规格版(Ω>20m ∨ 示向<2 ∨ 交会<30°, 即所有未认证频道);
    #               "uncert" = 收紧版(仅当前无法定位、会走补测/归航的频道)。
    OPP_TARGET = "spec"

    def __init__(self, client):
        self.c = client
        # 频道状态: None=待排查, 'found'=已发现, 'excluded'=已排除, 'cleared'=已清除
        self.state = {ch: None for ch in range(1, N_CH+1)}
        self.bearings = {ch: [] for ch in range(1, N_CH+1)}   # 已发现频道的示向观测
        self.near_pos = {ch: None for ch in range(1, N_CH+1)}  # near 时的位置
        self.visited_no_signal = {ch: set() for ch in range(1, N_CH+1)}  # 记录无信号搜索点
        self.cleared_count = 0
        self.locate_diag = {}     # 频道 -> 最近一次定位诊断(方式/Ω半径/交会角)
        self._meas_seen = set()   # (坐标, 频道) 去重: 模块O 不重复测量同一频道的同一坐标
        self.supp_channels = set()  # 进入过补测队列的频道(用于回填 avoided_supplement)

    # ---- 阶段标签 + 诊断记录(仅记录, 不改变任何决策) ----
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
        """记录一次清除尝试的定位来源与当时 Ω 半径(供配对实验判定是否收紧清除门限)。"""
        self._diag_list(self.c, "clear_diag").append({
            "ch": ch, "point": [round(x, 1), round(y, 1)], "src": src,
            "omega_radius_m": omega_r, "cross_angle_deg": cross_ang,
            "phase": getattr(self.c, "phase", None), "result": result,
        })

    # ---- 7 个保证搜索点 ----
    @staticmethod
    def search_points():
        pts = [(0.0, 0.0)]
        for k in range(6):
            a = k * 60 * DEG
            pts.append((HEX_R*math.cos(a), HEX_R*math.sin(a)))
        return pts

    def log(self, *a):
        print("[robot]", *a, flush=True)

    # ---- 模块A: 贝叶斯概率图 -> 覆盖点访问顺序 ----
    # 纯信息增益排序会忽略路程(实测路径 18.7km -> 22.7km), 故加入路程惩罚项。
    PROB_TRAVEL_W = 0.05     # 路程惩罚权重(米 -> 等价信息增益单位)

    @classmethod
    def _prob_order(cls, stations, n_grid=16):
        """用"信息增益 − 路程惩罚"给覆盖点排序(贪心)。

        均匀先验下, 一个点的信息增益 ∝ 其 1000m 接收圆盘新覆盖的目标区面积。
        按《汇总版》要求: 只调整访问次序, 不删除任何保证覆盖点, 不改变终止条件。
        """
        cells = []
        for i in range(n_grid):
            for j in range(n_grid):
                x = -R_AREA + (2*R_AREA) * i / (n_grid - 1)
                y = -R_AREA + (2*R_AREA) * j / (n_grid - 1)
                if math.hypot(x, y) <= R_AREA:
                    cells.append((x, y))
        unvisited = list(stations)
        order = []
        covered = [False] * len(cells)
        cur = (0.0, 0.0)
        while unvisited:
            best_i, best_score = 0, -float("inf")
            for i, p in enumerate(unvisited):
                gain = 0
                for k, c in enumerate(cells):
                    if not covered[k] and math.hypot(c[0]-p[0], c[1]-p[1]) <= R_GUARANTEE:
                        gain += 1
                score = gain - cls.PROB_TRAVEL_W * math.hypot(p[0]-cur[0], p[1]-cur[1])
                if score > best_score:
                    best_score, best_i = score, i
            p = unvisited.pop(best_i)
            order.append(p); cur = p
            for k, c in enumerate(cells):
                if math.hypot(c[0]-p[0], c[1]-p[1]) <= R_GUARANTEE:
                    covered[k] = True
        return order

    # ---- 模块B: DOP(交会角)预筛 + 极小极大/路程择优的补测点 ----
    def _dop_supplement_point(self, ch, toward):
        """在首示向垂线附近生成候选, 用交会角(DOP)预筛, 再按最坏定位直径+路程择优。"""
        dirs = [(p, t) for p, t in self.bearings[ch]]
        if not dirs:
            return None
        (x0, y0), th0 = dirs[0]
        cands = []
        for off in (60.0, 90.0, 120.0, -60.0, -90.0, -120.0):
            for d in (300.0, 500.0, 700.0):
                a = (th0 + off) * DEG
                cands.append((x0 + d*math.cos(a), y0 + d*math.sin(a)))
        # 1) DOP 预筛: 与首示向的交会角 >= 30° 才保留(方向退化点先删掉)
        cands = [q for q in cands
                 if crossing_angle(math.degrees(math.atan2(q[1]-y0, q[0]-x0)), th0) >= 30.0]
        if not cands:
            return None
        # 2) 极小极大: 用两站楔形交的最坏直径评价(离散源距离)
        best, best_key = None, (float("inf"), float("inf"))
        for q in cands:
            worst = 0.0
            for rr in (100.0, 500.0, 900.0, 1300.0, 1500.0):
                g = (x0 + rr*math.cos(th0*DEG), y0 + rr*math.sin(th0*DEG))
                t2 = math.degrees(math.atan2(g[1]-q[1], g[0]-q[0])) % 360
                v = _wedge_vertices((x0, y0), th0, q, t2)
                if len(v) >= 2:
                    worst = max(worst, max(math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                                           for p1 in v for p2 in v))
            key = (worst, math.hypot(q[0]-toward[0], q[1]-toward[1]))
            if key < best_key:
                best_key, best = key, q
        return best

    # ---- 主流程: 保证层(覆盖) + 状态层(可行域) + 调度层(事件驱动任务队列) ----
    def run(self):
        c = self.c
        self.log("调用 /enter ...")
        c.enter()
        self.log(f"进入成功, 剩余现实时间 {c.remaining_real}s")

        pts = self.search_points()
        if self.ORDER_BY_PROB:
            pts = self._prob_order(pts)   # 模块A: 概率图(信息增益)排序访问顺序
            self.log("覆盖点访问顺序(概率图排序): " +
                     " -> ".join("(%.0f,%.0f)" % p for p in pts))

        # 保证层: 依次访问 7 个覆盖点, 蛇形扫描(发现 + 免费交会)
        self._phase("coverage")
        for i, (px, py) in enumerate(pts):
            order = list(range(1, N_CH+1)) if i % 2 == 0 else list(range(N_CH, 0, -1))
            self.log(f"搜索点 {i}: ({px:.0f},{py:.0f})  频道序 {'1->20' if i%2==0 else '20->1'}")
            for ch in order:
                if self.state[ch] in ("excluded", "cleared"):
                    continue
                # 已发现频道的冗余测量过滤: 当前点交会角无明显改善则跳过
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
                    self.visited_no_signal[ch].add(i)
            # 受限顺路清除: 已定位目标若"几乎在"去下一个覆盖点的路上, 顺路清除
            if self.ON_WAY_DELTA is not None and i + 1 < len(pts):
                nxt = pts[i + 1]
                self._phase("on_way")
                for c2 in range(1, N_CH+1):
                    if self.state[c2] != "found":
                        continue
                    if self.near_pos[c2] is not None:
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
                    if dL <= self.ON_WAY_DELTA:
                        # 模块O: 即将停靠前预选机会频道(先清除、后顺带观测)
                        sel = self._opp_select(Q[0], Q[1]) if self.OPP_MEASURE else []
                        ok, res = c.clear(Q[0], Q[1], c2)
                        self._note_clear(c2, Q[0], Q[1], "on_way:"+qsrc, qr, qa,
                                         res if ok else "rejected")
                        if ok and res == "success":
                            self.state[c2] = "cleared"
                            self.cleared_count += 1
                            self.log(f"顺路清除: 频道 {c2} @ ({Q[0]:.0f},{Q[1]:.0f})  "
                                     f"[{self.cleared_count}]")
                        else:
                            self._homing_clear(c2, Q[0], Q[1], phase="on_way_homing",
                                               src="on_way:"+qsrc, omega_r=qr, cross_ang=qa)
                        if sel and tuple(c.position) == (Q[0], Q[1]):
                            self._opp_run(Q[0], Q[1], sel, "on_way")
                self._phase("coverage")

        # 状态层: 排除判定(遍历完全部覆盖点且均无信号)
        for ch in range(1, N_CH+1):
            if self.state[ch] is None and len(self.visited_no_signal[ch]) >= len(pts):
                self.state[ch] = "excluded"
        self.log(f"扫描完成: 已发现 {sum(1 for s in self.state.values() if s=='found')} 个频道, "
                 f"已排除 {sum(1 for s in self.state.values() if s=='excluded')} 个频道")

        # 状态层: 快速定位(不移动) —— 可分两类: 已可清除 / 需补测
        # 任务元组: (ch, x, y, src, omega_radius_m, cross_angle_deg)
        clear_tasks = []   # 已可清除
        supp_tasks = []    # 需补测
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found":
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
                sup = (self._dop_supplement_point(ch, c.position)
                       if self.DOP_PRESCREEN
                       else self._supplement_point(ch, c.position))
                if sup is not None:
                    self.supp_channels.add(ch)     # 模块O: 记录进入补测的频道(回填用)
                    supp_tasks.append((ch, sup[0], sup[1]))

        # 调度层: 批量补测 —— 全部补测点做开放路径最优排序后统一执行
        # (注: 实测"补测终点与清除路线联合选择"无收益, 因清除点集在补测后仍会增长,
        #  联合评价只能基于不完整信息; 而清除 2-opt 路径对起点不敏感)
        if supp_tasks:
            self.log(f"补测任务 {len(supp_tasks)} 个, 开放路径优化")
            self._phase("supplement")
            spts = [(x, y) for _, x, y in supp_tasks]
            sorder = self._optimal_open_path(spts, tuple(c.position))
            for idx in sorder:
                ch, x, y = supp_tasks[idx]        # 直接用索引取任务, 不再用坐标查表
                ok, res, svd = c.measure(x, y, ch)
                if ok and res == "near":
                    clear_tasks.append((ch, x, y, "near", None, None))
                    continue
                if ok and res == "direction":
                    self.bearings[ch].append(((x, y), svd))
                pt = self._locate_quick(ch, tag=f"补测后({x:.0f},{y:.0f})")
                dg = self.locate_diag.get(ch, {})
                if pt is not None:
                    clear_tasks.append((ch, pt[0], pt[1], dg.get("method") or "none",
                                        dg.get("omega_radius_m"),
                                        dg.get("cross_angle_deg")))
                else:
                    # 补测后仍不足以定位 -> 二分归航(保证)
                    self._phase("homing")
                    bt = self._binary_homing(ch)
                    if bt is not None:
                        clear_tasks.append((ch, bt[0], bt[1], "binary", None, None))
                    self._phase("supplement")

        # 调度层: 任务队列就近清除 —— 最近邻生成初始路径 + 2-opt 局部交换优化
        self.log(f"清除任务队列 {len(clear_tasks)} 个")
        self._phase("queue_clear")
        # 最近邻初始顺序(按任务而非坐标, 避免多频道同坐标时相互覆盖)
        ordered = []
        unvisited = set(range(len(clear_tasks)))
        cur = c.position
        while unvisited:
            k = min(unvisited, key=lambda i: math.hypot(clear_tasks[i][1]-cur[0],
                                                        clear_tasks[i][2]-cur[1]))
            ordered.append(clear_tasks[k])
            cur = (clear_tasks[k][1], clear_tasks[k][2])
            unvisited.discard(k)
        # 2-opt(任务级, 含开放路径尾段反转)
        qseq = self._two_opt_tasks(ordered, c.position)
        for i_task, (ch, x, y, src, omr, cra) in enumerate(qseq):
            # 模块O: 到达前预选机会频道(下一任务频道用于减少切换)
            next_ch = qseq[i_task+1][0] if i_task + 1 < len(qseq) else None
            sel = self._opp_select(x, y, next_ch) if self.OPP_MEASURE else []
            ok, res = c.clear(x, y, ch)
            self._note_clear(ch, x, y, "queue:"+str(src), omr, cra,
                             res if ok else "rejected")
            if ok and res == "success":
                self.state[ch] = "cleared"
                self.cleared_count += 1
                self.log(f"清除成功: 频道 {ch} @ ({x:.0f},{y:.0f})  来源={src} "
                         f"Ω半径={omr}m  [{self.cleared_count}]")
            else:
                self.log(f"清除未发现: 频道 {ch} @ ({x:.0f},{y:.0f})  来源={src} "
                         f"Ω半径={omr}m, 做就近精定位")
                self._homing_clear(ch, x, y, phase="queue_homing", src="queue:"+str(src),
                                   omega_r=omr, cross_ang=cra)
                self._phase("queue_clear")
            # 先清除、后顺带观测; 仅当仍停留在该停靠点时才执行(绝不返回已离开的点)
            if sel and tuple(c.position) == (x, y):
                self._opp_run(x, y, sel, "queue_clear")

        self._opp_finalize()

        # 退出前校验: 不应存在"既未清除、也未被排除"的频道
        unresolved = [ch for ch in range(1, N_CH+1)
                      if self.state[ch] not in ("cleared", "excluded")]
        if unresolved:
            self.log(f"警告: 仍有未解决频道 {unresolved}(可能未找到/未清除), "
                     f"将在退出摘要中记录")
        else:
            self.log("20 个频道均已了结(已清除或已排除)")

        # 回填结构化元信息(与问题4 同 schema)
        c.meta = {
            "cleared_count": self.cleared_count,
            "found_channels": sum(1 for s in self.state.values() if s == "found"),
            "excluded_channels": sum(1 for s in self.state.values() if s == "excluded"),
            "unresolved_channels": sum(1 for s in self.state.values()
                                      if s not in ("cleared", "excluded")),
            "unresolved_list": unresolved,
            "channels": {str(ch): {"state": self.state[ch],
                                   "bearings": len(self.bearings[ch]),
                                   "near": self.near_pos[ch] is not None,
                                   "locate": self.locate_diag.get(ch)}
                         for ch in range(1, N_CH+1)},
        }
        self._phase("exit")
        self.log("调用 /exit ...")
        c.exit()
        self.log(f"结束, 清除 {self.cleared_count} 个干扰源")
        return self.cleared_count

    # ---- 快速定位(不移动): 可行域最小覆盖圆<=20m 或 最小二乘交会 ----
    def _locate_eval(self, ch):
        """纯计算(不改状态/不记录日志): 返回 (诊断 dict, 可清除点或 None)。

        方式 'mec': Ω 最小覆盖圆半径 <= R_CLEAR(判定即清除);
        方式 'ls' : Ω 半径超限或可行域退化(顶点<3)时, 改用交会角最大的两条示向交会。
        """
        dirs = [(pos, th) for pos, th in self.bearings[ch]]
        rec = {"ch": ch, "n_dirs": len(dirs), "method": None,
               "omega_radius_m": None, "cross_angle_deg": None, "point": None}
        pt = None
        if len(dirs) >= 2:
            poly = feasible_region(dirs)
            if len(poly) >= 3:
                center, radius = minimal_enclosing_circle(poly)
                rec["omega_radius_m"] = round(radius, 1)
                if radius <= R_CLEAR:
                    rec["method"] = "mec"
                    pt = center
            if pt is None:
                best_i, best_j, best_ang = 0, 1, -1
                for i in range(len(dirs)):
                    for j in range(i+1, len(dirs)):
                        a = crossing_angle(dirs[i][1], dirs[j][1])
                        if a > best_ang:
                            best_ang, best_i, best_j = a, i, j
                rec["cross_angle_deg"] = round(best_ang, 1)
                if best_ang >= 30.0:
                    rec["method"] = "ls"
                    pt = bearing_intersection([dirs[best_i][0], dirs[best_j][0]],
                                              [dirs[best_i][1], dirs[best_j][1]])
                    # 模块E(仅实验): 门限生效时, Ω 半径超限/未知的 LS 点不直接用于盲清除
                    gate = getattr(Problem3Robot, "LS_CLEAR_GATE", None)
                    if gate is not None and (rec["omega_radius_m"] is None
                                             or rec["omega_radius_m"] > gate):
                        rec["gate_blocked"] = True
                        pt = None
        if pt is not None:
            rec["point"] = [round(pt[0], 1), round(pt[1], 1)]
        return rec, pt

    def _locate_quick(self, ch, tag=""):
        """定位并记录诊断(方式/Ω 半径/交会角), 返回可清除点或 None。"""
        rec, pt = self._locate_eval(ch)
        rec["tag"] = tag
        self._note_locate(rec)
        if tag:
            self.log(f"定位[{tag}] 频道 {ch}: 方式={rec['method']} "
                     f"Ω半径={rec['omega_radius_m']}m 交会角={rec['cross_angle_deg']}° "
                     f"示向数={rec['n_dirs']}"
                     + ("  [门限拦截->改走补测]" if rec.get("gate_blocked") else ""))
        return pt

    # ---- 垂直补测点(取靠近 toward 的一侧) ----
    def _supplement_point(self, ch, toward):
        dirs = [(pos, th) for pos, th in self.bearings[ch]]
        if not dirs:
            return None
        (x0, y0), th0 = dirs[0]
        best, best_d = None, float("inf")
        for d in (300.0, 500.0):
            for sgn in (+1, -1):
                a = (th0 + sgn*90.0) * DEG
                sx, sy = x0 + d*math.cos(a), y0 + d*math.sin(a)
                dd = math.hypot(sx-toward[0], sy-toward[1])
                if dd < best_d:
                    best_d, best = dd, (sx, sy)
        return best

    # ---- 二分归航(沿首示向找到源, 全向源稳健) ----
    def _binary_homing(self, ch):
        dirs = [(pos, th) for pos, th in self.bearings[ch]]
        (x0, y0), th0 = dirs[0]
        a = th0 * DEG
        ux, uy = math.cos(a), math.sin(a)
        lo, hi = 0.0, 1500.0
        for _ in range(22):
            mid = (lo + hi) / 2.0
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

    # ---- 模块O: 机会性顺带观测(清除停靠点兼作观测站, 默认关) ----
    #  目标频道 C_opp = { state=found, 未清除, 未认证可清除, 且 (Ω>20m ∨ 示向<2 ∨ 最佳交会<30°) }
    #  触发时机: 即将在清除点停靠时预选频道 -> 先完成原清除 -> 若仍在原地再执行顺带观测。
    #  绝不返回已离开的停靠点; no_signal 只记录, 不裁剪可行域(保守)。
    def _opp_hard(self, ch):
        """判定频道是否属于机会集合; 是则返回 (诊断 dict, 代表位置)。"""
        if self.state[ch] != "found" or self.near_pos[ch] is not None:
            return None                      # 非 found / near 可直接清, 不需观测
        rec, pt = self._locate_eval(ch)
        if rec["method"] == "mec":
            return None                      # 已认证可清除(Ω<=20m), 不再测
        if self.OPP_TARGET == "uncert":
            # 收紧版: 只对"当前根本定不了位"的频道(即会走补测/归航的那一群)
            if rec["method"] is not None:
                return None
        hard = (rec["omega_radius_m"] is None or rec["omega_radius_m"] > R_CLEAR
                or rec["n_dirs"] < 2
                or (rec["cross_angle_deg"] is not None
                    and rec["cross_angle_deg"] < 30.0))
        if not hard:
            return None
        rep = pt
        if rep is None and self.bearings[ch]:
            (px, py), th = self.bearings[ch][0]
            rep = (px + 500.0*math.cos(th*DEG), py + 500.0*math.sin(th*DEG))
        if rep is None:
            return None
        return rec, rep

    def _opp_pred_cross(self, ch, X, rep):
        """预测在 X 处观测频道 ch 的最佳有效交会角(度)。

        示向数>=2: 以代表位置为顶点, 计算 X 与各历史观测点的交会角;
        示向数==1: 无交会角可言, 用"该示向线 与 P1->X 连线"的夹角作代理。
        """
        dirs = list(self.bearings[ch])
        if len(dirs) >= 2:
            best = 0.0
            for (P, _th) in dirs:
                a1 = math.atan2(rep[1]-P[1], rep[0]-P[0])
                a2 = math.atan2(rep[1]-X[1], rep[0]-X[0])
                a = math.degrees(abs(a1 - a2)) % 180.0
                best = max(best, min(a, 180.0 - a))
            return best
        if len(dirs) == 1:
            (P, th) = dirs[0]
            a = math.degrees(math.atan2(X[1]-P[1], X[0]-P[0])) % 360.0
            return crossing_angle(th, a)
        return 0.0

    def _opp_pred_radius(self, ch, X, poly, n_rep=6):
        """预测观测后 Ω 的最小覆盖圆半径(对代表点取最坏), 用于"预计能否直接认证"。"""
        if len(poly) < 3:
            return None
        step = max(1, len(poly)//n_rep)
        worst = 0.0
        for S in poly[::step]:
            th = math.degrees(math.atan2(S[1]-X[1], S[0]-X[0])) % 360
            p2 = poly
            for n, c in wedge_halfplanes(X, th):
                p2 = clip_polygon(p2, n, c)
                if len(p2) < 3:
                    return float("inf")
            _ctr, r = minimal_enclosing_circle(p2)
            if r is None:
                return float("inf")
            worst = max(worst, r)
        return worst

    def _opp_select(self, x, y, next_ch=None):
        """在清除点 (x,y) 预选机会频道(纯计算, 不改状态)。"""
        cands = []
        for ch in range(1, N_CH+1):
            hard = self._opp_hard(ch)
            if hard is None:
                continue
            rec, rep = hard
            dirs = self.bearings[ch]
            sep = min([math.hypot(x-P[0], y-P[1]) for P, _ in dirs] or [1e9])
            if sep < self.OPP_MIN_SEP_M:
                continue                     # 与历史观测点太近: 几何冗余
            cross = self._opp_pred_cross(ch, (x, y), rep)
            if cross < self.OPP_MIN_CROSS_DEG:
                continue
            poly = feasible_region(dirs)
            pred_r = self._opp_pred_radius(ch, (x, y), poly) if len(poly) >= 3 else None
            certifiable = (pred_r is not None and pred_r <= R_CLEAR)
            cur_r = rec["omega_radius_m"] if rec["omega_radius_m"] is not None else 1e9
            sw = 0 if ch == self.c.channel else 1
            if next_ch is not None and ch == next_ch:
                sw -= 1                      # 下一任务正好用该频道: 无需切回
            cands.append({"ch": ch, "cross_pred_deg": round(cross, 1),
                          "mec_before": rec["omega_radius_m"], "dirs_before": rec["n_dirs"],
                          "min_sep_m": round(sep, 1), "pred_radius_m": (None if pred_r is None
                                                                        else round(pred_r, 1)),
                          "predicted_certifiable": bool(certifiable),
                          "switch_cost": sw, "cur_r": cur_r})
        # 分层排序(确定性, 不引入权重): 预计可认证 > 交会角接近90° > Ω 半径大 > 基线长 > 切换少
        cands.sort(key=lambda d: (d["predicted_certifiable"], -abs(d["cross_pred_deg"] - 90.0),
                                  d["cur_r"], d["min_sep_m"], -d["switch_cost"]), reverse=True)
        return cands[:self.OPP_MAX_PER_POINT]

    def _opp_run(self, x, y, sel, site):
        """执行预选频道的机会观测(先清除、后观测; 已在调用处保证仍在原地)。"""
        if not sel:
            return
        # 降低切换次数: 当前频道优先, 下一任务频道放最后(不再切回)
        sel = sorted(sel, key=lambda d: (0 if d["ch"] == self.c.channel else 1))
        snap0 = {k: getattr(self.c, k, None)
                 for k in ("dist", "n_measure", "n_switch", "n_clear", "n_clear_ok")}
        for d in sel:
            ch = d["ch"]
            if (round(x, 1), round(y, 1), ch) in self._meas_seen:
                d["result"] = "skipped_dup"
                continue
            before_dirs = len(self.bearings[ch])
            ok, res, svd = self.c.measure(x, y, ch)
            self._meas_seen.add((round(x, 1), round(y, 1), ch))
            if not ok:
                d["result"] = "rejected"
                continue
            d["result"] = res
            if res in ("direction", "near"):
                if res == "direction":
                    self.bearings[ch].append(((x, y), svd))
                else:
                    self.near_pos[ch] = (x, y)
                rec2, _pt = self._locate_eval(ch)
                d["mec_after"] = rec2["omega_radius_m"]
                d["dirs_after"] = rec2["n_dirs"]
                d["cross_actual_deg"] = rec2["cross_angle_deg"]
                d["became_certified"] = bool(rec2["method"] == "mec")
            else:
                # no_signal: 只记录(保守, 不裁剪可行域、不改状态)
                d["mec_after"] = d["mec_before"]
                d["dirs_after"] = before_dirs
                d["became_certified"] = False
            d.setdefault("cross_actual_deg", None)
            d["site"] = site
            d["point"] = [round(x, 1), round(y, 1)]
            d["phase"] = getattr(self.c, "phase", None)
            snap1 = {k: getattr(self.c, k, None) for k in snap0}
            d["switch_count"] = ((snap1["n_switch"] - snap0["n_switch"])
                                 if isinstance(snap0["n_switch"], (int, float)) else None)
            d["time_cost"] = (round((snap1["dist"]-snap0["dist"])/5.0
                                    + (snap1["n_measure"]-snap0["n_measure"])*5.0
                                    + (snap1["n_switch"]-snap0["n_switch"])*1.0, 1)
                              if isinstance(snap0["dist"], (int, float)) else None)
            d["avoided_supplement"] = None   # 反事实字段, 由 run() 结束时回填(见 _opp_finalize)
            self._diag_list(self.c, "opp_diag").append(d)
        self.log(f"机会观测[{site}] @ ({x:.0f},{y:.0f}): "
                 + ", ".join(f"{d['ch']}({d.get('result')})" for d in sel))

    def _opp_finalize(self):
        """回填 avoided_supplement: 该频道此后未再进入补测队列则记为 True(可观测代理量)。"""
        for d in self._diag_list(self.c, "opp_diag"):
            ch = d.get("ch")
            d["avoided_supplement"] = bool(d.get("became_certified")
                                           and ch not in self.supp_channels)

    # ---- 开放路径最优顺序: n<=9 全排列精确, 否则最近邻 ----
    @staticmethod
    def _optimal_open_path(points, start):
        n = len(points)
        if n <= 1:
            return list(range(n))
        def d(i, j):
            return math.hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
        if n <= 9:
            import itertools
            best, best_len = None, float("inf")
            for perm in itertools.permutations(range(n)):
                L = math.hypot(points[perm[0]][0]-start[0], points[perm[0]][1]-start[1])
                for k in range(1, n):
                    L += d(perm[k-1], perm[k])
                if L < best_len:
                    best_len, best = L, perm
            return list(best)
        order = []
        unvisited = set(range(n))
        cur = start
        while unvisited:
            k = min(unvisited, key=lambda i: math.hypot(points[i][0]-cur[0], points[i][1]-cur[1]))
            order.append(k); cur = points[k]; unvisited.discard(k)
        return order

    # ---- 从 start 出发清除 cpts 的开放路径长度(最近邻+2-opt 近似) ----
    def _clear_route_len(self, start, cpts):
        if not cpts:
            return 0.0
        order = []
        unvisited = set(range(len(cpts)))
        cur = start
        while unvisited:
            k = min(unvisited, key=lambda i: math.hypot(cpts[i][0]-cur[0], cpts[i][1]-cur[1]))
            order.append(cpts[k]); cur = cpts[k]; unvisited.discard(k)
        path = self._two_opt([tuple(start)] + order)
        return sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                   for i in range(len(path)-1))

    # ---- 联合选择: 补测顺序+终点, 使 (补测路径 + 后续清除路线) 最短 ----
    def _joint_order(self, spts, cpts, start):
        import itertools
        n = len(spts)
        if n == 0:
            return []
        # 预计算: 从每个补测点出发的清除路线长度(仅 n 次, 避免在排列里重复算)
        clear_cache = [self._clear_route_len(spts[j], cpts) for j in range(n)]

        def supp_len(seq):
            L = math.hypot(spts[seq[0]][0]-start[0], spts[seq[0]][1]-start[1])
            for k in range(1, len(seq)):
                L += math.hypot(spts[seq[k]][0]-spts[seq[k-1]][0],
                                spts[seq[k]][1]-spts[seq[k-1]][1])
            return L
        if n <= 8:
            best, best_tot = None, float("inf")
            for perm in itertools.permutations(range(n)):
                tot = supp_len(perm) + clear_cache[perm[-1]]
                if tot < best_tot:
                    best_tot, best = tot, perm
            return list(best)
        # n>8 兜底: 最近邻
        order = []
        unvisited = set(range(n))
        cur = start
        while unvisited:
            k = min(unvisited, key=lambda i: math.hypot(spts[i][0]-cur[0], spts[i][1]-cur[1]))
            order.append(k); cur = spts[k]; unvisited.discard(k)
        return order

    # ---- 就近精定位(兜底): 反复测向+前进 ----
    def _homing_clear(self, ch, x, y, phase="homing", src="homing",
                      omega_r=None, cross_ang=None):
        """就近精定位后清除。返回 True 仅当模拟器确实返回 success(不再"以为清除成功")。

        每次清除尝试都记入 clear_diag(带定位来源与当时 Ω 半径), 供诊断/配对实验。
        """
        self._phase(phase)
        pos = (x, y)
        step = 15.0
        for _ in range(8):
            ok, res, svd = self.c.measure(pos[0], pos[1], ch)
            if ok and res == "near":
                # 距离<=5m: 必须检查 accepted 与 clear_result
                okc, rc = self.c.clear(pos[0], pos[1], ch)
                self._note_clear(ch, pos[0], pos[1], src+"|归航near", omega_r, cross_ang,
                                 rc if okc else "rejected")
                if okc and rc == "success":
                    self.state[ch] = "cleared"
                    self.cleared_count += 1
                    return True
                return False                 # 清除未成功: 保持 found 状态
            if ok and res == "direction":
                a = svd * DEG
                pos = (pos[0] + step*math.cos(a), pos[1] + step*math.sin(a))
                okc, rc = self.c.clear(pos[0], pos[1], ch)
                self._note_clear(ch, pos[0], pos[1], src+"|归航direction", omega_r,
                                 cross_ang, rc if okc else "rejected")
                if okc and rc == "success":
                    self.state[ch] = "cleared"
                    self.cleared_count += 1
                    return True
                step *= 0.7
            else:
                return False
        return False

    # ---- 2-opt 局部交换: 对开放路径 [start, p1, ..., pn] 反转段缩短总长 ----
    @staticmethod
    def _two_opt(path):
        def d(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])
        n = len(path) - 1          # 清除点数
        improved = True
        while improved:
            improved = False
            for i in range(1, n):          # 段起点下标
                # j 可取到 n: 开放路径的"尾段反转"只改变一条边 (v_{i-1},v_i)->(v_{i-1},v_n)
                for j in range(i+1, n+1):
                    if j < n:
                        old = d(path[i-1], path[i]) + d(path[j], path[j+1])
                        new = d(path[i-1], path[j]) + d(path[i], path[j+1])
                    else:                  # j == n: 尾段, 无 path[j+1]
                        old = d(path[i-1], path[i])
                        new = d(path[i-1], path[n])
                    if new < old - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]
                        improved = True
        return path

    # ---- 任务级 2-opt: 输入 [(ch,x,y), ...], 整体重排(避免用坐标查表导致碰撞) ----
    @staticmethod
    def _two_opt_tasks(tasks, start):
        path = [(-1, start[0], start[1])] + [tuple(t) for t in tasks]
        n = len(path) - 1

        def d(a, b):
            return math.hypot(a[1]-b[1], a[2]-b[2])
        improved = True
        while improved:
            improved = False
            for i in range(1, n):
                for j in range(i+1, n+1):
                    if j < n:
                        old = d(path[i-1], path[i]) + d(path[j], path[j+1])
                        new = d(path[i-1], path[j]) + d(path[i], path[j+1])
                    else:                     # 尾段反转
                        old = d(path[i-1], path[i])
                        new = d(path[i-1], path[n])
                    if new < old - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]
                        improved = True
        return path[1:]

    # ---- 已发现频道的冗余测量过滤: 当前点交会角有明显改善才值得测 ----
    def _worth_measuring(self, ch, qx, qy):
        dirs = [(p, t) for p, t in self.bearings[ch]]
        if len(dirs) < 2:
            return True                       # 不足两条示向, 需要补
        est = bearing_intersection([p for p, _ in dirs], [t for _, t in dirs])
        if est is None:
            return True                       # 估计退化, 保守测量
        ex, ey = est
        thQ = math.degrees(math.atan2(ey - qy, ex - qx)) % 360
        for _, ti in dirs:
            if crossing_angle(thQ, ti) >= 30.0:
                return True                   # 当前点能形成 >=30° 交会
        return False                          # 冗余, 跳过


def main(argv=None):
    import argparse
    import os
    import datetime
    parser = argparse.ArgumentParser(
        description="问题3 无线电干扰源自动定位与清除机器人")
    parser.add_argument("--robot-id", dest="robot_id", default=os.environ.get("ROBOT_ID"),
                        help="参赛队号(也可用环境变量 ROBOT_ID 提供)")
    parser.add_argument("--base-url", dest="base_url", default=DEFAULT_BASE_URL,
                        help=f"模拟器接口地址(默认 {DEFAULT_BASE_URL})")
    parser.add_argument("--arena-id", dest="arena_id", default=DEFAULT_ARENA_ID,
                        help=f"竞技场ID(默认 {DEFAULT_ARENA_ID})")
    parser.add_argument("--log-file", dest="log_file", default=None,
                        help="本地行为日志输出路径(默认 robot_log_<时间戳>.jsonl)")
    args = parser.parse_args(argv)

    if not args.robot_id:
        print("错误: 未提供参赛队号。用法: python robot.py --robot-id 你的队号 "
              "(或设置环境变量 ROBOT_ID)", file=sys.stderr)
        sys.exit(2)

    # 默认日志写到本程序目录下的 logs/ (与问题4 一致)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(
        log_dir, "p3_log_%s.jsonl" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    client = SimClient(args.base_url, args.robot_id, args.arena_id)
    robot = Problem3Robot(client)
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
        # 无论正常/异常结束, 都落盘本地日志(单个 JSONL 文件:
        # 逐条动作 + 末行 __summary__ 结构化汇总, 不再另写第二份文件)
        if exit_code != 0 and not client.actions:
            client.error = "未连接上模拟器接口(测试未开始或已结束), 本局无有效数据"
        p = client.dump_log(log_file)
        print(f"[日志] 已写入 {p}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
