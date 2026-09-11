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
        """结构化 JSON 汇总(与问题4 同 schema: 配置/接口计数/移动距离/逐频道结果)。"""
        moves = 0.0
        prev = (0.0, 0.0)
        for a in self.actions:
            pos = a.get("payload", {}).get("position")
            if pos and a.get("path") in ("/measure", "/clear"):
                moves += math.hypot(pos["x"]-prev[0], pos["y"]-prev[1])
                prev = (pos["x"], pos["y"])
        clears = [a for a in self.actions if a["path"] == "/clear"]
        s = {
            "problem": 3,
            "team_no": self.robot_id,
            "base_url": self.base_url,
            "config": {"hex_r": HEX_R, "n_cover_points": 7,
                       "on_way_delta": getattr(Problem3Robot, "ON_WAY_DELTA", None),
                       "r_clear": R_CLEAR},
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
        if self.error:
            s["error"] = self.error
        return s

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
        self.position = (x, y)
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
    """用正 m 边形保守近似目标圆盘。"""
    return [(R*math.cos(2*math.pi*k/m), R*math.sin(2*math.pi*k/m)) for k in range(m)]

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

    def __init__(self, client):
        self.c = client
        # 频道状态: None=待排查, 'found'=已发现, 'excluded'=已排除, 'cleared'=已清除
        self.state = {ch: None for ch in range(1, N_CH+1)}
        self.bearings = {ch: [] for ch in range(1, N_CH+1)}   # 已发现频道的示向观测
        self.near_pos = {ch: None for ch in range(1, N_CH+1)}  # near 时的位置
        self.visited_no_signal = {ch: set() for ch in range(1, N_CH+1)}  # 记录无信号搜索点
        self.cleared_count = 0

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
                for c2 in range(1, N_CH+1):
                    if self.state[c2] != "found":
                        continue
                    if self.near_pos[c2] is not None:
                        Q = self.near_pos[c2]
                    else:
                        Q = self._locate_quick(c2)
                        if Q is None:
                            continue
                    cur = c.position
                    dL = (math.hypot(Q[0]-cur[0], Q[1]-cur[1])
                          + math.hypot(Q[0]-nxt[0], Q[1]-nxt[1])
                          - math.hypot(cur[0]-nxt[0], cur[1]-nxt[1]))
                    if dL <= self.ON_WAY_DELTA:
                        ok, res = c.clear(Q[0], Q[1], c2)
                        if ok and res == "success":
                            self.state[c2] = "cleared"
                            self.cleared_count += 1
                            self.log(f"顺路清除: 频道 {c2} @ ({Q[0]:.0f},{Q[1]:.0f})  "
                                     f"[{self.cleared_count}]")
                        else:
                            self._homing_clear(c2, Q[0], Q[1])

        # 状态层: 排除判定(遍历完全部覆盖点且均无信号)
        for ch in range(1, N_CH+1):
            if self.state[ch] is None and len(self.visited_no_signal[ch]) >= len(pts):
                self.state[ch] = "excluded"
        self.log(f"扫描完成: 已发现 {sum(1 for s in self.state.values() if s=='found')} 个频道, "
                 f"已排除 {sum(1 for s in self.state.values() if s=='excluded')} 个频道")

        # 状态层: 快速定位(不移动) —— 可分两类: 已可清除 / 需补测
        clear_tasks = []   # (ch, x, y) 已可清除
        supp_tasks = []    # (ch, x, y) 需补测
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found":
                continue
            if self.near_pos[ch] is not None:
                clear_tasks.append((ch, self.near_pos[ch][0], self.near_pos[ch][1]))
                continue
            pt = self._locate_quick(ch)
            if pt is not None:
                clear_tasks.append((ch, pt[0], pt[1]))
            else:
                sup = (self._dop_supplement_point(ch, c.position)
                       if self.DOP_PRESCREEN
                       else self._supplement_point(ch, c.position))
                if sup is not None:
                    supp_tasks.append((ch, sup[0], sup[1]))

        # 调度层: 批量补测 —— 全部补测点做开放路径最优排序后统一执行
        # (注: 实测"补测终点与清除路线联合选择"无收益, 因清除点集在补测后仍会增长,
        #  联合评价只能基于不完整信息; 而清除 2-opt 路径对起点不敏感)
        if supp_tasks:
            self.log(f"补测任务 {len(supp_tasks)} 个, 开放路径优化")
            spts = [(x, y) for _, x, y in supp_tasks]
            spos2ch = {(x, y): ch for ch, x, y in supp_tasks}
            sorder = self._optimal_open_path(spts, tuple(c.position))
            for idx in sorder:
                x, y = spts[idx]
                ch = spos2ch[(x, y)]
                ok, res, svd = c.measure(x, y, ch)
                if ok and res == "near":
                    clear_tasks.append((ch, x, y))
                    continue
                if ok and res == "direction":
                    self.bearings[ch].append(((x, y), svd))
                pt = self._locate_quick(ch)
                if pt is not None:
                    clear_tasks.append((ch, pt[0], pt[1]))
                else:
                    # 补测后仍不足以定位 -> 二分归航(保证)
                    bt = self._binary_homing(ch)
                    if bt is not None:
                        clear_tasks.append((ch, bt[0], bt[1]))

        # 调度层: 任务队列就近清除 —— 最近邻生成初始路径 + 2-opt 局部交换优化
        self.log(f"清除任务队列 {len(clear_tasks)} 个")
        pts = [(x, y) for _, x, y in clear_tasks]
        pos2ch = {(x, y): ch for ch, x, y in clear_tasks}
        # 最近邻初始顺序
        order = []
        unvisited = set(range(len(pts)))
        cur = c.position
        while unvisited:
            k = min(unvisited, key=lambda i: math.hypot(pts[i][0]-cur[0], pts[i][1]-cur[1]))
            order.append(pts[k]); cur = pts[k]; unvisited.discard(k)
        # 2-opt 局部交换
        path = self._two_opt([tuple(c.position)] + order)
        for (x, y) in path[1:]:
            ch = pos2ch[(x, y)]
            ok, res = c.clear(x, y, ch)
            if ok and res == "success":
                self.state[ch] = "cleared"
                self.cleared_count += 1
                self.log(f"清除成功: 频道 {ch} @ ({x:.0f},{y:.0f})  [{self.cleared_count}]")
            else:
                self.log(f"清除未发现: 频道 {ch} @ ({x:.0f},{y:.0f}), 做就近精定位")
                self._homing_clear(ch, x, y)

        # 回填结构化元信息(与问题4 同 schema)
        c.meta = {
            "cleared_count": self.cleared_count,
            "found_channels": sum(1 for s in self.state.values() if s == "found"),
            "excluded_channels": sum(1 for s in self.state.values() if s == "excluded"),
            "unresolved_channels": sum(1 for s in self.state.values()
                                      if s not in ("cleared", "excluded")),
            "channels": {str(ch): {"state": self.state[ch],
                                   "bearings": len(self.bearings[ch]),
                                   "near": self.near_pos[ch] is not None}
                         for ch in range(1, N_CH+1)},
        }
        self.log("调用 /exit ...")
        c.exit()
        self.log(f"结束, 清除 {self.cleared_count} 个干扰源")
        return self.cleared_count

    # ---- 快速定位(不移动): 可行域最小覆盖圆<=20m 或 最小二乘交会 ----
    def _locate_quick(self, ch):
        dirs = [(pos, th) for pos, th in self.bearings[ch]]
        if len(dirs) >= 2:
            poly = feasible_region(dirs)
            if len(poly) >= 3:
                center, radius = minimal_enclosing_circle(poly)
                if radius <= R_CLEAR:
                    return center
            best_i, best_j, best_ang = 0, 1, -1
            for i in range(len(dirs)):
                for j in range(i+1, len(dirs)):
                    a = crossing_angle(dirs[i][1], dirs[j][1])
                    if a > best_ang:
                        best_ang, best_i, best_j = a, i, j
            if best_ang >= 30.0:
                return bearing_intersection([dirs[best_i][0], dirs[best_j][0]],
                                            [dirs[best_i][1], dirs[best_j][1]])
        return None

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
    def _homing_clear(self, ch, x, y):
        pos = (x, y)
        step = 15.0
        for _ in range(8):
            ok, res, svd = self.c.measure(pos[0], pos[1], ch)
            if ok and res == "near":
                self.c.clear(pos[0], pos[1], ch)
                self.state[ch] = "cleared"
                self.cleared_count += 1
                return
            if ok and res == "direction":
                a = svd * DEG
                pos = (pos[0] + step*math.cos(a), pos[1] + step*math.sin(a))
                okc, rc = self.c.clear(pos[0], pos[1], ch)
                if okc and rc == "success":
                    self.state[ch] = "cleared"
                    self.cleared_count += 1
                    return
                step *= 0.7
            else:
                return

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
                for j in range(i+1, n):    # 段终点下标(保证 j+1<=n)
                    old = d(path[i-1], path[i]) + d(path[j], path[j+1])
                    new = d(path[i-1], path[j]) + d(path[i], path[j+1])
                    if new < old - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]
                        improved = True
        return path

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
