# -*- coding: utf-8 -*-
"""问题三 执行方案 V2: **概率快速发现 + GDOP/MPC 单源定位 + TSP 清除调度 + 七点覆盖兜底**。

严格按方案实现(§4-§8), 关键决策点与方案条文的对应:
  * §4.1 候选点集: 保证覆盖七点(中心 + r=1500 正六边形) + 概率搜索点(同心环 600/1200/1800,
    六边形对称, 相邻点距 <= 1000 m);
  * §4.2 贝叶斯更新: 似然按方案给出(P(near)=1 当 d<=5; P(direction)=1 当 5<d<=1000;
    P(direction)=(1500-d)/500 当 1000<d<=1500; d>1500 时 no_signal 必然)。
    **实现上补一个 ±1° 楔形因子**(方案 §4.2 未写, 但 §5.1 的 Ω 定义已含该约束; 不加则方向观测
    不会更新位置后验, 概率图退化) -> P(direction|g_i,s,θ)=P(R_c>=d_i)·1[|Δθ|<=1°+格子角宽];
  * §4.3 信息增益: 存在概率的二元熵 H(p_c) 与 IG(s,c)=H-E_e H, 选点按 IG/(T_move+5+1[c≠c0]);
  * §5.3 GDOP 选点: GDOP≈1/|sin γ|, 约束 max_{S∈Ω}‖P-S‖<=1000;
  * §5.4 MPC: 滚动重规划(H=1 贪心; H=2 用预测示向做一步前瞻);
  * §5.5 停止: 最小覆盖圆半径 R_c<=20 -> 清除点加入任务; 中途 near 即地清除;
  * §6 TSP: 最近邻 + 2-opt(复用既有 _optimal_open_path), 顺路插入判据 ΔL;
  * §7 兜底: 未发现频道必须在七点覆盖集全部 no_signal 才可判空。

本模块**不修改** robot.py 的任何既有行为: 只在 CLI 加 --scheme-v2 时替换机器人实现。
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_ROOT = os.path.dirname(HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                   # noqa: E402

import robot as R                                    # noqa: E402
from prob_map import ProbMap                         # noqa: E402

N_CH = R.N_CH
R_AREA = R.R_AREA
R_CLEAR = R.R_CLEAR
R_NEAR = R.R_NEAR
R_GUAR = R.R_GUARANTEE


# ---------------- 候选点集(方案 §4.1) ----------------
def ring_points(r, max_spacing=1000.0, hexsym=True):
    """半径 r 上的环点; 相邻点距 <= max_spacing; hexsym=True 时点数取 6 的倍数(六边形对称)。"""
    if r <= 0:
        return []
    n = max(6, int(math.ceil(2*math.pi*r/max_spacing)))
    if hexsym:
        n = int(math.ceil(n/6.0))*6
    return [(r*math.cos(2*math.pi*k/n), r*math.sin(2*math.pi*k/n)) for k in range(n)]


def cover7(r_cover=1500.0):
    """保证覆盖七点: 中心 + 半径 r_cover 的正六边形六顶点。

    覆盖性(方案 §7 复核): 内部最远 = r/√3(等边三角形外接圆半径), 边界最远 =
    sqrt(R²+r²-2Rr·cos30°); r=1500 时分别为 866.0 m 与 901.7 m, 均 < 1000 m。
    """
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k*60.0*R.DEG
        pts.append((r_cover*math.cos(a), r_cover*math.sin(a)))
    return pts


# ---------------- 存在概率 + 位置后验(方案 §4.2/§4.3) ----------------
class PresenceMap:
    """单频道的 (存在概率 p_c, 位置后验 p_i) 联合贝叶斯模型。

    - 位置后验条件于"该频道存在", 由 ProbMap 维护;
    - 存在概率按贝叶斯更新: 观测 e 下
          P(e|c 存在) = Σ_i p_i·P(e|g_i),   P(e|c 不存在) = 1[e = no_signal]
          p_c' = p_c·P(e|存在) / (p_c·P(e|存在) + (1-p_c)·P(e|不存在))
    - 信息增益用**存在概率的二元熵**(方案 §4.3)。
    """

    def __init__(self, p0=0.5, n_theta=12):
        self.m = ProbMap(p_channel=p0)
        self.p0 = float(p0)
        self.p = float(p0)
        self.n_theta = n_theta
        self.n_obs = 0

    def update(self, e, s, theta):
        lik = self.m.likelihood(e, s, theta or 0.0)
        pe_present = float(np.dot(lik, self.m.p))
        pe_absent = 1.0 if e == "no_signal" else 0.0
        denom = self.p*pe_present + (1.0-self.p)*pe_absent
        if denom > 0:
            self.p = self.p*pe_present/denom
        if pe_present > 0:
            self.m.p = lik*self.m.p/pe_present
        self.n_obs += 1

    @staticmethod
    def _h(p):
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p*math.log(p) - (1.0-p)*math.log(1.0-p)

    def ig(self, s):
        """IG(s,c) = H(p_c) - E_e H(p_c | e, s)。

        三值结果可归约为二元"是否发现": 只有 no_signal 会保留存在不确定性, 而
        near/direction 一旦出现则"该频道存在"变为确定(H(1)=0) —— 故
            P_detect(s) = Σ_i p_i·P(R_c >= d_i)      (方案 §4.2 的距离似然)
            P(no_signal) = 1 - p_c·P_detect(s)
            IG(s,c) = H(p_c) - P(no_signal)·H(p_c | no_signal)
        这正是"去最可能收到信号的点测"的信息论形式(定位后验仍由 update() 用楔形更新)。
        """
        pd = self.m.detect_prob(s)
        p_ns = self.p*(1.0-pd)
        p_after = p_ns/(1.0 - self.p*pd) if (1.0 - self.p*pd) > 1e-12 else self.p
        return max(0.0, self._h(self.p) - (1.0-self.p*pd)*self._h(p_after))

    def mecm(self):
        return self.m.mecm()

    def detect_prob(self, s):
        return self.m.detect_prob(s)


# ---------------- V2 机器人 ----------------
class SchemeV2Robot(R.Problem3Robot):
    """方案 V2 主循环(§8 流程)。"""

    COVER_R = 1500.0            # 七点兜底环半径(方案值)
    RING_RADII = (600.0, 1200.0, 1800.0)
    MAX_SPACING = 1000.0
    ONWAY_DL = 300.0            # 顺路插入判据 ΔL(米)
    GDOP_MAX_MEAS = 4           # 单源定位最多追加测量次数
    MPC_H = 1                   # 滚动优化视界(1=贪心, 2=一步前瞻)
    MIN_IG_RATIO = 1e-4         # 概率层收益下限(低于则转入兜底覆盖)
    MAX_DISCOVERY_MEAS = 120    # 概率层测量预算(超出即转兜底, 防止无界探索)
    COVER_SNAP = 60.0           # 视为"某覆盖点已测过"的坐标容差(米)

    # ---- 几何小工具 ----
    @staticmethod
    def _mec_of(bearings):
        poly = R.feasible_region(bearings)
        if len(poly) < 3:
            return None, None
        ctr, r = R.minimal_enclosing_circle(poly)
        return ctr, r

    def _best_dir_pair(self, ch):
        """返回 (最近一次方位, 与历史方位的最佳交会角)。"""
        dirs = list(self.bearings[ch])
        if not dirs:
            return None, None
        th_new = dirs[-1][1]
        best = 0.0
        for (p, t) in dirs[:-1]:
            best = max(best, R.crossing_angle(t, th_new))
        return th_new, (best if len(dirs) > 1 else None)

    def _gdop_candidates(self, ch):
        """围绕 Ω 的候选观测点(方案 §5.3): 约束 max_{S∈Ω}‖P-S‖ <= 1000。"""
        poly = R.feasible_region(self.bearings[ch])
        if len(poly) < 3:
            return []
        ctr, rad = R.minimal_enclosing_circle(poly)
        out = []
        for rr in (300.0, 600.0, 900.0):
            for k in range(18):
                a = 20.0*k*R.DEG
                q = (ctr[0] + rr*math.cos(a), ctr[1] + rr*math.sin(a))
                if math.hypot(*q) > R_AREA + 400.0:
                    continue
                dmax = max(math.hypot(q[0]-p[0], q[1]-p[1]) for p in poly)
                if dmax <= R_GUAR:                      # 保证可接收
                    out.append(q)
        return out

    def _gdop_score(self, ch, q):
        """GDOP(q) = 1/|sin γ|, γ = 新示向与最佳历史示向的交会角(取 Ω 代表点预测)。"""
        poly = R.feasible_region(self.bearings[ch])
        if len(poly) < 3:
            return float("inf")
        reps = poly if len(poly) <= 8 else poly[::max(1, len(poly)//8)]
        worst = 0.0
        dirs = [t for _, t in self.bearings[ch]]
        for S in reps:
            th = math.degrees(math.atan2(S[1]-q[1], S[0]-q[0])) % 360.0
            g = 0.0
            for t in dirs:
                g = max(g, R.crossing_angle(t, th))
            if g <= 1e-6:
                return float("inf")
            worst = max(worst, 1.0/math.sin(g*R.DEG))
        return worst

    def _best_localize_point(self, ch, here):
        """MPC(H=1/2) 选点: 目标 J = T_measure - λ·(预测最坏直径下降)/K, 越小越好。"""
        cands = self._gdop_candidates(ch)
        if not cands:
            return None
        poly0 = R.feasible_region(self.bearings[ch])
        d0 = R.polygon_diameter(poly0) if len(poly0) >= 2 else 0.0
        best, best_j = None, None
        for q in cands:
            t_meas = math.hypot(q[0]-here[0], q[1]-here[1])/5.0 + 5.0
            if self.MPC_H >= 2:
                worst_d = 0.0
                reps = poly0 if len(poly0) <= 8 else poly0[::max(1, len(poly0)//8)]
                for S in reps:
                    th = math.degrees(math.atan2(S[1]-q[1], S[0]-q[0])) % 360.0
                    v = R._wedge_vertices(self.bearings[ch][-1][0],
                                          self.bearings[ch][-1][1], q, th)
                    if len(v) >= 2:
                        worst_d = max(worst_d, max(math.hypot(a[0]-b[0], a[1]-b[1])
                                                   for a in v for b in v))
                gain = max(0.0, d0 - worst_d)
            else:
                g = self._gdop_score(ch, q)
                gain = 0.0 if math.isinf(g) else max(0.0, d0 - d0*min(1.0, 1.0/g))
            j = t_meas - 0.02*gain                      # λ 取 0.02 s/m(方案 §5.4 的 λ)
            if best_j is None or j < best_j:
                best, best_j = q, j
        return best

    # ---- 动作原语 ----
    def _do_measure(self, s, ch, phase):
        self._phase(phase)
        sw = 1.0 if ch != self._cur_ch else 0.0
        ok, res, svd = self.c.measure(s[0], s[1], ch)
        if not ok:
            return None
        self._cur_ch = ch
        self._mcount[ch] = self._mcount.get(ch, 0) + 1
        self._n_switch += sw
        return res, svd

    def _do_clear(self, ch, Q, phase="queue_clear", src="scheme_v2"):
        self._phase(phase)
        ok, res = self.c.clear(Q[0], Q[1], ch)
        self._note_clear(ch, Q[0], Q[1], src, None, None, res if ok else "rejected")
        if ok and res == "success":
            self.state[ch] = "cleared"
            self.cleared_count += 1
            self.near_pos[ch] = None
            return True
        return False

    # ---- 第一层: 概率引导快速发现(方案 §4) ----
    def _discovery(self):
        cand = [(0.0, 0.0)] + ring_points(600.0, self.MAX_SPACING) \
            + ring_points(1200.0, self.MAX_SPACING) \
            + ring_points(1800.0, self.MAX_SPACING)
        self.cand_pts = cand
        visited = {s: 0 for s in cand}
        while True:
            active = [ch for ch in range(1, N_CH+1)
                      if self.state[ch] is None and 1e-3 < self.pmaps[ch].p < 1-1e-3]
            if not active:
                break
            here = self.c.position
            best = None
            for s in cand:
                for ch in active:
                    ig = self.pmaps[ch].ig(s)
                    if ig <= 0.0:
                        continue
                    t = (math.hypot(s[0]-here[0], s[1]-here[1])/5.0 + 5.0
                         + (1.0 if ch != self._cur_ch else 0.0))
                    score = ig/t
                    if best is None or score > best[0]:
                        best = (score, s, ch, ig)
            if best is None or best[0] <= self.MIN_IG_RATIO:
                self.log("概率层收益耗尽(或无候选点)")
                break
            if self._discovery_meas >= self.MAX_DISCOVERY_MEAS:
                self.log(f"概率层达到测量预算 {self.MAX_DISCOVERY_MEAS} 次, 转入兜底覆盖")
                break
            _, s, ch, ig = best
            here = self.c.position
            mv = math.hypot(s[0]-here[0], s[1]-here[1])
            res = self._do_measure(s, ch, "prob_discovery")
            visited[s] += 1
            self._discovery_meas += 1
            if res is None:
                break
            e, svd = res
            self.pmaps[ch].update(e, s, svd)
            if e == "direction":
                if self.state[ch] is None:
                    self.state[ch] = "found"
                self.bearings[ch].append((s, svd))
                self.log(f"发现频道 {ch} @ ({s[0]:.0f},{s[1]:.0f}) 方位 {svd:.1f}° "
                         f"IG={ig:.3f} 移动 {mv:.0f} m")
            elif e == "near":
                self.state[ch] = "found"
                self.near_pos[ch] = s
                self.log(f"近距发现频道 {ch} @ ({s[0]:.0f},{s[1]:.0f}), 就地清除")
                self._do_clear(ch, s, phase="on_way", src="near")
            else:
                if self._is_cover_point(s):
                    self.ns_cover[ch].add(self._cover_index(s))
            # 顺路清除机会(方案 §6.3)
            self._onway_clear(s)

    def _is_cover_point(self, s):
        return any(math.hypot(s[0]-p[0], s[1]-p[1]) <= self.COVER_SNAP
                   for p in self.cover_pts)

    def _cover_index(self, s):
        for i, p in enumerate(self.cover_pts):
            if math.hypot(s[0]-p[0], s[1]-p[1]) <= self.COVER_SNAP:
                return i
        return -1

    def _onway_clear(self, frm):
        """已定位到 R_c<=20 的频道, 若顺路(ΔL <= 判据)则立刻清除。"""
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found" or self.ready_pos.get(ch) is None:
                continue
            Q = self.ready_pos[ch]
            dL = (math.hypot(Q[0]-frm[0], Q[1]-frm[1]))
            if dL <= self.ONWAY_DL:
                self._do_clear(ch, Q, phase="on_way", src="mec_frozen")

    # ---- 第二层: GDOP/MPC 单源定位(方案 §5) ----
    def _localize(self, ch):
        while self.state[ch] == "found" and self.ready_pos.get(ch) is None:
            if len(self.bearings[ch]) >= 1:
                ctr, rad = self._mec_of(self.bearings[ch])
                if ctr is not None and rad is not None and rad <= R_CLEAR:
                    self.ready_pos[ch] = ctr
                    self.log(f"频道 {ch} 定位完成: Ω 半径 {rad:.1f} m <= {R_CLEAR:.0f} m, "
                             f"清除点 ({ctr[0]:.0f},{ctr[1]:.0f})")
                    return
            if self.near_pos[ch] is not None:
                self.ready_pos[ch] = self.near_pos[ch]
                return
            if self._meas_count(ch) >= self.GDOP_MAX_MEAS:
                break
            q = self._best_localize_point(ch, self.c.position)
            if q is None:
                break
            res = self._do_measure(q, ch, "localize")
            if res is None:
                break
            e, svd = res
            if e == "direction":
                self.bearings[ch].append((q, svd))
            elif e == "near":
                self.near_pos[ch] = q
                self.ready_pos[ch] = q
                return
            elif e == "no_signal":
                # 定向/超距导致无信号: 该点不提供信息, 记录后换点
                self.log(f"定位点 ({q[0]:.0f},{q[1]:.0f}) 返回 no_signal, 换点")
                break
        # 兜底: 用既有补测点 + 二分归航(保证不遗留)
        if self.state[ch] == "found" and self.ready_pos.get(ch) is None:
            sp = self._supplement_point(ch, self.c.position)
            if sp is not None:
                res = self._do_measure(sp, ch, "supplement")
                if res is not None:
                    e, svd = res
                    if e == "direction":
                        self.bearings[ch].append((sp, svd))
                    elif e == "near":
                        self.near_pos[ch] = sp
                        self.ready_pos[ch] = sp
                        return
            ctr, rad = self._mec_of(self.bearings[ch])
            if ctr is not None and rad is not None and rad <= R_CLEAR:
                self.ready_pos[ch] = ctr
                return
            Q = self._binary_homing(ch)
            if Q is not None:
                self.ready_pos[ch] = Q

    def _meas_count(self, ch):
        try:
            return self._mcount[ch]
        except (AttributeError, KeyError):
            return 0

    # ---- 兜底: 七点覆盖(方案 §7) ----
    def _fallback(self):
        need = [ch for ch in range(1, N_CH+1) if self.state[ch] is None]
        if not need:
            return
        self.log(f"兜底覆盖: {len(need)} 个未发现频道需在七点集完成排查")
        order = R.Problem3Robot._optimal_open_path(self.cover_pts, tuple(self.c.position))
        for idx in order:
            s = self.cover_pts[idx]
            still = [ch for ch in need if self.state[ch] is None]
            if not still:
                break
            for ch in still:
                res = self._do_measure(s, ch, "cover_fallback")
                if res is None:
                    continue
                e, svd = res
                self.pmaps[ch].update(e, s, svd)
                if e == "direction":
                    self.state[ch] = "found"          # 关键: 兜底阶段发现也必须置位
                    self.bearings[ch].append((s, svd))
                    self.log(f"兜底阶段发现频道 {ch} @ ({s[0]:.0f},{s[1]:.0f})")
                elif e == "near":
                    self.state[ch] = "found"
                    self.near_pos[ch] = s
                    self.ready_pos[ch] = s
                    self._do_clear(ch, s, phase="on_way", src="near")
                else:
                    self.ns_cover[ch].add(idx)
            for ch in need:
                if self.state[ch] is None and len(self.ns_cover[ch]) >= len(self.cover_pts):
                    self.state[ch] = "excluded"
                    self.log(f"频道 {ch} 在七点覆盖集全部 no_signal -> 判定不存在")

    # ---- 第三层: TSP 清除调度(方案 §6) ----
    def _clear_tour(self):
        todo = {ch: self.ready_pos[ch] for ch in range(1, N_CH+1)
                if self.state[ch] == "found" and self.ready_pos.get(ch) is not None}
        if not todo:
            return
        chs = list(todo.keys())
        pts = [todo[ch] for ch in chs]
        order = R.Problem3Robot._optimal_open_path(pts, tuple(self.c.position))
        for i in order:
            ch, Q = chs[i], pts[i]
            if self.state[ch] != "found":
                continue
            if not self._do_clear(ch, Q):
                Q2 = self._binary_homing(ch)
                if Q2 is not None and self._do_clear(ch, Q2, src="homing"):
                    continue
                self.log(f"频道 {ch} 清除失败(已归航重试)")

    # ---- 主流程(方案 §8) ----
    def run(self):
        c = self.c
        self.log("调用 /enter ...")
        c.enter()
        self._phase("init")
        self.cover_pts = cover7(self.COVER_R)
        self.pmaps = {ch: PresenceMap(0.5) for ch in range(1, N_CH+1)}
        self.ns_cover = {ch: set() for ch in range(1, N_CH+1)}
        self.ready_pos = {ch: None for ch in range(1, N_CH+1)}
        self._mcount = {ch: 0 for ch in range(1, N_CH+1)}
        self._cur_ch = 1                     # 题设: 初始频道为 1
        self._n_switch = 0.0
        self._discovery_meas = 0
        self.log(f"七点覆盖集: 中心 + r={self.COVER_R:.0f} 六边形; 覆盖余量 "
                 f"{R_GUAR - math.sqrt(R_AREA**2 + self.COVER_R**2 - 2*R_AREA*self.COVER_R*math.cos(30*R.DEG)):.1f} m")

        # 第一层: 概率引导发现 + 七点兜底(把"哪些频道有源"彻底判完)
        self._phase("prob_discovery")
        self._discovery()
        self._phase("cover_fallback")
        self._fallback()
        # 第二层: 对**全部**已发现频道做 GDOP/MPC 定位(必须在发现完成之后, 否则兜底阶段
        #         才发现的频道会被遗漏 —— 首版即踩此坑, 导致 7 个源"发现但未定位")
        self._phase("localize")
        for ch in range(1, N_CH+1):
            if self.state[ch] == "found":
                self._localize(ch)
                self._onway_clear(self.c.position)
        # 第三层: TSP 清除
        self._phase("queue_clear")
        self._clear_tour()
        # 收尾: 仍有未清除频道 -> 归航重试一轮(保证不留残)
        left = [ch for ch in range(1, N_CH+1) if self.state[ch] == "found"]
        if left:
            self._phase("recovery")
            for ch in left:
                Q = self._binary_homing(ch)
                if Q is not None and self._do_clear(ch, Q, src="homing_retry"):
                    self.log(f"收尾清除成功: 频道 {ch}")
        # 终止校验(与既有语义一致: 仍有未清除/未排查频道则标记 incomplete)
        self._phase("exit")
        unresolved = []
        for ch in range(1, N_CH+1):
            if self.state[ch] == "found":
                unresolved.append(ch)
        if unresolved:
            self.exit_status = "incomplete"
            self.unresolved_kind["found_uncleared"] = unresolved
        c.exit()
        return self.cleared_count
