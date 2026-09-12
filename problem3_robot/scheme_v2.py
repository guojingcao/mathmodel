# -*- coding: utf-8 -*-
"""问题三 执行方案 V2.1: **单一巡回**(改法1) + **保证集即定位集**(改法2) + **现行恢复闭环**(改法3)。

相对 V2.0 的结构性修改(按用户三步改法):
  改法1 合并层: 只用**一条滚动巡回**同时承载 覆盖证书任务 / 概率搜索任务 / 定位任务 / 清除任务;
        每一步按"类别优先 + 最近代价 + 顺路插入(ΔL <= ONWAY_DL)"选动作, 不再分三阶段各跑一趟;
  改法2 保证集即定位集: 覆盖环直接用现行几何优化环(1030 m × 8 点 = search_points()), 
        使"发现即可就地交会", 未解决频道数大幅下降; 概率点仅作**顺路**(ΔL 小)时的额外观测;
  改法3 恢复闭环: 收尾直接复用现行 Problem3Robot._recovery()(多轮 + 邻域兜底 + 未解决判定),
        并同步 visited_no_signal 索引, 保证"未发现频道必须覆盖点全 no_signal 才判空"。

滚动规则(单一巡回的具体形式):
  A) 顺路清除: 任一 ready 频道的插入增量 ΔL <= ONWAY_DL -> 立即清除;
  B) 顺路观测: 某个定位点/概率点满足 ΔL <= ONWAY_DL 且 IG > 0 -> 顺路去测;
  C) 否则: 在**未访问覆盖点**中按"信息增益/代价"选下一个覆盖点, 到点上对全部未发现频道逐个测量
     (蛇形频道序, 已发现频道不再重复扫描)。
停止: 全部频道 cleared/excluded; 之后进入改法3 的恢复闭环并做终止判定。
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
R_GUAR = R.R_GUARANTEE


def ring_points(r, max_spacing=1000.0):
    """半径 r 上的环点(六边形对称), 相邻点距 <= max_spacing。"""
    if r <= 0:
        return []
    n = max(6, int(math.ceil(2*math.pi*r/max_spacing)))
    n = int(math.ceil(n/6.0))*6
    return [(r*math.cos(2*math.pi*k/n), r*math.sin(2*math.pi*k/n)) for k in range(n)]


class PresenceMap:
    """单频道 (存在概率, 位置后验) 联合贝叶斯模型(似然对 R_c∈[1000,1500] 积分 + ±1° 楔形)。"""

    def __init__(self, p0=0.5):
        self.m = ProbMap(p_channel=p0)
        self.p = float(p0)
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
        return 0.0 if (p <= 0.0 or p >= 1.0) else -p*math.log(p) - (1.0-p)*math.log(1.0-p)

    def ig(self, s):
        """IG = H(p) - P(no_signal)·H(p | no_signal) (发现即确定存在, 故只需二元)。"""
        pd = self.m.detect_prob(s)
        p_ns = self.p*(1.0-pd)
        den = 1.0 - self.p*pd
        p_after = p_ns/den if den > 1e-12 else self.p
        return max(0.0, self._h(self.p) - (1.0-self.p*pd)*self._h(p_after))


class SchemeV2Robot(R.Problem3Robot):
    """V2.1: 单一巡回 + 保证集即定位集 + 复现恢复闭环。"""

    ONWAY_DL = 300.0            # 顺路插入判据(米)
    MIN_IG = 1e-3               # 覆盖点排序用的信息增益门槛
    GDOP_MAX_MEAS = 4           # 单源定位最多追加测量
    PROB_EXTRA_R = (1080.0, 1800.0)   # 额外观测环(仅顺路时访问)
    PROB_MAX_DETOUR = 300.0     # 额外概率点的最大绕行增量(米)

    # ---------- 几何 ----------
    @staticmethod
    def _mec_of(bearings):
        poly = R.feasible_region(bearings)
        if len(poly) < 3:
            return None, None
        return R.minimal_enclosing_circle(poly)

    def _gdop_point(self, ch):
        """GDOP 选点(方案 §5.3): 约束 max_{S∈Ω}‖P-S‖ <= 1000, 取 1/|sinγ| 最小。"""
        poly = R.feasible_region(self.bearings[ch])
        if len(poly) < 3:
            return None
        ctr, _rad = R.minimal_enclosing_circle(poly)
        reps = poly if len(poly) <= 8 else poly[::max(1, len(poly)//8)]
        dirs = [t for _, t in self.bearings[ch]]
        best, best_g = None, None
        for rr in (300.0, 600.0, 900.0):
            for k in range(18):
                a = 20.0*k*R.DEG
                q = (ctr[0] + rr*math.cos(a), ctr[1] + rr*math.sin(a))
                dmax = max(math.hypot(q[0]-p[0], q[1]-p[1]) for p in poly)
                if dmax > R_GUAR:
                    continue
                g = 0.0
                for S in reps:
                    th = math.degrees(math.atan2(S[1]-q[1], S[0]-q[0])) % 360.0
                    gg = max((R.crossing_angle(t, th) for t in dirs), default=0.0)
                    if gg <= 1e-6:
                        g = float("inf")
                        break
                    g = max(g, 1.0/math.sin(gg*R.DEG))
                if g != float("inf") and (best_g is None or g < best_g):
                    best, best_g = q, g
        return best

    # ---------- 动作原语 ----------
    def _measure(self, s, ch, phase):
        self._phase(phase)
        sw = 1.0 if ch != self._cur_ch else 0.0
        ok, res, svd = self.c.measure(s[0], s[1], ch)
        if not ok:
            return None
        self._cur_ch = ch
        self._mcount[ch] = self._mcount.get(ch, 0) + 1
        self._n_switch += sw
        self._mark_seen(ch, s)
        return res, svd

    def _clear(self, ch, Q, src="scheme_v21"):
        self._phase("on_way" if src == "on_way" else "queue_clear")
        ok, res = self.c.clear(Q[0], Q[1], ch)
        self._note_clear(ch, Q[0], Q[1], src, None, None, res if ok else "rejected")
        if ok and res == "success":
            self.state[ch] = "cleared"
            self.cleared_count += 1
            self.ready_pos[ch] = None
            self.near_pos[ch] = None
            return True
        return False

    def _apply(self, ch, e, s, svd, cover_idx=None):
        """统一状态/概率更新。"""
        self.pmaps[ch].update(e, s, svd)
        if e == "direction":
            if self.state[ch] is None:
                self.state[ch] = "found"
            self.bearings[ch].append((s, svd))
        elif e == "near":
            self.state[ch] = "found"
            self.near_pos[ch] = s
            self.ready_pos[ch] = s
        else:
            if cover_idx is not None:
                self.ns_cover[ch].add(cover_idx)
                self.visited_no_signal[ch].add(cover_idx)   # 与现行 _recovery 口径对齐
        if self.state[ch] == "found" and self.ready_pos[ch] is None:
            ctr, rad = self._mec_of(self.bearings[ch])
            if ctr is not None and rad is not None and rad <= R_CLEAR:
                self.ready_pos[ch] = ctr

    # ---------- 单一巡回的决策(改法1) ----------
    def _resolved(self, ch):
        return self.state[ch] in ("cleared", "excluded")

    def _all_resolved(self):
        return all(self._resolved(ch) for ch in range(1, N_CH+1))

    def _unresolved(self):
        return [ch for ch in range(1, N_CH+1)
                if self.state[ch] is None or self.state[ch] == "found"]

    def _dl(self, frm, Q, to=None):
        """把 Q 插到 frm→to 之间的增量(无 to 时退化为到 Q 的距离)。"""
        if to is None:
            return math.hypot(Q[0]-frm[0], Q[1]-frm[1])
        return (math.hypot(Q[0]-frm[0], Q[1]-frm[1])
                + math.hypot(Q[0]-to[0], Q[1]-to[1])
                - math.hypot(frm[0]-to[0], frm[1]-to[1]))

    def _next_action(self, measured, next_cover):
        """返回下一步动作: ('clear',ch,Q) / ('measure',ch,pos,phase) / ('cover',idx)。"""
        here = self.c.position
        # A) 顺路清除(清除点优先插入当前路径附近)
        best = None
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found" or self.ready_pos[ch] is None:
                continue
            Q = self.ready_pos[ch]
            dL = self._dl(here, Q, next_cover)
            if dL <= self.ONWAY_DL:
                return ("clear", ch, Q)
            if best is None or dL < best[0]:
                best = (dL, ch, Q)
        # B) 顺路观测: 定位点(已发现未就绪) 或 概率点(未发现且 IG>门槛)
        for ch in range(1, N_CH+1):
            if self.state[ch] != "found" or self.ready_pos[ch] is not None:
                continue
            if self._mcount.get(ch, 0) >= self.GDOP_MAX_MEAS:
                continue
            q = self._gdop_point(ch)
            if q is not None and not self._is_seen(ch, q) \
                    and self._dl(here, q, next_cover) <= self.ONWAY_DL:
                return ("measure", ch, q, "localize")
        for ch in range(1, N_CH+1):
            if self.state[ch] is not None:
                continue
            for p in self.prob_pts:
                if self._is_seen(ch, p) or self.pmaps[ch].ig(p) <= self.MIN_IG:
                    continue
                if self._dl(here, p, next_cover) <= self.PROB_MAX_DETOUR:
                    return ("measure", ch, p, "prob")
        # C) 推进覆盖证书: 未访问覆盖点中按 IG/代价选一个(该点上的频道测量由执行器完成)
        if next_cover is not None:
            return ("cover", next_cover)
        return None

    def _is_seen(self, ch, pos):
        """(频道, 坐标) 是否已测过(纯查询, 无副作用)。"""
        return (int(ch), round(pos[0], 3), round(pos[1], 3)) in self._sites

    def _mark_seen(self, ch, pos):
        self._sites.add((int(ch), round(pos[0], 3), round(pos[1], 3)))

    def _pick_cover(self, measured):
        """未完全覆盖的点里, 按 Σ_c IG(c,p)/代价 选下一个(不改变"每点必访"的硬约束)。"""
        here = self.c.position
        best, best_v = None, None
        for idx, p in enumerate(self.cover_pts):
            pend = [ch for ch in range(1, N_CH+1)
                    if self.state[ch] is None and (ch, idx) not in measured]
            if not pend:
                continue
            gain = sum(self.pmaps[ch].ig(p) for ch in pend)
            cost = math.hypot(p[0]-here[0], p[1]-here[1])/5.0 + 5.0*len(pend)
            v = gain/max(1e-9, cost)
            if best_v is None or v > best_v:
                best, best_v = idx, v
        return best

    # ---------- 主循环 ----------
    def run(self):
        c = self.c
        self.log("调用 /enter ...")
        c.enter()
        self._phase("init")
        self.cover_pts = list(R.Problem3Robot.search_points())      # 改法2: 1030x8 环
        self.prob_pts = [(0.0, 0.0)] + ring_points(self.PROB_EXTRA_R[0], 1000.0) \
            + ring_points(self.PROB_EXTRA_R[1], 1000.0)
        self.pmaps = {ch: PresenceMap(0.5) for ch in range(1, N_CH+1)}
        self.ns_cover = {ch: set() for ch in range(1, N_CH+1)}
        self.ready_pos = {ch: None for ch in range(1, N_CH+1)}
        self._mcount = {ch: 0 for ch in range(1, N_CH+1)}
        self._cur_ch = 1
        self._n_switch = 0.0
        self._sites = set()
        self.log(f"单一巡回: 覆盖环 {len(self.cover_pts)} 点(即保证集与定位集同一套), "
                 f"额外观测点 {len(self.prob_pts)} 个(仅 ΔL <= {self.PROB_MAX_DETOUR:.0f} m 时顺路访问)")
        measured = set()
        guard = 0
        while not self._all_resolved() and guard < 4000:
            guard += 1
            nxt = self._pick_cover(measured)
            if nxt is None:
                break
            act = self._next_action(measured, self.cover_pts[nxt])
            if act is None:
                break
            if act[0] == "clear":
                _, ch, Q = act
                if not self._clear(ch, Q, src="on_way"):
                    self.log(f"频道 {ch} 顺路清除失败, 转由恢复闭环处理")
                continue
            if act[0] == "measure":
                _, ch, pos, phase = act
                r = self._measure(pos, ch, phase)
                if r is None:
                    continue
                e, svd = r
                cidx = None
                for idx, p in enumerate(self.cover_pts):
                    if math.hypot(pos[0]-p[0], pos[1]-p[1]) <= 1.0:
                        cidx = idx
                        break
                self._apply(ch, e, pos, svd, cidx)
                if self.state[ch] == "found" and self.ready_pos[ch] is not None:
                    if self._dl(self.c.position, self.ready_pos[ch], None) <= self.ONWAY_DL:
                        self._clear(ch, self.ready_pos[ch], src="on_way")
                continue
            # ('cover', p): 到覆盖点, 对全部未发现频道逐个测量(该点的证书一次做完)
            _, p = act
            idx = min(range(len(self.cover_pts)),
                      key=lambda i: math.hypot(p[0]-self.cover_pts[i][0],
                                               p[1]-self.cover_pts[i][1]))
            self._phase("coverage")
            order = (list(range(1, N_CH+1)) if idx % 2 == 0 else list(range(N_CH, 0, -1)))
            for ch in order:
                if self.state[ch] is not None or (ch, idx) in measured:
                    continue
                r = self._measure(p, ch, "coverage")
                if r is None:
                    continue
                e, svd = r
                measured.add((ch, idx))
                self._apply(ch, e, p, svd, idx)
            for ch in range(1, N_CH+1):
                if (self.state[ch] is None
                        and len(self.ns_cover[ch]) >= len(self.cover_pts)):
                    self.state[ch] = "excluded"
                    self.log(f"频道 {ch}: 覆盖环全部 no_signal -> 判定不存在")
            # 覆盖点上也顺路清除一次
            for ch in range(1, N_CH+1):
                if (self.state[ch] == "found" and self.ready_pos[ch] is not None
                        and self._dl(self.c.position, self.ready_pos[ch], None)
                        <= self.ONWAY_DL):
                    self._clear(ch, self.ready_pos[ch], src="on_way")
        # ---- 改法3: 复用现行恢复闭环(多轮 + 邻域兜底 + 未解决判定) ----
        if any(not self._resolved(ch) for ch in range(1, N_CH+1)):
            self._phase("recovery")
            self.log("进入恢复闭环 ...")
            self._recovery()
        self._phase("exit")
        unresolved = [ch for ch in range(1, N_CH+1) if not self._resolved(ch)]
        if unresolved:
            self.exit_status = "incomplete"
            self.unresolved_kind["found_uncleared"] = unresolved
        c.exit()
        return self.cleared_count
