# -*- coding: utf-8 -*-
"""问题三/四: **贝叶斯概率图**(网格化后验 + 熵减最大的下一个检测点) —— numpy 版。

规格(用户给定的框架):
  1) 目标区域离散为网格, 每个格子 j **对每个频道 c** 维护"该格含该频道源"的概率 p_c(j);
  2) 在位置 s 对频道 c 测得 z ∈ {direction, near, no_signal} 时按贝叶斯规则更新
         p_c(j) <- P(z | 源在 j, s) · p_c(j) / P(z),   P(z) = Σ_j P(z|源在 j, s) p_c(j)
  3) 下一个检测点取使**信息熵减少最大**者
         s* = arg max_s [ H(p_c) - E_z H(p_c | z, s) ]            (式 19)

本实现的实质改进: 题设只给接收半径区间 R_c ∈ [1000,1500] m, 故似然对 R_c **积分**:
    P(no_signal | 源在 j, s) = P(R_c < dist) = clip((dist-1000)/500, 0, 1)
    P(direction | 源在 j, s) = P(R_c >= dist) · 1[落在示向楔形内]
于是 1000~1500 m 的环形区域保留**部分概率**(而不是被当成不可能), 并且"距离 > 1500 m 的
观测点" 的信息增益自动为 0(确定性裁剪无法给出这一点)。
定向源(问题四)再乘"是否落在发射半平面内"的因子。
"""
import math

import numpy as np

R_AREA = 1800.0
R_NEAR = 5.0
R_RX_LO, R_RX_HI = 1000.0, 1500.0
CELL = 200.0
BEAR_HALF = 1.0
EPS = 1e-12


class ProbMap:
    """单频道存在概率图(numpy 向量化)。"""

    def __init__(self, p_channel=0.65, cell=CELL, r_area=R_AREA,
                 r_support=None, out_prior=0.0):
        """r_support: 后验支持集半径(默认 = 目标区半径); out_prior: 落在
        (r_area, r_support] 环带内的先验质量份额。

        **为什么需要 out_prior(问题四的教训)**: 题设把源放在 1800 m 目标区内, 但病理集
        ("边界外指") 会把源放到盘外。若支持集严格等于圆盘, 盘外源的后验质量恒为 0,
        `detect_prob` 也随之恒为 0 —— 于是"零信息增益不测"会把该频道的测量几乎全部跳过,
        反而变慢(实测该病理类 +1 841 s)。故支持集可外扩, 并用 out_prior 保留少量盘外先验。
        """
        self.cell = cell
        self.p_channel = float(p_channel)
        self.r_area = r_area
        self.r_support = r_area if r_support is None else float(r_support)
        self.out_prior = float(out_prior)
        n = int(2*self.r_support/cell) + 1
        xs, ys, ins = [], [], []
        for i in range(n):
            for j in range(n):
                x = -self.r_support + (i + 0.5)*cell
                y = -self.r_support + (j + 0.5)*cell
                rr = math.hypot(x, y)
                if rr <= self.r_support:
                    xs.append(x); ys.append(y); ins.append(rr <= r_area)
        X = np.array(xs); Y = np.array(ys); ins = np.array(ins)
        self.X, self.Y = X, Y
        w = np.where(ins, (1.0-self.out_prior)/max(1, int(ins.sum())),
                     (self.out_prior/max(1, int((~ins).sum()))) if self.out_prior > 0
                     else 0.0)
        if w.sum() <= 0:
            w = np.full(len(xs), 1.0/len(xs))
        self.p = w/w.sum()
        self.n_obs = 0

    # ---------- 似然(向量化) ----------
    def _prx_ge(self, d):
        return np.clip((R_RX_HI - d)/(R_RX_HI - R_RX_LO), 0.0, 1.0)

    def _wedge(self, s, theta):
        dx, dy = self.X - s[0], self.Y - s[1]
        r = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        diff = np.abs((ang - theta + 180.0) % 360.0 - 180.0)
        half = np.degrees(np.arctan2(self.cell*0.71, np.maximum(1e-6, r)))
        return diff <= (BEAR_HALF + half)

    def likelihood(self, z, s, theta, pointing=None, directional=False):
        d = np.hypot(self.X - s[0], self.Y - s[1])
        if z == "near":
            prx = self._prx_ge(d)
            if directional:
                prx = self._half_plane(prx, s, pointing)
            return np.where((d <= R_NEAR) & (prx > 0), 1.0, EPS)
        prx = self._prx_ge(d)
        if directional:
            prx = self._half_plane(prx, s, pointing)
        if z == "direction":
            return np.maximum(EPS, prx*np.where(self._wedge(s, theta), 1.0, EPS))
        if z == "no_signal":
            return np.maximum(EPS, 1.0 - prx)
        raise ValueError(z)

    def _half_plane(self, prx, s, pointing):
        """定向源的发射半平面因子。

        pointing=None 表示**指向未知**(机器人不知道源的朝向): 对均匀指向取边际,
        检测点落在该半平面内的概率 = 1/2 -> 接收概率乘 1/2。这正是问题四中
        `no_signal` 三义("无源/超距/盲区")的概率表达: P(no_signal) >= 1/2 恒成立。
        """
        if pointing is None:
            return 0.5*prx
        ang = np.degrees(np.arctan2(s[1]-self.Y, s[0]-self.X))
        gate = np.where(np.abs((ang - pointing + 180.0) % 360.0 - 180.0) <= 90.0,
                        1.0, 0.0)
        return prx*gate

    # ---------- 贝叶斯更新 ----------
    def update(self, z, s, theta, pointing=None, directional=False):
        lik = self.likelihood(z, s, theta, pointing, directional)
        tot = float(np.dot(lik, self.p))
        if tot <= 0:
            return 0.0
        self.p = lik*self.p/tot
        self.n_obs += 1
        return tot                                     # = P(z)

    # ---------- 熵 / 信息增益(式 19) ----------
    def entropy(self):
        q = self.p[self.p > 0]
        return float(-np.sum(q*np.log(q)))

    def info_gain(self, s, pointing=None, directional=False, n_theta=6):
        """IG(s) = H(p) - E_z H(p|z,s); θ 用 n_theta 个方向近似期望。"""
        h0 = self.entropy()
        exp_h = 0.0
        lik = self.likelihood("near", s, 0.0, pointing, directional)
        pz = float(np.dot(lik, self.p))
        if pz > 1e-9:
            post = lik*self.p/pz
            q = post[post > 0]
            exp_h += pz*float(-np.sum(q*np.log(q)))
        for z in ("direction", "no_signal"):
            for k in range(n_theta):
                th = 360.0*k/n_theta
                lik = self.likelihood(z, s, th, pointing, directional)
                pz = float(np.dot(lik, self.p))
                if pz <= 1e-9:
                    continue
                post = lik*self.p/pz
                q = post[post > 0]
                exp_h += pz*float(-np.sum(q*np.log(q)))/n_theta
        return h0 - exp_h

    def detect_prob(self, s, pointing=None, directional=False):
        """P(该点能收到源) = Σ_j p(j)·P(接收 | j, s) —— 快速判据(信息增益的主导项)。"""
        d = np.hypot(self.X - s[0], self.Y - s[1])
        prx = self._prx_ge(d)
        if directional:
            prx = self._half_plane(prx, s, pointing)
        return float(np.dot(prx, self.p))

    def mecm(self):
        cx = float(np.dot(self.X, self.p))
        cy = float(np.dot(self.Y, self.p))
        mask = self.p > 1e-6
        r = float(np.max(np.hypot(self.X[mask]-cx, self.Y[mask]-cy))) if mask.any() else 0.0
        return (cx, cy), r

    def map_estimate(self):
        j = int(np.argmax(self.p))
        return (float(self.X[j]), float(self.Y[j])), float(self.p[j])


def selftest():
    truth, r_rx = (600.0, -300.0), 1300.0
    m = ProbMap()
    print(f"网格 {len(m.p)} 格(格边 {m.cell:.0f} m), H0 = {m.entropy():.4f} nats")
    for s in ((0.0, 0.0), (1500.0, 0.0), (0.0, 1500.0), (-1500.0, -1500.0)):
        d = math.hypot(truth[0]-s[0], truth[1]-s[1])
        if d <= R_NEAR:
            z, th = "near", 0.0
        elif d <= r_rx:
            z, th = "direction", math.degrees(math.atan2(truth[1]-s[1],
                                                          truth[0]-s[0])) % 360.0
        else:
            z, th = "no_signal", 0.0
        pz = m.update(z, s, th)
        (cx, cy), r = m.mecm()
        (mx, my), pm = m.map_estimate()
        print(f"  {z:>10} @ ({s[0]:6.0f},{s[1]:6.0f}) P(z)={pz:.4f} H={m.entropy():.4f} "
              f"MEC={r:5.0f} m  众数=({mx:5.0f},{my:5.0f}) p={pm:.4f}  "
              f"距真值 {math.hypot(mx-truth[0], my-truth[1]):5.0f} m")
    print("候选点信息增益(式 19)与快速判据:")
    for q in ((500.0, 0.0), (1500.0, 0.0), (0.0, -1500.0), (2000.0, 0.0)):
        print(f"  s=({q[0]:6.0f},{q[1]:6.0f})  IG={m.info_gain(q):.4f}  "
              f"P(收到)={m.detect_prob(q):.4f}")


if __name__ == "__main__":
    selftest()
