# -*- coding: utf-8 -*-
"""
问题2 检验脚本
已知一个定向干扰源在某检测点 S1 的示向度，给出第二检测点 S2 的选择策略与候选区域。
核心：交会定位精度(DOP) 与 定向覆盖概率 的权衡。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEG = np.pi / 180.0
SIGMA = 1.0 * DEG          # 测向误差 1°(弧度)

# ---------------------------------------------------------------
# (1) 数值验证 DOP 解析式: det(C) = sigma^4 r1^2 r2^2 / sin^2(gamma)
# ---------------------------------------------------------------
def bearing_jacobian(G, S):
    dx = G[0] - S[0]; dy = G[1] - S[1]
    r2 = dx*dx + dy*dy
    # d theta / d(x,y) = (-dy/r2, dx/r2)
    return np.array([[-dy/r2, dx/r2]])

def cross_bearing_cov(G, S1, S2):
    H = np.vstack([bearing_jacobian(G, S1), bearing_jacobian(G, S2)])
    return np.linalg.inv(H.T @ H)   # 忽略 sigma^2 缩放

print("=" * 66)
print("[检验1] DOP 解析式 det(C)=sigma^4 r1^2 r2^2 / sin^2(gamma) 核对")
rng = np.random.default_rng(1)
for k in range(5):
    G = np.array([300.0, 200.0])
    S1 = np.array([0.0, 0.0])
    S2 = S1 + rng.uniform(-800, 800, 2)
    C = cross_bearing_cov(G, S1, S2)
    det_num = np.linalg.det(C)
    r1 = np.linalg.norm(G - S1); r2 = np.linalg.norm(G - S2)
    # 交会角 gamma = |ang(G-S1) - ang(G-S2)|
    a1 = np.arctan2(G[1]-S1[1], G[0]-S1[0])
    a2 = np.arctan2(G[1]-S2[1], G[0]-S2[0])
    g = abs(a1 - a2); g = min(g, np.pi - g) if g > np.pi/2 else g  # 取锐交会角
    det_th = r1**2 * r2**2 / np.sin(g)**2
    print(f"  case{k}: det(C)数值={det_num:.4e}, 解析={det_th:.4e}, 比值={det_num/det_th:.6f}")

# ---------------------------------------------------------------
# (2) 固定源位置, DOP 随 S2 变化 -> 最优点在 gamma=90°
# ---------------------------------------------------------------
print("=" * 66)
print("[检验2] 固定源, DOP ∝ r1 r2/sin(gamma) 最小值位置")
S1 = np.array([0.0, 0.0]); G = np.array([800.0, 0.0]); r1 = 800.0
# 在 G 周围一圈等距(r2=400)采样 S2
best = None
for ang in np.linspace(0, 360, 721)[:-1]:
    S2 = G + 400 * np.array([np.cos(ang*DEG), np.sin(ang*DEG)])
    a1 = np.arctan2(G[1]-S1[1], G[0]-S1[0])
    a2 = np.arctan2(G[1]-S2[1], G[0]-S2[0])
    g = abs(a1-a2); g = min(g, np.pi-g)
    dop = r1 * 400 / np.sin(g)
    if best is None or dop < best[0]:
        best = (dop, ang, np.degrees(g))
print(f"  固定 r1=800,r2=400, 最优 S2 方位(相对源)={best[1]:.1f}°, 对应交会角 gamma={best[2]:.1f}°")
print(f"  => 最优交会角接近 90°, 验证 DOP 在 gamma=90° 最小")

# ---------------------------------------------------------------
# (3) 蒙特卡洛验证定向覆盖概率 P = 1 - gamma/180°
# ---------------------------------------------------------------
print("=" * 66)
print("[检验3] 定向干扰源覆盖概率 P = 1 - gamma/180° (蒙特卡洛)")
def coverage_prob_mc(gamma_deg, N=200000):
    rng = np.random.default_rng(3)
    u = np.array([1.0, 0.0])                 # G->S1 方向(设源在原点, S1在+x)
    v = np.array([np.cos(gamma_deg*DEG), np.sin(gamma_deg*DEG)])  # G->S2 方向
    # d 均匀于 {u·d>=0} 半圆
    cnt = 0
    for _ in range(N):
        phi = rng.uniform(-90, 90) * DEG
        d = np.array([np.cos(phi), np.sin(phi)])   # 相对 u 旋转 phi
        d = np.array([np.cos(phi), np.sin(phi)])
        if v @ d >= 0:
            cnt += 1
    return cnt / N
for gd in [0, 30, 45, 60, 75, 90]:
    pm = coverage_prob_mc(gd)
    pt = 1 - gd/180
    print(f"  gamma={gd:3d}°: 蒙特卡洛 P={pm:.4f}, 理论 P=1-gamma/180={pt:.4f}")

# ---------------------------------------------------------------
# (4) 画图: DOP 热力图 + 候选区域(未知 r1, 保证 gamma>=60°)
# ---------------------------------------------------------------
print("=" * 66)
print("[图] DOP 热力图与候选区域")
S1 = np.array([0.0, 0.0]); G = np.array([800.0, 0.0]); r1 = 800.0
xs = np.linspace(-400, 1800, 220)
ys = np.linspace(-1100, 1100, 220)
XX, YY = np.meshgrid(xs, ys)
DOP = np.zeros_like(XX)
for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        S2 = np.array([XX[i, j], YY[i, j]])
        r2 = np.linalg.norm(G - S2)
        if r2 < 5:  # 近距阈值
            DOP[i, j] = np.nan
            continue
        a1 = np.arctan2(G[1]-S1[1], G[0]-S1[0])
        a2 = np.arctan2(G[1]-S2[1], G[0]-S2[0])
        g = abs(a1-a2); g = min(g, np.pi-g)
        if np.sin(g) < 1e-6:
            DOP[i, j] = np.nan
        else:
            DOP[i, j] = np.log10(r1 * r2 / np.sin(g))

fig, ax = plt.subplots(figsize=(9, 7))
cm = ax.pcolormesh(XX, YY, DOP, cmap="viridis", shading="auto")
cb = plt.colorbar(cm, ax=ax)
cb.set_label("log10( DOP ~ r1*r2/sin(gamma) )")
ax.plot(*S1, 'ws', ms=9, mec='k', label='S1 (given detection point)')
ax.plot(*G, 'r*', ms=16, label='source G (unknown, for illustration)')
ax.plot([S1[0], G[0]], [S1[1], G[1]], 'k--', lw=1, label='bearing direction')
# unknown r1 in [500,1500]: guarantee crossing angle >= 60 deg region
rmin, rmax = 500.0, 1500.0
gm = 60.0
xline = np.linspace(rmin, rmax, 400)
yup = np.tan(gm*DEG) * np.maximum(xline - rmin, rmax - xline)
ax.fill_between(xline, yup, 1100, color='cyan', alpha=0.25)
ax.fill_between(xline, -yup, -1100, color='cyan', alpha=0.25)
ax.plot(xline, yup, 'c-', lw=1.5)
ax.plot(xline, -yup, 'c-', lw=1.5,
        label=f'candidate region boundary (guarantee gamma>=60deg, r1 in [{rmin:.0f},{rmax:.0f}])')
ax.set_xlim(-400, 1800); ax.set_ylim(-1100, 1100)
ax.set_aspect('equal')
ax.set_title("Q2: DOP field and candidate region of 2nd detection point")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc='lower left', fontsize=8)
plt.tight_layout()
out = r"D:\My_MathModeling_Project\2026B_solution\verify\problem2_dop.png"
plt.savefig(out, dpi=130)
print("  已保存:", out)
