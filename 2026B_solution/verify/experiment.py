# -*- coding: utf-8 -*-
"""
模型竞技实验平台 (M0-M4) —— 终版
统一环境 + 8项指标 + 蒙特卡洛对比 + 消融(DOP/概率/PSO) + 鲁棒性(定向比例)
M0 随机 | M1 固定覆盖 | M2 +DOP | M3 +概率(动态删除) | M4 混合
"""
import numpy as np
from itertools import combinations

DEG = np.pi / 180.0
SPEED = 5.0; T_MEAS = 5.0; T_SWITCH = 1.0; T_CLEAR_OK = 5.0
R_AREA = 1800.0; R_CLEAR = 20.0; R_NEAR = 5.0; N_CH = 20

# ================= 统一环境 =================
class Env:
    def __init__(self, rng, n_src=None, directional=False, p_dir=0.5, r_min=1000.0, r_max=1500.0):
        self.rng = rng; self.directional = directional
        self.n_src = n_src if n_src is not None else int(rng.integers(10, 17))
        chans = rng.choice(np.arange(1, N_CH+1), size=self.n_src, replace=False)
        self.sources = []
        for c in chans:
            r = R_AREA*np.sqrt(rng.uniform(0,1)); a = rng.uniform(0, 2*np.pi)
            pointing = (rng.uniform(0,2*np.pi) if (directional and rng.uniform()<p_dir) else None)
            self.sources.append(dict(pos=np.array([r*np.cos(a), r*np.sin(a)]),
                                     ch=int(c), r_rx=float(rng.uniform(r_min, r_max)), pointing=pointing))
        self.cleared = set(); self.ch_by_id = {s['ch']: s for s in self.sources}
    def _in_cov(self, s, pos):
        if s['pointing'] is None: return True
        d = pos - s['pos']; ang = np.arctan2(d[1], d[0])
        diff = abs(ang - s['pointing']); diff = min(diff, 2*np.pi-diff)
        return diff <= np.pi/2 + 1e-12
    def measure(self, pos, ch):
        for s in self.sources:
            if s['ch'] != ch or ch in self.cleared: continue
            d = s['pos'] - pos; dist = np.linalg.norm(d)
            if dist > s['r_rx'] or not self._in_cov(s, pos): continue
            if dist <= R_NEAR: return 'near', None
            true = np.degrees(np.arctan2(d[1], d[0])) % 360
            return 'direction', (true + self.rng.uniform(-1, 1)) % 360
        return 'no_signal', None
    def clear(self, pos, ch):
        for s in self.sources:
            if s['ch'] == ch and ch not in self.cleared:
                if np.linalg.norm(s['pos']-pos) <= R_CLEAR:
                    self.cleared.add(ch); return 'success'
        return 'no_target_in_range'

# ================= 几何 =================
def bearing_intersection(stations, bearings_deg):
    A = np.zeros((2,2)); b = np.zeros(2)
    for (x,y), th in zip(stations, bearings_deg):
        a = th*DEG; d = np.array([np.cos(a), np.sin(a)])
        P = np.eye(2) - np.outer(d, d); S = np.array([x,y]); A += P; b += P@S
    try: return np.linalg.solve(A, b)
    except np.linalg.LinAlgError: return None

def dist(a, b): return float(np.linalg.norm(np.array(a, float)-np.array(b, float)))
def crossing(t1, t2):
    d = abs(t1-t2) % 180.0; return min(d, 180.0-d)

def _wedge_vertices(P, t1, Q, t2, eps=1.0):
    lines = []
    for S, t in ((np.array(P,float),t1),(np.array(Q,float),t2)):
        for e in (+eps,-eps):
            a = (t+e)*DEG; lines.append((np.array([np.cos(a),np.sin(a)]), np.array(S,float)))
    verts = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            d1,S1 = lines[i]; d2,S2 = lines[j]
            M = np.column_stack([d1,-d2])
            try:
                u = np.linalg.solve(M, S2-S1); verts.append(S1+u[0]*d1)
            except np.linalg.LinAlgError: pass
    return verts

def worst_diameter(P, t1, Q, t2, r_min=5.0, r_max=1500.0):
    worst = 0.0
    for rr in (r_min, 400.0, 900.0, 1400.0, r_max):
        g = np.array(P,float) + rr*np.array([np.cos(t1*DEG), np.sin(t1*DEG)])
        t2t = np.degrees(np.arctan2(g[1]-Q[1], g[0]-Q[0])) % 360
        for de in (-1.0, 0.0, 1.0):
            v = _wedge_vertices(P, t1+de, Q, t2t)
            if len(v) >= 2:
                worst = max(worst, max(np.linalg.norm(a-b) for a,b in combinations(v,2)))
    return worst

def dop_2nd_point(P, th, r_max=1500.0):
    best_q, best_d = None, np.inf
    for d in (300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0):
        for sgn in (+1,-1):
            a = (th+sgn*90)*DEG
            q = np.array(P,float) + d*np.array([np.cos(a), np.sin(a)])
            wd = worst_diameter(P, th, q, th, r_max=r_max)
            if wd < best_d: best_d, best_q = wd, q
    return best_q

# ================= 机器人 =================
class Robot:
    def __init__(self, env):
        self.env = env; self.pos = np.array([0.0,0.0]); self.ch = 1
        self.time = 0.0; self.dist_travel = 0.0; self.n_measure = 0; self.n_clear = 0
        self.obs = {c: [] for c in range(1, N_CH+1)}
        self.first_found_time = None; self.est_err = {}
    def move(self, p):
        p = np.array(p, float); d = dist(self.pos, p)
        self.time += d/SPEED; self.dist_travel += d; self.pos = p
    def measure(self, p, ch):
        self.move(p)
        if ch != self.ch: self.time += T_SWITCH; self.ch = ch
        self.time += T_MEAS; self.n_measure += 1
        res, deg = self.env.measure(self.pos, ch)
        if res in ('direction','near'):
            self.obs[ch].append((tuple(self.pos), deg if res=='direction' else 'NEAR'))
            if self.first_found_time is None: self.first_found_time = self.time
        return res, deg
    def homing_clear(self, ch, center):
        self.move(center); self.time += T_CLEAR_OK; self.n_clear += 1
        if self.env.clear(self.pos, ch) == 'success': return True
        step = 15.0
        for _ in range(6):
            res, deg = self.measure(self.pos, ch)
            if res == 'near':
                if self.env.clear(self.pos, ch) == 'success':
                    self.time += T_CLEAR_OK; self.n_clear += 1; return True
                continue
            if res == 'direction':
                a = deg*DEG
                self.move(self.pos + step*np.array([np.cos(a), np.sin(a)]))
                self.time += T_CLEAR_OK; self.n_clear += 1
                if self.env.clear(self.pos, ch) == 'success': return True
                step *= 0.7; continue
            for rad in (20.0, 40.0, 60.0):
                for k in range(8):
                    a = k*45*DEG; q = center + rad*np.array([np.cos(a), np.sin(a)])
                    self.move(q); r2, d2 = self.measure(self.pos, ch)
                    if r2 == 'near':
                        if self.env.clear(self.pos, ch) == 'success':
                            self.time += T_CLEAR_OK; self.n_clear += 1; return True
                        continue
                    if r2 == 'direction':
                        a2 = d2*DEG
                        for s2 in (12.0, 8.0, 5.0):
                            self.move(self.pos + s2*np.array([np.cos(a2), np.sin(a2)]))
                            self.time += T_CLEAR_OK; self.n_clear += 1
                            if self.env.clear(self.pos, ch) == 'success': return True
                        continue
            return False
        return False
    def metrics(self):
        n = len(self.env.cleared)
        return dict(CR=n/self.env.n_src, T=self.time,
                    T_f=(self.first_found_time if self.first_found_time is not None else self.time),
                    e=(float(np.mean(list(self.est_err.values()))) if self.est_err else 0.0),
                    N_d=self.n_measure, L=self.dist_travel,
                    eta=(n/self.time if self.time > 0 else 0.0))

# ================= 定位+清除 (dop_opt 决定低交会源补测策略) =================
def _locate_and_clear(rb, dop_opt=False):
    estimates = {}; pending = []
    for c in range(1, N_CH+1):
        ob = rb.obs[c]
        if not ob: continue
        dirs = [(st,d) for st,d in ob if d != 'NEAR']
        if any(d == 'NEAR' for _, d in ob):
            for st, d in ob:
                if d == 'NEAR': estimates[c] = np.array(st, float); break
            continue
        if len(dirs) >= 2:
            best = max(crossing(a,b) for _,a in dirs for _,b in dirs)
            if best >= 35.0:
                est = bearing_intersection([st for st,_ in dirs], [d for _,d in dirs])
                if est is not None: estimates[c] = est
                else: pending.append((c, dirs))
            else: pending.append((c, dirs))
        else: pending.append((c, dirs))
    for c, dirs in pending:
        S1 = np.array(dirs[0][0], float); th = dirs[0][1]; done = False
        if dop_opt:
            q = dop_2nd_point(tuple(S1), th)
            res, deg = rb.measure(q, c)
            if res == 'direction':
                est = bearing_intersection([tuple(S1), tuple(q)], [th, deg])
                if est is not None: estimates[c] = est; done = True
            elif res == 'near': estimates[c] = q.copy(); done = True
        if not done:
            for dd in (300.0, 500.0):
                for sgn in (+1,-1):
                    a = (th+sgn*90)*DEG; q = S1 + dd*np.array([np.cos(a), np.sin(a)])
                    res, deg = rb.measure(q, c)
                    if res == 'direction':
                        est = bearing_intersection([tuple(S1), tuple(q)], [th, deg])
                        if est is not None: estimates[c] = est; done = True; break
                    elif res == 'near': estimates[c] = q.copy(); done = True; break
                if done: break
    for c, est in estimates.items():
        s = rb.env.ch_by_id.get(c)
        if s is not None: rb.est_err[c] = dist(est, s['pos'])
    ordered = []; cur = rb.pos.copy(); unvisited = set(estimates.keys())
    while unvisited:
        c = min(unvisited, key=lambda k: dist(cur, estimates[k]))
        ordered.append(c); cur = estimates[c]; unvisited.discard(c)
    for c in ordered: rb.homing_clear(c, estimates[c])
    return rb.metrics()

def _stations(ring_r, n_ring):
    st = [(0.0, 0.0)]
    for k in range(n_ring):
        a = 2*np.pi*k/n_ring; st.append((ring_r*np.cos(a), ring_r*np.sin(a)))
    return st

# ================= M0 随机 =================
def model_M0(env, K=9):
    rb = Robot(env)
    pts = [np.array([0.0,0.0])]
    for _ in range(K):
        r = R_AREA*np.sqrt(env.rng.uniform(0,1)); a = env.rng.uniform(0,2*np.pi)
        pts.append(np.array([r*np.cos(a), r*np.sin(a)]))
    for p in pts:
        for c in range(1, N_CH+1): rb.measure(p, c)
    return _locate_and_clear(rb)

# ================= M1 固定覆盖(全量扫描) =================
def model_M1(env, ring_r=1200.0, n_ring=8):
    rb = Robot(env)
    for sp in _stations(ring_r, n_ring):
        for c in range(1, N_CH+1): rb.measure(sp, c)
    return _locate_and_clear(rb, dop_opt=False)

# ================= M2 +DOP =================
def model_M2(env, ring_r=1200.0, n_ring=8):
    rb = Robot(env)
    for sp in _stations(ring_r, n_ring):
        for c in range(1, N_CH+1): rb.measure(sp, c)
    return _locate_and_clear(rb, dop_opt=True)

# ================= M3 +概率(动态删除) =================
def model_M3(env, ring_r=1200.0, n_ring=8):
    """信念驱动: 已发现频道从全局存在性扫描中删除(节省检测), 其余同M1。"""
    rb = Robot(env)
    found = set()
    for sp in _stations(ring_r, n_ring):
        for c in range(1, N_CH+1):
            if c in found: continue          # 已发现, 跳过存在性检测(概率地图: 熵≈0)
            res, deg = rb.measure(sp, c)
            if res in ('direction', 'near'): found.add(c)
    return _locate_and_clear(rb, dop_opt=False)

# ================= M4 混合 =================
def model_M4(env, ring_r=1200.0, n_ring=8):
    rb = Robot(env)
    found = set()
    for sp in _stations(ring_r, n_ring):
        for c in range(1, N_CH+1):
            if c in found: continue
            res, deg = rb.measure(sp, c)
            if res in ('direction', 'near'): found.add(c)
    return _locate_and_clear(rb, dop_opt=True)

MODELS = {'M0': model_M0, 'M1': model_M1, 'M2': model_M2, 'M3': model_M3, 'M4': model_M4}

def monte_carlo(model, n_cases=30, seed=2026, directional=False, p_dir=0.5,
                n_src=None, r_min=1000.0, r_max=1500.0):
    rng = np.random.default_rng(seed)
    keys = ['CR', 'T', 'T_f', 'e', 'N_d', 'L', 'eta']
    acc = {k: [] for k in keys}
    for _ in range(n_cases):
        env = Env(rng, directional=directional, p_dir=p_dir, n_src=n_src, r_min=r_min, r_max=r_max)
        m = MODELS[model](env)
        for k in keys: acc[k].append(m[k])
    return {k: (float(np.mean(a)), float(np.std(a))) for k, a in acc.items()}

# 多环布局(问题4 定向用)
def _stations_multi(rings):
    st = [(0.0, 0.0)]
    for (r, n) in rings:
        for k in range(n):
            a = 2*np.pi*k/n
            st.append((r*np.cos(a), r*np.sin(a)))
    return st

def model_M1_multi(env, rings, dop_opt=False, dynamic=False):
    rb = Robot(env)
    found = set()
    for sp in _stations_multi(rings):
        for c in range(1, N_CH+1):
            if dynamic and c in found: continue
            res, deg = rb.measure(sp, c)
            if res in ('direction', 'near'): found.add(c)
    return _locate_and_clear(rb, dop_opt=dop_opt)

# ================= PSO vs LS 定位对照 =================
def pso_localize(stations, bearings_deg, n_p=40, iters=80, rng=None):
    rng = rng or np.random.default_rng(0)
    def residual(pos):
        s = 0.0
        for (x,y), th in zip(stations, bearings_deg):
            a = np.degrees(np.arctan2(pos[1]-y, pos[0]-x)) % 360
            d = abs(a-th) % 360; d = min(d, 360-d)
            s += d*d
        return s
    init = bearing_intersection(stations, bearings_deg)
    init = init if init is not None else np.array([0.0,0.0])
    P = init + rng.normal(0, 200, (n_p, 2))
    P = np.clip(P, -2000, 2000)
    V = rng.normal(0, 30, (n_p, 2))
    pb = P.copy(); pbv = np.array([residual(p) for p in P])
    gb = pb[np.argmin(pbv)]; gbv = pbv.min()
    w, c1, c2 = 0.5, 1.2, 1.2
    for _ in range(iters):
        r1, r2 = rng.uniform(0,1,(n_p,1)), rng.uniform(0,1,(n_p,1))
        V = w*V + c1*r1*(pb-P) + c2*r2*(gb-P)
        V = np.clip(V, -200, 200)
        P = np.clip(P + V, -2000, 2000)
        f = np.array([residual(p) for p in P])
        imp = f < pbv
        pb[imp] = P[imp]; pbv[imp] = f[imp]
        if f.min() < gbv: gbv = f.min(); gb = P[np.argmin(f)]
    return gb

def pso_vs_ls(n_cases=300):
    rng = np.random.default_rng(99)
    e_ls, e_pso = [], []
    for _ in range(n_cases):
        G = rng.uniform(-800, 800, 2)              # 源在近中区域
        n = int(rng.integers(2, 5))
        # 围绕源布置, 距离 400~1200m, 保证较好交会几何
        st, bd = [], []
        for k in range(n):
            ang = rng.uniform(0, 2*np.pi)
            rr = rng.uniform(400, 1200)
            S = G + rr*np.array([np.cos(ang), np.sin(ang)])
            st.append(S)
            d = G - S
            th = np.degrees(np.arctan2(d[1], d[0])) % 360
            bd.append((th + rng.uniform(-1, 1)) % 360)
        ls = bearing_intersection(st, bd)
        pso = pso_localize(st, bd, rng=rng)
        if ls is not None: e_ls.append(dist(ls, G))
        e_pso.append(dist(pso, G))
    return float(np.mean(e_ls)), float(np.mean(e_pso))

if __name__ == "__main__":
    import time
    print("=" * 100)
    print("【初步对比】30 案例, 全向源(问题3)")
    hdr = f"{'模型':<4}{'完成率':>11}{'总时间':>8}{'首发现':>7}{'定位误差':>9}{'检测次数':>8}{'移动':>9}{'η(个/ks)':>10}"
    print(hdr)
    for m in ['M0','M1','M2','M3','M4']:
        t0 = time.time(); r = monte_carlo(m, n_cases=30)
        print(f"{m:<4}{r['CR'][0]*100:>10.1f}%{r['T'][0]:>8.0f}{r['T_f'][0]:>7.0f}"
              f"{r['e'][0]:>8.1f}m{r['N_d'][0]:>8.0f}{r['L'][0]:>9.0f}{r['eta'][0]*1000:>9.2f}"
              f"  [{time.time()-t0:.1f}s]")
    print("\n【PSO vs LS 定位】%d 次随机场景" % 200)
    e_ls, e_pso = pso_vs_ls(200)
    print(f"  LS 平均定位误差={e_ls:.2f}m   PSO 平均定位误差={e_pso:.2f}m")
