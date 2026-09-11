# -*- coding: utf-8 -*-
"""
问题3 仿真检验 (v3)
策略: 原点+圆环观测站扫描 -> 交会定位(低交会角则垂直补测) -> 最近邻TSP清除(带导引精定位)
"""
import numpy as np

DEG = np.pi / 180.0
SPEED = 5.0
T_MEAS = 5.0
T_SWITCH = 1.0
T_CLEAR_OK = 5.0
R_AREA = 1800.0
R_CLEAR = 20.0
R_NEAR = 5.0
N_CH = 20

class Env:
    def __init__(self, rng, n_src=None, r_min=1000.0, r_max=1500.0):
        self.rng = rng
        self.n_src = n_src if n_src is not None else int(rng.integers(10, 17))
        chans = rng.choice(np.arange(1, N_CH + 1), size=self.n_src, replace=False)
        self.sources = []
        for c in chans:
            r = R_AREA * np.sqrt(rng.uniform(0, 1))
            a = rng.uniform(0, 2*np.pi)
            self.sources.append(dict(pos=np.array([r*np.cos(a), r*np.sin(a)]),
                                      ch=int(c), r_rx=float(rng.uniform(r_min, r_max))))
        self.cleared = set()

    def measure(self, pos, ch):
        for s in self.sources:
            if s['ch'] != ch or ch in self.cleared:
                continue
            d = s['pos'] - pos
            dist = np.linalg.norm(d)
            if dist > s['r_rx']:
                continue
            if dist <= R_NEAR:
                return 'near', None
            true = np.degrees(np.arctan2(d[1], d[0])) % 360
            return 'direction', (true + self.rng.uniform(-1, 1)) % 360
        return 'no_signal', None

    def clear(self, pos, ch):
        for s in self.sources:
            if s['ch'] == ch and ch not in self.cleared:
                if np.linalg.norm(s['pos'] - pos) <= R_CLEAR:
                    self.cleared.add(ch)
                    return 'success'
        return 'no_target_in_range'

    @property
    def total_sources(self):
        return len(self.sources)

def bearing_intersection(stations, bearings_deg):
    A = np.zeros((2, 2)); b = np.zeros(2)
    for (x, y), th in zip(stations, bearings_deg):
        a = th * DEG
        d = np.array([np.cos(a), np.sin(a)])
        P = np.eye(2) - np.outer(d, d)
        S = np.array([x, y])
        A += P; b += P @ S
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

def dist(a, b):
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))

def crossing(th1, th2):
    d = abs(th1 - th2) % 180.0
    return min(d, 180.0 - d)

def run_strategy(env, ring_radius=1200.0, n_ring=8, rng=None):
    stations = [(0.0, 0.0)]
    for k in range(n_ring):
        a = 2*np.pi * k / n_ring
        stations.append((ring_radius*np.cos(a), ring_radius*np.sin(a)))

    pos = np.array([0.0, 0.0]); ch_cur = 1; total_time = 0.0
    obs = {c: [] for c in range(1, N_CH + 1)}

    def move_to(p):
        nonlocal pos, total_time
        total_time += dist(pos, p) / SPEED
        pos = np.array(p, float)

    def measure_at(p, c):
        nonlocal ch_cur, total_time
        if c != ch_cur:
            total_time += T_SWITCH; ch_cur = c
        total_time += T_MEAS
        return env.measure(p, c)

    # 相位1: 扫描
    for sp in stations:
        move_to(sp)
        for c in range(1, N_CH + 1):
            res, deg = measure_at(pos, c)
            if res == 'direction':
                obs[c].append((tuple(pos), deg))
            elif res == 'near':
                obs[c].append((tuple(pos), 'NEAR'))

    # 相位2: 定位(先处理NEAR与高交会角, 低交会角/单示向度进入pending)
    estimates = {}
    pending = []
    for c in range(1, N_CH + 1):
        ob = obs[c]
        if not ob:
            continue
        dirs = [(st, d) for st, d in ob if d != 'NEAR']
        if any(d == 'NEAR' for _, d in ob):
            for st, d in ob:
                if d == 'NEAR':
                    estimates[c] = np.array(st, float)
                    break
            continue
        if len(dirs) >= 2:
            best = max(crossing(a, b) for _, a in dirs for _, b in dirs)
            if best >= 45.0:
                est = bearing_intersection([st for st, _ in dirs], [d for _, d in dirs])
                if est is not None:
                    estimates[c] = est
                else:
                    pending.append((c, dirs))
            else:
                pending.append((c, dirs))
        else:
            pending.append((c, dirs))

    # 相位3: 自适应补测(低交会角/单示向度)
    for c, dirs in pending:
        done = False
        if len(dirs) >= 2:
            # 近共线: 参考站用第一站 S1(其示向度在S1处测得),
            # 但垂直补测点 Q 取两站中点附近(靠近源, 保证在接收半径内)
            S1 = np.array(dirs[0][0], float); S2 = np.array(dirs[1][0], float)
            mid = (S1 + S2) / 2
            base_th = dirs[0][1]
        else:
            S1 = np.array(dirs[0][0], float); S2 = None
            mid = S1
            base_th = dirs[0][1]
        for dd in (300.0, 500.0):
            for sgn in (+1, -1):
                a = (base_th + sgn*90.0) * DEG
                q = mid + dd * np.array([np.cos(a), np.sin(a)])
                move_to(q)
                res, deg = measure_at(pos, c)
                if res == 'direction':
                    est = bearing_intersection([tuple(S1), tuple(pos)], [base_th, deg])
                    if est is not None:
                        estimates[c] = est; done = True; break
                elif res == 'near':
                    estimates[c] = pos.copy(); done = True; break
            if done:
                break
        if not done and S2 is not None:
            # 换到第二站一侧垂直补测
            base_th = dirs[1][1]
            for dd in (300.0, 500.0):
                for sgn in (+1, -1):
                    a = (base_th + sgn*90.0) * DEG
                    q = S2 + dd * np.array([np.cos(a), np.sin(a)])
                    move_to(q)
                    res, deg = measure_at(pos, c)
                    if res == 'direction':
                        est = bearing_intersection([tuple(S2), tuple(pos)], [base_th, deg])
                        if est is not None:
                            estimates[c] = est; done = True; break
                    elif res == 'near':
                        estimates[c] = pos.copy(); done = True; break
                if done:
                    break

    # 相位4: 最近邻TSP清除(带导引精定位)
    order = []; unvisited = set(estimates.keys()); cur = pos.copy()
    while unvisited:
        c = min(unvisited, key=lambda k: dist(cur, estimates[k]))
        order.append(c); cur = estimates[c]; unvisited.discard(c)

    cleared_count = 0
    for c in order:
        move_to(estimates[c])
        total_time += T_CLEAR_OK
        if env.clear(pos, c) == 'success':
            cleared_count += 1; continue
        # 归航精定位: 反复测向并前进, 直到清除成功或到达 near
        ok = False
        step = 15.0
        for _ in range(8):
            res, deg = measure_at(pos, c)
            if res == 'near':
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    cleared_count += 1
                ok = True; break
            if res == 'direction':
                a = deg * DEG
                move_to(pos + step * np.array([np.cos(a), np.sin(a)]))
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    cleared_count += 1; ok = True; break
                step *= 0.7
            else:
                break

    det = sum(1 for c in range(1, N_CH+1) if len(obs[c]) >= 1)
    cross = sum(1 for c in range(1, N_CH+1) if len(obs[c]) >= 2)
    avg = total_time / cleared_count if cleared_count else float('inf')
    return dict(cleared=cleared_count, n_src=env.total_sources,
                total=total_time, avg=avg, n_est=len(estimates),
                det=det, cross=cross)

def monte_carlo(n_cases=300, **kw):
    rng = np.random.default_rng(2026)
    ratios, avgs, totals = [], [], []
    miss = 0; det_ratio = []; cross_ratio = []
    for _ in range(n_cases):
        env = Env(rng)
        r = run_strategy(env, rng=rng, **kw)
        ratios.append(r['cleared']/r['n_src'])
        avgs.append(r['avg']); totals.append(r['total'])
        det_ratio.append(r['det']/r['n_src']); cross_ratio.append(r['cross']/r['n_src'])
        if r['cleared'] < r['n_src']:
            miss += 1
    return dict(n=n_cases, clear=np.mean(ratios), min_clear=np.min(ratios),
                avg=np.mean(avgs), total=np.mean(totals), miss=miss,
                det=np.mean(det_ratio), cross=np.mean(cross_ratio))

if __name__ == "__main__":
    print("=" * 80)
    print("问题3 全向干扰源: 策略仿真 v3 (300 随机案例)")
    print(f"{'布局':<22}{'检测%':>7}{'交会%':>7}{'清除%':>7}{'最差%':>7}{'平均时间s':>10}{'总时间s':>9}{'未清完':>6}")
    for name, rr, nr in [
        ("原点+环R1200×8", 1200, 8),
        ("原点+环R1300×8", 1300, 8),
        ("原点+环R1400×8", 1400, 8),
        ("原点+环R1200×6", 1200, 6),
        ("原点+环R1300×6", 1300, 6),
    ]:
        s = monte_carlo(n_cases=300, ring_radius=rr, n_ring=nr)
        print(f"{name:<22}{s['det']*100:>6.1f}{s['cross']*100:>6.1f}{s['clear']*100:>6.1f}"
              f"{s['min_clear']*100:>6.0f}{s['avg']:>9.0f}{s['total']:>8.0f}{s['miss']:>5}")
