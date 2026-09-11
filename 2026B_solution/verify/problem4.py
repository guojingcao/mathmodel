# -*- coding: utf-8 -*-
"""
问题4 仿真检验
既有全向又有定向干扰源(定向方向未知), 总数10~16未知。
定向源: 覆盖范围=指向方向两侧各90°(含), 盲区180°; /measure 在盲区返回 no_signal;
        /clear 只要<=20m 即可清除(与覆盖无关)。
策略: 原点 + 内环(R1200x8) + 外环(包围盘外, 使边界定向源也被"环绕"而可测) -> 交会 -> TSP清除
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

class Env4:
    def __init__(self, rng, n_src=None, p_dir=0.5, r_min=1000.0, r_max=1500.0):
        self.rng = rng
        self.n_src = n_src if n_src is not None else int(rng.integers(10, 17))
        chans = rng.choice(np.arange(1, N_CH + 1), size=self.n_src, replace=False)
        self.sources = []
        for c in chans:
            r = R_AREA * np.sqrt(rng.uniform(0, 1))
            a = rng.uniform(0, 2*np.pi)
            directional = rng.uniform() < p_dir
            pointing = rng.uniform(0, 2*np.pi) if directional else None
            self.sources.append(dict(pos=np.array([r*np.cos(a), r*np.sin(a)]),
                                      ch=int(c), r_rx=float(rng.uniform(r_min, r_max)),
                                      pointing=pointing))
        self.cleared = set()

    def _in_coverage(self, s, pos):
        if s['pointing'] is None:
            return True
        d = pos - s['pos']            # 源 -> 检测点 方向
        ang = np.arctan2(d[1], d[0])
        diff = abs(ang - s['pointing'])
        diff = min(diff, 2*np.pi - diff)
        return diff <= np.pi/2 + 1e-12

    def measure(self, pos, ch):
        for s in self.sources:
            if s['ch'] != ch or ch in self.cleared:
                continue
            d = s['pos'] - pos
            dist = np.linalg.norm(d)
            if dist > s['r_rx']:
                continue
            if not self._in_coverage(s, pos):
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

    @property
    def n_directional(self):
        return sum(1 for s in self.sources if s['pointing'] is not None)

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

def run_strategy(env, r_in=1200.0, n_in=8, r_out=2200.0, n_out=8, r_mid=0.0, n_mid=0, rng=None):
    stations = [(0.0, 0.0)]
    if n_mid:
        for k in range(n_mid):
            a = 2*np.pi*k/n_mid
            stations.append((r_mid*np.cos(a), r_mid*np.sin(a)))
    for k in range(n_in):
        a = 2*np.pi*k/n_in
        stations.append((r_in*np.cos(a), r_in*np.sin(a)))
    for k in range(n_out):
        a = 2*np.pi*k/n_out
        stations.append((r_out*np.cos(a), r_out*np.sin(a)))

    pos = np.array([0.0, 0.0]); ch_cur = 1; total_time = 0.0
    obs = {c: [] for c in range(1, N_CH + 1)}

    def move_to(p):
        nonlocal pos, total_time
        total_time += dist(pos, p)/SPEED
        pos = np.array(p, float)

    def measure_at(p, c):
        nonlocal ch_cur, total_time
        if c != ch_cur:
            total_time += T_SWITCH; ch_cur = c
        total_time += T_MEAS
        return env.measure(p, c)

    for sp in stations:
        move_to(sp)
        for c in range(1, N_CH + 1):
            res, deg = measure_at(pos, c)
            if res == 'direction':
                obs[c].append((tuple(pos), deg))
            elif res == 'near':
                obs[c].append((tuple(pos), 'NEAR'))

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
                    estimates[c] = np.array(st, float); break
            continue
        if len(dirs) >= 2:
            best = max(crossing(a, b) for _, a in dirs for _, b in dirs)
            if best >= 35.0:
                est = bearing_intersection([st for st, _ in dirs], [d for _, d in dirs])
                if est is not None:
                    estimates[c] = est
                else:
                    pending.append((c, dirs))
            else:
                pending.append((c, dirs))
        else:
            pending.append((c, dirs))

    for c, dirs in pending:
        done = False
        if len(dirs) >= 2:
            S1 = np.array(dirs[0][0], float); S2 = np.array(dirs[1][0], float)
            mid = (S1 + S2)/2; base_th = dirs[0][1]
        else:
            S1 = np.array(dirs[0][0], float); S2 = None; mid = S1; base_th = dirs[0][1]
        for dd in (300.0, 500.0):
            for sgn in (+1, -1):
                a = (base_th + sgn*90.0)*DEG
                q = mid + dd*np.array([np.cos(a), np.sin(a)])
                move_to(q)
                res, deg = measure_at(pos, c)
                if res == 'direction':
                    est = bearing_intersection([tuple(S1), tuple(pos)], [base_th, deg])
                    if est is not None:
                        estimates[c] = est; done = True; break
                elif res == 'near':
                    estimates[c] = pos.copy(); done = True; break
            if done: break
        if not done and S2 is not None:
            base_th = dirs[1][1]
            for dd in (300.0, 500.0):
                for sgn in (+1, -1):
                    a = (base_th + sgn*90.0)*DEG
                    q = S2 + dd*np.array([np.cos(a), np.sin(a)])
                    move_to(q)
                    res, deg = measure_at(pos, c)
                    if res == 'direction':
                        est = bearing_intersection([tuple(S2), tuple(pos)], [base_th, deg])
                        if est is not None:
                            estimates[c] = est; done = True; break
                    elif res == 'near':
                        estimates[c] = pos.copy(); done = True; break
                if done: break
        if done:
            continue
        # 二分归航兜底: 沿示向度方向对源距离二分, 越过(no_signal=进入盲区)则回退
        S0 = np.array(dirs[0][0], float)
        th = dirs[0][1]
        move_to(S0)
        a = th * DEG
        dvec = np.array([np.cos(a), np.sin(a)])
        lo, hi = 0.0, 1500.0
        for _ in range(20):
            mid = (lo + hi) / 2
            move_to(S0 + mid * dvec)
            res, deg = measure_at(pos, c)
            if res == 'near':
                estimates[c] = pos.copy(); break
            if res == 'direction':
                lo = mid
            else:
                hi = mid
            if hi - lo < 4.0:
                total_time += T_CLEAR_OK
                env.clear(pos, c)
                estimates[c] = pos.copy()
                break
        else:
            estimates[c] = pos.copy()

    order = []; unvisited = set(estimates.keys()); cur = pos.copy()
    while unvisited:
        c = min(unvisited, key=lambda k: dist(cur, estimates[k]))
        order.append(c); cur = estimates[c]; unvisited.discard(c)

    def homing_clear(c, center):
        """从 center 出发, 归航+螺旋搜索清除频道 c, 返回是否成功。"""
        nonlocal total_time
        move_to(center)
        total_time += T_CLEAR_OK
        if env.clear(pos, c) == 'success':
            return True
        step = 15.0
        for _ in range(6):
            res, deg = measure_at(pos, c)
            if res == 'near':
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    return True
                continue
            if res == 'direction':
                a = deg * DEG
                move_to(pos + step * np.array([np.cos(a), np.sin(a)]))
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    return True
                step *= 0.7
                continue
            # no_signal: 估计点落在定向源盲区, 螺旋搜索找回
            for rad in (20.0, 40.0, 60.0):
                for k in range(8):
                    a = k * 45 * DEG
                    q = center + rad * np.array([np.cos(a), np.sin(a)])
                    move_to(q)
                    res2, deg2 = measure_at(pos, c)
                    if res2 == 'near':
                        total_time += T_CLEAR_OK
                        if env.clear(pos, c) == 'success':
                            return True
                        continue
                    if res2 == 'direction':
                        a2 = deg2 * DEG
                        for s2 in (12.0, 8.0, 5.0):
                            move_to(pos + s2 * np.array([np.cos(a2), np.sin(a2)]))
                            total_time += T_CLEAR_OK
                            if env.clear(pos, c) == 'success':
                                return True
                        continue
            return False
        return False

    cleared_count = 0
    for c in order:
        if homing_clear(c, estimates[c]):
            cleared_count += 1

    # 恢复兜底: 已检测但未清除的频道, 从其首个观测站沿示向度做二分归航
    for c in range(1, N_CH + 1):
        if c in env.cleared or not obs[c]:
            continue
        st, d = obs[c][0]
        if d == 'NEAR':
            move_to(st)
            total_time += T_CLEAR_OK
            if env.clear(pos, c) == 'success':
                cleared_count += 1
            continue
        S0 = np.array(st, float)
        th = d
        a = th * DEG
        dvec = np.array([np.cos(a), np.sin(a)])
        lo, hi = 0.0, 1500.0
        for _ in range(20):
            mid = (lo + hi) / 2
            move_to(S0 + mid * dvec)
            res, deg = measure_at(pos, c)
            if res == 'near':
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    cleared_count += 1
                break
            if res == 'direction':
                lo = mid
            else:
                hi = mid
            if hi - lo < 4.0:
                total_time += T_CLEAR_OK
                if env.clear(pos, c) == 'success':
                    cleared_count += 1
                break

    det = sum(1 for c in range(1, N_CH+1) if len(obs[c]) >= 1)
    cross = sum(1 for c in range(1, N_CH+1) if len(obs[c]) >= 2)
    avg = total_time/cleared_count if cleared_count else float('inf')
    return dict(cleared=cleared_count, n_src=env.total_sources,
                total=total_time, avg=avg, n_est=len(estimates),
                det=det, cross=cross, n_dir=env.n_directional)

def monte_carlo(n_cases=200, **kw):
    rng = np.random.default_rng(2027)
    ratios, avgs, totals = [], [], []
    miss = 0; det_r = []; cross_r = []
    for _ in range(n_cases):
        env = Env4(rng)
        r = run_strategy(env, rng=rng, **kw)
        ratios.append(r['cleared']/r['n_src'])
        avgs.append(r['avg']); totals.append(r['total'])
        det_r.append(r['det']/r['n_src']); cross_r.append(r['cross']/r['n_src'])
        if r['cleared'] < r['n_src']:
            miss += 1
    return dict(n=n_cases, clear=np.mean(ratios), min_clear=np.min(ratios),
                avg=np.mean(avgs), total=np.mean(totals), miss=miss,
                det=np.mean(det_r), cross=np.mean(cross_r))

if __name__ == "__main__":
    print("=" * 86)
    print("问题4 全向+定向混合: 策略仿真 (200 随机案例, 定向比例50%)")
    print(f"{'布局':<28}{'检测%':>7}{'交会%':>7}{'清除%':>7}{'最差%':>7}{'平均时间s':>10}{'总时间s':>9}{'未清完':>6}")
    for name, kw in [
        ("原点+R1200x8+R2200x12", dict(r_in=1200, n_in=8, r_out=2200, n_out=12)),
        ("原点+R500x8+R1200x8+R2200x12", dict(r_mid=500, n_mid=8, r_in=1200, n_in=8, r_out=2200, n_out=12)),
        ("原点+R600x8+R1200x8+R2200x12", dict(r_mid=600, n_mid=8, r_in=1200, n_in=8, r_out=2200, n_out=12)),
        ("原点+R500x6+R1000x8+R2200x12", dict(r_mid=500, n_mid=6, r_in=1000, n_in=8, r_out=2200, n_out=12)),
    ]:
        s = monte_carlo(n_cases=200, **kw)
        print(f"{name:<28}{s['det']*100:>6.1f}{s['cross']*100:>6.1f}{s['clear']*100:>6.1f}"
              f"{s['min_clear']*100:>6.0f}{s['avg']:>9.0f}{s['total']:>8.0f}{s['miss']:>5}")
