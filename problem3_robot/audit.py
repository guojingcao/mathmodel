# -*- coding: utf-8 -*-
"""架构参考路线审计 (architecture reference-route audit)

**口径勘误(本文件此前版本的错误)**
  此前把三个量都叫做"下界(LB)", 这是错的:
    * 访问"源中心"的开放路径最优长度, 相对真实的**清除**问题(允许在距源 <=20m 处清除)
      是**上界**, 不是下界 —— 圆盘访问问题比中心访问问题更宽松。
    * 用"最近邻+2-opt"得到的路线是**可行解**, 其长度 ≥ 最优值, 因此是**上界**, 不是下界。
  本版改用中性命名, 并给出一个**严格成立**的下界 LOWER_disk。

本文件输出(同一批随机案例):
  UB_center   从扫描终点出发、访问全部【真实源中心】的开放路径长度
              (n<=EXACT_CAP 时由 Held-Karp 精确给出并标 exact=True; 否则为 2-opt 启发式并标 exact=False)
  LOWER_disk  = max(0, UB_center - 2*n*R_CLEAR)
              清除问题的**严格下界**: 任一条"圆盘访问"路线长度 L, 把每个圆盘点换成走到中心
              至多多走 2*R_CLEAR, 故 UB_center <= L + 2*n*R_CLEAR
  REF_separated = COVER_OPT + UB_center
              固定"7 点覆盖骨架 + 扫描结束后再清除"这一**分阶段约束**下的参考路线
  REF_fused   从原点出发、联合访问【7 个覆盖点 + 全部真实源中心】的 2-opt 可行路线(上界)
  ALG         当前在线算法实际移动距离(不受上述分阶段约束, 含顺路清除)
指标(全部按**参考路线**口径表述, 不称"损失已精确量化"):
  R_online = (ALG - REF_separated)/REF_separated   相对分阶段参考的比值
  R_arch   = (REF_separated - REF_fused)/REF_separated  融合参考相对分阶段的改善率
  G_info   = L_post_ALG - LOWER_disk              信息未知带来的额外距离(相对严格下界)
正确性核对(每个案例):
  机器人返回清除数 == 客户端 /clear 成功数 == 环境真值 len(env.cleared) == 真实源数
"""
import importlib.util
import sys
import math
import io
import contextlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("robotmod", HERE / "robot.py")
robotmod = importlib.util.module_from_spec(spec)
sys.modules["robotmod"] = robotmod
spec.loader.exec_module(robotmod)
spec2 = importlib.util.spec_from_file_location("exp", ROOT / "2026B_solution" / "verify" / "experiment.py")
exp = importlib.util.module_from_spec(spec2)
sys.modules["exp"] = exp
spec2.loader.exec_module(exp)

EXACT_CAP = 14
R_CLEAR = robotmod.R_CLEAR


def d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def exact_open_path(pts, start, cap=EXACT_CAP):
    """Held-Karp 精确开放路径(终点自由)。n > cap 时返回 None(不静默降级)。"""
    n = len(pts)
    if n == 0:
        return 0.0
    if n > cap:
        return None
    INF = float("inf")
    size = 1 << n
    D = [[d(pts[i], pts[j]) for j in range(n)] for i in range(n)]
    Ds = [d(pts[i], start) for i in range(n)]
    dp = [[INF]*n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = Ds[i]
    for mask in range(size):
        row = dp[mask]
        for i in range(n):
            base = row[i]
            if base == INF or not (mask >> i) & 1:
                continue
            Di = D[i]
            for j in range(n):
                if (mask >> j) & 1:
                    continue
                v = base + Di[j]
                nm = mask | (1 << j)
                if v < dp[nm][j]:
                    dp[nm][j] = v
    return min(dp[size-1])


def path_len(path):
    return sum(d(path[i], path[i+1]) for i in range(len(path)-1))


def nn_path(pts, start):
    order = []
    unv = set(range(len(pts)))
    cur = start
    while unv:
        k = min(unv, key=lambda i: d(pts[i], cur))
        order.append(k)
        cur = pts[k]
        unv.discard(k)
    return [tuple(start)] + [tuple(pts[k]) for k in order]


def two_opt(path):
    """开放路径 2-opt: **终点自由**, 因此必须包含 j == n 的尾段反转。

    尾段反转只改一条边: (v_{i-1},v_i) -> (v_{i-1},v_n)。
    此前版本 j 只到 n-1, 等价于把终点固定为最近邻初始路的终点, 会系统性高估路线长度。
    """
    n = len(path) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i+1, n+1):
                if j < n:
                    old = d(path[i-1], path[i]) + d(path[j], path[j+1])
                    new = d(path[i-1], path[j]) + d(path[i], path[j+1])
                else:
                    old = d(path[i-1], path[i])
                    new = d(path[i-1], path[n])
                if new < old - 1e-9:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
    return path


def heuristic_open_path(pts, start):
    """可行解(上界): 最近邻 + 2-opt(终点自由)。"""
    return path_len(two_opt(nn_path(pts, start)))


def cover_tour_len():
    """7 点覆盖骨架的最短开放路径(6! 全排列精确), 并验证与现用顺序一致。"""
    import itertools
    pts = robotmod.Problem3Robot.search_points()
    best = float("inf")
    for perm in itertools.permutations(range(1, 7)):
        seq = [pts[0]] + [pts[k] for k in perm]
        best = min(best, path_len(seq))
    return best


class MockClient:
    """与冻结版 SimClient 同接口; 额外记录阶段切换时刻的累计距离与动作计数。"""

    def __init__(self, env):
        self.env = env
        self.position = (0.0, 0.0)
        self.channel = 1
        self.remaining_real = 1200
        self.log = []
        self.dist = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_ok = 0
        self.n_switch = 0
        self.fail = 0
        self._phase = "init"
        self.dist_at_scan_end = None
        self.ph = {}          # 阶段 -> [移动, 检测, 清除, 换频]

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, name):
        # 只在**真正进入扫描后阶段**(补测/队列/归航)时记录累计距离;
        # 注意不能把 on_way 当扫描结束 —— 第 0 个覆盖点就是原点, 首次 on_way 切换发生在 dist=0。
        if (self.dist_at_scan_end is None
                and name in ("supplement", "queue_clear", "queue_homing", "homing")):
            self.dist_at_scan_end = self.dist
        self._phase = name

    def _p(self):
        return self.ph.setdefault(self._phase, [0.0, 0, 0, 0])

    def _move(self, x, y):
        dd = d((x, y), self.position)
        self.dist += dd
        self._p()[0] += dd
        self.position = (x, y)

    def enter(self):
        pass

    def measure(self, x, y, ch):
        self._move(x, y)
        if ch != self.channel:
            self.n_switch += 1
            self._p()[3] += 1
        self.channel = ch
        self.n_measure += 1
        self._p()[1] += 1
        r, svd = self.env.measure(np.array([x, y]), ch)
        return True, r, svd

    def clear(self, x, y, ch):
        self._move(x, y)
        # 题设: /clear 不换频, 也不改变测向机频道状态(附件1/附件2)
        self.n_clear += 1
        self._p()[2] += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r == 'success':
            self.n_clear_ok += 1
        else:
            self.fail += 1
        return True, r

    def exit(self):
        pass


T_CLEAR_OK, T_ENTER_EXIT = 5.0, 0.0     # 题设标准: 成功清除 3+2=5 s(附件1/附件2)

def sim_time(cli):
    """统一计时口径(实机标定): 移动/5 + 检测*5 + 换频*1 + 成功清除*4 + 失败清除*3。"""
    return (cli.dist/5.0 + cli.n_measure*5 + cli.n_switch*1
            + cli.n_clear_ok*T_CLEAR_OK + cli.fail*3 + T_ENTER_EXIT)


def phase_time(cli, name):
    """按**同一口径**给出的阶段时间(此前只算 移动/5+检测*5, 会与总时间不一致)。"""
    mv, nm, nc, sw = cli.ph.get(name, [0.0, 0, 0, 0])
    return mv/5.0 + nm*5 + sw*1    # 阶段内的清除成功/失败次数另计(见 phase_clear_cost)


def scenario_hash(env):
    """场景指纹: 源数/频道/位置/接收半径/指向 -> 稳定整数, 用于验证配对是否真的同场景。"""
    h = 1469598103934665603
    for s in sorted(env.sources, key=lambda s: s['ch']):
        vals = [s['ch'], round(float(s['pos'][0]), 6), round(float(s['pos'][1]), 6),
                round(float(s['r_rx']), 6),
                -1.0 if s['pointing'] is None else round(float(s['pointing']), 6)]
        for v in vals:
            h ^= hash(repr(v)) & 0xFFFFFFFFFFFFFFFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng([2026, 11])          # 场景随机源固定(与其它脚本一致口径)
    cover_pts = robotmod.Problem3Robot.search_points()
    COVER_OPT = cover_tour_len()
    print(f"7 点覆盖骨架最短开放路径(精确) = {COVER_OPT:.0f} m")
    rows = []
    bad_clear = 0
    for t in range(N):
        env = exp.Env(rng, directional=False)
        targets = [tuple(float(v) for v in s['pos']) for s in env.sources]
        cli = MockClient(env)
        rb = robotmod.Problem3Robot(cli)
        with contextlib.redirect_stdout(io.StringIO()):
            n_clear_ret = rb.run()
        # 正确性核对: 机器人返回 == 客户端成功 == 环境真值 == 源数
        env_cleared = len(env.cleared)
        if not (n_clear_ret == cli.n_clear_ok == env_cleared == env.n_src):
            bad_clear += 1
        if not (env_cleared == env.n_src):
            print(f"  !! case {t+1}: 环境真值 {env_cleared}/{env.n_src}, 机器人返回 {n_clear_ret}")
        L_alg = cli.dist
        scan_end_dist = cli.dist_at_scan_end
        L_post = (L_alg - scan_end_dist) if scan_end_dist is not None else float("nan")
        scan_end = cli.position if False else cover_pts[-1]
        ub_center = exact_open_path(targets, scan_end)
        exact_flag = ub_center is not None
        ub_heuristic = heuristic_open_path(targets, scan_end)   # 同点集 2-opt(终点自由)上界
        if not exact_flag:
            # n > EXACT_CAP: **显式标记为启发式上界**, 不与精确值混用
            ub_center = ub_heuristic
        lower_disk = max(0.0, ub_center - 2*len(targets)*R_CLEAR)
        ref_sep = COVER_OPT + ub_center
        ref_fused = heuristic_open_path(list(cover_pts) + targets, (0.0, 0.0))
        rows.append(dict(n_src=env.n_src, L_alg=L_alg, ub_center=ub_center, exact=exact_flag,
                         ub_heuristic=ub_heuristic,
                         lower_disk=lower_disk, ref_sep=ref_sep, ref_fused=ref_fused,
                         L_post=L_post, meas=cli.n_measure, switches=cli.n_switch,
                         clears=cli.n_clear_ok, fails=cli.fail, T=sim_time(cli),
                         scan_end_dist=(scan_end_dist if scan_end_dist is not None else np.nan),
                         sh=scenario_hash(env), ph=dict(cli.ph)))
        print(f"  case {t+1}/{N}: n={env.n_src} ALG={L_alg:.0f} UB_center={ub_center:.0f}"
              f"{'' if exact_flag else '(启发式)'} LOWER_disk={lower_disk:.0f} "
              f"REF_sep={ref_sep:.0f} REF_fused={ref_fused:.0f}", flush=True)

    K = [k for k in rows[0].keys() if k != "ph"]      # ph 是 dict, 不参与数值汇总
    A = {k: np.array([r[k] for r in rows], float) for k in K}
    n_exact = int(A['exact'].sum())
    gap_online = (A['L_alg'] - A['ref_sep']) / A['ref_sep']
    gap_arch = (A['ref_sep'] - A['ref_fused']) / A['ref_sep']
    g_info = A['L_post'] - A['lower_disk']
    print("\n=== 架构参考路线审计汇总(参考口径, 非严格下界) ===")
    print(f"案例数            : {N}  (清除一致性核对失败 {bad_clear} 例)")
    print(f"源数均值          : {A['n_src'].mean():.1f}")
    print(f"ALG 实际移动      : {A['L_alg'].mean():.0f} m")
    print(f"UB_center         : {A['ub_center'].mean():.0f} m   (精确 {n_exact}/{N} 例; "
          f"其余为 2-opt 启发式上界)")
    print(f"LOWER_disk(严格)  : {A['lower_disk'].mean():.0f} m   (清除问题下界 = UB_center - 2nR_clear)")
    print(f"REF_separated     : {A['ref_sep'].mean():.0f} m   (覆盖 {COVER_OPT:.0f} + UB_center)")
    print(f"REF_fused(2-opt上界): {A['ref_fused'].mean():.0f} m")
    print(f"R_online=(ALG-REF_sep)/REF_sep : {gap_online.mean()*100:.2f}% (中位 {np.median(gap_online)*100:.2f}%)")
    print(f"R_arch =(REF_sep-REF_fused)/REF_sep : {gap_arch.mean()*100:.2f}% (中位 {np.median(gap_arch)*100:.2f}%)")
    print(f"G_info = L_post - LOWER_disk  : {g_info.mean():.0f} m")
    print(f"扫描结束距离(阶段表真值) : 覆盖 {A['scan_end_dist'].mean():.0f} m "
          f"(占 ALG {A['scan_end_dist'].mean()/A['L_alg'].mean()*100:.1f}%)")
    print(f"统一计时口径总时间     : {A['T'].mean():.0f} s "
          f"(移动 {A['L_alg'].mean()/5:.0f} + 检测 {A['meas'].mean()*5:.0f} + "
          f"换频 {A['switches'].mean():.0f} + 清除 {A['clears'].mean()*T_CLEAR_OK:.0f} + "
          f"失败 {A['fails'].mean()*3:.0f} + enter/exit {T_ENTER_EXIT:.0f})")
    # 分账一致性: 各阶段移动之和必须等于总移动(统一记账的自检)
    ph_sum = np.array([sum(v[0] for v in r['ph'].values()) for r in rows])
    print(f"分账一致性: 阶段移动之和 vs 总移动 最大偏差 = {np.abs(ph_sum - A['L_alg']).max():.3f} m")
    mask_ex = A['exact'] > 0.5
    if mask_ex.any():
        gap2 = (A['ub_heuristic'][mask_ex] - A['ub_center'][mask_ex]) / A['ub_center'][mask_ex]
        print(f"2-opt(终点自由) vs 精确开放TSP (仅 {int(mask_ex.sum())} 例可精确): "
              f"均值 {gap2.mean()*100:.2f}%, 中位 {np.median(gap2)*100:.2f}%, 最大 {gap2.max()*100:.2f}%")
    print(f"阶段移动分解(均值): " + ", ".join(
        f"{k} {np.mean([r['ph'].get(k,[0,0,0,0])[0] for r in rows]):.0f}m"
        for k in ("coverage", "on_way", "supplement", "queue_clear", "queue_homing", "on_way_homing")))
    print("注: ALG 在线算法含**顺路清除**, 不受 REF_separated 的'先扫描后清除'分阶段约束, "
          "故 R_online 只是比值, 不能解释为'已量化的架构损失'。")
