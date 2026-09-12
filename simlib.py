# -*- coding: utf-8 -*-
"""共享离线实验库 (problem3_robot / problem4_robot 共用)

解决的问题(来自审查):
  P0-2 配置隔离: 每个实验臂 = **完整冻结配置 + 本臂覆盖项**, 并在 finally 里恢复;
       同时落盘 配置哈希 / 场景哈希 / 案例编号, 使"同场景配对"可被独立验证。
  P0-3 统一计时: 只保留一个计时函数 sim_time(), 阶段时间用同一口径(此前
       selfcheck 只算"移动+检测", 阶段统计又漏换频与清除成本, 跨脚本不可比)。
  P0-4 真值核验: 清除率同时给出
         * 案例全清率   = 全部源被清除的案例数 / 案例数
         * 平均源清除比例 = 各案例(已清除/真实源数)的均值
         * 漏清案例数 / 漏清源数
       并强制三方一致: 机器人返回 == 客户端 /clear 成功数 == len(env.cleared) == 源数。

用法:
    import simlib
    with simlib.config_scope(rb.Problem3Robot, simlib.FROZEN3, OPP_MEASURE=True):
        rows = simlib.run_cases(...)
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp = load_module("exp", ROOT / "2026B_solution" / "verify" / "experiment.py")

# ---------------- 计费模型(由实机 86 局逐局回算标定, 见 time_model_audit.py) ----------------
SPEED = 5.0            # 移动速度 m/s
T_MEASURE = 5.0        # 单次检测(含测向)
T_SWITCH = 1.0         # 频道切换
T_CLEAR_OK = 5.0       # 成功清除: 3(执行)+2(确认) = 5 s, 见附件1/附件2 的标准计时公式
                       # 注: 曾"标定为 4 s", 实为客户端把 /clear 误记一次换频(+1 s)所致,
                       #     已更正(clear 不换频), 计时回到题设公式。
T_CLEAR_FAIL = 3.0     # 清除未发现
T_ENTER_EXIT = 0.0     # 无固定项: 实测残余为**高估**方向, 加常数项只会更偏(见 time_model_audit.py)
# 标定结果(同一局回算 vs 模拟器虚拟时间, 86 局):
#   原始口径(5 s 清除/无固定项) e = T-实际: 问题三 +14.3 s、问题四 +16.9 s, 相对 0.33%/0.17%
#   标定口径(4 s 清除, 无常数项) 残余: 问题三 +0.9 s、问题四 +3.7 s, 相对 0.02%/0.04%, SD 1.6/2.1 s
#   注: 偏差与路程无关(速度恰为 5 m/s), 只随成功清除次数线性增长 +1.0 s/次。

# ---------------- 冻结配置(唯一真源) ----------------
FROZEN3 = {
    "ORDER_BY_PROB": False, "DOP_PRESCREEN": True, "LENS_TRAVEL_W": 0.05,
    "ON_WAY_DELTA": 300.0, "LS_CLEAR_GATE": None,
    # 覆盖环(§5.5 候选: 半径 1150 m, 9 点 40 度等分; 几何余量必增, 实机时间优势未确认)。
    # 回退到旧冻结配置: RING_R=1200.0, RING_N=6
    "RING_R": 1150.0, "RING_N": 9,
    "OPP_MEASURE": False, "OPP_TARGET": "spec", "OPP_MAX_PER_POINT": 2,
    "OPP_MIN_CROSS_DEG": 45.0, "OPP_MIN_SEP_M": 200.0,
    "SUPP_REUSE": False, "SUPP_REUSE_ROUTE_GATE_M": 2000.0,
    "SUPP_REUSE_MIN_SAVING_M": 100.0, "SUPP_REUSE_MAX_PER_POINT": 1,
    "SUPP_REUSE_MIN_CROSS_DEG": 45.0, "SUPP_REUSE_DELETE_MODE": "cert",
}
FROZEN4_CLS = {"USE_NEG_INFO": False, "USE_PSO": False, "DO_VERIFY": False,
               "NEIGHBOR_RINGS": (8.0, 15.0), "SUPP_MAX_DIST": None,
               "MEC_FREEZE": True,      # MEC 就绪冻结(已采纳; 消融各臂需显式覆盖以免串味)
               "DEFER_ONWAY_HOMING": True,   # 顺路 LS 失败后暂缓归航(已采纳)
               # 网格补齐点(覆盖空洞修复): 必须纳入冻结配置, 否则消融各臂会互相串味
               "MESH_EXTRA_PTS": [(-1174.6, -1363.8), (1773.2, -308.8),
                                  (-1167.6, -1372.5), (1768.7, -342.5)]}
# 冻结网格 = robot4 模块级当前默认(θ20 + 平移), 与实机默认保持一致;
# 旧网格 900/800/θ0 只作为消融对照臂, 不再作为"冻结"值。
FROZEN4_MOD = {"ON_WAY_DELTA": 300.0, "MESH_A": 970.0, "MESH_MARGIN": 700.0,
               "MESH_THETA": 20.0, "MESH_OFFSET": (460.0, 398.0)}


def apply_cfg(target, cfg):
    """把 cfg 逐项写入 target(类或模块), 返回旧值以便恢复。"""
    old = {}
    for k, v in cfg.items():
        old[k] = getattr(target, k, None)
        setattr(target, k, v)
    return old


def restore_cfg(target, old):
    for k, v in old.items():
        setattr(target, k, v)


@contextlib.contextmanager
def config_scope(*pairs):
    """config_scope((cls1, cfg1), (mod2, cfg2), ...) —— 进入时应用, 退出时**无条件恢复**。"""
    saved = []
    try:
        for target, cfg in pairs:
            saved.append((target, apply_cfg(target, cfg)))
        yield
    finally:
        for target, old in reversed(saved):
            restore_cfg(target, old)


def cfg_hash(*cfgs):
    """配置指纹: 便于把结果与具体配置绑定。"""
    h = hashlib.sha256()
    for cfg in cfgs:
        for k in sorted(cfg):
            h.update(f"{k}={cfg[k]!r}|".encode())
    return h.hexdigest()[:16]


def frozen3(**override):
    c = dict(FROZEN3); c.update(override); return c


def frozen4_cls(**override):
    c = dict(FROZEN4_CLS); c.update(override); return c


def frozen4_mod(**override):
    c = dict(FROZEN4_MOD); c.update(override); return c


# ---------------- 场景随机源 ----------------
def case_env(seed, k, **kw):
    """按 (seed, 案例编号) 派生场景随机源。

    必须如此: exp.Env 用同一个 rng 既生成场景、又在每次 measure 抽 ±1° 噪声,
    各臂共用一个 rng 时, 臂间动作数不同会错开随机流 -> 同编号变成不同场景。
    注意: 本函数只保证**场景**配对; 相同物理观测对应相同噪声**不保证**(噪声调用序号
    仍随策略动作数变化), 因此"置信区间变窄"不能当作配对正确性的证明, 需用 scene_hash 核对。
    """
    return exp.Env(np.random.default_rng([int(seed), int(k)]), **kw)


def scene_hash(env):
    """场景指纹(源频道/位置/接收半径/指向), 用于逐案例核对各臂是否真同场景。"""
    h = hashlib.sha256()
    for s in sorted(env.sources, key=lambda s: int(s["ch"])):
        h.update(f"{int(s['ch'])}|{float(s['pos'][0]):.6f}|{float(s['pos'][1]):.6f}|"
                 f"{float(s['r_rx']):.6f}|"
                 f"{'None' if s['pointing'] is None else format(float(s['pointing']), '.6f')}|".encode())
    return h.hexdigest()[:16]


# ---------------- 统一客户端与计时 ----------------
class SimClient:
    """离线客户端: 与正式 SimClient 同接口, 逐动作记账(阶段+总计), 便于统一口径核账。"""

    def __init__(self, env):
        self.env = env
        self.position = (0.0, 0.0)
        self.channel = 1
        self.remaining_real = 1200
        self.meta = {}
        self.dist = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_ok = 0
        self.n_switch = 0
        self.fail = 0
        self._phase = "init"
        self.ph = {}                 # 阶段 -> [移动, 检测, 清除ok, 清除fail, 换频]
        self.ledger = []             # 逐动作 (phase, kind, cost_s) 便于分账核对
        self.trace = []              # 逐动作 (phase, kind, ch, result, x, y) 供策略审计
        self.repeat_same_point_channel = 0   # "同点同频道重复测量"计数器(硬约束检查)

    # --- 阶段 ---
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, name):
        self._phase = name

    def _p(self):
        return self.ph.setdefault(self._phase, [0.0, 0, 0, 0, 0])

    def _move(self, x, y):
        dd = math.hypot(x-self.position[0], y-self.position[1])
        if dd:
            self.dist += dd
            self._p()[0] += dd
            self.ledger.append((self._phase, "move", dd/5.0))
        self.position = (x, y)

    def enter(self):
        pass

    def measure(self, x, y, ch):
        key = (round(x, 3), round(y, 3), int(ch))
        if key in getattr(self, "_seen", set()):
            self.repeat_same_point_channel += 1
        else:
            self._seen = getattr(self, "_seen", set()) | {key}
        self._move(x, y)
        if ch != self.channel:
            self.n_switch += 1
            self._p()[4] += 1
            self.ledger.append((self._phase, "switch", 1.0))
        self.channel = ch
        self.n_measure += 1
        self._p()[1] += 1
        self.ledger.append((self._phase, "measure", 5.0))
        r, svd = self.env.measure(np.array([x, y]), ch)
        self.trace.append((self._phase, "measure", int(ch), r, x, y))
        return True, r, svd

    def clear(self, x, y, ch):
        self._move(x, y)
        # 题设: /clear **不换频**, 也不改变测向机频道状态(附件1/附件2);
        # 此前在此记 +1 换频并以"清除 4 s"补偿, 属错误建模, 已更正为 5 s 清除。
        self.n_clear += 1
        r = self.env.clear(np.array([x, y]), ch)
        if r == "success":
            self.n_clear_ok += 1
            self._p()[2] += 1
            self.ledger.append((self._phase, "clear_ok", 5.0))
        else:
            self.fail += 1
            self._p()[3] += 1
            self.ledger.append((self._phase, "clear_fail", 3.0))
        self.trace.append((self._phase, "clear", int(ch), r, x, y))
        return True, r

    def exit(self):
        pass


def sim_time(cli):
    """**唯一**计时口径(实机标定): 移动/5 + 检测*5 + 换频*1 + 成功清除*6 + 失败清除*3 + 13(enter/exit)。"""
    return (cli.dist/SPEED + T_MEASURE*cli.n_measure + T_SWITCH*cli.n_switch
            + T_CLEAR_OK*cli.n_clear_ok + T_CLEAR_FAIL*cli.fail + T_ENTER_EXIT)


def ledger_time(cli):
    """按逐动作账本重算总时间(与 sim_time 必须一致, 用作分账自检)。"""
    return sum(c for _, _, c in cli.ledger)


def phase_time(cli, name):
    """阶段时间(同一口径: 含换频与清除成本)。"""
    mv, nm, cok, cf, sw = cli.ph.get(name, [0.0, 0, 0, 0, 0])
    return (mv/SPEED + nm*T_MEASURE + sw*T_SWITCH
            + cok*T_CLEAR_OK + cf*T_CLEAR_FAIL)   # 固定 enter/exit 项不计入任何阶段


def check_clearance(n_returned, cli, env):
    """清除正确性核验(真值口径)。返回 dict, 含三方一致性与两个不同的"清除率"。"""
    env_cleared = len(env.cleared)
    n_src = int(env.n_src)
    consistent = (n_returned == cli.n_clear_ok == env_cleared == n_src)
    return {
        "n_src": n_src,
        "cleared_env": env_cleared,
        "cleared_ret": int(n_returned),
        "cleared_client": int(cli.n_clear_ok),
        "case_full_clear": 1 if env_cleared == n_src else 0,     # 案例全清
        "src_clear_ratio": env_cleared/n_src if n_src else 0.0,  # 平均源清除比例
        "missing_src": max(0, n_src - env_cleared),              # 漏清源数
        "consistent": bool(consistent),
    }


def summarize(rows, keys=("T",)):
    """空集/单例安全的统计: 0 例返回 None, 1 例不给零宽区间。"""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k] = None
        elif v.size == 1:
            out[k] = dict(n=1, mean=float(v[0]), median=float(v[0]),
                          p90=float(v[0]), max=float(v[0]))
        else:
            out[k] = dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                          p90=float(np.percentile(v, 90)), max=float(v.max()))
    return out


def paired_stat(arm, base):
    """配对差值统计。样本不足时明确返回 None, 不输出伪精确区间。"""
    d = np.array(arm, float) - np.array(base, float)
    if d.size == 0:
        return None
    if d.size == 1:
        return dict(n=1, mean=float(d[0]), lo=None, hi=None,
                    median=float(d[0]), win=float(d[0] < 0), p10=None, p90=None)
    se = d.std(ddof=1)/math.sqrt(d.size)
    return dict(n=int(d.size), mean=float(d.mean()), lo=float(d.mean()-1.96*se),
                hi=float(d.mean()+1.96*se), median=float(np.median(d)),
                win=float((d < 0).mean()),
                p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)))


def bootstrap_ci(arm, base, n_boot=2000, seed=7):
    """按案例成对重采样 bootstrap 区间(不拆散案例对), 用于稀疏触发/少数大收益模块。"""
    d = np.array(arm, float) - np.array(base, float)
    if d.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
