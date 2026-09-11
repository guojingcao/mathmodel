"""自适应四叉网格：不以“格心无效”为理由丢弃整个网格。"""
from dataclasses import dataclass
import logging
import numpy as np
try:  # 作为 Problem1_Localization 包被问题二复用时走相对导入。
    from ..config import GridConfig
except ImportError:  # 直接运行 Problem1_Localization/main.py 时保持兼容。
    from config import GridConfig
from .geometry import wedge_halfplanes, valid_mask, convex_hull

LOG = logging.getLogger(__name__)
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)


@dataclass
class RegionResult:
    """离散可行点、内外凸包及可审计网格数据；外包不是精确可行集。"""

    valid_points: np.ndarray
    inner_hull: np.ndarray
    outer_hull: np.ndarray
    cells: np.ndarray
    status: str
    boundary_width: float
    levels: int
    visited_cells: int


def classify_cells(cells, normals, offsets, radius):
    """格心 c、半边长 h 的线性区间为 a·c-b ± h||a||_1。

    某约束最大值仍为负才丢弃；所有约束最小值非负才确认整格内部。
    圆约束使用矩形到原点最近/最远距离，保证不能漏掉穿格角域。
    """
    centers, half = cells[:, :2], cells[:, 2]
    near = np.maximum(np.abs(centers) - half[:, None], 0)
    far = np.abs(centers) + half[:, None]
    possible = np.sum(near**2, axis=1) <= radius**2 + 1e-8
    interior = np.sum(far**2, axis=1) <= radius**2 - 1e-8
    for normal, offset in zip(normals, offsets):
        value = centers @ normal - offset
        spread = half * np.abs(normal).sum()
        possible &= value + spread >= -1e-9
        interior &= value - spread >= 1e-9
    return possible, interior & possible


def calculate_feasible_region(observation_points, angles, error=1.0, *, config=None,
                              return_details=False):
    """从 200×200 初始格出发，只细分不能确定的边界格，返回 valid_points。

    return_details=True 提供内外包，用来评估连续区域指标及离散误差。
    无采样点但存在边界格时标为 unresolved，绝不宣称区域为空。
    初始格心被检查；内格无需继续细分，保留角点帮助计算凸包。
    """
    cfg = config or GridConfig()
    if (not np.isfinite([cfg.radius, cfg.resolution]).all() or cfg.radius <= 0
            or cfg.resolution <= 0 or not isinstance(cfg.initial_grid, int)
            or cfg.initial_grid < 2 or cfg.max_cells < 1):
        raise ValueError("网格参数必须有效、有限且为正；initial_grid 为 >=2 的整数")
    normals, offsets = wedge_halfplanes(observation_points, angles, error)
    observers = np.asarray(observation_points, float)
    width = 2 * cfg.radius / cfg.initial_grid
    axis = -cfg.radius + (np.arange(cfg.initial_grid) + .5) * width
    xx, yy = np.meshgrid(axis, axis)
    cells = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, width / 2)])
    accepted, boundary, level, visited = [], [], 0, 0
    while len(cells):
        visited += len(cells)
        if visited > cfg.max_cells:
            raise RuntimeError("网格计算达到 max_cells 限制；请提高限额或放宽精度")
        possible, interior = classify_cells(cells, normals, offsets, cfg.radius)
        accepted.append(cells[interior])
        uncertain = cells[possible & ~interior]
        LOG.debug("层 %d：检查 %d 格，内部 %d，边界 %d", level, len(cells), interior.sum(), len(uncertain))
        if not len(uncertain) or width <= cfg.resolution:
            boundary.append(uncertain)
            break
        child_half = uncertain[:, 2] / 2
        centers = uncertain[:, None, :2] + SIGNS[None] * child_half[:, None, None]
        cells = np.column_stack([centers.reshape(-1, 2), np.repeat(child_half, 4)])
        width /= 2
        level += 1
    leaves = np.concatenate(accepted + boundary, axis=0)
    corners = (leaves[:, None, :2] + leaves[:, None, 2:3] * SIGNS).reshape(-1, 2)
    candidates = np.concatenate([corners, leaves[:, :2]])
    valid = np.unique(candidates[valid_mask(candidates, observers, angles, error, cfg.radius)], axis=0)
    status = "sampled" if len(valid) else "unresolved" if len(leaves) else "empty"
    result = RegionResult(valid, convex_hull(valid), convex_hull(corners), leaves,
                          status, width, level + 1, visited)
    LOG.debug("求解结束：%s；可行点 %d，叶格 %d，边界格边长 %.6f m", status, len(valid), len(leaves), width)
    return result if return_details else result.valid_points
