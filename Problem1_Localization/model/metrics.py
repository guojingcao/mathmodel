"""对自适应网格内外包计算区域指标，不将不等面积叶格作等权平均。"""
import numpy as np


def polygon_area_centroid(polygon):
    """鞋带公式面积和面积质心；退化线段/点返回点集均值，空集返回 None。"""
    if not len(polygon):
        return 0.0, None
    if len(polygon) < 3:
        return 0.0, polygon.mean(axis=0)
    # 平移到局部原点降低大坐标下鞋带差分的数值抵消。
    origin = polygon[0]
    p = polygon - origin
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    twice_area = cross.sum()
    if abs(twice_area) < 1e-14:
        return 0.0, polygon.mean(axis=0)
    centroid = origin + ((p + q) * cross[:, None]).sum(axis=0) / (3 * twice_area)
    return abs(float(twice_area)) / 2, centroid


def maximum_diameter(points):
    """凸包顶点间最大欧氏距离；分块计算，避免完整 n×n 距离矩阵。"""
    best = 0.0
    for start in range(0, len(points), 256):
        diff = points[start:start + 256, None] - points[None]
        best = max(best, float(np.sum(diff**2, axis=2).max(initial=0)))
    return best**.5


def circle_from_three(a, b, c):
    """三点外接圆；近共线时选择能覆盖三点的最小两点直径圆。"""
    u, v = b - a, c - a
    det = 2 * (u[0] * v[1] - u[1] * v[0])
    if abs(det) <= 1e-12 * max(np.linalg.norm(u) * np.linalg.norm(v), 1):
        choices = []
        for p, q in ((a, b), (a, c), (b, c)):
            center = (p + q) / 2
            radius = np.linalg.norm(p - q) / 2
            if max(np.linalg.norm(center - x) for x in (a, b, c)) <= radius + 1e-8:
                choices.append((center, radius))
        if choices:
            return min(choices, key=lambda item: item[1])
        raise ArithmeticError("近共线三点的覆盖圆数值不稳定")
    uu, vv = u @ u, v @ v
    center = a + np.array([v[1] * uu - u[1] * vv, u[0] * vv - v[0] * uu]) / det
    return center, float(np.linalg.norm(center - a))


def minimum_enclosing_circle(points):
    """固定种子随机增量最小覆盖圆；边界由至多三个点决定，可复现。

    仅作用于凸包，不引入定位搜索算法。返回圆心和半径；空集返回 (None,None)。
    """
    if not len(points):
        return None, None
    ordered = np.random.default_rng(17).permutation(np.asarray(points, float))
    center, radius = ordered[0].copy(), 0.0
    for i, p in enumerate(ordered):
        if np.linalg.norm(p - center) <= radius + 1e-9:
            continue
        center, radius = p.copy(), 0.0
        for j, q in enumerate(ordered[:i]):
            if np.linalg.norm(q - center) <= radius + 1e-9:
                continue
            center, radius = (p + q) / 2, float(np.linalg.norm(p - q)) / 2
            for r in ordered[:j]:
                if np.linalg.norm(r - center) > radius + 1e-9:
                    center, radius = circle_from_three(p, q, r)
    # 向外补偿浮点残差，确保输出圆覆盖所有输入顶点。
    radius = max(radius, float(np.linalg.norm(ordered - center, axis=1).max()))
    return center, radius


def calculate_metrics(region, true_position=None):
    """内包⊆Ω闭包⊆外包，因此面积、直径、最小覆盖半径均有上下界。

    area 取界中点，diameter 取下界近似，cover_radius 取安全外包圆半径。
    质心取内包面积质心；误差是描述性指标，不替代整个可行区域。
    """
    lower_area, centroid = polygon_area_centroid(region.inner_hull)
    upper_area, _ = polygon_area_centroid(region.outer_hull)
    lower_d = maximum_diameter(region.inner_hull)
    upper_d = maximum_diameter(region.outer_hull)
    _, lower_r = minimum_enclosing_circle(region.inner_hull)
    cover_center, upper_r = minimum_enclosing_circle(region.outer_hull)
    has_sample = len(region.valid_points) > 0
    centroid_bound = upper_d * (upper_area - lower_area) / upper_area if lower_area > 0 else None
    return {
        "status": region.status,
        "area": (lower_area + upper_area) / 2 if region.status != "unresolved" else None,
        "area_lower": lower_area, "area_upper": upper_area,
        "diameter": lower_d if has_sample else None,
        "diameter_upper": upper_d if len(region.outer_hull) else None,
        "cover_radius": upper_r, "cover_radius_lower": lower_r,
        "centroid_x": float(centroid[0]) if has_sample else None,
        "centroid_y": float(centroid[1]) if has_sample else None,
        "centroid_error_bound": centroid_bound,
        "cover_center_x": float(cover_center[0]) if cover_center is not None else None,
        "cover_center_y": float(cover_center[1]) if cover_center is not None else None,
        "localization_error": float(np.linalg.norm(centroid - true_position))
        if has_sample and true_position is not None else None,
        "valid_point_count": len(region.valid_points), "leaf_count": len(region.cells),
        "boundary_width": region.boundary_width, "visited_cells": region.visited_cells,
    }
