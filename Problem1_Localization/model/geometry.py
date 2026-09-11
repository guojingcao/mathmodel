"""角度均用度，逆时针为正，正 x 轴为零度。"""
import numpy as np


def angle_between_points(point1, point2):
    """计算从 point1 指向 point2 的方向；重合点方向未定义，显式报错。"""
    delta = np.asarray(point2, dtype=float) - np.asarray(point1, dtype=float)
    if delta.shape != (2,) or not np.isfinite(delta).all() or np.linalg.norm(delta) == 0:
        raise ValueError("两点必须是有限二维坐标，且不能重合")
    return float(np.degrees(np.arctan2(delta[1], delta[0])) % 360)


def angle_difference(angle1, angle2):
    """圆周最短角距：|((a-b+180) mod 360)-180|，支持数组。"""
    return np.abs((np.asarray(angle1) - np.asarray(angle2) + 180) % 360 - 180)


def check_direction_constraint(candidate_point, observer_point, observed_angle, error):
    """直接按圆周角距判断单点；不把观测站本身当成有效测向点。"""
    if not np.isfinite([observed_angle, error]).all() or not 0 <= error < 90:
        raise ValueError("本凸角域模型要求 0 <= error < 90 度")
    delta = np.asarray(candidate_point, float) - np.asarray(observer_point, float)
    if delta.shape != (2,) or not np.isfinite(delta).all() or np.linalg.norm(delta) == 0:
        return False
    theta = np.degrees(np.arctan2(delta[1], delta[0])) % 360
    return bool(angle_difference(theta, observed_angle) <= error + 1e-10)


def wedge_halfplanes(observers, angles, error):
    """将窄角域写成 A X >= b；两个法向量分别限制左右边界，避免反向射线。

    cross(u_low, X-P)>=0，cross(u_high, X-P)<=0。
    该闭包表达在站点本身也成立；采样输出另行排除站点。
    """
    observers = np.asarray(observers, float)
    angles = np.asarray(angles, float)
    if observers.ndim != 2 or observers.shape[1] != 2 or len(observers) == 0:
        raise ValueError("observation_points 必须是非空 n×2 数组")
    if angles.shape != (len(observers),) or not np.isfinite(observers).all() or not np.isfinite(angles).all():
        raise ValueError("观测点和有限角度必须一一对应")
    if not np.isfinite(error) or not 0 <= error < 90:
        raise ValueError("error 必须为 [0,90) 内的有限数")
    low, high = np.deg2rad(angles - error), np.deg2rad(angles + error)
    center = np.deg2rad(angles)
    normals = np.stack([np.column_stack([-np.sin(low), np.cos(low)]),
                        np.column_stack([np.sin(high), -np.cos(high)]),
                        np.column_stack([np.cos(center), np.sin(center)])], axis=1).reshape(-1, 2)
    # 正向约束在 error>0 时冗余，但 error=0 时必须加入，否则退化成双向直线。
    offsets = np.sum(normals * np.repeat(observers, 3, axis=0), axis=1)
    return normals, offsets


def valid_mask(points, observers, angles, error, radius):
    """向量化原始 atan2 约束，用于独立于半平面筛格的点级复核。"""
    points = np.asarray(points, float).reshape(-1, 2)
    keep = np.sum(points**2, axis=1) <= radius**2 + 1e-8
    for observer, angle in zip(observers, angles):
        delta = points - observer
        theta = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        keep &= (angle_difference(theta, angle) <= error + 1e-10)
        keep &= np.any(delta != 0, axis=1)
    return keep


def convex_hull(points):
    """Andrew 单调链凸包；去重复和共线内点，支持空集、单点及线段。"""
    points = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(points) <= 2:
        return points
    lower, upper = [], []
    for chain, sequence in ((lower, points), (upper, points[::-1])):
        for point in sequence:
            while len(chain) >= 2:
                a, b = chain[-1] - chain[-2], point - chain[-1]
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1] + upper[:-1])
