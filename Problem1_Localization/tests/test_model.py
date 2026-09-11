"""独立角度判断、解析几何与穷举覆盖圆交叉验证；运行 unittest discover。"""
import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GridConfig
from model.geometry import (angle_between_points, angle_difference, check_direction_constraint,
                            wedge_halfplanes, convex_hull, valid_mask)
from model.feasible_region import calculate_feasible_region
from model.metrics import (polygon_area_centroid, maximum_diameter, minimum_enclosing_circle,
                           circle_from_three, calculate_metrics)


def independent_polygon(observers, angles, error):
    """仅用于测试的多边形半平面裁剪参考，不作为生产定位求解器。

    初始方框[-1800,1800]²；调用者须验证最终顶点均在圆盘内。
    """
    polygon = np.array([[-1800., -1800.], [1800., -1800.], [1800., 1800.], [-1800., 1800.]])
    normals, offsets = wedge_halfplanes(observers, angles, error)
    for normal, offset in zip(normals, offsets):
        result = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            a, b = start @ normal - offset, end @ normal - offset
            if a >= 0:
                result.append(start)
            if (a >= 0) != (b >= 0):
                result.append(start + a / (a - b) * (end - start))
        polygon = np.asarray(result).reshape(-1, 2)
    return polygon


class ModelTests(unittest.TestCase):
    """关注集合安全性和数值误差，而不是仅检验代码不报错。"""

    def test_angle_wrap(self):
        """跨零度与各象限方向使用 atan2 而不是 atan(y/x)。"""
        self.assertEqual(angle_difference(359, 1), 2)
        self.assertEqual(angle_difference(0, 180), 180)
        for p, expected in [((1, 0), 0), ((0, 1), 90), ((-1, 0), 180), ((0, -1), 270)]:
            self.assertEqual(angle_between_points((0, 0), p), expected)
        with self.assertRaises(ValueError):
            angle_between_points((0, 0), (0, 0))

    def test_constraint(self):
        """边界包含、站点排除以及反向射线拒绝。"""
        self.assertTrue(check_direction_constraint([1, 0], [0, 0], 359, 1))
        self.assertFalse(check_direction_constraint([-1, 0], [0, 0], 0, 1))
        self.assertFalse(check_direction_constraint([0, 0], [0, 0], 0, 1))

    def test_halfplanes_vs_angles(self):
        """随机点上用独立原始角距判别交叉检验半平面表达。"""
        rng = np.random.default_rng(19)
        for error in (0, .2, 1, 45, 89):
            for angle in (0, 359, 120):
                a, b = wedge_halfplanes([[3, -4]], [angle], error)
                points = rng.uniform(-100, 100, (1000, 2))
                fast = np.all(points @ a.T >= b, axis=1)
                direct = valid_mask(points, [[3, -4]], [angle], error, 1000)
                np.testing.assert_array_equal(fast, direct)
        a, b = wedge_halfplanes([[0, 0]], [0], 0)
        self.assertFalse(np.all(np.array([-1, 0]) @ a.T >= b))

    def test_invalid_inputs(self):
        """不允许无约束、错配、NaN、过宽角域或负网格参数静默运行。"""
        for points, angles, error in [([], [], 1), ([[0, 0]], [], 1), ([[0, 0]], [0], 90),
                                       ([[np.nan, 0]], [0], 1), ([[0, 0]], [0], -1)]:
            with self.assertRaises(ValueError):
                calculate_feasible_region(points, angles, error)

    def test_exact_square_metrics(self):
        """正方形面积4、质心0、直径2√2、覆盖半径√2。"""
        polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        area, centroid = polygon_area_centroid(polygon)
        self.assertAlmostEqual(area, 4)
        np.testing.assert_allclose(centroid, [0, 0])
        self.assertAlmostEqual(maximum_diameter(polygon), 2*np.sqrt(2))
        center, radius = minimum_enclosing_circle(polygon)
        self.assertAlmostEqual(radius, np.sqrt(2))
        np.testing.assert_allclose(center, [0, 0])

    def test_diameter_circle_counterexample(self):
        """等边三角形证明 D/2 半径不能一般性覆盖可行域。"""
        points = np.array([[0., 0.], [2., 0.], [1., np.sqrt(3)]])
        _, radius = minimum_enclosing_circle(points)
        self.assertAlmostEqual(radius, 2/np.sqrt(3))
        self.assertGreater(radius, maximum_diameter(points)/2)

    def test_circle_against_exhaustive(self):
        """小点集穷举所有两点圆与三点圆，独立检验增量最小圆。"""
        rng = np.random.default_rng(24)
        for _ in range(40):
            points = rng.normal(size=(7, 2))
            circles = [((a+b)/2, np.linalg.norm(a-b)/2) for a, b in itertools.combinations(points, 2)]
            circles += [circle_from_three(a, b, c) for a, b, c in itertools.combinations(points, 3)]
            radius = min(r for c, r in circles if np.all(np.linalg.norm(points-c, axis=1) <= r+1e-8))
            center, computed = minimum_enclosing_circle(points)
            self.assertAlmostEqual(computed, radius, places=8)
            self.assertTrue(np.all(np.linalg.norm(points-center, axis=1) <= computed+1e-8))

    def test_degenerate_metrics(self):
        """空集、单点、重复点、共线点不制造不存在的位置误差。"""
        self.assertEqual(minimum_enclosing_circle(np.empty((0, 2))), (None, None))
        points = convex_hull([[2, 0], [0, 0], [1, 0], [2, 0]])
        self.assertEqual(len(points), 2)
        center, radius = minimum_enclosing_circle(points)
        self.assertEqual(radius, 1)
        np.testing.assert_allclose(center, [1, 0])

    def test_reference_and_convergence(self):
        """独立裁剪多边形的面积/直径/最小圆/质心应落在网格误差包络内。"""
        observers = np.array([[-1000, 0], [0, -1000]])
        polygon = independent_polygon(observers, [0, 90], 1)
        self.assertLess(np.linalg.norm(polygon, axis=1).max(), 1800)
        exact_area, exact_centroid = polygon_area_centroid(polygon)
        exact_d = maximum_diameter(polygon)
        _, exact_r = minimum_enclosing_circle(polygon)
        gaps = []
        for resolution in (1.125, .28125, .0703125):
            region = calculate_feasible_region(observers, [0, 90], 1,
                         config=GridConfig(resolution=resolution), return_details=True)
            m = calculate_metrics(region, [0, 0])
            self.assertLessEqual(m["area_lower"], exact_area)
            self.assertGreaterEqual(m["area_upper"], exact_area)
            self.assertLessEqual(m["diameter"], exact_d)
            self.assertGreaterEqual(m["diameter_upper"], exact_d)
            self.assertLessEqual(m["cover_radius_lower"], exact_r)
            self.assertGreaterEqual(m["cover_radius"], exact_r)
            self.assertLessEqual(np.linalg.norm(exact_centroid-[m["centroid_x"], m["centroid_y"]]), m["centroid_error_bound"])
            self.assertTrue(valid_mask(region.valid_points, observers, [0, 90], 1, 1800).all())
            gaps.append(m["area_upper"]-m["area_lower"])
        self.assertTrue(np.all(np.diff(gaps) < 0))

    def test_empty_and_narrow(self):
        """相反角域为空；远窄于初始18m网格的区域仍能被边界细分发现。"""
        empty = calculate_feasible_region([[0, 0], [0, 0]], [0, 180], 1, return_details=True)
        # 两个闭角域只可能交于站点，原测向模型排除站点；允许保守标记未决。
        self.assertIn(empty.status, ("empty", "unresolved"))
        self.assertEqual(len(empty.valid_points), 0)
        narrow = calculate_feasible_region([[-1000, .13], [.17, -1000]], [0, 90], .01,
                            config=GridConfig(resolution=.03515625), return_details=True)
        self.assertEqual(narrow.status, "sampled")
        self.assertGreater(len(narrow.valid_points), 0)
        truly_empty = calculate_feasible_region([[0, 0], [-10, 0]], [0, 180], 1, return_details=True)
        self.assertEqual(truly_empty.status, "empty")

    def test_disk_and_parallel(self):
        """四分之一圆的面积解析值落入包络；平行方向由场地圆盘限制为有界。"""
        cfg = GridConfig(radius=10, initial_grid=20, resolution=.0625)
        region = calculate_feasible_region([[0, 0]], [45], 45, config=cfg, return_details=True)
        m = calculate_metrics(region)
        self.assertLessEqual(m["area_lower"], np.pi*25)
        self.assertGreaterEqual(m["area_upper"], np.pi*25)
        region = calculate_feasible_region([[-5, 0], [-4, 0]], [0, 0], 1, config=cfg, return_details=True)
        self.assertEqual(region.status, "sampled")
        self.assertLessEqual(np.linalg.norm(region.valid_points, axis=1).max(), 10+1e-9)

    def test_zero_error_no_false_certainty(self):
        """零误差且交点未落格点时，未决是合法结果，不能返回空域结论。"""
        region = calculate_feasible_region([[-1000, .123], [.456, -1000]], [0, 90], 0,
                                          config=GridConfig(resolution=.1), return_details=True)
        self.assertIn(region.status, ("unresolved", "sampled"))


if __name__ == "__main__":
    unittest.main()
