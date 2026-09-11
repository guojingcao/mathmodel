"""布局候选、几何、贪心、DOP诊断和问题一调用的单元测试。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.candidate_points import generate_candidate_points, check_route_feasibility
from model.geometry_metric import (calculate_intersection_angle, pairwise_intersection_angles,
                                   bearing_dop, calculate_area, calculate_diameter)
from model.layout_optimizer import optimize_layout, optimize_dop_baseline
from model.localization import localize


class Problem2Tests(unittest.TestCase):
    """每项测试对应一个建模假设或工程接口。"""

    @classmethod
    def setUpClass(cls):
        """构造15°间隔的固定候选集，供确定性测试复用。"""
        cls.candidates = generate_candidate_points(angular_step=15)

    def test_candidates(self):
        """圆周候选应有24个、无重复、半径1000m且位于场地。"""
        self.assertEqual(self.candidates.shape, (24, 2))
        self.assertEqual(len(np.unique(np.round(self.candidates, 9), axis=0)), 24)
        np.testing.assert_allclose(np.linalg.norm(self.candidates, axis=1), 1000)
        self.assertTrue(check_route_feasibility(self.candidates[:2]))

    def test_candidate_validation(self):
        """非法中心、半径、步长和场地不足应明确报错。"""
        for kwargs in ({"radius": 0}, {"angular_step": 0}, {"angular_step": 200},
                       {"center": [np.nan, 0]}, {"radius": 1000, "arena_radius": 100}):
            with self.assertRaises(ValueError):
                generate_candidate_points(**kwargs)

    def test_intersection_angle(self):
        """正交为90°，反向共线按条件角定义为0°。"""
        self.assertAlmostEqual(calculate_intersection_angle([[1, 0], [0, 1]], [0, 0]), 90)
        self.assertAlmostEqual(calculate_intersection_angle([[1, 0], [-1, 0]], [0, 0]), 0)
        values = pairwise_intersection_angles([[1, 0], [0, 1], [-1, 0]], [0, 0])
        np.testing.assert_allclose(np.sort(values), [0, 90, 90])

    def test_dop(self):
        """共线方向DOP无穷，正交两站DOP为√2。"""
        self.assertTrue(np.isinf(bearing_dop([[1, 0], [-1, 0]], [0, 0])))
        self.assertAlmostEqual(bearing_dop([[1, 0], [0, 1]], [0, 0]), np.sqrt(2))

    def test_problem1_inheritance(self):
        """适配器返回问题一RegionResult，真值包含且指标封装一致。"""
        region, metrics, angles = localize([[-1000, 0], [0, -1000]], [0, 0],
            measurement_errors=[.2, -.3], grid_resolution=.28125,
            truth_for_evaluation=[0, 0])
        self.assertEqual(region.status, "sampled")
        self.assertEqual(len(angles), 2)
        area, low, high = calculate_area(region)
        diameter, upper = calculate_diameter(region)
        self.assertEqual(area, metrics["area"])
        self.assertLessEqual(low, area); self.assertGreaterEqual(high, area)
        self.assertLessEqual(diameter, upper)

    def test_error_validation(self):
        """测量误差维数必须与站点一致，站点不可与参考目标重合。"""
        with self.assertRaises(ValueError):
            localize([[1, 0], [0, 1]], [0, 0], measurement_errors=[0])
        with self.assertRaises(ValueError):
            localize([[0, 0]], [0, 0])

    def test_optimizer_deterministic_and_nested(self):
        """相同输入结果完全可复现，选择无重复且面积包络随加点不扩大。"""
        first = optimize_layout(self.candidates, [0, 0], 4, seed=123,
                                grid_resolution=1.125)
        second = optimize_layout(self.candidates, [0, 0], 4, seed=123,
                                 grid_resolution=1.125)
        self.assertEqual(first.selected_indices, second.selected_indices)
        self.assertEqual(len(set(first.selected_indices)), 4)
        upper = [row["area_upper"] for row in first.trace]
        self.assertTrue(np.all(np.diff(upper) <= 1e-8))
        self.assertTrue(check_route_feasibility(first.selected_points))

    def test_optimizer_candidates_evaluated(self):
        """第二轮应评价除起点外所有23个可达候选点。"""
        result = optimize_layout(self.candidates, [0, 0], 2, seed=1,
                                 grid_resolution=1.125)
        self.assertEqual(len(result.candidate_evaluations), 23)
        self.assertEqual([row["round"] for row in result.trace], [1, 2])

    def test_dop_is_only_baseline(self):
        """DOP基线共享起点和候选集，但不调用或替换主优化结果。"""
        layout, indices = optimize_dop_baseline(self.candidates, [0, 0], 4, initial_index=3)
        self.assertEqual(indices[0], 3)
        self.assertEqual(len(set(indices)), 4)
        np.testing.assert_allclose(layout, self.candidates[indices])


if __name__ == "__main__":
    unittest.main()
