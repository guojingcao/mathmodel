"""Smoke and output tests for the CUMCM plotting toolkit."""
from pathlib import Path
import sys
import tempfile
import unittest
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import CUMCM_Style as cs

class TestCumcmStyle(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory()
    def tearDown(self): self.temp.cleanup(); plt.close("all")
    @property
    def out(self): return Path(self.temp.name)
    def test_import_and_fonts(self):
        self.assertIn("english", cs.check_fonts())
    def test_bar_and_all_exports(self):
        cs.cumcm_bar(["A", "B"], [1, 2], filename="bar", output_dir=self.out)
        for suffix in ("pdf", "svg", "png"): self.assertTrue((self.out / f"bar.{suffix}").is_file())
    def test_core_plots(self):
        x = np.arange(5); y = x * 2 + 1
        calls = [
            lambda: cs.cumcm_line(x, y), lambda: cs.cumcm_scatter(x, y),
            lambda: cs.cumcm_heatmap([[1, .2], [.2, 1]]),
            lambda: cs.cumcm_sensitivity(["a", "b"], [-.2, .5]),
            lambda: cs.cumcm_model_compare(["a", "b"], [.8, .9]),
            lambda: cs.cumcm_residual(y, y-y), lambda: cs.cumcm_prediction(x, y, y+.2),
            lambda: cs.cumcm_box([[1, 2, 3], [2, 3, 4]], ["a", "b"]),
            lambda: cs.cumcm_multi_line(x, {"a": y, "b": y+1}),
        ]
        for call in calls:
            fig, _ = call(); self.assertIsNotNone(fig)
    def test_bad_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            cs.cumcm_bar(["a"], [1, 2])

if __name__ == "__main__": unittest.main()
