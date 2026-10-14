"""
colorlab/lens.py 단위 테스트 (카메라가 필요 없다).

    cd ~/src
    python3 -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorlab import lens  # noqa: E402


class TestLens(unittest.TestCase):
    def test_pixel_height_example(self):
        # focal 500 px, 0.3 m 물체, 2 m 거리 -> 75 px
        self.assertAlmostEqual(lens.pixel_height(500, 0.3, 2.0), 75.0)

    def test_focal_from_measure_example(self):
        # 같은 예를 거꾸로: 0.3 m 물체가 2 m에서 75 px -> focal 500 px
        self.assertAlmostEqual(lens.focal_from_measure(75, 2.0, 0.3), 500.0)

    def test_a4_example(self):
        # A4 세로 (0.297 m), 2.3 m 거리에서 높이 120 px -> focal = 120 * 2.3 / 0.297
        self.assertAlmostEqual(lens.focal_from_measure(120, 2.3, 0.297), 120 * 2.3 / 0.297)

    def test_round_trip(self):
        f = lens.focal_from_measure(88.0, 1.75, 0.25)
        self.assertAlmostEqual(lens.pixel_height(f, 0.25, 1.75), 88.0)

    def test_median(self):
        self.assertEqual(lens.median([3, 1, 2]), 2)
        self.assertEqual(lens.median([4, 1, 3, 2]), 2.5)
        with self.assertRaises(ValueError):
            lens.median([])

    def test_px_per_degree(self):
        self.assertAlmostEqual(lens.px_per_degree(500), 500 * math.pi / 180)

    def test_fov(self):
        # 폭 1280 px, focal 640 px -> 2 * atan(1) = 90 deg
        self.assertAlmostEqual(lens.fov_deg(1280, 640), 90.0)

    def test_marks(self):
        self.assertEqual(lens.mark(39.9), "LOW")
        self.assertEqual(lens.mark(40.0), "--")
        self.assertEqual(lens.mark(59.9), "--")
        self.assertEqual(lens.mark(60.0), "OK")

    def test_table_shape_and_values(self):
        t = lens.size_table(500)
        self.assertEqual(len(t), len(lens.SIGN_SIZES_M))
        self.assertTrue(all(len(r) == len(lens.DISTANCES_M) for r in t))
        # 0.30 m 표지판, 2.0 m 거리 -> 75 px
        i, j = lens.SIGN_SIZES_M.index(0.30), lens.DISTANCES_M.index(2.0)
        self.assertAlmostEqual(t[i][j], 75.0)

    def test_table_text_is_ascii(self):
        text = "\n".join(lens.format_size_table(500))
        text.encode("ascii")  # ASCII가 아니면 예외가 발생한다
        self.assertIn("LOW", text)
        self.assertIn("OK", text)

    def test_bad_input(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                lens.focal_from_measure(bad, 2.0, 0.3)
            with self.assertRaises(ValueError):
                lens.pixel_height(500, 0.3, bad)


if __name__ == "__main__":
    unittest.main()
