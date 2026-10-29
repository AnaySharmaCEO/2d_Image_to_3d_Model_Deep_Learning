"""Unit tests for image preprocessing, view assignment, and quality analysis."""

import unittest
import numpy as np
from PIL import Image

from src.preprocessing import (
    analyze_image,
    assign_views,
    has_real_alpha,
    preprocess_image,
    simple_bg_removal,
)


class TestPreprocessing(unittest.TestCase):
    def test_has_real_alpha(self):
        # RGB image without alpha
        rgb_img = Image.new("RGB", (100, 100), (255, 0, 0))
        self.assertFalse(has_real_alpha(rgb_img))

        # RGBA image with fully opaque alpha
        opaque_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        self.assertFalse(has_real_alpha(opaque_img))

        # RGBA image with meaningful transparent pixels
        trans_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        arr = np.asarray(trans_img).copy()
        arr[40:60, 40:60] = [255, 0, 0, 255]
        trans_img = Image.fromarray(arr)
        self.assertTrue(has_real_alpha(trans_img))

    def test_assign_views(self):
        img1 = Image.new("RGB", (64, 64), (255, 0, 0))
        img2 = Image.new("RGB", (64, 64), (0, 255, 0))
        img3 = Image.new("RGB", (64, 64), (0, 0, 255))

        input_dict = {"my_front_view": img1, "camera_left": img2, "rear_back": img3}
        assigned = assign_views(input_dict)

        self.assertIn("front", assigned)
        self.assertIn("left", assigned)
        self.assertIn("back", assigned)
        self.assertEqual(assigned["front"], img1)
        self.assertEqual(assigned["left"], img2)
        self.assertEqual(assigned["back"], img3)

        raw_list = [("photo1.jpg", img1), ("photo2.jpg", img2)]
        assigned_list = assign_views(raw_list)
        self.assertIn("front", assigned_list)
        self.assertEqual(len(assigned_list), 2)

    def test_simple_bg_removal(self):
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        arr = np.asarray(img).copy()
        arr[30:70, 30:70] = [20, 120, 220]
        test_img = Image.fromarray(arr)

        rgba = simple_bg_removal(test_img)
        self.assertEqual(rgba.mode, "RGBA")
        alpha = np.asarray(rgba)[..., 3]

        self.assertGreater(alpha[50, 50], 0)
        self.assertEqual(alpha[0, 0], 0)

    def test_analyze_image(self):
        orig = Image.new("RGB", (512, 512), (255, 255, 255))
        rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        arr = np.asarray(rgba).copy()
        arr[150:350, 150:350] = [255, 50, 50, 255]
        rgba = Image.fromarray(arr)

        report = analyze_image(orig, rgba)
        self.assertIn("sharpness", report)
        self.assertIn("mask_cover", report)
        self.assertEqual(report["width"], 512)
        self.assertEqual(report["height"], 512)
        self.assertTrue(0.0 < report["mask_cover"] < 1.0)


if __name__ == "__main__":
    unittest.main()
