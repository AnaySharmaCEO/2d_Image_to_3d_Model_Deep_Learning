"""Unit tests for Tier 3 procedural relief math engine."""

import unittest
import numpy as np
from PIL import Image

from src.backends.procedural_backend import ProceduralMathBackend

try:
    import trimesh
except ImportError:
    trimesh = None


class TestProceduralEngine(unittest.TestCase):
    @unittest.skipIf(trimesh is None, "trimesh is not installed in the test environment")
    def test_procedural_math_backend_execution(self):
        size = 128
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        arr = np.asarray(img).copy()
        y, x = np.ogrid[:size, :size]
        mask = (x - size // 2) ** 2 + (y - size // 2) ** 2 <= (size // 3) ** 2
        arr[mask] = [200, 100, 50, 255]
        disk_img = Image.fromarray(arr, "RGBA")

        backend = ProceduralMathBackend()
        mesh = backend.generate(views={"front": disk_img}, seed=0, config={})

        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.vertices), 10)
        self.assertGreater(len(mesh.faces), 10)
        self.assertTrue(hasattr(mesh, "visual"))

    @unittest.skipIf(trimesh is None, "trimesh is not installed in the test environment")
    def test_procedural_empty_mask_fallback(self):
        empty_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        backend = ProceduralMathBackend()
        mesh = backend.generate(views={"front": empty_img}, seed=0, config={})

        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)


if __name__ == "__main__":
    unittest.main()
