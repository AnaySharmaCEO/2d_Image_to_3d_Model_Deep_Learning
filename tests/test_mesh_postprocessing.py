"""Unit tests for mesh cleanup, decimation, and vertex arrays."""

import unittest
import numpy as np

from src.postprocessing import clean_mesh, mesh_arrays

try:
    import trimesh
except ImportError:
    trimesh = None


class TestMeshPostprocessing(unittest.TestCase):
    @unittest.skipIf(trimesh is None, "trimesh is not installed in the test environment")
    def test_clean_mesh_and_normalization(self):
        box = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        cleaned = clean_mesh(box, max_faces=100)

        v, f = mesh_arrays(cleaned)
        self.assertGreater(len(v), 0)
        self.assertGreater(len(f), 0)

        extents = v.max(axis=0) - v.min(axis=0)
        self.assertLessEqual(float(extents.max()), 1.0001)


if __name__ == "__main__":
    unittest.main()
