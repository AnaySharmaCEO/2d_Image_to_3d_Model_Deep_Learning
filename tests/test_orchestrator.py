"""Unit tests for the orchestrator, watchdog configuration, and tier cascade."""

import os
import shutil
import tempfile
import unittest
from PIL import Image

from src.config import AgentConfig
from src.orchestrator import ImageTo3DAgent

try:
    import trimesh
except ImportError:
    trimesh = None


class TestOrchestrator(unittest.TestCase):
    @unittest.skipIf(trimesh is None, "trimesh is not installed in the test environment")
    def test_agent_cascade_with_synthetic_input(self):
        temp_dir = tempfile.mkdtemp()
        try:
            config = AgentConfig(
                timeout_seconds=80,
                output_dir=temp_dir,
                max_faces=5000,
            )
            agent = ImageTo3DAgent(config=config)

            # Create synthetic input image
            img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            for x in range(32, 96):
                for y in range(32, 96):
                    img.putpixel((x, y), (50, 150, 250, 255))

            res = agent.run(items=[("front", img)], name="test_obj")

            self.assertIn("report", res)
            self.assertIn("glb", res)
            self.assertIn("obj", res)
            self.assertIn("stl", res)
            self.assertIn("preview", res)

            self.assertTrue(os.path.exists(res["glb"]))
            self.assertTrue(os.path.exists(res["report_json"]))

            report = res["report"]
            self.assertEqual(report["name"], "test_obj")
            self.assertGreater(report["faces"], 0)
            self.assertIn(report["backend"], ["triposr", "gemini_3d_api", "procedural_math"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
