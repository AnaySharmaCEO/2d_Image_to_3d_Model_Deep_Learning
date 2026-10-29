"""Instant Procedural 3D Mesh Generation Demo (Pure CPU, Zero GPU, No Downloads Required).

Run this script to immediately verify the 3D pipeline on any machine.
It generates a synthetic colored shape, converts it into a watertight 3D relief mesh,
and exports GLB, OBJ, STL, and a 4-angle orbit preview sprite sheet.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backends.procedural_backend import ProceduralMathBackend
from src.postprocessing import clean_mesh, mesh_arrays
from src.rendering import make_sheet, render_orbit


def create_synthetic_test_image(size: int = 512) -> Image.Image:
    """Create a high-contrast synthetic star/shield logo with transparent background."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r_outer = size * 0.42
    r_inner = size * 0.20
    points = []

    # 8-point geometric star
    for i in range(16):
        angle = i * np.pi / 8.0
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + r * np.sin(angle)
        y = cy - r * np.cos(angle)
        points.append((x, y))

    draw.polygon(points, fill=(65, 130, 245, 255), outline=(255, 215, 0, 255))
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(255, 90, 95, 255))
    return img


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "procedural_demo"))
    os.makedirs(out_dir, exist_ok=True)

    print("🎨 Creating synthetic RGBA input image...")
    input_img = create_synthetic_test_image()
    input_path = os.path.join(out_dir, "input_shape.png")
    input_img.save(input_path)
    print(f"   Saved input image to: {input_path}")

    print("\n⚡ Executing Tier 3: Zero-Failure Procedural Relief Math Engine...")
    backend = ProceduralMathBackend()
    mesh = backend.generate(views={"front": input_img}, seed=0, config={})

    print("🔧 Cleaning and normalizing 3D mesh...")
    mesh = clean_mesh(mesh, max_faces=10000)
    v, f = mesh_arrays(mesh)
    print(f"   Resulting mesh: {len(v):,} vertices, {len(f):,} faces")

    print("\n📦 Exporting 3D deliverables...")
    glb_path = os.path.join(out_dir, "procedural_star.glb")
    obj_path = os.path.join(out_dir, "procedural_star.obj")
    stl_path = os.path.join(out_dir, "procedural_star.stl")
    preview_path = os.path.join(out_dir, "procedural_star_preview.png")

    mesh.export(glb_path)
    mesh.export(obj_path)
    mesh.export(stl_path)
    print(f"   • GLB (Web/AR)  : {glb_path}")
    print(f"   • OBJ (General) : {obj_path}")
    print(f"   • STL (3D Print): {stl_path}")

    print("\n📸 Generating pure-CPU 4-angle orbit preview sheet...")
    vcol = np.asarray(mesh.visual.vertex_colors) if hasattr(mesh, "visual") else None
    tiles = render_orbit(v, f, colors=vcol)
    sheet = make_sheet(input_img, tiles)
    sheet.save(preview_path)
    print(f"   • Preview Sheet : {preview_path}")

    print("\n✅ Procedural 3D demo completed successfully!")


if __name__ == "__main__":
    main()
