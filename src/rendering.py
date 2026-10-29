"""Pure CPU orbit preview renderer, point-splatting, sprite sheets, and 3D viewers."""

import base64
import math
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n: int = 150000,
    rng: Optional[np.random.Generator] = None,
    vcol: Optional[np.ndarray] = None,
):
    """Uniformly sample points, surface normals, and interpolated colors over mesh triangles."""
    rng = rng or np.random.default_rng(0)
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = area.sum()

    if total_area <= 1e-12:
        return np.zeros((n, 3)), np.zeros((n, 3)), None

    prob = area / total_area
    idx = rng.choice(len(faces), size=n, p=prob)
    u = rng.random(n)
    v = rng.random(n)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    pts = tri[idx, 0] + u[:, None] * e1[idx] + v[:, None] * e2[idx]
    norm = cross[idx] / (np.linalg.norm(cross[idx], axis=1, keepdims=True) + 1e-12)

    col = None
    if vcol is not None and len(vcol) == len(vertices):
        c = vcol[faces][idx].astype(np.float32)
        col = (1.0 - u - v)[:, None] * c[:, 0] + u[:, None] * c[:, 1] + v[:, None] * c[:, 2]

    return pts, norm, col


def render_orbit(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    n_views: int = 4,
    size: int = 300,
    n_pts: int = 150000,
) -> List[Image.Image]:
    """Pure CPU point-splatting orbit preview generator.
    
    Renders orthogonal multi-angle viewpoints without requiring OpenGL, PyTorch, or GPU contexts.
    """
    pts, norm, col = sample_surface(
        vertices,
        faces,
        n=n_pts,
        rng=np.random.default_rng(1),
        vcol=colors[:, :3] if colors is not None else None,
    )

    if col is None:
        col = np.full((len(pts), 3), (185, 195, 215), dtype=np.float32)

    extent = float(np.abs(pts).max()) * 1.05 or 1.0
    tiles: List[Image.Image] = []

    for t in range(n_views):
        theta = 2.0 * math.pi * t / n_views
        tw = np.array([math.sin(theta), 0.0, math.cos(theta)])
        rt = np.array([math.cos(theta), 0.0, -math.sin(theta)])

        x = pts @ rt
        y = pts[:, 1]
        depth = pts @ tw

        ix = np.clip(((x / (2.0 * extent) + 0.5) * (size - 1)).astype(int), 0, size - 1)
        iy = np.clip(((0.5 - y / (2.0 * extent)) * (size - 1)).astype(int), 0, size - 1)

        shade = (0.40 + 0.60 * np.abs(norm @ tw))[:, None]
        rgb = np.clip(col * shade, 0, 255)

        order = np.argsort(depth)
        canvas = np.full((size, size, 3), 245.0, dtype=np.float32)
        canvas[iy[order], ix[order]] = rgb[order]

        mask = np.zeros((size, size), dtype=bool)
        mask[iy, ix] = True

        if ndi is not None:
            holes = ndi.binary_closing(mask, iterations=2) & ~mask
            if holes.any():
                _, (ri, ci) = ndi.distance_transform_edt(~mask, return_indices=True)
                canvas[holes] = canvas[ri[holes], ci[holes]]

        tiles.append(Image.fromarray(canvas.astype(np.uint8)))

    return tiles


def make_sheet(input_rgba: Image.Image, tiles: List[Image.Image], size: int = 300) -> Image.Image:
    """Compose input source and multi-angle 3D renders into a single sprite sheet."""
    bg = Image.new("RGB", (size, size), (245, 245, 245))
    inp = input_rgba.resize((size, size), Image.LANCZOS)
    bg.paste(inp, (0, 0), inp)

    total_width = size * (len(tiles) + 1)
    sheet = Image.new("RGB", (total_width, size), (245, 245, 245))
    sheet.paste(bg, (0, 0))

    for i, t in enumerate(tiles):
        sheet.paste(t, (size * (i + 1), 0))

    return sheet


def generate_model_viewer_html(glb_path: str, height: int = 480) -> str:
    """Generate interactive <model-viewer> HTML block for 3D asset inspection."""
    with open(glb_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return (
        f'<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>'
        f'<model-viewer src="data:model/gltf-binary;base64,{b64}" camera-controls auto-rotate '
        f'shadow-intensity="1" style="width:100%;height:{height}px;background:#f2f2f2;border-radius:8px;"></model-viewer>'
    )
