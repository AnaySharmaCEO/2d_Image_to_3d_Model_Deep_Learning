"""Post-processing, quadric decimation, normal repairs, and photometric color projection."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
except ImportError:
    ndi = None
    cKDTree = None

try:
    import trimesh
except ImportError:
    trimesh = None


def mesh_arrays(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertex and face arrays as float64 and int64 numpy arrays."""
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def clean_mesh(mesh, max_faces: int = 20000):
    """Clean degenerate geometry, fill non-manifold holes, decimate, and normalize unit scale."""
    if trimesh is None:
        return mesh

    m = mesh.copy()
    try:
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    except Exception:
        pass

    if len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            try:
                m = m.simplify_quadric_decimation(max_faces)
            except Exception:
                pass

    vertices = np.asarray(m.vertices, dtype=np.float64)
    if len(vertices) > 0:
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max()) or 1.0
        m.vertices = (vertices - center) / extent

    return m


def project_colors(
    vertices: np.ndarray,
    normals: np.ndarray,
    rgba: Image.Image,
    base_colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Project high-resolution photometric texture from the front image onto 3D mesh vertices.
    
    Uses Z-buffer depth testing, normal culling (N · Z > 0.15), and KD-Tree
    nearest-neighbor color diffusion for self-occluded back surfaces.
    """
    arr = np.asarray(rgba)
    h, w = arr.shape[:2]

    if ndi is not None:
        alpha = ndi.binary_erosion(arr[..., 3] > 127, iterations=2)
    else:
        alpha = arr[..., 3] > 127

    ys, xs = np.where(arr[..., 3] > 127)
    if len(xs) == 0:
        return np.full((len(vertices), 4), 180, dtype=np.uint8), 0.0

    side_img = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    cx, cy = (xs.min() + xs.max() + 1) / 2.0, (ys.min() + ys.max() + 1) / 2.0

    u, v, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    side_m = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-9)
    cu, cv = (u.max() + u.min()) / 2.0, (v.max() + v.min()) / 2.0

    px = np.clip(np.floor(cx + (u - cu) / side_m * side_img).astype(int), 0, w - 1)
    py = np.clip(np.floor(cy - (v - cv) / side_m * side_img).astype(int), 0, h - 1)

    # 256x256 discrete Z-buffer raster
    r = 256
    gx = np.clip(((u - cu) / side_m + 0.5) * (r - 1), 0, r - 1).astype(int)
    gy = np.clip((0.5 - (v - cv) / side_m) * (r - 1), 0, r - 1).astype(int)
    cell = gy * r + gx

    zbuf = np.full(r * r, -np.inf)
    np.maximum.at(zbuf, cell, z)

    visible = (z >= zbuf[cell] - 0.03 * side_m) & (normals[:, 2] > 0.15) & alpha[py, px]
    col = np.zeros((len(vertices), 4), dtype=np.uint8)
    col[:, 3] = 255
    col[visible, :3] = arr[py[visible], px[visible], :3]

    if (~visible).any():
        if base_colors is not None and len(base_colors) == len(vertices):
            col[~visible, :3] = base_colors[~visible, :3]
        elif visible.any() and cKDTree is not None:
            tree = cKDTree(vertices[visible])
            _, nn = tree.query(vertices[~visible])
            col[~visible, :3] = col[visible][nn, :3]
        else:
            col[~visible, :3] = 180

    return col, float(visible.mean())
