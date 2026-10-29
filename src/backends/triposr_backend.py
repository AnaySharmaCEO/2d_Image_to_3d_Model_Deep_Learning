"""Tier 1: TripoSR Ultra-Fast Single-View Feed-Forward 3D Backend (~1.5s)."""

import os
import sys
import types
from typing import Any, Dict
import numpy as np
from PIL import Image

from .base import Base3DBackend

try:
    import torch
except ImportError:
    torch = None


def apply_skimage_marching_cubes_shim():
    """Apply pure Python marching cubes shims to prevent C++ extension compile errors on NumPy 2 / Python 3.13."""
    if torch is None:
        return
    try:
        from skimage.measure import marching_cubes as _sk_mc

        def _torchmcubes_shim(density, threshold):
            d_np = density.detach().cpu().numpy()
            thresh = float(threshold)
            min_v, max_v = float(d_np.min()), float(d_np.max())
            if thresh <= min_v or thresh >= max_v:
                thresh = (min_v + max_v) * 0.5
            try:
                v_np, f_np, _, _ = _sk_mc(d_np, level=thresh)
            except Exception:
                v_np, f_np, _, _ = _sk_mc(d_np, level=float(d_np.mean()))
            verts = torch.from_numpy(v_np.copy()).float().to(density.device)
            faces = torch.from_numpy(f_np.copy()).long().to(density.device)
            return verts, faces

        torchmcubes_module = types.ModuleType("torchmcubes")
        torchmcubes_module.marching_cubes = _torchmcubes_shim
        sys.modules["torchmcubes"] = torchmcubes_module
        sys.modules["mcubes"] = torchmcubes_module
    except Exception:
        pass


class TripoSRBackend(Base3DBackend):
    """Tier 1: TripoSR Large Reconstruction Model (Feed-forward LRM Transformer).
    
    Generates 3D meshes from a single viewpoint in ~1-2 seconds with minimal VRAM usage (~4GB).
    """

    name: str = "triposr"

    def __init__(self, repo_dir: str = ""):
        super().__init__()
        self.repo_dir = repo_dir
        self.model = None

    @classmethod
    def is_available(cls) -> bool:
        if torch is None:
            return False
        try:
            from tsr.system import TSR  # noqa: F401
            return True
        except ImportError:
            # Check if TripoSR repo is in path or subfolder
            return any(
                os.path.isdir(p)
                for p in ["/kaggle/working/TripoSR", "./TripoSR", "../TripoSR"]
            )

    def load(self) -> None:
        apply_skimage_marching_cubes_shim()

        # Add clone directory to path if present
        for candidate in ["/kaggle/working/TripoSR", "./TripoSR", "../TripoSR", self.repo_dir]:
            if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

        from tsr.system import TSR

        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(8192)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def unload(self) -> None:
        self.model = None
        self.free_gpu()

    def generate(self, views: Dict[str, Image.Image], seed: int, config: Dict[str, Any]):
        if self.model is None:
            raise RuntimeError("TripoSR backend is not loaded. Call load() first.")

        front = views.get("front") or next(iter(views.values()))
        arr = np.asarray(front).astype(np.float32) / 255.0
        
        # Blend alpha channel over mid-gray background if transparent
        if arr.shape[-1] == 4:
            rgb = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4]) * 0.5
        else:
            rgb = arr[..., :3]

        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        with torch.no_grad():
            scene_codes = self.model([pil_img], device=device)

        resolution = config.get("TRIPOSR_RES", 192)
        try:
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
        except TypeError:
            meshes = self.model.extract_mesh(scene_codes, resolution=resolution)

        return meshes[0]
