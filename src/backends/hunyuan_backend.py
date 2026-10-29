"""Tier 1: Hunyuan3D-2 Turbo Multi-View 3D Generation Pipeline."""

import os
import sys
from typing import Any, Dict
from PIL import Image

from .base import Base3DBackend

try:
    import torch
except ImportError:
    torch = None


class HunyuanBackend(Base3DBackend):
    """Tier 1: Tencent Hunyuan3D-2 Turbo multi-view diffusion flow-matching pipeline.
    
    Synthesizes 3D meshes from 2-4 orthographic viewpoints (front, left, back, right)
    using a fast 5-step DiT model with octree marching cubes extraction.
    """

    def __init__(self, kind: str = "mv", repo_dir: str = ""):
        super().__init__()
        self.kind = kind
        self.name = f"hunyuan_{kind}"
        self.repo_dir = repo_dir
        self.pipe = None

    @classmethod
    def is_available(cls) -> bool:
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return True
        except ImportError:
            return any(
                os.path.isdir(p)
                for p in ["/kaggle/working/Hunyuan3D-2", "./Hunyuan3D-2", "../Hunyuan3D-2"]
            )

    def load(self) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("Hunyuan3D-2 requires CUDA GPU hardware acceleration.")

        # Add clone directory to path if present
        for candidate in ["/kaggle/working/Hunyuan3D-2", "./Hunyuan3D-2", "../Hunyuan3D-2", self.repo_dir]:
            if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Pipe

        repo = "tencent/Hunyuan3D-2mv" if self.kind == "mv" else "tencent/Hunyuan3D-2mini"
        self.pipe = Pipe.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="cuda",
            low_cpu_mem_usage=True,
            subfolder="hunyuan3d-dit-v2-0-turbo",
        )
        try:
            self.pipe.enable_flashvdm(mc_algo="mc")
        except Exception:
            pass

    def unload(self) -> None:
        self.pipe = None
        self.free_gpu()

    def generate(self, views: Dict[str, Image.Image], seed: int, config: Dict[str, Any]):
        if self.pipe is None:
            raise RuntimeError("Hunyuan3D backend is not loaded. Call load() first.")

        valid_keys = ("front", "left", "back", "right")
        imgs = {k: v for k, v in views.items() if k in valid_keys}

        kwargs = {
            "num_inference_steps": 5,
            "octree_resolution": config.get("OCTREE_RES", 192),
            "num_chunks": config.get("CHUNKS", 1500),
            "output_type": "trimesh",
        }

        generator = torch.Generator("cpu").manual_seed(seed)
        output = self.pipe(image=imgs, generator=generator, **kwargs)
        return output[0]
