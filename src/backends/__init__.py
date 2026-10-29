"""Backends package providing Tier 1, 2, and 3 3D reconstruction engines."""

from typing import Dict, Type
from .base import Base3DBackend
from .triposr_backend import TripoSRBackend
from .hunyuan_backend import HunyuanBackend
from .gemini_backend import Gemini3DBackend
from .procedural_backend import ProceduralMathBackend

BACKEND_REGISTRY: Dict[str, Type[Base3DBackend]] = {
    "triposr": TripoSRBackend,
    "hunyuan_mv": lambda **kwargs: HunyuanBackend(kind="mv", **kwargs),
    "hunyuan_mini": lambda **kwargs: HunyuanBackend(kind="mini", **kwargs),
    "gemini_3d_api": Gemini3DBackend,
    "procedural_math": ProceduralMathBackend,
}


def get_backend(name: str, **kwargs) -> Base3DBackend:
    """Retrieve an instantiated 3D generation backend by registered name."""
    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    factory = BACKEND_REGISTRY[name]
    return factory(**kwargs)


__all__ = [
    "Base3DBackend",
    "TripoSRBackend",
    "HunyuanBackend",
    "Gemini3DBackend",
    "ProceduralMathBackend",
    "BACKEND_REGISTRY",
    "get_backend",
]
