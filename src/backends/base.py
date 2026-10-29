"""Base class interface for 3D generation backends."""

from abc import ABC, abstractmethod
import gc
from typing import Any, Dict
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


class Base3DBackend(ABC):
    """Abstract interface for all 3D mesh reconstruction engines."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """Initialize and load model weights into compute device memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Evict model from VRAM and free GPU cache."""
        pass

    @abstractmethod
    def generate(self, views: Dict[str, Image.Image], seed: int, config: Dict[str, Any]) -> Any:
        """Synthesize a 3D mesh given processed multi-view or single-view images."""
        pass

    def free_gpu(self) -> None:
        """Utility to trigger garbage collection and flush CUDA cache."""
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def is_available(cls) -> bool:
        """Check if backend dependencies and hardware are available."""
        return True
