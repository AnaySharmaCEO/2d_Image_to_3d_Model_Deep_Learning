"""Configuration and hardware detection utilities for Image-to-3D Mesh Agent."""

from dataclasses import dataclass, field
import os
import shutil
import subprocess
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_gpu_info() -> str:
    """Query NVIDIA GPU hardware details via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class AgentConfig:
    """Central configuration for 3D Mesh Generation Agent."""

    # Watchdog & Execution Limits
    timeout_seconds: int = 80
    primary_backend: str = "triposr"
    enable_hunyuan: bool = True

    # Mesh & Voxel Resolution
    triposr_resolution: int = 192
    octree_resolution: int = 192
    hunyuan_chunks: int = 1500
    max_faces: int = 20000
    colorize: bool = True

    # Directories
    base_dir: str = field(
        default_factory=lambda: (
            "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./outputs")
        )
    )
    output_dir: str = field(
        default_factory=lambda: (
            os.path.join("/kaggle/working", "outputs")
            if os.path.exists("/kaggle/working")
            else os.path.abspath("./outputs")
        )
    )
    cache_dir: str = field(
        default_factory=lambda: os.environ.get(
            "HF_HOME",
            (
                os.path.join("/kaggle/working", "cache", "hf")
                if os.path.exists("/kaggle/working")
                else os.path.abspath("./cache/hf")
            ),
        )
    )

    # API Keys & Cloud Services
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))

    # Hardware Info
    gpu_description: str = field(default_factory=get_gpu_info)

    def __post_init__(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["HF_HOME"] = self.cache_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def to_dict(self):
        return {
            "TIMEOUT_S": self.timeout_seconds,
            "PRIMARY_BACKEND": self.primary_backend,
            "ENABLE_HUNYUAN": self.enable_hunyuan,
            "TRIPOSR_RES": self.triposr_resolution,
            "OCTREE_RES": self.octree_resolution,
            "CHUNKS": self.hunyuan_chunks,
            "MAX_FACES": self.max_faces,
            "COLORIZE": self.colorize,
            "GEMINI_API_KEY": self.gemini_api_key,
            "OUT_DIR": self.output_dir,
            "CACHE_DIR": self.cache_dir,
        }
