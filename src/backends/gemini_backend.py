"""Tier 2: Gemini 3D Computational Geometry API Backend (Cloud Fail-Safe)."""

import base64
import io
import json
import math
import os
import re
from typing import Any, Dict
import urllib.request
import numpy as np
from PIL import Image

from .base import Base3DBackend

try:
    import trimesh
except ImportError:
    trimesh = None


class Gemini3DBackend(Base3DBackend):
    """Tier 2: Gemini 2.5 Flash Multimodal Computational Geometry Agent.
    
    Triggered when local GPU inference exceeds the watchdog limit (default 80s)
    or encounters CUDA OOM. Synthesizes a watertight 3D trimesh representation
    using analytical geometry and CSG operations with 0 GB local VRAM.
    """

    name: str = "gemini_3d_api"

    def __init__(self, api_key: str = ""):
        super().__init__()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, views: Dict[str, Image.Image], seed: int, config: Dict[str, Any]):
        key = self.api_key or config.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Provide an API key via environment variable "
                "or configuration to enable Tier 2 Cloud Geometry synthesis."
            )

        if trimesh is None:
            raise ImportError("trimesh is required for Gemini 3D geometry execution.")

        front = views.get("front") or next(iter(views.values()))
        front_rgb = front.convert("RGB")

        buffer = io.BytesIO()
        front_rgb.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={key}"
        )

        prompt = (
            "You are an expert 3D computational geometry mathematician.\n"
            "Analyze this image and generate executable Python code using `trimesh` to construct "
            "a 3D mesh representation of the object.\n"
            "Rules:\n"
            "1. Define a function: def build_mesh() -> trimesh.Trimesh:\n"
            "2. Construct the shape using trimesh primitives (box, cylinder, icosphere, cone, capsule), "
            "booleans, or vertex/face arrays matching the proportions and structure of the image.\n"
            "3. Set vertex_colors or face_colors on the mesh matching dominant colors seen in the image.\n"
            "4. Return ONLY executable Python code inside a ```python ``` block with NO other commentary."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]

        match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
        code = match.group(1) if match else text

        scope = {"trimesh": trimesh, "np": np, "math": math}
        exec(code, scope, scope)

        if "build_mesh" in scope and callable(scope["build_mesh"]):
            return scope["build_mesh"]()

        raise RuntimeError("Gemini generated script did not contain a callable build_mesh() function.")
