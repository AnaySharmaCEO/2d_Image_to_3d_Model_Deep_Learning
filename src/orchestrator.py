"""Orchestrator managing the 3-Tier cascade, 80-second watchdog, and asset exports."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .config import AgentConfig
from .preprocessing import analyze_image, assign_views, preprocess_image
from .backends import get_backend
from .backends.triposr_backend import TripoSRBackend
from .backends.hunyuan_backend import HunyuanBackend
from .postprocessing import clean_mesh, mesh_arrays, project_colors
from .rendering import make_sheet, render_orbit

try:
    import trimesh
except ImportError:
    trimesh = None


class ImageTo3DAgent:
    """Intelligent 3-Tier Multi-Backend 3D Reconstruction Agent.
    
    Coordinates fast feed-forward models (Tier 1), strict watchdog timeouts to LLM 3D
    APIs (Tier 2), and zero-failure procedural relief math fallback (Tier 3).
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def run(
        self,
        items: Union[Dict[str, Image.Image], List[Tuple[str, Image.Image]]],
        name: str = "object",
        api_key: str = "",
    ) -> Dict[str, Any]:
        """Execute full 3D reconstruction pipeline on provided images.
        
        Args:
            items: Single image or dictionary/list of view images.
            name: Object identifier for saved asset names.
            api_key: Optional Gemini API key override for Tier 2 fallback.
            
        Returns:
            Dictionary containing export filepaths, mesh object, and metadata report.
        """
        t_start = time.time()
        views_in = assign_views(items)

        # 1. Preprocessing and Quality Analysis
        prep: Dict[str, Image.Image] = {}
        reports: Dict[str, Dict] = {}
        for k, im in views_in.items():
            prep[k] = preprocess_image(im)
            reports[k] = analyze_image(im, prep[k])

        timeout_limit = self.config.timeout_seconds
        mesh_out = None
        used_backend = "none"

        # 2. Tier 1: Fast Local AI Engine (TripoSR for 1 view, Hunyuan for multi-view)
        elapsed = time.time() - t_start
        if elapsed < timeout_limit:
            if len(prep) > 1 and self.config.enable_hunyuan and HunyuanBackend.is_available():
                try:
                    be = HunyuanBackend(kind="mv")
                    be.load()
                    mesh_out = be.generate(prep, seed=0, config=self.config.to_dict())
                    used_backend = "hunyuan_mv"
                    be.unload()
                except Exception as e:
                    print(f"[Agent Warning] Hunyuan multi-view backend error: {e}")

            elif TripoSRBackend.is_available():
                try:
                    be = TripoSRBackend()
                    be.load()
                    mesh_out = be.generate(prep, seed=0, config=self.config.to_dict())
                    used_backend = "triposr"
                    be.unload()
                except Exception as e:
                    print(f"[Agent Warning] TripoSR backend error: {e}")

        # 3. Tier 2: Gemini 3D Geometry API (Triggered on watchdog timeout or local failure)
        active_key = api_key or self.config.gemini_api_key
        if mesh_out is None and active_key:
            print(
                f"[Agent Watchdog] Triggered (Elapsed: {time.time()-t_start:.1f}s). "
                "Routing to Tier 2: Gemini 3D Geometry API..."
            )
            try:
                be = get_backend("gemini_3d_api", api_key=active_key)
                mesh_out = be.generate(prep, seed=0, config=self.config.to_dict())
                used_backend = "gemini_3d_api"
            except Exception as e:
                print(f"[Agent Warning] Tier 2 (Gemini API) error: {e}")

        # 4. Tier 3: Zero-Failure Instant Procedural Relief Math Engine
        if mesh_out is None:
            print("[Agent Fallback] Engaging Tier 3: Procedural Relief Math Engine (Zero-Failure Guarantee)...")
            be = get_backend("procedural_math")
            mesh_out = be.generate(prep, seed=0, config=self.config.to_dict())
            used_backend = "procedural_math"

        # 5. Post-Processing & Geometry Optimization
        mesh = clean_mesh(mesh_out, max_faces=self.config.max_faces)
        vn = np.asarray(mesh.vertex_normals)
        v, f = mesh_arrays(mesh)

        if float((vn * v).sum()) < 0:
            mesh.invert()
            vn = -vn

        base_colors = None
        try:
            if hasattr(mesh, "visual") and mesh.visual.kind == "vertex":
                base_colors = np.asarray(mesh.visual.vertex_colors)
        except Exception:
            pass

        colors = None
        if self.config.colorize and "front" in prep:
            colors, _ = project_colors(v, vn, prep["front"], base_colors)
            if trimesh is not None:
                mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)

        # 6. Render Orbit Sprite Sheet
        tiles = render_orbit(v, f, colors)
        front_img = prep.get("front") or next(iter(prep.values()))
        sheet = make_sheet(front_img, tiles)

        # 7. Asset Export
        out_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        paths = {ext: os.path.join(out_dir, f"{name}.{ext}") for ext in ("glb", "obj", "stl")}
        paths["preview"] = os.path.join(out_dir, f"{name}_preview.png")
        paths["report_json"] = os.path.join(out_dir, f"{name}_report.json")

        mesh.export(paths["glb"])
        if trimesh is not None:
            plain = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
            )
            plain.export(paths["obj"])
            plain.export(paths["stl"])
        sheet.save(paths["preview"])

        total_seconds = round(time.time() - t_start, 2)
        report = {
            "name": name,
            "backend": used_backend,
            "views": list(prep.keys()),
            "vertices": int(len(v)),
            "faces": int(len(f)),
            "seconds": total_seconds,
            "image_analysis": reports,
            "watchdog_timeout": timeout_limit,
        }

        with open(paths["report_json"], "w", encoding="utf-8") as f_json:
            json.dump(report, f_json, indent=2)

        return {
            "report": report,
            "sheet": sheet,
            "mesh": mesh,
            **paths,
        }


def run_agent(
    items: Union[Dict[str, Image.Image], List[Tuple[str, Image.Image]]],
    name: str = "object",
    api_key: str = "",
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """Convenience helper to instantiate and execute the 3D agent."""
    agent = ImageTo3DAgent(config=config)
    return agent.run(items=items, name=name, api_key=api_key)
