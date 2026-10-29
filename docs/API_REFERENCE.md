# 📚 Python API Reference

The `src` package provides a programmatic interface for executing 3D reconstruction, configuring multi-tier fallbacks, and manipulating 3D mesh assets.

---

## 1. Quick Example

```python
from src import ImageTo3DAgent, AgentConfig, load_image

# 1. Initialize configuration
config = AgentConfig(
    timeout_seconds=80,
    max_faces=20000,
    gemini_api_key="AIzaSy...",
)

# 2. Instantiate agent
agent = ImageTo3DAgent(config=config)

# 3. Load input image
img = load_image("examples/sample.png")

# 4. Generate 3D mesh
result = agent.run(
    items=[("front", img)],
    name="my_3d_object"
)

print("Generated GLB path:", result["glb"])
print("Backend utilized:", result["report"]["backend"])
print("Inference time:", result["report"]["seconds"])
```

---

## 2. Core Modules & Classes

### `src.config.AgentConfig`
Dataclass holding runtime hyper-parameters, timeout watchdogs, and directory structures.

```python
@dataclass
class AgentConfig:
    timeout_seconds: int = 80          # Strict watchdog threshold
    primary_backend: str = "triposr"   # Default single-view engine
    enable_hunyuan: bool = True        # Multi-view engine enabled
    triposr_resolution: int = 192      # Marching cubes grid
    octree_resolution: int = 192       # Hunyuan octree grid
    hunyuan_chunks: int = 1500         # VRAM streaming chunk size
    max_faces: int = 20000             # Quadric decimation budget
    colorize: bool = True              # Enable vertex color projection
    output_dir: str = "./outputs"      # Directory for saved models
    cache_dir: str = "./cache/hf"      # Model weights cache
    gemini_api_key: str = ""           # API key for Tier 2 fallback
```

---

### `src.orchestrator.ImageTo3DAgent`
The central orchestrator managing view mapping, preprocessing, multi-tier watchdog routing, post-processing, and multi-format asset export.

#### `agent.run(items, name="object", api_key="") -> Dict[str, Any]`
- **Arguments**:
  - `items`: Single PIL image, list of `(view_label, PIL.Image)`, or dictionary mapping view names to images.
  - `name` *(str)*: Identifier for exported files.
  - `api_key` *(str, optional)*: Overrides the configured Gemini API key.
- **Returns**: Dictionary with keys:
  - `glb` *(str)*: Filepath to Binary glTF model with vertex colors.
  - `obj` *(str)*: Filepath to Wavefront OBJ format mesh.
  - `stl` *(str)*: Filepath to STL 3D printing geometry.
  - `preview` *(str)*: Filepath to 4-angle PNG sprite sheet.
  - `report_json` *(str)*: Filepath to structured JSON metadata.
  - `report` *(dict)*: Execution summary, face/vertex counts, backend used.
  - `mesh` *(trimesh.Trimesh)*: In-memory Trimesh object.
  - `sheet` *(PIL.Image)*: Rendered sprite sheet image.

---

### `src.preprocessing`

#### `load_image(src: Union[str, Image.Image, bytes]) -> Image.Image`
Loads and applies EXIF orientation transposes to ensure correct orientation.

#### `preprocess_image(img: Image.Image, size: int = 512, margin: float = 0.08) -> Image.Image`
Removes background (via lightweight `u2netp` or instant flood fill), centers the foreground, and scales it into a square RGBA canvas with specified margin.

#### `analyze_image(orig: Image.Image, rgba: Image.Image) -> Dict`
Calculates Laplacian sharpness variance, foreground coverage ratio, connected component integrity, and emits warnings for edge cases.

#### `assign_views(items) -> Dict[str, Image.Image]`
Categorizes arbitrary uploads or lists into canonical orthographic keys: `"front"`, `"left"`, `"back"`, `"right"`.

---

### `src.postprocessing`

#### `clean_mesh(mesh: trimesh.Trimesh, max_faces: int = 20000) -> trimesh.Trimesh`
Performs non-manifold face removal, hole filling, normal vector correction, quadric decimation simplification, and unit bounding box normalization.

#### `project_colors(vertices, normals, rgba, base_colors=None) -> Tuple[np.ndarray, float]`
Applies discrete Z-buffer visibility testing and KD-Tree nearest-neighbor color diffusion to texture 3D vertices from the input viewpoint.

---

### `src.rendering`

#### `render_orbit(vertices, faces, colors=None, n_views=4, size=300) -> List[Image.Image]`
Pure CPU point-splatting renderer producing orthogonal preview images without GPU requirements.

#### `generate_model_viewer_html(glb_path: str, height: int = 480) -> str`
Generates self-contained HTML embedding `<model-viewer>` for Jupyter notebook or web display.
