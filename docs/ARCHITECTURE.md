# 🏗️ System Architecture & Engineering Design

This document details the internal engineering, mathematical formulation, and algorithmic pipeline of the **Image-to-3D Mesh Agent (V5)**.

---

## 1. High-Level System Workflow

The architecture addresses the classic bottleneck of 3D deep learning pipelines: **generation latency, heavy VRAM consumption, and unpredictable runtime failures** (e.g., CUDA OOM, diffusion process timeouts, package compilation issues).

```mermaid
flowchart TD
    A[Input Image: Single or 2-4 Views] --> B[Preprocessing & Validation]
    B --> C{EXIF & Aspect Normalization}
    C --> D[4MB u2netp / Instant Flood BG Removal]
    D --> E[Laplacian Variance & Coverage Quality Analysis]
    
    E --> F[Watchdog Cascade Orchestrator (Cap: 80s)]
    
    F -->|Timer < 80s & Single View| G1[Tier 1: TripoSR Transformer LRM]
    F -->|Timer < 80s & Multi View| G2[Tier 1: Hunyuan3D-2 Turbo DiT]
    
    G1 -.->|Timeout / CUDA OOM| H[Tier 2: Gemini 3D Geometry API]
    G2 -.->|Timeout / CUDA OOM| H
    
    H -.->|No Key / API Error| I[Tier 3: Procedural Relief Math Engine]
    
    G1 --> J[Raw Trimesh Geometry]
    G2 --> J
    H --> J
    I --> J
    
    J --> K[Post-Processing: Quadric Decimation & Mesh Repair]
    K --> L[Z-Buffer Visibility & Normal Alignment]
    L --> M[KD-Tree Nearest-Neighbor Color Projection]
    M --> N[Point-Splat CPU Orbit Renderer]
    N --> O[Multi-Format Export: GLB, OBJ, STL, PNG, JSON]
```

---

## 2. The 3-Tier Multi-Engine Cascade

### Tier 1: Primary Neural Reconstruction (Local GPU)
1. **Single-View Engine (TripoSR)**:
   - **Model Class**: Large Reconstruction Model (LRM) feed-forward transformer.
   - **Latency**: ~1.2 to 2.0 seconds.
   - **VRAM**: ~4.2 GB.
   - **Principle**: TripoSR encodes the single input image via a Vision Transformer (ViT) into a triplane neural representation, followed by a lightweight NeRF-style implicit decoder and Marching Cubes extraction.
   - **NumPy 2.x & Python 3.13 Shim**: Replaces native C++ `torchmcubes` compilation with a pure-Python `skimage.measure.marching_cubes` bridge, avoiding build-tool dependencies.

2. **Multi-View Engine (Hunyuan3D-2 Turbo)**:
   - **Model Class**: Flow-Matching Diffusion Transformer (DiT).
   - **Views**: Front, Left, Back, Right (2 to 4 angles).
   - **Latency**: ~18 to 28 seconds (5 turbo inference steps).
   - **VRAM**: ~9.5 to 11.5 GB.
   - **Extraction**: Octree resolution 192–320 for high-detail geometric relief.

---

### Tier 2: 80-Second Watchdog & Gemini 3D Geometry API (Cloud Fallback)
When a GPU environment is constrained, experiences CUDA Out-Of-Memory, or exceeds the strict 80-second threshold, the **Watchdog Timer** triggers a circuit breaker that transitions execution to **Google Gemini 2.5 Flash**.

- **Prompt Structure**: Instructs Gemini as an expert computational geometry mathematician.
- **Payload**: Base64 encoded PNG of the input object.
- **Output**: Executable Python code defining a callable `build_mesh() -> trimesh.Trimesh` using CSG (Constructive Solid Geometry), primitive booleans (icosphere, box, cylinder, capsule), and analytic vertex/face calculations.
- **Resource Footprint**: **0 GB VRAM, 0 GB GPU memory**, ~5–8s latency over HTTPS.

---

### Tier 3: Zero-Failure Procedural Silhouette & Depth-Relief Engine
If network connectivity is unavailable, no API key is supplied, or local deep learning fails, Tier 3 provides a **100% mathematical guarantee** that a watertight, textured 3D mesh is generated.

#### Mathematical Formulation:
1. **Silhouette Mask Extraction**:
   $$\mathcal{M}(x, y) = \begin{cases} 1 & \text{if } \alpha(x, y) > 60 \\ 0 & \text{otherwise} \end{cases}$$

2. **Euclidean Distance Transform (EDT)**:
   $$\mathcal{D}(x, y) = \min_{(x', y') \notin \mathcal{M}} \sqrt{(x - x')^2 + (y - y')^2}$$

3. **Non-Linear Depth Profile**:
   $$\mathcal{Z}(x, y) = \left( \frac{\mathcal{D}(x, y)}{\max \mathcal{D}} \right)^{0.65} \cdot 0.35$$

4. **Watertight Double-Shell Assembly**:
   - For every sampled grid point $(x_k, y_k) \in \mathcal{M}$, two vertices are instantiated:
     $$V_{\text{front}} = \left( x_k, y_k, \mathcal{Z}(x_k, y_k) \right), \quad V_{\text{back}} = \left( x_k, y_k, -0.5 \cdot \mathcal{Z}(x_k, y_k) \right)$$
   - Quads are decomposed into clockwise front-facing and counter-clockwise back-facing triangles to create a closed, manifold 3D volume.
   - Vertex colors are sampled directly from the input image pixel coordinates $(y_k, x_k)$.
   - **Execution Time**: **< 0.08 seconds on standard CPU**.

---

## 3. Preprocessing & Quality Assurance

1. **Lightweight Background Removal**:
   - Rather than downloading massive 1.02 GB general models, V5 initializes `rembg` with the **4MB `u2netp` model**.
   - If `rembg` is absent or encounters network issues, it invokes an instant morphological flood fill (`simple_bg_removal`) that samples background color across 4 corners, filters Euclidean distance deviations $> 35$, and extracts the primary connected component in ~0.02s.

2. **Image Quality Analysis**:
   - **Sharpness Metric**: Computes the variance of the discrete Laplacian operator $\nabla^2 I$:
     $$\text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x, y} \left( \nabla^2 I(x, y) - \mu \right)^2$$
     Values $< 30$ trigger a `"image looks blurry"` warning.
   - **Foreground Coverage Ratio**:
     $$\mathcal{C} = \frac{1}{W \cdot H} \sum_{x, y} \mathcal{M}(x, y)$$
     Values $< 0.10$ or $> 0.92$ trigger anomaly warnings.

---

## 4. Photometric Texture Projection & Visibility Culling

To ensure photorealistic fidelity on the reconstructed mesh, a discrete depth buffer (Z-buffer) texture projection algorithm is executed:

1. **Forward Orthographic Projection**:
   Vertices $(u, v, z)$ are rasterized onto a $256 \times 256$ virtual grid.
2. **Depth Testing**:
   $$\text{Visible}(v_i) \iff z_i \ge \mathcal{Z}_{\text{buf}}(\text{cell}_i) - 0.03 \cdot \text{size} \quad \land \quad \vec{N}_i \cdot \hat{z} > 0.15$$
3. **Occlusion Filling via KD-Tree Diffusion**:
   For self-occluded back surfaces ($\text{Visible}(v_j) = \text{False}$), a $k$-d tree structure searches the nearest visible vertex neighbor in Euclidean $\mathbb{R}^3$ space:
   $$\text{Color}(v_j) = \text{Color}\left(\arg\min_{v_k \in \mathcal{V}_{\text{vis}}} \|v_j - v_k\|\right)$$

---

## 5. Pure CPU Point-Splatting Renderer

For instantaneous previews without requiring GPU OpenGL or X11 display contexts, a point-splatting surface renderer samples 150,000 points over triangular faces weighted by surface area:

$$P(\Delta_i) = \frac{\text{Area}(\Delta_i)}{\sum_j \text{Area}(\Delta_j)}$$

Points are projected onto 4 orthogonal camera angles ($\theta \in \{0, \frac{\pi}{2}, \pi, \frac{3\pi}{2}\}$), shaded with Lambertian directional illumination:

$$I_{\text{shade}} = 0.40 + 0.60 \cdot |\vec{N} \cdot \vec{d}_{\text{view}}|$$

Sub-pixel discontinuities are closed via morphological binary closing and distance transform interpolation, producing a clean 4-angle sprite sheet in $<0.4$ seconds.
