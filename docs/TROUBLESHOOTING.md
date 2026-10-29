# 🔧 Troubleshooting & Compatibility Guide

This guide covers known environment issues, platform compatibility quirks (e.g., NumPy 2.x, Python 3.13), and solutions for cloud environments (Kaggle / Colab).

---

## 1. NumPy 2.x & Python 3.13 Compatibility

### Issue: `AttributeError: module 'numpy' has no attribute 'core'`
**Cause**: NumPy 2.0+ deprecated and reorganized `numpy.core` into `numpy._core`. Legacy deep learning packages (such as older PyTorch, TripoSR, or fast-simplification builds) expect `numpy.core` to exist.

**Solution**:
Our codebase includes dynamic shims in `src/backends/triposr_backend.py` and the notebook:
```python
import sys, numpy as np
if hasattr(np, "_core"):
    sys.modules["numpy.core"] = np._core
```

---

### Issue: `torchmcubes` C++ Build Fails During `pip install`
**Cause**: The native `torchmcubes` package requires a C++ compiler (MSVC on Windows, GCC on Linux) and often fails against Python 3.13 or newer CUDA toolchains.

**Solution**:
We bypass `torchmcubes` entirely with a high-performance **pure-Python `skimage.measure.marching_cubes` bridge**:
```python
from skimage.measure import marching_cubes as _sk_mc
import types

def _skimage_shim(density, threshold):
    d_np = density.detach().cpu().numpy()
    v_np, f_np, _, _ = _sk_mc(d_np, level=float(threshold))
    return torch.from_numpy(v_np).float().to(density.device), torch.from_numpy(f_np).long().to(density.device)

torchmcubes = types.ModuleType("torchmcubes")
torchmcubes.marching_cubes = _skimage_shim
sys.modules["torchmcubes"] = torchmcubes
```
This shim delivers identical geometric extraction fidelity with zero C++ compilation dependencies.

---

## 2. Cloud GPU Environments (Kaggle & Colab)

### Recommended Kaggle Settings:
1. **Accelerator**: Select **`GPU T4 x2`** or **`GPU P100`**.
2. **Internet Access**: Set **`Internet: ON`** (essential for downloading model weights and calling the Gemini API).
3. **Session Persistence**: Select `Variables and Files` to avoid redownloading checkpoints across session reconnects.

### CUDA Out-Of-Memory (OOM) Errors:
If your GPU runs out of VRAM when using Hunyuan3D-2:
1. Enable PyTorch segment expansion:
   ```python
   import os
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
2. Reduce the chunk size:
   Set `CONFIG["CHUNKS"] = 1000` or `1200` (default is 1500).
3. Switch to single-view TripoSR mode, which consumes only ~4.1 GB of VRAM.

---

## 3. Background Removal Hanging

### Issue: Notebook hangs on cell 3 / `rembg.remove`
**Cause**: Standard `rembg` defaults to `u2net`, attempting to download a **1.02 GB model** from GitHub releases. Cloud networks frequently throttle or drop this download, causing silent infinite hangs.

**Solution**:
The V5 pipeline specifies the ultra-lightweight **`u2netp` model (4 MB)**:
```python
import rembg
session = rembg.new_session("u2netp")
rgba = rembg.remove(image, session=session)
```
If network downloads fail entirely, the system automatically falls back to `simple_bg_removal()`, an instant (~0.02s) color-flood and morphological segmenter that requires zero network downloads.

---

## 4. Gradio Interface Issues

### Issue: "Gradio Request Timed Out (60s)"
**Cause**: Gradio defaults to a 60-second request timeout if workers are blocked by heavy multi-step diffusion models.

**Solution**:
1. V5's **80-second watchdog** ensures that if local generation slows down, execution is immediately diverted to the sub-10s Gemini API or sub-0.1s procedural engine.
2. Launch Gradio with request queueing enabled:
   ```python
   demo.queue().launch(share=True)
   ```
