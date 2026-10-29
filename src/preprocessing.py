"""Preprocessing, background removal, and image quality analysis utilities."""

from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

VIEW_KEYS = ("front", "left", "back", "right")
_REMBG_SESSION = None


def load_image(src: Union[str, Image.Image, bytes]) -> Image.Image:
    """Load an image from filepath, bytes, or Image object and apply EXIF orientation transpose."""
    if isinstance(src, Image.Image):
        im = src
    elif isinstance(src, (str, bytes)):
        im = Image.open(src)
    else:
        raise TypeError(f"Unsupported image input type: {type(src)}")
    return ImageOps.exif_transpose(im)


def has_real_alpha(img: Image.Image) -> bool:
    """Check if image already possesses an informative alpha transparency channel."""
    if img.mode != "RGBA":
        return False
    alpha = np.asarray(img)[..., 3]
    return float((alpha < 250).mean()) > 0.02


def simple_bg_removal(img: Image.Image) -> Image.Image:
    """Fast background removal using corner color estimation and morphological filtering.
    
    Executes in ~0.02s without requiring external neural network downloads.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = rgb.shape
    corner_size = max(2, min(10, h // 10, w // 10))

    corners = np.concatenate([
        rgb[:corner_size, :corner_size].reshape(-1, 3),
        rgb[:corner_size, -corner_size:].reshape(-1, 3),
        rgb[-corner_size:, :corner_size].reshape(-1, 3),
        rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)
    diff = np.linalg.norm(rgb - bg_color, axis=2)
    mask = diff > 35.0

    if ndi is not None:
        mask = ndi.binary_opening(mask, iterations=2)
        mask = ndi.binary_fill_holes(mask)
        lab, n = ndi.label(mask)
        if n > 1:
            sizes = ndi.sum(mask, lab, range(1, n + 1))
            mask = lab == (1 + int(np.argmax(sizes)))
    else:
        # Pure numpy basic fallback for morphology
        mask = mask.astype(bool)

    rgba = np.dstack([rgb.astype(np.uint8), (mask * 255).astype(np.uint8)])
    return Image.fromarray(rgba)


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using lightweight 4MB u2netp model or instant morphological fallback."""
    global _REMBG_SESSION
    try:
        import rembg
        if _REMBG_SESSION is None:
            _REMBG_SESSION = rembg.new_session("u2netp")
        return rembg.remove(img.convert("RGB"), session=_REMBG_SESSION).convert("RGBA")
    except Exception:
        return simple_bg_removal(img)


def preprocess_image(img: Image.Image, size: int = 512, margin: float = 0.08) -> Image.Image:
    """Normalize input image to standardized centered canvas with transparency.
    
    Args:
        img: Input PIL Image.
        size: Target square dimension (default 512px).
        margin: Outer padding ratio (default 0.08).
    
    Returns:
        Square RGBA Image with centered, aspect-preserved foreground object.
    """
    img = img.copy()
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)

    rgba = img.convert("RGBA") if has_real_alpha(img) else remove_background(img)
    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.where(alpha > 20)

    if xs.size == 0:
        # Fallback if background removal was too aggressive
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3]
        ys, xs = np.where(alpha > 0)
        if xs.size == 0:
            raise ValueError("Background removal removed all content from the image.")

    obj = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = round(size * (1.0 - 2 * margin)) / max(obj.size)
    new_w = max(1, round(obj.width * scale))
    new_h = max(1, round(obj.height * scale))
    obj = obj.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(obj, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def analyze_image(orig: Image.Image, rgba: Image.Image) -> Dict:
    """Assess image quality metrics: sharpness, mask coverage, and blob integrity."""
    gray = np.asarray(orig.convert("L").resize((512, 512)), dtype=np.float32)
    
    if ndi is not None:
        blur = float(ndi.laplace(gray).var())
    else:
        # Discrete Laplacian kernel fallback in pure NumPy
        lap = (
            gray[1:-1, 1:-1] * -4.0
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        blur = float(lap.var())

    mask = np.asarray(rgba)[..., 3] > 127
    cover = float(mask.mean())
    main = 1.0

    if ndi is not None:
        lab, n = ndi.label(mask)
        if n > 1:
            sizes = ndi.sum(mask, lab, range(1, n + 1))
            total = sizes.sum()
            main = float(sizes.max() / total) if total > 0 else 1.0

    w, h = orig.size
    warnings = []
    if min(w, h) < 256:
        warnings.append("low resolution (<256px)")
    if blur < 30.0:
        warnings.append("image looks blurry")
    if cover < 0.10:
        warnings.append("object mask very small (<10%)")
    if cover > 0.92:
        warnings.append("mask covers almost entire image (>92%)")

    return {
        "width": w,
        "height": h,
        "sharpness": round(blur, 1),
        "mask_cover": round(cover, 3),
        "main_blob": round(main, 3),
        "warnings": warnings,
    }


def assign_views(items: Union[Dict, List[Tuple[str, Image.Image]]]) -> Dict[str, Image.Image]:
    """Map arbitrary image lists or dicts into canonical viewing angles (front, left, back, right)."""
    items_list = list(items.items()) if isinstance(items, dict) else list(items)
    out: Dict[str, Image.Image] = {}
    rest: List[Image.Image] = []

    for label, im in items_list:
        key = next((k for k in VIEW_KEYS if k in str(label).lower() and k not in out), None)
        if key:
            out[key] = im
        else:
            rest.append(im)

    free_keys = [k for k in VIEW_KEYS if k not in out]
    for im in rest:
        if free_keys:
            out[free_keys.pop(0)] = im

    if "front" not in out and out:
        first = next(iter(out))
        out["front"] = out.pop(first)

    return {k: out[k] for k in VIEW_KEYS if k in out}
