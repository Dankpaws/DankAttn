"""Hy4-flash image preprocessing (matches HYV4VLImageProcessor in processor_config.json).

Pure Pillow + numpy (no PyTorch/transformers dependency on the Mac Studio front end).
"""
import base64, hashlib, io, math, os, urllib.request
import numpy as np
from PIL import Image

PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
MIN_PIXELS = 4096
MAX_PIXELS = 4194304  # 2048 x 2048
MAX_EDGE = 2048


def _smart_resize(height: int, width: int, factor: int = FACTOR,
                  min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS,
                  max_edge: int = MAX_EDGE) -> tuple[int, int]:
    """Resize (H, W) to multiples of factor (32) within [min_pixels, max_pixels] and <= max_edge."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image dimensions {width}x{height}")
    # Cap longest edge first so we never exceed position_embedding's 128x128 patch grid (2048px)
    if max(height, width) > max_edge:
        s = max_edge / max(height, width)
        height, width = max(factor, int(round(height * s))), max(factor, int(round(width * s)))
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    h_bar = min(h_bar, max_edge)
    w_bar = min(w_bar, max_edge)
    return int(h_bar), int(w_bar)


def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """IEEE-754 round-to-nearest-even float32 -> bfloat16 (little-endian uint16 bytes)."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16 = ((u32 + rounding) >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def load_image_bytes(url_or_path: str) -> bytes:
    """Load raw image bytes from a data: URI, file:// URI, local path, or http(s):// URL."""
    if url_or_path.startswith("data:"):
        _, b64 = url_or_path.split(",", 1)
        return base64.b64decode(b64)
    if url_or_path.startswith("file://"):
        with open(url_or_path[7:], "rb") as f:
            return f.read()
    if url_or_path.startswith(("http://", "https://")):
        req = urllib.request.Request(url_or_path, headers={"User-Agent": "DankAttn-Hy4/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    if os.path.exists(url_or_path):
        with open(url_or_path, "rb") as f:
            return f.read()
    raise ValueError(f"unsupported or missing image_url: {url_or_path[:80]}")


def preprocess_image(url_or_path: str, max_pixels: int = MAX_PIXELS) -> dict:
    """Preprocess one image into Hy4-flash ViT bf16 patches.

    Returns dict with keys: T, h, w, n_tokens, hash_u64, pixels_bf16 (bytes).
    """
    raw = load_image_bytes(url_or_path)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size
    new_h, new_w = _smart_resize(orig_h, orig_w, max_pixels=max_pixels)
    if (new_w, new_h) != (orig_w, orig_h):
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Normalize: (x / 255.0 - 0.5) / 0.5 = x * (2/255) - 1.0, shape [H, W, 3]
    arr = np.asarray(img, dtype=np.float32) * (2.0 / 255.0) - 1.0
    hp, wp = new_h // PATCH_SIZE, new_w // PATCH_SIZE
    # [hp, 16, wp, 16, 3] -> [hp, wp, 3, 16, 16] -> [hp * wp, 768]
    patches = arr.reshape(hp, PATCH_SIZE, wp, PATCH_SIZE, 3).transpose(0, 2, 4, 1, 3).reshape(hp * wp, -1)
    pixels_bf16 = _f32_to_bf16_bytes(patches)
    digest = hashlib.sha256(pixels_bf16).digest()
    hash_u64 = int.from_bytes(digest[:8], "little")
    n_tokens = (hp // MERGE_SIZE) * (wp // MERGE_SIZE + 1)
    return {
        "T": 1,
        "h": hp,
        "w": wp,
        "n_tokens": n_tokens,
        "hash_u64": hash_u64,
        "pixels_bf16": pixels_bf16,
    }


def extract_image_urls(messages: list[dict]) -> list[str]:
    """Extract all OpenAI image_url / image entries from messages in template order."""
    urls = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for p in c:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "image_url" or "image_url" in p:
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if not isinstance(url, str) or not url:
                    raise ValueError("image_url must contain a non-empty url string")
                urls.append(url)
            elif t == "image" or "image" in p:
                url = p.get("image")
                if not isinstance(url, str) or not url:
                    raise ValueError("image content part must contain a non-empty string")
                urls.append(url)
    return urls
