"""Hy4-flash image preprocessing (matches HYV4VLImageProcessor in processor_config.json).

Pure Pillow + numpy (no PyTorch/transformers dependency on the Mac Studio front end).
"""
import base64, hashlib, http.client, io, ipaddress, math, os, socket, ssl, urllib.parse
import numpy as np
from PIL import Image

PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
MIN_PIXELS = 4096
MAX_PIXELS = 4194304  # 2048 x 2048
MAX_EDGE = 2048

MAX_IMAGES = int(os.environ.get("DANKATTN_MAX_IMAGES", "32"))
MAX_DATA_IMAGE_BYTES = int(os.environ.get("DANKATTN_MAX_IMAGE_MB", "25")) * 1024 * 1024
MAX_REMOTE_IMAGE_BYTES = int(os.environ.get("DANKATTN_MAX_REMOTE_IMAGE_MB", "20")) * 1024 * 1024
MAX_REDIRECTS = 5
REMOTE_TIMEOUT_S = 20


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def _resolve_public_ip(hostname: str, port: int) -> tuple[str, int]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(f"failed to resolve image host: {e}") from e
    if not infos:
        raise ValueError("image host resolved to no addresses")
    chosen = None
    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        if not _is_public_ip(ip_str):
            raise ValueError("remote image URLs targeting private or local addresses are not allowed")
        if chosen is None:
            chosen = (ip_str, family)
    return chosen


def _fetch_public_http(url: str, max_bytes: int = MAX_REMOTE_IMAGE_BYTES) -> bytes:
    cur_url = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urllib.parse.urlparse(cur_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported remote image scheme: {parsed.scheme}")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("remote image URL missing hostname")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        ip_str, _ = _resolve_public_ip(hostname, port)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        host_hdr = hostname if (
            (parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)
        ) else f"{hostname}:{port}"

        conn = None
        try:
            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                raw_sock = socket.create_connection((ip_str, port), timeout=REMOTE_TIMEOUT_S)
                try:
                    tls_sock = ctx.wrap_socket(raw_sock, server_hostname=hostname)
                except Exception:
                    raw_sock.close()
                    raise
                conn = http.client.HTTPSConnection(hostname, port, timeout=REMOTE_TIMEOUT_S, context=ctx)
                conn.sock = tls_sock
            else:
                conn = http.client.HTTPConnection(ip_str, port, timeout=REMOTE_TIMEOUT_S)
            conn.request("GET", target, headers={"Host": host_hdr, "User-Agent": "DankAttn-Hy4/1.0"})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location")
                if not loc:
                    raise ValueError(f"remote image redirect ({resp.status}) missing Location")
                cur_url = urllib.parse.urljoin(cur_url, loc)
                continue
            if resp.status != 200:
                raise ValueError(f"remote image fetch failed with HTTP {resp.status}")
            cl = (resp.getheader("Content-Length") or "").strip()
            if cl and cl.isdigit() and int(cl) > max_bytes:
                raise ValueError(f"remote image exceeds {max_bytes // (1024 * 1024)} MB limit")
            chunks = []
            total = 0
            while True:
                buf = resp.read(65536)
                if not buf:
                    break
                total += len(buf)
                if total > max_bytes:
                    raise ValueError(f"remote image exceeds {max_bytes // (1024 * 1024)} MB limit")
                chunks.append(buf)
            return b"".join(chunks)
        finally:
            if conn is not None:
                conn.close()
    raise ValueError("too many redirects fetching remote image")


def _read_allowed_local_file(path: str, allow_local_media: str, max_bytes: int = MAX_DATA_IMAGE_BYTES) -> bytes:
    if not allow_local_media:
        raise ValueError("local file image paths are disabled; pass a data: URI or public http(s):// URL")
    root = os.path.realpath(allow_local_media)
    resolved = os.path.realpath(path)
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError("local image path is outside --allow-local-media directory")
    if not os.path.isfile(resolved):
        raise ValueError("local image path does not exist or is not a regular file")
    st = os.stat(resolved)
    if st.st_size > max_bytes:
        raise ValueError(f"local image exceeds {max_bytes // (1024 * 1024)} MB limit")
    with open(resolved, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"local image exceeds {max_bytes // (1024 * 1024)} MB limit")
    return data


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


def load_image_bytes(url_or_path: str, allow_local_media: str = "") -> bytes:
    """Load raw image bytes from a data: URI, public http(s):// URL, or (only if opted in) a local path."""
    if url_or_path.startswith("data:"):
        _, b64 = url_or_path.split(",", 1)
        if len(b64) > (MAX_DATA_IMAGE_BYTES * 4) // 3 + 64:
            raise ValueError(f"data: image exceeds {MAX_DATA_IMAGE_BYTES // (1024 * 1024)} MB limit")
        raw = base64.b64decode(b64, validate=False)
        if len(raw) > MAX_DATA_IMAGE_BYTES:
            raise ValueError(f"data: image exceeds {MAX_DATA_IMAGE_BYTES // (1024 * 1024)} MB limit")
        return raw
    if url_or_path.startswith(("http://", "https://")):
        return _fetch_public_http(url_or_path)
    if url_or_path.startswith("file://"):
        parsed = urllib.parse.urlparse(url_or_path)
        return _read_allowed_local_file(urllib.parse.unquote(parsed.path), allow_local_media)
    if allow_local_media:
        return _read_allowed_local_file(url_or_path, allow_local_media)
    raise ValueError("unsupported image_url (expected data: URI or public http(s):// URL)")


def preprocess_image(url_or_path: str, max_pixels: int = MAX_PIXELS, allow_local_media: str = "") -> dict:
    """Preprocess one image into Hy4-flash ViT bf16 patches.

    Returns dict with keys: T, h, w, n_tokens, hash_u64, pixels_bf16 (bytes).
    """
    raw = load_image_bytes(url_or_path, allow_local_media=allow_local_media)
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
