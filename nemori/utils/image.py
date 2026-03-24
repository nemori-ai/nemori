"""Image compression utilities for multimodal memory processing."""
from __future__ import annotations

import base64
import io
import logging
from typing import Sequence

from PIL import Image

logger = logging.getLogger("nemori")

_FORMAT_SIGNATURES = {
    "/9j/": "jpeg",
    "iVBORw0KGgo": "png",
    "UklGR": "webp",
    "R0lGOD": "gif",
}


def ensure_image_data_url(raw_or_url: str) -> str:
    """Ensure input is a proper data URL. Wraps raw base64 if needed."""
    if raw_or_url.startswith("data:"):
        return raw_or_url
    mime = "image/png"
    for sig, fmt in _FORMAT_SIGNATURES.items():
        if raw_or_url.startswith(sig):
            mime = f"image/{fmt}"
            break
    return f"data:{mime};base64,{raw_or_url}"


def compress_image_for_llm(
    data_url: str,
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 70,
) -> str:
    """Compress image for LLM. Converts to JPEG, resizes, returns data URL."""
    if "," in data_url:
        b64_data = data_url.split(",", 1)[1]
    else:
        b64_data = data_url

    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))

    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if img.width > max_width or img.height > max_height:
        ratio = min(max_width / img.width, max_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def compress_images_for_llm(
    data_urls: Sequence[str],
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 70,
) -> list[str]:
    """Compress multiple images."""
    return [compress_image_for_llm(url, max_width, max_height, quality) for url in data_urls]
