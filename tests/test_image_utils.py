import base64
import io
import pytest
from PIL import Image
from nemori.utils.image import compress_image_for_llm, compress_images_for_llm, ensure_image_data_url


def _make_test_png(width: int = 200, height: int = 200) -> str:
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_compress_returns_jpeg_data_url():
    result = compress_image_for_llm(_make_test_png())
    assert result.startswith("data:image/jpeg;base64,")


def test_compress_respects_max_dimensions():
    result = compress_image_for_llm(_make_test_png(3000, 2000), max_width=1280, max_height=720)
    b64 = result.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.width <= 1280
    assert img.height <= 720


def test_ensure_image_data_url_wraps_raw_base64():
    img = Image.new("RGB", (10, 10), "blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode()
    result = ensure_image_data_url(raw_b64)
    assert result.startswith("data:image/jpeg;base64,")


def test_compress_handles_rgba():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = compress_image_for_llm(f"data:image/png;base64,{b64}")
    assert result.startswith("data:image/jpeg;base64,")


def test_compress_images_for_llm_batch():
    urls = [_make_test_png(100, 100), _make_test_png(50, 50)]
    results = compress_images_for_llm(urls)
    assert len(results) == 2
    assert all(r.startswith("data:image/jpeg;base64,") for r in results)


def test_small_image_not_resized():
    result = compress_image_for_llm(_make_test_png(100, 100))
    b64 = result.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.width == 100
    assert img.height == 100
