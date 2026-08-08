import base64
from pathlib import Path

import pytest

from llama_cpp_py_sync._cffi_bindings import ffi
from llama_cpp_py_sync.multimodal import (
    ImageValidationError,
    MultimodalLimits,
    _decode_data_url,
    _normalise_content,
    _validate_image,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_mtmd_cffi_types_are_present():
    assert ffi.sizeof("struct mtmd_context_params") > 0
    assert ffi.sizeof("struct mtmd_decoder_pos") == 16
    assert ffi.typeof("mtmd_progress_callback").kind == "function"
    assert ffi.typeof("mtmd_bitmap_lazy_callback").kind == "function"
    assert ffi.typeof("mtmd_helper_post_decode_callback").kind == "function"


def test_png_data_url_and_ordered_parts():
    limits = MultimodalLimits(max_images=2)
    data_url = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")
    payload = _decode_data_url(data_url, limits)
    assert payload.mime_type == "image/png"
    assert (payload.width, payload.height) == (1, 1)

    text, images = _normalise_content(
        [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "after"},
        ],
        limits,
        "<__media__>",
    )
    assert text == "before<__media__>after"
    assert len(images) == 1


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image", "mime_type": "image/gif", "data": b"x"},
        {"type": "input_image", "mime_type": "image/png", "data": b""},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-base64"}},
    ],
)
def test_invalid_image_parts_are_rejected(part):
    with pytest.raises(ImageValidationError):
        _normalise_content([part], MultimodalLimits(), "<__media__>")


def test_corrupt_image_and_limits_are_rejected():
    with pytest.raises(ImageValidationError):
        _validate_image("image/png", b"not-an-image", MultimodalLimits())
    with pytest.raises(ImageValidationError):
        _validate_image(
            "image/png", PNG_1X1, MultimodalLimits(max_image_bytes=len(PNG_1X1) - 1)
        )
    with pytest.raises(ImageValidationError):
        _normalise_content(
            [{"type": "text", "text": "<__media__>"}],
            MultimodalLimits(),
            "<__media__>",
        )


def test_projector_discovery_patterns(tmp_path: Path):
    from llama_cpp_py_sync.multimodal import MultimodalContext

    model_path = tmp_path / "vision-model.gguf"
    projector = tmp_path / "vision-model-mmproj.gguf"
    model_path.write_bytes(b"model")
    projector.write_bytes(b"projector")

    model = type("Model", (), {"model_path": str(model_path)})()
    resolved = MultimodalContext._resolve_projector_path(model, None, True)
    assert resolved == projector.resolve()
