"""Safe Python-facing access to upstream llama.cpp ``mtmd``.

The native image processor is deliberately kept separate from :class:`Llama`.
It owns the projector context and all temporary image/chunk handles, while
``Llama`` remains usable for text-only inference without a projector.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from llama_cpp_py_sync._cffi_bindings import get_ffi, get_mtmd_lib


class MultimodalError(RuntimeError):
    """Base class for validation, preprocessing, and inference failures."""


class ProjectorCompatibilityError(MultimodalError):
    """Raised when a projector is missing, corrupt, unsupported, or incompatible."""


class ImageValidationError(MultimodalError, ValueError):
    """Raised when an image payload violates the multimodal input contract."""


class AudioValidationError(MultimodalError, ValueError):
    """Raised when an audio payload violates the multimodal input contract."""


class MultimodalCancelledError(MultimodalError):
    """Raised when a caller cancels multimodal preprocessing or generation."""


CancellationCallback = Callable[[], bool]


@dataclass(frozen=True)
class MultimodalLimits:
    """Resource limits applied before native image decoding."""

    max_images: int = 8
    max_image_bytes: int = 16 * 1024 * 1024
    max_total_image_bytes: int = 64 * 1024 * 1024
    max_width: int = 8192
    max_height: int = 8192
    max_total_pixels: int = 64 * 1024 * 1024
    max_audio_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_images",
            "max_image_bytes",
            "max_total_image_bytes",
            "max_width",
            "max_height",
            "max_total_pixels",
            "max_audio_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _ImagePayload:
    mime_type: str
    data: bytes
    width: int
    height: int


def _check_cancelled(cancel_callback: CancellationCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise MultimodalCancelledError("Multimodal request was cancelled")


def _read_png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None

    index = 2
    sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8))
    sof_markers |= set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
    while index < len(data):
        while index < len(data) and data[index] != 0xFF:
            index += 1
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA:  # Start of scan: dimensions should have appeared before it.
            break
        if index + 2 > len(data):
            return None
        segment_len = struct.unpack(">H", data[index : index + 2])[0]
        if segment_len < 2 or index + segment_len > len(data):
            return None
        if marker in sof_markers and segment_len >= 7:
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            return width, height
        index += segment_len
    return None


def _image_dimensions(mime_type: str, data: bytes) -> tuple[int, int]:
    if mime_type == "image/png":
        dimensions = _read_png_dimensions(data)
    else:
        dimensions = _read_jpeg_dimensions(data)
    if dimensions is None:
        raise ImageValidationError("Image data is corrupt or does not match its MIME type")
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ImageValidationError("Image dimensions must be non-zero")
    return width, height


def _validate_image(mime_type: str, data: bytes | bytearray | memoryview, limits: MultimodalLimits) -> _ImagePayload:
    if not isinstance(mime_type, str):
        raise ImageValidationError("Image MIME type must be a string")
    mime = mime_type.strip().lower()
    if mime not in {"image/png", "image/jpeg"}:
        raise ImageValidationError("Unsupported image MIME type; expected image/png or image/jpeg")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ImageValidationError("Image data must be bytes-like")

    raw = bytes(data)
    if not raw:
        raise ImageValidationError("Image data must not be empty")
    if len(raw) > limits.max_image_bytes:
        raise ImageValidationError("Image payload exceeds the configured byte limit")

    width, height = _image_dimensions(mime, raw)
    if width > limits.max_width or height > limits.max_height:
        raise ImageValidationError("Image dimensions exceed the configured limit")
    if width > (1 << 63) // max(1, height):
        raise ImageValidationError("Image dimensions overflow the supported pixel-count range")
    if width * height > limits.max_total_pixels:
        raise ImageValidationError("Image pixel count exceeds the configured limit")
    return _ImagePayload(mime, raw, width, height)


_DATA_URL_RE = re.compile(r"^data:(?P<mime>image/(?:png|jpeg));base64,(?P<data>[A-Za-z0-9+/]*={0,2})$")


def _decode_data_url(url: str, limits: MultimodalLimits) -> _ImagePayload:
    if not isinstance(url, str) or not url:
        raise ImageValidationError("image_url.url must be a non-empty data URL")
    if url.startswith(("http://", "https://")):
        raise ImageValidationError("Remote image URLs are not fetched; provide validated image bytes")
    match = _DATA_URL_RE.fullmatch(url)
    if match is None:
        raise ImageValidationError("Malformed image data URL; expected a base64-encoded PNG or JPEG")

    encoded = match.group("data")
    # Reject oversized input before base64 decoding allocates the byte buffer.
    max_encoded = ((limits.max_image_bytes + 2) // 3) * 4
    if len(encoded) > max_encoded:
        raise ImageValidationError("Encoded image payload exceeds the configured byte limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError("Malformed base64 image data") from exc
    return _validate_image(match.group("mime"), raw, limits)


def _part_to_image(part: Mapping[str, Any], limits: MultimodalLimits) -> _ImagePayload:
    kind = part.get("type")
    if kind == "input_image":
        return _validate_image(part.get("mime_type", ""), part.get("data", b""), limits)
    if kind == "image_url":
        image_url = part.get("image_url")
        if not isinstance(image_url, Mapping):
            raise ImageValidationError("image_url content must contain an image_url object")
        return _decode_data_url(image_url.get("url", ""), limits)
    raise ImageValidationError(f"Unsupported image content type: {kind!r}")


def _normalise_content(content: Any, limits: MultimodalLimits, marker: str) -> tuple[str, list[_ImagePayload]]:
    if isinstance(content, str):
        if marker in content:
            raise ImageValidationError("Text content may not contain the reserved multimodal marker")
        return content, []
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray, memoryview)):
        raise ImageValidationError("Message content must be text or an ordered content array")

    pieces: list[str] = []
    images: list[_ImagePayload] = []
    total_bytes = 0
    total_pixels = 0
    for part in content:
        if not isinstance(part, Mapping):
            raise ImageValidationError("Every message content part must be an object")
        kind = part.get("type")
        if kind == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ImageValidationError("Text content parts require a string text field")
            if marker in text:
                raise ImageValidationError("Text content may not contain the reserved multimodal marker")
            pieces.append(text)
        elif kind in {"image_url", "input_image"}:
            if len(images) >= limits.max_images:
                raise ImageValidationError("Image count exceeds the configured limit")
            payload = _part_to_image(part, limits)
            total_bytes += len(payload.data)
            if total_bytes > limits.max_total_image_bytes:
                raise ImageValidationError("Total image payload exceeds the configured byte limit")
            total_pixels += payload.width * payload.height
            if total_pixels > limits.max_total_pixels:
                raise ImageValidationError("Total image pixel count exceeds the configured limit")
            images.append(payload)
            pieces.append(marker)
        else:
            raise ImageValidationError(f"Unsupported message content type: {kind!r}")
    return "".join(pieces), images


class MultimodalChunk:
    """A view over one native chunk; valid only while its prompt is open."""

    def __init__(self, prompt: MultimodalPrompt, pointer: Any):
        self._prompt = prompt
        self._pointer = pointer

    @property
    def type(self) -> str:
        self._prompt._ensure_open()
        value = int(self._prompt._lib.mtmd_input_chunk_get_type(self._pointer))
        return {0: "text", 1: "image", 2: "audio"}.get(value, f"unknown:{value}")

    @property
    def n_tokens(self) -> int:
        self._prompt._ensure_open()
        return int(self._prompt._lib.mtmd_input_chunk_get_n_tokens(self._pointer))

    @property
    def n_pos(self) -> int:
        self._prompt._ensure_open()
        return int(self._prompt._lib.mtmd_input_chunk_get_n_pos(self._pointer))

    @property
    def identifier(self) -> str | None:
        self._prompt._ensure_open()
        pointer = self._prompt._lib.mtmd_input_chunk_get_id(self._pointer)
        return None if pointer == self._prompt._ffi.NULL else self._prompt._ffi.string(pointer).decode("utf-8", "replace")

    @property
    def text_tokens(self) -> tuple[int, ...]:
        self._prompt._ensure_open()
        if self.type != "text":
            return ()
        count = self._prompt._ffi.new("size_t *")
        tokens = self._prompt._lib.mtmd_input_chunk_get_tokens_text(self._pointer, count)
        if tokens == self._prompt._ffi.NULL:
            return ()
        return tuple(int(tokens[index]) for index in range(int(count[0])))

    @property
    def image_token_count(self) -> int:
        self._prompt._ensure_open()
        if self.type != "image":
            return 0
        image_tokens = self._prompt._lib.mtmd_input_chunk_get_tokens_image(self._pointer)
        return int(self._prompt._lib.mtmd_image_tokens_get_n_tokens(image_tokens))

    @property
    def embeddings(self) -> tuple[tuple[float, ...], ...]:
        """Encode this image chunk and return one input embedding per token."""
        self._prompt._ensure_open()
        return self._prompt._context.encode_chunk_embeddings(self)


class MultimodalPrompt:
    """Owned tokenized chunks and bitmaps produced by ``mtmd_tokenize``."""

    def __init__(self, context: MultimodalContext, chunks: Any, bitmaps: list[Any]):
        self._context = context
        self._ffi = context._ffi
        self._lib = context._lib
        self._chunks = chunks
        self._bitmaps = bitmaps
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise MultimodalError("Multimodal prompt has already been closed")

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._lib.mtmd_input_chunks_size(self._chunks))

    def __iter__(self) -> Iterator[MultimodalChunk]:
        self._ensure_open()
        for index in range(len(self)):
            pointer = self._lib.mtmd_input_chunks_get(self._chunks, index)
            if pointer == self._ffi.NULL:
                raise MultimodalError("mtmd returned a null input chunk")
            yield MultimodalChunk(self, pointer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._chunks != self._ffi.NULL:
            self._lib.mtmd_input_chunks_free(self._chunks)
        for bitmap in self._bitmaps:
            if bitmap != self._ffi.NULL:
                self._lib.mtmd_bitmap_free(bitmap)
        self._chunks = self._ffi.NULL
        self._bitmaps.clear()

    def __enter__(self) -> MultimodalPrompt:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> Any:
        raise TypeError("MultimodalPrompt contains process-local native handles and cannot be pickled")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MultimodalContext:
    """Own an upstream ``mtmd_context`` for one :class:`~.Llama` instance."""

    DISCOVERY_PATTERNS = (
        "{stem}-mmproj.gguf",
        "{stem}.mmproj.gguf",
        "mmproj-{stem}.gguf",
        "mmproj.gguf",
    )

    def __init__(
        self,
        model: Any,
        projector_path: str | os.PathLike[str] | None = None,
        *,
        discover_projector: bool = True,
        use_gpu: bool = True,
        n_threads: int | None = None,
        flash_attn_type: int | None = None,
        warmup: bool = True,
        image_min_tokens: int = -1,
        image_max_tokens: int = -1,
        batch_max_tokens: int = 1024,
        limits: MultimodalLimits | None = None,
        cancel_callback: CancellationCallback | None = None,
    ):
        if getattr(model, "_model", None) is None or getattr(model, "_ctx", None) is None:
            raise MultimodalError("The Llama model must be open before creating MultimodalContext")
        self.model = model
        self._ffi = get_ffi()
        self._lib = get_mtmd_lib()
        self._ctx = self._ffi.NULL
        self.limits = limits or MultimodalLimits()
        self.use_gpu = bool(use_gpu)
        self.projector_path = self._resolve_projector_path(model, projector_path, discover_projector)
        self._progress_callback = None
        self._closed = False
        self._progress_cancelled = False

        caps = self._lib.mtmd_get_cap_from_file(str(self.projector_path).encode("utf-8"))
        if not bool(caps.inp_vision) and not bool(caps.inp_audio):
            raise ProjectorCompatibilityError(
                f"Projector {self.projector_path} does not provide supported media input or is corrupt"
            )

        params = self._lib.mtmd_context_params_default()
        params.use_gpu = self.use_gpu
        params.print_timings = False
        if n_threads is not None:
            if n_threads <= 0:
                raise ValueError("n_threads must be positive")
            params.n_threads = int(n_threads)
        if flash_attn_type is not None:
            params.flash_attn_type = int(flash_attn_type)
        params.warmup = bool(warmup)
        params.image_min_tokens = int(image_min_tokens)
        params.image_max_tokens = int(image_max_tokens)
        params.batch_max_tokens = int(batch_max_tokens)
        if params.batch_max_tokens <= 0:
            raise ValueError("batch_max_tokens must be positive")
        if cancel_callback is not None:
            def _progress(_progress: float, _user_data: Any) -> bool:
                try:
                    return not bool(cancel_callback())
                except BaseException:
                    self._progress_cancelled = True
                    return False

            self._progress_callback = self._ffi.callback(
                "mtmd_progress_callback",
                _progress,
            )
            params.progress_callback = self._progress_callback
            params.progress_callback_user_data = self._ffi.NULL

        self._ctx = self._lib.mtmd_init_from_file(
            str(self.projector_path).encode("utf-8"), model._model, params
        )
        if self._ctx == self._ffi.NULL:
            if self._progress_cancelled:
                raise MultimodalCancelledError("Multimodal request was cancelled")
            _check_cancelled(cancel_callback)
            raise ProjectorCompatibilityError(
                f"Projector {self.projector_path} is incompatible with the loaded language model"
            )
        if not bool(self._lib.mtmd_support_vision(self._ctx)) and not bool(
            self._lib.mtmd_support_audio(self._ctx)
        ) and int(self._lib.mtmd_gen_audio_get_info(self._ctx).type) == 0:
            self._lib.mtmd_free(self._ctx)
            self._ctx = self._ffi.NULL
            raise ProjectorCompatibilityError(
                f"Projector {self.projector_path} initialized without a supported capability"
            )
        _check_cancelled(cancel_callback)
        self.model._multimodal_capabilities = self.capabilities

    @staticmethod
    def _resolve_projector_path(model: Any, path: str | os.PathLike[str] | None, discover: bool) -> Path:
        if path is not None:
            projector = Path(path)
            if not projector.is_file():
                raise FileNotFoundError(f"Projector file not found: {projector}")
            return projector.resolve()
        if not discover:
            raise FileNotFoundError("A projector_path is required when projector discovery is disabled")

        model_path = Path(getattr(model, "model_path", ""))
        if not model_path:
            model_path = Path(getattr(model, "_model_path", ""))
        if not model_path:
            raise FileNotFoundError("Cannot discover a projector without the model path")
        stem = model_path.stem
        for pattern in MultimodalContext.DISCOVERY_PATTERNS:
            candidate = model_path.parent / pattern.format(stem=stem)
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "No compatible projector found beside the model. Expected one of: "
            + ", ".join(pattern.format(stem=stem) for pattern in MultimodalContext.DISCOVERY_PATTERNS)
        )

    @property
    def marker(self) -> str:
        self._ensure_open()
        marker = self._lib.mtmd_get_marker(self._ctx)
        if marker == self._ffi.NULL:
            marker = self._lib.mtmd_default_marker()
        return self._ffi.string(marker).decode("utf-8")

    @property
    def capabilities(self) -> dict[str, Any]:
        self._ensure_open()
        gen_info = self._lib.mtmd_gen_audio_get_info(self._ctx)
        supports_generation = int(gen_info.type) != 0
        modalities = ["text"]
        if self.supports_vision:
            modalities.append("image")
        if self.supports_audio:
            modalities.append("audio")
        if supports_generation:
            modalities.append("audio_output")
        projector_type = "+".join(item for item in ("vision" if self.supports_vision else "", "audio" if self.supports_audio else "", "audio-generation" if supports_generation else "") if item)
        capabilities = {
            "multimodal": True,
            "modalities": modalities,
            "multiple_images": self.supports_vision,
            "projector_path": str(self.projector_path),
            "projector_type": projector_type,
            "marker": self.marker,
            "mrope": bool(self._lib.mtmd_decode_use_mrope(self._ctx)),
            "non_causal_decode": bool(self._lib.mtmd_decode_use_non_causal(self._ctx, self._ffi.NULL)),
            "input_audio": self.supports_audio,
            "audio_generation": supports_generation,
            "audio_sample_rate": int(gen_info.sample_rate) if supports_generation else self.audio_sample_rate,
            "audio_model_variant": None if gen_info.model_variant == self._ffi.NULL else self._ffi.string(gen_info.model_variant).decode("utf-8", "replace"),
            "projector_offload": {"use_gpu": self.use_gpu},
            "embedding_capabilities": self.model._embedding_capabilities()
            if hasattr(self.model, "_embedding_capabilities")
            else {},
            "generation_options": (["language", "speaker_reference", "top_k", "top_p", "seed", "max_frames", "output_format"] if supports_generation else []),
            "stt_capabilities": {
                "audio_input": self.supports_audio,
            },
            "tts_capabilities": {
                "audio_generation": supports_generation,
                "speaker_reference": supports_generation and self.supports_audio,
            },
        }
        self.model._multimodal_capabilities = dict(capabilities)
        return capabilities

    @property
    def supports_vision(self) -> bool:
        self._ensure_open()
        return bool(self._lib.mtmd_support_vision(self._ctx))

    @property
    def supports_audio(self) -> bool:
        self._ensure_open()
        return bool(self._lib.mtmd_support_audio(self._ctx))

    @property
    def audio_sample_rate(self) -> int:
        self._ensure_open()
        return int(self._lib.mtmd_get_audio_sample_rate(self._ctx))

    def _ensure_open(self) -> None:
        if self._closed or self._ctx == self._ffi.NULL:
            raise MultimodalError("MultimodalContext has already been closed")

    def create_bitmap(self, mime_type: str, data: bytes | bytearray | memoryview) -> Any:
        """Validate and decode one bounded image into an owned native bitmap."""
        self._ensure_open()
        payload = _validate_image(mime_type, data, self.limits)
        buf = self._ffi.new("unsigned char[]", payload.data)
        wrapper = self._lib.mtmd_helper_bitmap_init_from_buf(
            self._ctx, buf, len(payload.data), False
        )
        if wrapper.bitmap == self._ffi.NULL:
            raise ImageValidationError("Native image preprocessing failed")
        return wrapper.bitmap

    def create_audio_bitmap(self, audio: str | os.PathLike[str] | bytes | bytearray | memoryview) -> Any:
        """Decode validated audio through upstream mtmd/miniaudio.

        File formats and codecs intentionally remain owned by the synchronized
        native implementation.  Python only bounds and validates the input.
        """
        self._ensure_open()
        if isinstance(audio, (str, os.PathLike)):
            path = Path(audio)
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise AudioValidationError("Could not inspect audio file") from exc
            if size <= 0:
                raise AudioValidationError("Audio file must not be empty")
            if size > self.limits.max_audio_bytes:
                raise AudioValidationError("Audio file exceeds the configured byte limit")
            wrapper = self._lib.mtmd_helper_bitmap_init_from_file(
                self._ctx, str(path.resolve()).encode("utf-8"), False
            )
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            raw = bytes(audio)
            if not raw:
                raise AudioValidationError("Audio data must not be empty")
            if len(raw) > self.limits.max_audio_bytes:
                raise AudioValidationError("Audio data exceeds the configured byte limit")
            buf = self._ffi.new("unsigned char[]", raw)
            wrapper = self._lib.mtmd_helper_bitmap_init_from_buf(
                self._ctx, buf, len(raw), False
            )
        else:
            raise TypeError("audio must be a local path or bytes-like encoded audio")
        if wrapper.bitmap == self._ffi.NULL or not bool(
            self._lib.mtmd_bitmap_is_audio(wrapper.bitmap)
        ):
            if wrapper.bitmap != self._ffi.NULL:
                self._lib.mtmd_bitmap_free(wrapper.bitmap)
            raise AudioValidationError("Native audio decoding failed or the format is unsupported")
        return wrapper.bitmap

    def tokenize_audio_prompt(
        self,
        text: str,
        audio: str | os.PathLike[str] | bytes | bytearray | memoryview,
        *,
        add_special: bool = True,
        parse_special: bool = True,
        cancel_callback: CancellationCallback | None = None,
    ) -> MultimodalPrompt:
        """Tokenize one audio input using the projector's native media marker."""
        self._ensure_open()
        if not self.supports_audio:
            raise ProjectorCompatibilityError("The loaded projector does not support audio input")
        _check_cancelled(cancel_callback)
        marker = self.marker
        prompt = text if marker in text else text + marker
        bitmap = self.create_audio_bitmap(audio)
        chunks = self._lib.mtmd_input_chunks_init()
        if chunks == self._ffi.NULL:
            self._lib.mtmd_bitmap_free(bitmap)
            raise MultimodalError("Could not allocate native input chunks")
        try:
            encoded = prompt.encode("utf-8")
            text_buf = self._ffi.new("char[]", encoded)
            input_text = self._ffi.new("mtmd_input_text *")
            input_text.text = text_buf
            input_text.text_len = len(encoded)
            input_text.add_special = bool(add_special)
            input_text.parse_special = bool(parse_special)
            bitmaps = self._ffi.new("const mtmd_bitmap *[]", [bitmap])
            result = self._lib.mtmd_tokenize(self._ctx, chunks, input_text, bitmaps, 1)
            if result != 0:
                raise MultimodalError(f"Native audio tokenization failed with code {result}")
            _check_cancelled(cancel_callback)
            return MultimodalPrompt(self, chunks, [bitmap])
        except BaseException:
            self._lib.mtmd_input_chunks_free(chunks)
            self._lib.mtmd_bitmap_free(bitmap)
            raise

    def load_image_file(self, path: str | os.PathLike[str]) -> Any:
        """Load a local PNG/JPEG file after applying the same limits as byte input."""
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            size = image_path.stat().st_size
        except OSError as exc:
            raise ImageValidationError("Could not inspect image file") from exc
        if size > self.limits.max_image_bytes:
            raise ImageValidationError("Image file exceeds the configured byte limit")
        suffix = image_path.suffix.lower()
        mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
        if mime_type is None:
            raise ImageValidationError("Unsupported image file type; expected PNG or JPEG")
        return self.create_bitmap(mime_type, image_path.read_bytes())

    def tokenize_prompt(
        self,
        text: str,
        images: Sequence[tuple[str, bytes | bytearray | memoryview]] = (),
        *,
        add_special: bool = True,
        parse_special: bool = True,
        cancel_callback: CancellationCallback | None = None,
    ) -> MultimodalPrompt:
        """Tokenize text and ordered images using upstream marker semantics."""
        self._ensure_open()
        if not isinstance(text, str):
            raise TypeError("Multimodal prompt text must be a string")
        marker = self.marker
        marker_count = text.count(marker)
        if marker_count != len(images):
            raise ImageValidationError(
                f"Prompt contains {marker_count} media markers but {len(images)} images were supplied"
            )
        if len(images) > self.limits.max_images:
            raise ImageValidationError("Image count exceeds the configured limit")

        bitmaps: list[Any] = []
        total_bytes = 0
        chunks = self._lib.mtmd_input_chunks_init()
        if chunks == self._ffi.NULL:
            raise MultimodalError("Could not allocate mtmd input chunks")
        try:
            total_pixels = 0
            for mime_type, data in images:
                _check_cancelled(cancel_callback)
                payload = _validate_image(mime_type, data, self.limits)
                total_bytes += len(payload.data)
                if total_bytes > self.limits.max_total_image_bytes:
                    raise ImageValidationError("Total image payload exceeds the configured byte limit")
                total_pixels += payload.width * payload.height
                if total_pixels > self.limits.max_total_pixels:
                    raise ImageValidationError("Total image pixel count exceeds the configured limit")
                bitmaps.append(self.create_bitmap(payload.mime_type, payload.data))
            _check_cancelled(cancel_callback)

            text_bytes = text.encode("utf-8")
            text_buffer = self._ffi.new("char[]", text_bytes)
            input_text = self._ffi.new("mtmd_input_text *")
            input_text.text = text_buffer
            input_text.text_len = len(text_bytes)
            input_text.add_special = bool(add_special)
            input_text.parse_special = bool(parse_special)
            bitmap_array = self._ffi.new("const mtmd_bitmap *[]", len(bitmaps))
            for index, bitmap in enumerate(bitmaps):
                bitmap_array[index] = bitmap
            result = self._lib.mtmd_tokenize(
                self._ctx, chunks, input_text, bitmap_array, len(bitmaps)
            )
            if result != 0:
                raise MultimodalError(f"mtmd tokenization failed with code {int(result)}")
            image_chunks = sum(1 for chunk in _iter_chunk_pointers(self, chunks) if _chunk_type(self, chunk) == 1)
            if image_chunks != len(bitmaps):
                raise MultimodalError("mtmd did not preserve every supplied image in the prompt")
            prompt = MultimodalPrompt(self, chunks, bitmaps)
            chunks = self._ffi.NULL
            bitmaps = []
            return prompt
        except Exception:
            if chunks != self._ffi.NULL:
                self._lib.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                if bitmap != self._ffi.NULL:
                    self._lib.mtmd_bitmap_free(bitmap)
            raise

    def encode_chunk_embeddings(self, chunk: MultimodalChunk) -> tuple[tuple[float, ...], ...]:
        """Encode one media chunk and return its processor embeddings."""
        self._ensure_open()
        if chunk._prompt._context is not self:
            raise MultimodalError("Multimodal chunk belongs to a different context")
        if chunk.type != "image":
            raise MultimodalError("Only image chunks have vision embeddings")
        result = self._lib.mtmd_encode_chunk(self._ctx, chunk._pointer)
        if result != 0:
            raise MultimodalError(f"mtmd image encoding failed with code {int(result)}")
        embeddings = self._lib.mtmd_get_output_embd(self._ctx)
        if embeddings == self._ffi.NULL:
            raise MultimodalError("mtmd did not return image embeddings")
        n_tokens = chunk.n_tokens
        n_embd = int(self.model._lib.llama_model_n_embd_inp(self.model._model))
        if n_embd <= 0:
            raise MultimodalError("The language model reported an invalid input embedding size")
        return tuple(
            tuple(float(embeddings[token * n_embd + dimension]) for dimension in range(n_embd))
            for token in range(n_tokens)
        )

    def evaluate_prompt(
        self,
        prompt: MultimodalPrompt,
        *,
        n_past: int = 0,
        seq_id: int = 0,
        cancel_callback: CancellationCallback | None = None,
    ) -> int:
        """Evaluate ordered text/image chunks and return the next prompt position."""
        self._ensure_open()
        if prompt._context is not self:
            raise MultimodalError("Multimodal prompt belongs to a different context")
        current = int(n_past)
        batch_size = max(1, int(getattr(self.model, "_n_batch", 512)))
        for index, chunk in enumerate(prompt):
            _check_cancelled(cancel_callback)
            next_position = self._ffi.new("llama_pos *")
            next_position[0] = current
            result = self._lib.mtmd_helper_eval_chunk_single(
                self._ctx,
                self.model._ctx,
                chunk._pointer,
                current,
                int(seq_id),
                batch_size,
                index == len(prompt) - 1,
                next_position,
            )
            if result != 0:
                raise MultimodalError(f"mtmd chunk evaluation failed with code {int(result)}")
            current = int(next_position[0])
        _check_cancelled(cancel_callback)
        return current

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ctx != self._ffi.NULL:
            self._lib.mtmd_free(self._ctx)
            self._ctx = self._ffi.NULL
        self._progress_callback = None
        capabilities = getattr(self.model, "_multimodal_capabilities", None)
        if isinstance(capabilities, Mapping) and capabilities.get("projector_path") == str(self.projector_path):
            try:
                del self.model._multimodal_capabilities
            except AttributeError:
                pass

    def __enter__(self) -> MultimodalContext:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> Any:
        raise TypeError("MultimodalContext contains process-local native handles and cannot be pickled")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _iter_chunk_pointers(context: MultimodalContext, chunks: Any) -> Iterable[Any]:
    count = int(context._lib.mtmd_input_chunks_size(chunks))
    for index in range(count):
        pointer = context._lib.mtmd_input_chunks_get(chunks, index)
        if pointer == context._ffi.NULL:
            raise MultimodalError("mtmd returned a null input chunk")
        yield pointer


def _chunk_type(context: MultimodalContext, chunk: Any) -> int:
    return int(context._lib.mtmd_input_chunk_get_type(chunk))


__all__ = [
    "ImageValidationError",
    "MultimodalCancelledError",
    "MultimodalChunk",
    "MultimodalContext",
    "MultimodalError",
    "MultimodalLimits",
    "MultimodalPrompt",
    "ProjectorCompatibilityError",
]
