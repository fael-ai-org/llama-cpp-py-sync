"""
High-level Llama class for easy model interaction.

This provides a thin wrapper around the llama.cpp C API for common operations
like loading models, tokenizing text, and generating completions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib

# `ggml_type` values accepted for the KV cache. Kept to the types that are
# actually useful for type_k / type_v rather than mirroring the whole enum.
GGML_KV_TYPES: dict[str, int] = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "bf16": 30,
}


# `llama_load_mode` values, keyed by the names llama_load_mode_from_str accepts.
LLAMA_LOAD_MODES: dict[str, int] = {
    "none": 0,
    "mmap": 1,
    "mlock": 2,
    "dio": 3,
}


def _resolve_ggml_type(value: int | str) -> int:
    """Map a ggml type name to its enum value, passing ints through."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in GGML_KV_TYPES:
            raise ValueError(
                f"Unknown KV cache type {value!r}; expected one of "
                f"{', '.join(sorted(GGML_KV_TYPES))} or a ggml_type int."
            )
        return GGML_KV_TYPES[key]
    return int(value)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int = -1
    stop_sequences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedAudio:
    """Audio returned by :meth:`Llama.generate_audio`."""

    data: bytes
    sample_rate: int
    format: str
    n_samples: int


class Llama:
    """
    High-level wrapper for llama.cpp model inference.

    This class provides a simple interface for loading GGUF models and generating text.
    It automatically manages the model context and provides convenient methods for
    common operations.

    Example:
        >>> llm = Llama("model.gguf", n_ctx=2048, n_gpu_layers=35)
        >>> response = llm.generate("Hello, world!", max_tokens=100)
        >>> print(response)
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        n_ubatch: int | None = None,
        n_threads_batch: int | None = None,
        seed: int = -1,
        use_mmap: bool = True,
        use_mlock: bool = False,
        load_mode: int | str | None = None,
        verbose: bool = False,
        embedding: bool = False,
        flash_attn_type: int | None = None,
        type_k: int | str | None = None,
        type_v: int | str | None = None,
        offload_kqv: bool | None = None,
    ):
        """
        Initialize the Llama model.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context size (max tokens in context window).
            n_batch: Logical maximum batch size for prompt processing (llama_decode).
            n_threads: Number of generation threads (default: auto-detect).
            n_gpu_layers: Number of layers to offload to GPU.
            n_ubatch: Physical microbatch size; capped by ``n_batch``. Defaults to ``n_batch``.
            n_threads_batch: Threads for prompt/batch processing; defaults to ``n_threads``.
            seed: Random seed for sampling (-1 for random).
            use_mmap: Whether to use memory mapping for model loading. Ignored
                when ``load_mode`` is given.
            use_mlock: Whether to lock model in memory. Ignored when
                ``load_mode`` is given.
            load_mode: ``llama_load_mode`` value or name (``"none"``, ``"mmap"``,
                ``"mlock"``, ``"dio"``). Supersedes ``use_mmap`` / ``use_mlock``,
                which llama.cpp folded into this enum. Note ``"mlock"`` implies
                mmap, and ``"dio"`` selects direct I/O.
            verbose: Whether to print verbose output.
            embedding: Whether to enable embedding mode.
            flash_attn_type: ``llama_flash_attn_type`` value (e.g. 0 off, 1 on, -1 auto).
                If ``None``, uses ``LLAMA_FLASH_ATTENTION`` env (same rules as before).
            type_k: KV cache data type for keys, as a ``ggml_type`` value or a
                name such as ``"f16"`` or ``"q8_0"``. ``None`` keeps the
                llama.cpp default. Quantizing the KV cache trades a little
                quality for a large reduction in KV memory at high ``n_ctx``.
            type_v: KV cache data type for values; see ``type_k``. Note that
                llama.cpp requires flash attention for a quantized V cache.
            offload_kqv: Whether to keep the KV cache on the GPU. ``None``
                keeps the llama.cpp default.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(model_path)
        self._lib = get_lib()
        self._ffi = get_ffi()
        self._model = None
        self._ctx = None
        self._sampler = None
        self._vocab = None
        self._abort_callback = None
        self._verbose = verbose
        self._embedding = embedding
        self._n_ctx = n_ctx
        # Prompt evaluation must be chunked to this width: llama_decode asserts
        # on any batch wider than the context's n_batch.
        self._n_batch = max(1, int(n_batch))

        self._lib.llama_backend_init()

        model_params = self._lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        # llama.cpp folded use_mmap / use_mlock / use_direct_io into a single
        # llama_load_mode enum. An explicit load_mode wins; otherwise the legacy
        # booleans are translated, with mlock implying mmap as upstream defines.
        if load_mode is not None:
            model_params.load_mode = self._resolve_load_mode(load_mode)
        elif use_mlock:
            model_params.load_mode = 2  # LLAMA_LOAD_MODE_MLOCK (implies mmap)
        elif use_mmap:
            model_params.load_mode = 1  # LLAMA_LOAD_MODE_MMAP
        else:
            model_params.load_mode = 0  # LLAMA_LOAD_MODE_NONE

        if self._verbose:
            print(f"Loading model from {model_path}...")

        if hasattr(self._lib, "llama_model_load_from_file"):
            load_model = self._lib.llama_model_load_from_file
        else:
            load_model = self._lib.llama_load_model_from_file
        self._model = load_model(
            model_path.encode("utf-8"),
            model_params,
        )

        if self._model == self._ffi.NULL:
            raise RuntimeError(f"Failed to load model from {model_path}")

        ctx_params = self._lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ubatch = n_ubatch if n_ubatch is not None else n_batch
        ctx_params.n_ubatch = max(1, min(n_batch, ubatch))
        n_thr = n_threads if n_threads else os.cpu_count() or 4
        ctx_params.n_threads = n_thr
        ctx_params.n_threads_batch = (
            n_threads_batch if n_threads_batch is not None else n_thr
        )
        # Native audio generation consumes the backbone's last hidden state via
        # llama_get_embeddings_ith().  A projector may be supplied only later
        # to generate_audio(), so the context must retain embeddings from the
        # outset.  The public text-embedding API remains gated by
        # ``self._embedding`` below.
        ctx_params.embeddings = True
        if type_k is not None:
            ctx_params.type_k = _resolve_ggml_type(type_k)
        if type_v is not None:
            ctx_params.type_v = _resolve_ggml_type(type_v)
        if offload_kqv is not None:
            ctx_params.offload_kqv = bool(offload_kqv)
        if flash_attn_type is not None:
            ctx_params.flash_attn_type = flash_attn_type
        else:
            flash_env = os.environ.get("LLAMA_FLASH_ATTENTION", "0").strip()
            if flash_env.lower() in ("auto", "-1"):
                ctx_params.flash_attn_type = -1
            elif flash_env.lower() not in ("0", "", "false", "off", "disabled"):
                ctx_params.flash_attn_type = 1
            else:
                ctx_params.flash_attn_type = 0

        if seed != -1:
            pass

        if hasattr(self._lib, "llama_init_from_model"):
            init_ctx = self._lib.llama_init_from_model
        else:
            init_ctx = self._lib.llama_new_context_with_model
        self._ctx = init_ctx(self._model, ctx_params)

        if self._ctx == self._ffi.NULL:
            if hasattr(self._lib, "llama_model_free"):
                free_model = self._lib.llama_model_free
            else:
                free_model = self._lib.llama_free_model
            free_model(self._model)
            raise RuntimeError("Failed to create model context")

        get_vocab = getattr(self._lib, "llama_model_get_vocab", None)
        if get_vocab is not None:
            self._vocab = get_vocab(self._model)
            if self._vocab == self._ffi.NULL:
                self._vocab = None

        self._setup_sampler(seed)

        if self._verbose:
            print("Model loaded successfully!")
            print(f"  Vocab size: {self.n_vocab}")
            print(f"  Context size: {self.n_ctx}")
            print(f"  Embedding size: {self.n_embd}")

    def _resolve_load_mode(self, value: int | str) -> int:
        """Map a load-mode name to its enum value, passing ints through.

        Resolved in Python on purpose: llama.cpp's ``llama_load_mode_from_str``
        throws ``std::invalid_argument`` for an unknown name, and a C++
        exception crossing the CFFI boundary terminates the process instead of
        raising. Names mirror that function's accepted set.
        """
        if not isinstance(value, str):
            return int(value)
        key = value.strip().lower()
        if key not in LLAMA_LOAD_MODES:
            raise ValueError(
                f"Unknown load mode {value!r}; expected one of "
                f"{', '.join(LLAMA_LOAD_MODES)} or a llama_load_mode int."
            )
        return LLAMA_LOAD_MODES[key]

    def _setup_sampler(self, seed: int = -1):
        """Set up the default sampler chain."""
        sampler_params = self._lib.llama_sampler_chain_default_params()
        self._sampler = self._lib.llama_sampler_chain_init(sampler_params)

        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_top_k(40)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_top_p(0.95, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_min_p(0.05, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_temp(0.8)
        )

        actual_seed = seed if seed != -1 else int.from_bytes(os.urandom(4), "little")
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_dist(actual_seed)
        )

    def __del__(self):
        """Clean up resources."""
        self.close()

    def close(self):
        """Explicitly release model resources."""
        if hasattr(self, "_sampler") and self._sampler is not None:
            self._lib.llama_sampler_free(self._sampler)
            self._sampler = None
        if hasattr(self, "_ctx") and self._ctx is not None:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if hasattr(self, "_model") and self._model is not None:
            if hasattr(self._lib, "llama_model_free"):
                free_model = self._lib.llama_model_free
            else:
                free_model = self._lib.llama_free_model
            free_model(self._model)
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def n_vocab(self) -> int:
        """Get vocabulary size."""
        if self._vocab is not None:
            n_tokens = getattr(self._lib, "llama_vocab_n_tokens", None)
            if n_tokens is not None:
                return int(n_tokens(self._vocab))
            n_vocab = getattr(self._lib, "llama_n_vocab", None)
            if n_vocab is not None:
                return int(n_vocab(self._vocab))
        raise RuntimeError("Vocab handle is not available; bindings may be out of sync")

    @property
    def n_ctx(self) -> int:
        """Get context size."""
        return self._lib.llama_n_ctx(self._ctx)

    @property
    def n_embd(self) -> int:
        """Get embedding dimension."""
        return self._lib.llama_n_embd(self._model)

    @property
    def capabilities(self) -> dict[str, Any]:
        """Report text and, after setup, multimodal model capabilities."""
        multimodal = getattr(self, "_multimodal_capabilities", None)
        if multimodal is not None:
            return dict(multimodal)
        return {
            "multimodal": False,
            "modalities": ["text"],
            "multiple_images": False,
            "projector_path": None,
            "projector_type": None,
        }

    def _model_metadata(self) -> dict[str, str]:
        """Read GGUF metadata through llama.cpp without interpreting model names."""
        result: dict[str, str] = {}
        count_fn = getattr(self._lib, "llama_model_meta_count", None)
        key_fn = getattr(self._lib, "llama_model_meta_key_by_index", None)
        value_fn = getattr(self._lib, "llama_model_meta_val_str_by_index", None)
        if count_fn is None or key_fn is None or value_fn is None:
            return result
        for index in range(max(0, int(count_fn(self._model)))):
            key_size = int(key_fn(self._model, index, self._ffi.NULL, 0))
            value_size = int(value_fn(self._model, index, self._ffi.NULL, 0))
            if key_size < 0 or value_size < 0:
                continue
            key_buf = self._ffi.new("char[]", key_size + 1)
            value_buf = self._ffi.new("char[]", value_size + 1)
            if key_fn(self._model, index, key_buf, key_size + 1) < 0:
                continue
            if value_fn(self._model, index, value_buf, value_size + 1) < 0:
                continue
            result[self._ffi.string(key_buf).decode("utf-8", "replace")] = (
                self._ffi.string(value_buf).decode("utf-8", "replace")
            )
        return result

    def _native_audio_languages(self) -> list[str]:
        """Expose language tokens supplied by the native vocabulary, when present."""
        if self._vocab is None:
            return []
        get_text = getattr(self._lib, "llama_vocab_get_text", None)
        if get_text is None:
            return []
        prefix = "<|codec_language_"
        suffix = "|>"
        languages: set[str] = set()
        for token in range(self.n_vocab):
            pointer = get_text(self._vocab, token)
            if pointer == self._ffi.NULL:
                continue
            piece = self._ffi.string(pointer).decode("utf-8", "replace")
            if piece.startswith(prefix) and piece.endswith(suffix):
                languages.add(piece[len(prefix) : -len(suffix)])
        return sorted(languages)

    def get_capabilities(
        self,
        projector_path: str | os.PathLike[str] | None = None,
        *,
        discover_projector: bool = True,
    ) -> dict[str, Any]:
        """Inspect GGUF and native llama.cpp/mtmd capabilities.

        Supplying a projector path performs full native compatibility checking.
        Automatic discovery follows the same conventions as multimodal chat.
        """
        metadata = self._model_metadata()
        has_encoder_fn = getattr(self._lib, "llama_model_has_encoder", None)
        has_decoder_fn = getattr(self._lib, "llama_model_has_decoder", None)
        result: dict[str, Any] = {
            "text_generation": bool(has_decoder_fn(self._model)) if has_decoder_fn else True,
            "embeddings": bool(self._embedding),
            "encoder": bool(has_encoder_fn(self._model)) if has_encoder_fn else None,
            "decoder": bool(has_decoder_fn(self._model)) if has_decoder_fn else None,
            "modalities": ["text"],
            "projector_path": None,
            "input_audio": False,
            "audio_generation": False,
            "supported_languages": self._native_audio_languages(),
            "preset_timbres": [],
            "speaker_references": False,
            "generation_options": [],
            "metadata": metadata,
        }
        try:
            from llama_cpp_py_sync.multimodal import MultimodalContext

            with MultimodalContext(
                self,
                projector_path,
                discover_projector=discover_projector,
                warmup=False,
            ) as context:
                result.update(context.capabilities)
                result["supported_languages"] = self._native_audio_languages()
                result["speaker_references"] = bool(result.get("audio_generation"))
        except FileNotFoundError:
            if projector_path is not None:
                raise
        return result

    @property
    def n_layer(self) -> int:
        """Get number of layers."""
        return self._lib.llama_n_layer(self._model)

    @property
    def bos_token(self) -> int:
        """Get beginning-of-sequence token ID."""
        if self._vocab is None:
            raise RuntimeError("Vocab handle is not available; bindings may be out of sync")
        fn = getattr(self._lib, "llama_vocab_bos", None) or getattr(self._lib, "llama_token_bos", None)
        if fn is None:
            raise RuntimeError("No BOS token API available in llama library")
        return int(fn(self._vocab))

    @property
    def eos_token(self) -> int:
        """Get end-of-sequence token ID."""
        if self._vocab is None:
            raise RuntimeError("Vocab handle is not available; bindings may be out of sync")
        fn = getattr(self._lib, "llama_vocab_eos", None) or getattr(self._lib, "llama_token_eos", None)
        if fn is None:
            raise RuntimeError("No EOS token API available in llama library")
        return int(fn(self._vocab))

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False
    ) -> list[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Text to tokenize.
            add_special: Whether to add special tokens (BOS, etc.).
            parse_special: Whether to parse special tokens in text.

        Returns:
            List of token IDs.
        """
        text_bytes = text.encode("utf-8")
        max_tokens = len(text_bytes) + 16

        tokens = self._ffi.new(f"llama_token[{max_tokens}]")

        n_tokens = self._lib.llama_tokenize(
            self._vocab,
            text_bytes,
            len(text_bytes),
            tokens,
            max_tokens,
            add_special,
            parse_special
        )

        if n_tokens < 0:
            max_tokens = -n_tokens
            tokens = self._ffi.new(f"llama_token[{max_tokens}]")
            n_tokens = self._lib.llama_tokenize(
                self._vocab,
                text_bytes,
                len(text_bytes),
                tokens,
                max_tokens,
                add_special,
                parse_special
            )

        return [tokens[i] for i in range(n_tokens)]

    def detokenize(
        self,
        tokens: list[int],
        remove_special: bool = False,
        unparse_special: bool = True
    ) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs.
            remove_special: Whether to remove special tokens.
            unparse_special: Whether to render special tokens as text.

        Returns:
            Decoded text string.
        """
        if not tokens:
            return ""

        tokens_arr = self._ffi.new(f"llama_token[{len(tokens)}]")
        for i, tok in enumerate(tokens):
            tokens_arr[i] = tok

        buf_size = len(tokens) * 16
        buf = self._ffi.new(f"char[{buf_size}]")

        n_chars = self._lib.llama_detokenize(
            self._vocab,
            tokens_arr,
            len(tokens),
            buf,
            buf_size,
            remove_special,
            unparse_special
        )

        if n_chars < 0:
            buf_size = -n_chars
            buf = self._ffi.new(f"char[{buf_size}]")
            n_chars = self._lib.llama_detokenize(
                self._vocab,
                tokens_arr,
                len(tokens),
                buf,
                buf_size,
                remove_special,
                unparse_special
            )

        return self._ffi.string(buf, n_chars).decode("utf-8", errors="replace")

    def token_to_piece(self, token: int) -> str:
        """Convert a single token to its string representation."""
        buf = self._ffi.new("char[128]")
        n = self._lib.llama_token_to_piece(self._vocab, token, buf, 128, 0, False)
        if n < 0:
            return ""
        return self._ffi.string(buf, n).decode("utf-8", errors="replace")

    def _eval_tokens(self, tokens: list[int], n_past: int) -> int:
        """Evaluate tokens and update the context.

        The batch is split into ``n_batch``-wide chunks. llama_decode asserts
        (aborting the process, not raising) when handed more tokens than the
        context's ``n_batch``, so a long prompt must never be submitted whole.
        Logits are requested only for the very last token, which is all the
        sampler needs.
        """
        if not tokens:
            return n_past

        total = len(tokens)
        offset = 0

        while offset < total:
            chunk = tokens[offset : offset + self._n_batch]
            is_last_chunk = offset + len(chunk) >= total
            batch = self._lib.llama_batch_init(len(chunk), 0, 1)

            try:
                batch.n_tokens = len(chunk)
                for i, token in enumerate(chunk):
                    batch.token[i] = token
                    batch.pos[i] = n_past + offset + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = 0

                if is_last_chunk:
                    batch.logits[len(chunk) - 1] = 1

                result = self._lib.llama_decode(self._ctx, batch)
                if result != 0:
                    raise RuntimeError(f"llama_decode failed with code {result}")
            finally:
                self._lib.llama_batch_free(batch)

            offset += len(chunk)

        return n_past + total

    def _sample_token(self) -> int:
        """Sample the next token from the model's output."""
        return self._lib.llama_sampler_sample(self._sampler, self._ctx, -1)

    def _clear_context_state(self) -> None:
        """Clear KV-cache / memory state so a new prompt can start at position 0."""
        clear_fn = getattr(self._lib, "llama_kv_cache_clear", None)
        if clear_fn is not None:
            clear_fn(self._ctx)
            return

        # Newer llama.cpp exposes KV-cache as a "memory module".
        get_mem = getattr(self._lib, "llama_get_memory", None)
        mem_clear = getattr(self._lib, "llama_memory_clear", None)
        if get_mem is not None and mem_clear is not None:
            mem = get_mem(self._ctx)
            if mem != self._ffi.NULL:
                mem_clear(mem, True)
                return

    def _configure_generation_sampler(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repeat_penalty: float,
        repeat_last_n: int,
        seed: int | None,
    ) -> None:
        """Create the sampler chain used by both text and multimodal requests."""
        if hasattr(self, "_sampler") and self._sampler is not None:
            self._lib.llama_sampler_free(self._sampler)

        sampler_params = self._lib.llama_sampler_chain_default_params()
        self._sampler = self._lib.llama_sampler_chain_init(sampler_params)
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_penalties(
                self.n_vocab,
                repeat_last_n,
                repeat_penalty,
                0.0,
                0.0,
            ),
        )
        self._lib.llama_sampler_chain_add(
            self._sampler, self._lib.llama_sampler_init_top_k(top_k)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler, self._lib.llama_sampler_init_top_p(top_p, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler, self._lib.llama_sampler_init_min_p(min_p, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler, self._lib.llama_sampler_init_temp(temperature)
        )
        dist_seed = (
            int.from_bytes(os.urandom(4), "little")
            if seed is None
            else (int(seed) & 0xFFFFFFFF)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler, self._lib.llama_sampler_init_dist(dist_seed)
        )

    def _set_abort_callback(self, cancel_callback: Callable[[], bool] | None) -> None:
        """Connect cancellation to llama.cpp's native decode abort hook."""
        setter = getattr(self._lib, "llama_set_abort_callback", None)
        if setter is None:
            return
        self._abort_callback = None
        if cancel_callback is None:
            setter(self._ctx, self._ffi.NULL, self._ffi.NULL)
            return

        def _abort(_data: Any) -> bool:
            try:
                return bool(cancel_callback())
            except BaseException:
                # Never let a Python exception cross the C callback boundary;
                # aborting is the safe outcome for a failed cancellation hook.
                return True

        self._abort_callback = self._ffi.callback(
            "ggml_abort_callback", _abort
        )
        setter(self._ctx, self._abort_callback, self._ffi.NULL)

    def _generate_from_n_past(
        self,
        n_past: int,
        max_tokens: int,
        stop_sequences: list[str] | None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        generated_text = ""
        for _ in range(max_tokens):
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("Generation was cancelled")

            new_token = self._sample_token()
            if self._lib.llama_vocab_is_eog(self._vocab, new_token):
                break

            self._lib.llama_sampler_accept(self._sampler, new_token)
            piece = self.token_to_piece(new_token)
            generated_text += piece

            if stop_sequences:
                stopped_at: int | None = None
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        stopped_at = generated_text.find(stop_seq)
                        generated_text = generated_text[:stopped_at]
                        break
                if stopped_at is not None:
                    yield piece
                    break

            yield piece
            n_past = self._eval_tokens([new_token], n_past)

    def _format_chat_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        multimodal_context: Any = None,
    ) -> tuple[str, list[tuple[str, bytes]]]:
        """Format ordered content parts while retaining mtmd markers and images."""
        if not messages:
            raise ValueError("messages must not be empty")

        marker = multimodal_context.marker if multimodal_context is not None else ""
        rendered: list[tuple[str, str]] = []
        images: list[tuple[str, bytes]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("Every chat message must be an object")
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("Every chat message requires a non-empty role")
            content = message.get("content", "")
            if multimodal_context is None:
                if isinstance(content, str):
                    text = content
                elif isinstance(content, Sequence) and not isinstance(
                    content, (bytes, bytearray, memoryview)
                ):
                    text_parts: list[str] = []
                    for part in content:
                        if not isinstance(part, Mapping) or part.get("type") != "text":
                            raise ValueError(
                                "Image content requires multimodal_context; request was not downgraded"
                            )
                        text = part.get("text")
                        if not isinstance(text, str):
                            raise ValueError("Text content parts require a string text field")
                        text_parts.append(text)
                    text = "".join(text_parts)
                else:
                    raise TypeError("Message content must be text or an ordered content array")
            else:
                from llama_cpp_py_sync.multimodal import _normalise_content

                text, payloads = _normalise_content(content, multimodal_context.limits, marker)
                images.extend((payload.mime_type, payload.data) for payload in payloads)
            rendered.append((role, text))

        message_array = self._ffi.new("llama_chat_message[]", len(rendered))
        role_buffers: list[Any] = []
        content_buffers: list[Any] = []
        for index, (role, content_text) in enumerate(rendered):
            role_bytes = role.encode("utf-8")
            content_bytes = content_text.encode("utf-8")
            role_buffer = self._ffi.new("char[]", role_bytes)
            content_buffer = self._ffi.new("char[]", content_bytes)
            role_buffers.append(role_buffer)
            content_buffers.append(content_buffer)
            message_array[index].role = role_buffer
            message_array[index].content = content_buffer

        template = None
        if hasattr(self._lib, "llama_model_chat_template"):
            template = self._lib.llama_model_chat_template(self._model, self._ffi.NULL)
        if template not in (None, self._ffi.NULL):
            total_chars = sum(len(item[1].encode("utf-8")) for item in rendered)
            size = max(256, total_chars * 2 + 256)
            output = self._ffi.new(f"char[{size}]")
            result = self._lib.llama_chat_apply_template(
                template, message_array, len(rendered), True, output, size
            )
            if result < 0:
                size = -int(result)
                output = self._ffi.new(f"char[{size}]")
                result = self._lib.llama_chat_apply_template(
                    template, message_array, len(rendered), True, output, size
                )
            if result >= 0:
                return self._ffi.string(output, result).decode("utf-8", "replace"), images

        # Models without a built-in template still get deterministic role
        # boundaries.  The exact content-part order is retained above.
        fallback = "".join(f"{role}: {content}\n" for role, content in rendered)
        return fallback + "assistant:", images

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        seed: int | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> str | Iterator[str]:
        """
        Generate text completion for a prompt.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            min_p: Min-p sampling parameter.
            repeat_penalty: Repetition penalty (passed to ``llama_sampler_init_penalties``).
            repeat_last_n: Token window for repetition penalty.
            stop_sequences: List of strings that stop generation.
            stream: If True, return an iterator yielding tokens.
            seed: RNG seed for the final ``dist`` sampler; ``None`` for non-deterministic.

        Returns:
            Generated text (or iterator if stream=True).
        """
        self._clear_context_state()
        self._configure_generation_sampler(
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            seed,
        )
        tokens = self.tokenize(prompt, add_special=True)
        if len(tokens) >= self._n_ctx:
            raise ValueError(f"Prompt too long: {len(tokens)} tokens exceeds context size {self._n_ctx}")
        self._set_abort_callback(cancel_callback)

        def _generate_tokens() -> Iterator[str]:
            try:
                n_past = self._eval_tokens(tokens, 0)
                yield from self._generate_from_n_past(
                    n_past, max_tokens, stop_sequences, cancel_callback
                )
            finally:
                self._set_abort_callback(None)

        if stream:
            return _generate_tokens()
        else:
            return "".join(_generate_tokens())

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        seed: int | None = None,
        multimodal_context: Any = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Create a chat completion with ordered text/image content parts.

        Images are accepted only when ``multimodal_context`` is supplied; a
        failed multimodal request is never silently converted to text-only
        inference.
        """
        if multimodal_context is not None and getattr(multimodal_context, "model", None) is not self:
            raise ValueError("multimodal_context belongs to a different Llama model")

        def _run() -> Iterator[str]:
            self._clear_context_state()
            self._configure_generation_sampler(
                temperature,
                top_k,
                top_p,
                min_p,
                repeat_penalty,
                repeat_last_n,
                seed,
            )
            prompt, images = self._format_chat_prompt(messages, multimodal_context)
            self._set_abort_callback(cancel_callback)
            try:
                if multimodal_context is None:
                    tokens = self.tokenize(prompt, add_special=True)
                    if len(tokens) >= self._n_ctx:
                        raise ValueError(
                            f"Prompt too long: {len(tokens)} tokens exceeds context size {self._n_ctx}"
                        )
                    n_past = self._eval_tokens(tokens, 0)
                else:
                    with multimodal_context.tokenize_prompt(
                        prompt,
                        images,
                        cancel_callback=cancel_callback,
                    ) as tokenized:
                        n_past = multimodal_context.evaluate_prompt(
                            tokenized, cancel_callback=cancel_callback
                        )
                if n_past + max_tokens >= self._n_ctx:
                    raise ValueError("Chat request exceeds the model context window")
                yield from self._generate_from_n_past(
                    n_past, max_tokens, stop_sequences, cancel_callback
                )
            finally:
                self._set_abort_callback(None)

        if not stream:
            content = "".join(_run())
            return {
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }

        def _stream() -> Iterator[dict[str, Any]]:
            first = True
            for piece in _run():
                yield {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if first else None,
                                "content": piece,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                first = False
            yield {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        return _stream()

    def create_multimodal_chat_completion(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
        """Explicit alias for :meth:`create_chat_completion` with multimodal messages."""
        return self.create_chat_completion(messages, **kwargs)

    def transcribe(
        self,
        audio: str | os.PathLike[str] | bytes | bytearray | memoryview,
        *,
        projector_path: str | os.PathLike[str] | None = None,
        discover_projector: bool = True,
        prompt: str | None = None,
        language: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> str:
        """Transcribe encoded audio using an upstream audio-input projector."""
        from llama_cpp_py_sync.multimodal import (
            MultimodalCancelledError,
            MultimodalContext,
        )

        if not isinstance(prompt, (str, type(None))):
            raise TypeError("prompt must be a string or None")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("language must be a non-empty string")
        request = prompt if prompt else "Transcribe the audio."
        if language:
            request += f" (language: {language.strip()})"
        self._clear_context_state()
        self._configure_generation_sampler(
            temperature, top_k, top_p, 0.0, 1.0, 0, seed
        )
        self._set_abort_callback(cancel_callback)
        try:
            with MultimodalContext(
                self,
                projector_path,
                discover_projector=discover_projector,
                cancel_callback=cancel_callback,
            ) as context:
                if not context.supports_audio:
                    raise RuntimeError("The companion projector does not support audio transcription")
                formatted_prompt, _ = self._format_chat_prompt(
                    [{"role": "user", "content": request + context.marker}],
                    None,
                )
                with context.tokenize_audio_prompt(
                    formatted_prompt,
                    audio,
                    cancel_callback=cancel_callback,
                ) as tokenized:
                    n_past = context.evaluate_prompt(
                        tokenized, cancel_callback=cancel_callback
                    )
                if n_past + max_tokens >= self._n_ctx:
                    raise ValueError("Transcription request exceeds the model context window")
                try:
                    return "".join(
                        self._generate_from_n_past(
                            n_past, max_tokens, None, cancel_callback
                        )
                    )
                except RuntimeError as exc:
                    if cancel_callback is not None and cancel_callback():
                        raise MultimodalCancelledError("Audio transcription was cancelled") from exc
                    raise
        finally:
            self._set_abort_callback(None)

    def generate_audio(
        self,
        text: str,
        *,
        projector_path: str | os.PathLike[str] | None = None,
        discover_projector: bool = True,
        language: str | None = None,
        speaker_reference: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        top_k: int = 40,
        top_p: float = 0.95,
        temperature: float = 0.8,
        seed: int | None = None,
        max_frames: int = 512,
        output_format: str = "wav",
        cancel_callback: Callable[[], bool] | None = None,
    ) -> GeneratedAudio:
        """Generate audio through upstream mtmd's in-process TTS helper."""
        from llama_cpp_py_sync.multimodal import (
            MultimodalCancelledError,
            MultimodalContext,
            ProjectorCompatibilityError,
        )

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("language must be a non-empty string")
        if top_k <= 0 or not 0.0 < top_p <= 1.0 or temperature < 0.0:
            raise ValueError("top_k, top_p, or temperature is outside its supported range")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        fmt = output_format.strip().lower()
        if fmt not in {"wav", "pcm"}:
            raise ValueError("output_format must be 'wav' or 'pcm'")

        self._clear_context_state()
        self._configure_generation_sampler(temperature, top_k, top_p, 0.0, 1.0, 0, seed)
        self._set_abort_callback(cancel_callback)
        try:
            with MultimodalContext(
                self,
                projector_path,
                discover_projector=discover_projector,
                cancel_callback=cancel_callback,
            ) as context:
                info = context._lib.mtmd_gen_audio_get_info(context._ctx)
                if int(info.type) == 0:
                    raise ProjectorCompatibilityError(
                        "The companion artifact does not support audio generation"
                    )
                speaker = context._ffi.NULL
                helper = context._ffi.NULL
                try:
                    if speaker_reference is not None:
                        speaker = context.create_audio_bitmap(speaker_reference)
                    helper = context._lib.mtmd_helper_gen_audio_init(self._ctx, context._ctx)
                    if helper == context._ffi.NULL:
                        raise RuntimeError("Could not initialize native audio generation")
                    prompt_bytes = text.encode("utf-8")
                    prompt_buf = context._ffi.new("char[]", prompt_bytes)
                    language_buf = (
                        context._ffi.NULL
                        if language is None
                        else context._ffi.new("char[]", language.strip().encode("utf-8"))
                    )
                    inp = context._ffi.new("struct mtmd_helper_gen_audio_inp *")
                    inp.seq_id = 0
                    inp.prompt = prompt_buf
                    inp.prompt_len = len(prompt_bytes)
                    inp.speaker_ref = speaker
                    inp.lang = language_buf
                    inp.top_k = int(top_k)
                    inp.top_p = float(top_p)
                    inp.seed = 0xFFFFFFFF if seed is None else int(seed) & 0xFFFFFFFF
                    inp.out_type = 1 if fmt == "wav" else 0
                    if context._lib.mtmd_helper_gen_audio_set_input(helper, inp) != 0:
                        raise ValueError("Native audio generation rejected the requested options")
                    while True:
                        if cancel_callback is not None and cancel_callback():
                            raise MultimodalCancelledError("Audio generation was cancelled")
                        remaining = int(
                            context._lib.mtmd_helper_gen_audio_step_prompt(helper, self._n_batch)
                        )
                        if remaining < 0:
                            raise RuntimeError("Native audio prompt processing failed")
                        if remaining == 0:
                            break

                    sampled = self._sample_token()
                    h_state = self._lib.llama_get_embeddings_ith(self._ctx, -1)
                    if h_state == self._ffi.NULL:
                        raise RuntimeError("Native model did not provide an audio generation state")
                    for _ in range(max_frames):
                        if cancel_callback is not None and cancel_callback():
                            raise MultimodalCancelledError("Audio generation was cancelled")
                        next_state = context._ffi.new("const float **")
                        stopped = context._ffi.new("bool *")
                        code = context._lib.mtmd_helper_gen_audio_step_gen(
                            helper, sampled, h_state, next_state, stopped
                        )
                        if code != 0:
                            raise RuntimeError("Native audio frame generation failed")
                        if bool(stopped[0]) or next_state[0] == context._ffi.NULL:
                            break
                        h_state = next_state[0]
                        self._lib.llama_sampler_accept(self._sampler, sampled)
                        sampled = self._sample_token()

                    sample_rate = context._ffi.new("int32_t *")
                    output = context._ffi.new("const char **")
                    output_len = context._ffi.new("size_t *")
                    n_samples = context._ffi.new("int64_t *")
                    if context._lib.mtmd_helper_gen_audio_get_output(
                        helper, sample_rate, output, output_len, n_samples
                    ) != 0:
                        raise RuntimeError("Native audio output encoding failed")
                    data = bytes(context._ffi.buffer(output[0], int(output_len[0])))
                    return GeneratedAudio(data, int(sample_rate[0]), fmt, int(n_samples[0]))
                finally:
                    if helper != context._ffi.NULL:
                        context._lib.mtmd_helper_gen_audio_free(helper)
                    if speaker != context._ffi.NULL:
                        context._lib.mtmd_bitmap_free(speaker)
        finally:
            self._set_abort_callback(None)

    def get_embeddings(self, text: str) -> list[float]:
        """
        Get embeddings for input text.

        Args:
            text: Input text to embed.

        Returns:
            List of embedding floats.

        Note:
            Model must be loaded with embedding=True for this to work properly.
        """
        if not self._embedding:
            raise RuntimeError("Model was not loaded with embedding=True")

        self._clear_context_state()

        tokens = self.tokenize(text, add_special=True)

        # Pooled embeddings need the whole sequence in a single encode, so this
        # batch cannot be chunked the way _eval_tokens is. Fail with a clear
        # message instead of letting llama_encode abort the process.
        if len(tokens) > self._n_batch:
            raise ValueError(
                f"Input is {len(tokens)} tokens but n_batch is {self._n_batch}; "
                f"load the model with n_batch >= {len(tokens)} to embed this text."
            )

        batch = self._lib.llama_batch_init(len(tokens), 0, 1)
        try:
            batch.n_tokens = len(tokens)
            for i, token in enumerate(tokens):
                batch.token[i] = token
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 0

            batch.logits[len(tokens) - 1] = 1

            has_encoder = getattr(self._lib, "llama_model_has_encoder", None)
            evaluate = (
                self._lib.llama_encode
                if has_encoder is not None and bool(has_encoder(self._model))
                else self._lib.llama_decode
            )
            result = evaluate(self._ctx, batch)
            if result != 0:
                operation = "llama_encode" if evaluate == self._lib.llama_encode else "llama_decode"
                raise RuntimeError(f"{operation} failed with code {result}")

            embd_ptr = self._lib.llama_get_embeddings_seq(self._ctx, 0)
            if embd_ptr == self._ffi.NULL:
                embd_ptr = self._lib.llama_get_embeddings(self._ctx)

            if embd_ptr == self._ffi.NULL:
                raise RuntimeError("Failed to get embeddings")

            n_embd = self.n_embd
            return [embd_ptr[i] for i in range(n_embd)]
        finally:
            self._lib.llama_batch_free(batch)

    def get_model_desc(self) -> str:
        """Get model description string."""
        buf = self._ffi.new("char[256]")
        self._lib.llama_model_desc(self._model, buf, 256)
        return self._ffi.string(buf).decode("utf-8")

    def get_model_size(self) -> int:
        """Get model size in bytes."""
        return self._lib.llama_model_size(self._model)

    def get_model_n_params(self) -> int:
        """Get number of model parameters."""
        return self._lib.llama_model_n_params(self._model)

    @staticmethod
    def print_system_info() -> str:
        """Print llama.cpp system info."""
        lib = get_lib()
        ffi = get_ffi()
        info = lib.llama_print_system_info()
        return ffi.string(info).decode("utf-8")
