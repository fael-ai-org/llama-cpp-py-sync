"""
High-level Llama class for easy model interaction.

This provides a thin wrapper around the llama.cpp C API for common operations
like loading models, tokenizing text, and generating completions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

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
            use_mmap: Whether to use memory mapping for model loading.
            use_mlock: Whether to lock model in memory.
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

        self._lib = get_lib()
        self._ffi = get_ffi()
        self._model = None
        self._ctx = None
        self._sampler = None
        self._vocab = None
        self._verbose = verbose
        self._embedding = embedding
        self._n_ctx = n_ctx
        # Prompt evaluation must be chunked to this width: llama_decode asserts
        # on any batch wider than the context's n_batch.
        self._n_batch = max(1, int(n_batch))

        self._lib.llama_backend_init()

        model_params = self._lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        model_params.use_mmap = use_mmap
        model_params.use_mlock = use_mlock

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
        ctx_params.embeddings = embedding
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
        self._lib.llama_sampler_reset(self._sampler)

        if hasattr(self, "_sampler") and self._sampler is not None:
            self._lib.llama_sampler_free(self._sampler)

        sampler_params = self._lib.llama_sampler_chain_default_params()
        self._sampler = self._lib.llama_sampler_chain_init(sampler_params)

        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_penalties(
                repeat_last_n,
                repeat_penalty,
                0.0,
                0.0,
            ),
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_top_k(top_k)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_top_p(top_p, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_min_p(min_p, 1)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_temp(temperature)
        )
        dist_seed = (
            int.from_bytes(os.urandom(4), "little")
            if seed is None
            else (int(seed) & 0xFFFFFFFF)
        )
        self._lib.llama_sampler_chain_add(
            self._sampler,
            self._lib.llama_sampler_init_dist(dist_seed),
        )

        tokens = self.tokenize(prompt, add_special=True)

        if len(tokens) >= self._n_ctx:
            raise ValueError(f"Prompt too long: {len(tokens)} tokens exceeds context size {self._n_ctx}")

        def _generate_tokens():
            n_past = 0
            generated_text = ""

            n_past = self._eval_tokens(tokens, n_past)

            for _ in range(max_tokens):
                new_token = self._sample_token()

                if new_token == self.eos_token:
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

        if stream:
            return _generate_tokens()
        else:
            return "".join(_generate_tokens())

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

            result = self._lib.llama_encode(self._ctx, batch)
            if result != 0:
                raise RuntimeError(f"llama_encode failed with code {result}")

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
