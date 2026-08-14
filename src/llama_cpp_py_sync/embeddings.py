"""
Embedding utilities for llama-cpp-py-sync.

Provides convenient functions for generating embeddings from text using
llama.cpp models that support embedding generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from llama_cpp_py_sync.llama import Llama

NORMALIZATION_MODES = {
    "none": -1,
    "max_abs": 0,
    "max-abs": 0,
    "taxicab": 1,
    "l1": 1,
    "euclidean": 2,
    "l2": 2,
}


def _resolve_normalization_mode(mode: bool | int | str | None) -> int:
    if mode is None or mode is False:
        return -1
    if mode is True:
        return 2
    if isinstance(mode, str):
        value = mode.strip().lower()
        if value.startswith("p-") or value.startswith("p:"):
            value = value[2:]
        if value not in NORMALIZATION_MODES:
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    "Unknown embedding normalization mode; use none, max_abs, "
                    "taxicab, euclidean, or an integer p-norm"
                ) from exc
        return NORMALIZATION_MODES[value]
    return int(mode)


def normalize_embedding(
    embedding: Sequence[float], mode: bool | int | str | None = 2
) -> list[float]:
    """
    Normalize an embedding vector to unit length.

    Args:
        embedding: Raw embedding vector.
        mode: llama.cpp's common embedding normalization mode. ``-1``/``none``
            leaves values unchanged, ``0`` is max-absolute int16 scaling,
            ``1`` is taxicab/L1, ``2`` is Euclidean/L2, and values greater than
            2 are p-norms.

    Returns:
        Normalized embedding vector.
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("embedding must be a one-dimensional vector")

    resolved = _resolve_normalization_mode(mode)
    if resolved == -1:
        return arr.tolist()
    if resolved == 0:
        norm = (float(np.max(np.abs(arr))) if arr.size else 0.0) / 32760.0
    elif resolved == 1:
        norm = float(np.sum(np.abs(arr), dtype=np.float64))
    elif resolved == 2:
        norm = float(np.linalg.norm(arr))
    elif resolved > 2:
        norm = float(np.sum(np.abs(arr) ** resolved, dtype=np.float64) ** (1.0 / resolved))
    else:
        raise ValueError("embedding normalization mode must be -1, 0, 1, 2, or greater than 2")
    if norm > 0.0:
        arr = arr / norm
    return arr.tolist()


def normalize_embeddings(embeddings: Sequence[Sequence[float]], mode: bool | int | str | None = 2) -> list[list[float]]:
    """Normalize a batch of embedding vectors using the same mode for each row."""
    return [normalize_embedding(embedding, mode=mode) for embedding in embeddings]


def get_embeddings(
    model: str | Llama,
    text: str,
    normalize: bool | int | str | None = True,
    n_ctx: int = 512,
    n_batch: int = 512,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
    offload_kqv: bool | None = None,
    op_offload: bool | None = None,
    n_ubatch: int | None = None,
    n_threads_batch: int | None = None,
    pooling_type: int | str | None = None,
    per_token: bool = False,
    n_seq_max: int | None = None,
    ) -> Any:
    """
    Get embeddings for a single text string.

    Args:
        model: Either a path to a GGUF model file or an existing Llama instance.
        text: Text to embed.
        normalize: Normalization mode. ``True``/``"l2"`` uses Euclidean
            normalization; ``False``/``"none"`` returns native values.
        n_ctx: Context size (only used if model is a path).
        n_batch: Max decode batch size (only used if model is a path).
        n_threads: Generation threads (only used if model is a path).
        n_gpu_layers: GPU layers (only used if model is a path).
        offload_kqv: Whether to offload KQV and KV-cache operations (only used
            if model is a path).
        op_offload: Whether to offload host tensor operations (only used if
            model is a path).
        n_ubatch: Microbatch size (only used if model is a path).
        n_threads_batch: Batch/prompt threads (only used if model is a path).
        pooling_type: Native llama.cpp pooling type, such as ``"mean"`` or
            ``"none"`` (only used if model is a path).
        per_token: Return one vector per token. Requires ``pooling_type="none"``.
        n_seq_max: Maximum native sequence count (only used if model is a path).

    Returns:
        Embedding vector as a list of floats.

    Example:
        >>> emb = get_embeddings("model.gguf", "Hello, world!")
        >>> print(len(emb))
        4096
    """
    if isinstance(model, str):
        with Llama(
            model,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            offload_kqv=offload_kqv,
            op_offload=op_offload,
            n_ubatch=n_ubatch,
            n_threads_batch=n_threads_batch,
            embedding=True,
            pooling_type="none" if per_token and pooling_type is None else pooling_type,
            n_seq_max=n_seq_max or (1 if not per_token else 1),
        ) as llm:
            embedding = llm.get_embeddings(text, normalize=normalize, per_token=per_token)
    else:
        embedding = model.get_embeddings(text, normalize=normalize, per_token=per_token)

    return embedding


def get_embeddings_batch(
    model: str | Llama,
    texts: list[str],
    normalize: bool | int | str | None = True,
    n_ctx: int = 512,
    n_batch: int = 512,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
    offload_kqv: bool | None = None,
    op_offload: bool | None = None,
    n_ubatch: int | None = None,
    n_threads_batch: int | None = None,
    pooling_type: int | str | None = None,
    per_token: bool = False,
    n_seq_max: int | None = None,
    ) -> list[Any]:
    """
    Get embeddings for multiple text strings.

    Args:
        model: Either a path to a GGUF model file or an existing Llama instance.
        texts: List of texts to embed.
        normalize: Normalization mode applied row-wise.
        n_ctx: Context size (only used if model is a path).
        n_batch: Max decode batch size (only used if model is a path).
        n_threads: Generation threads (only used if model is a path).
        n_gpu_layers: GPU layers (only used if model is a path).
        offload_kqv: Whether to offload KQV and KV-cache operations (only used
            if model is a path).
        op_offload: Whether to offload host tensor operations (only used if
            model is a path).
        n_ubatch: Microbatch size (only used if model is a path).
        n_threads_batch: Batch/prompt threads (only used if model is a path).
        pooling_type: Native llama.cpp pooling type, such as ``"mean"`` or
            ``"none"`` (only used if model is a path).
        per_token: Return one vector per token for each input.
        n_seq_max: Maximum native sequence count (only used if model is a path).

    Returns:
        List of embedding vectors.

    Example:
        >>> embs = get_embeddings_batch("model.gguf", ["Hello", "World"])
        >>> print(len(embs))
        2
    """
    if isinstance(model, str):
        with Llama(
            model,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            offload_kqv=offload_kqv,
            op_offload=op_offload,
            n_ubatch=n_ubatch,
            n_threads_batch=n_threads_batch,
            embedding=True,
            pooling_type="none" if per_token and pooling_type is None else pooling_type,
            n_seq_max=n_seq_max or max(1, len(texts)),
        ) as llm:
            embeddings = llm.get_embeddings_batch(
                texts, normalize=normalize, per_token=per_token
            )
    else:
        embeddings = model.get_embeddings_batch(
            texts, normalize=normalize, per_token=per_token
        )

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)

    dot_product = np.dot(arr_a, arr_b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """
    Compute Euclidean distance between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Euclidean distance (lower = more similar).
    """
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(arr_a - arr_b))
