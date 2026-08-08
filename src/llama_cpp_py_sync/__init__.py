"""
llama-cpp-py-sync: Auto-synchronized Python bindings for llama.cpp

This package provides Python bindings for llama.cpp that are automatically
generated from upstream headers using CFFI ABI mode. The bindings stay
synchronized with the latest llama.cpp releases without manual intervention.
"""

from llama_cpp_py_sync._version import __llama_cpp_commit__, __llama_cpp_tag__, __version__
from llama_cpp_py_sync.backends import (
    get_available_backends,
    get_backend_info,
    is_blas_available,
    is_cuda_available,
    is_metal_available,
    is_rocm_available,
    is_vulkan_available,
)
from llama_cpp_py_sync.embeddings import get_embeddings, get_embeddings_batch
from llama_cpp_py_sync.llama import Llama
from llama_cpp_py_sync.multimodal import (
    ImageValidationError,
    MultimodalCancelledError,
    MultimodalContext,
    MultimodalError,
    MultimodalLimits,
    ProjectorCompatibilityError,
)

__all__ = [
    "__version__",
    "__llama_cpp_commit__",
    "__llama_cpp_tag__",
    "Llama",
    "MultimodalContext",
    "MultimodalError",
    "ProjectorCompatibilityError",
    "ImageValidationError",
    "MultimodalCancelledError",
    "MultimodalLimits",
    "get_embeddings",
    "get_embeddings_batch",
    "get_available_backends",
    "is_cuda_available",
    "is_metal_available",
    "is_vulkan_available",
    "is_rocm_available",
    "is_blas_available",
    "get_backend_info",
]
