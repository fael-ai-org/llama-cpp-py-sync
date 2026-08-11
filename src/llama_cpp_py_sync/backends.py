"""
Backend detection and information for llama-cpp-py-sync.

Provides utilities to check which hardware backends are available
in the compiled llama.cpp library.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib


@dataclass
class BackendInfo:
    """Information about available backends."""
    cuda: bool = False
    metal: bool = False
    vulkan: bool = False
    rocm: bool = False
    blas: bool = False
    gpu_offload: bool = False
    mmap: bool = False
    mlock: bool = False
    max_devices: int = 0
    system_info: str = ""


def _check_backend_from_system_info(system_info: str) -> dict[str, bool]:
    """Parse system info string to detect backends."""
    info_lower = system_info.lower()

    return {
        "cuda": "cuda" in info_lower or "cublas" in info_lower,
        "metal": "metal" in info_lower,
        "vulkan": "vulkan" in info_lower,
        "rocm": "rocm" in info_lower or "hip" in info_lower,
        "blas": "blas" in info_lower or "openblas" in info_lower or "accelerate" in info_lower,
    }


def _check_bundled_backend_libraries() -> dict[str, bool]:
    """Detect packaged backends when upstream system info only lists CPU features."""
    library_names = [path.name.lower() for path in Path(__file__).parent.iterdir()]

    def has_fragment(*fragments: str) -> bool:
        return any(any(fragment in name for fragment in fragments) for name in library_names)

    return {
        "cuda": has_fragment("ggml-cuda"),
        "metal": has_fragment("ggml-metal"),
        "vulkan": has_fragment("ggml-vulkan"),
        "rocm": has_fragment("ggml-hip", "ggml-rocm"),
    }


def get_backend_info() -> BackendInfo:
    """
    Get comprehensive information about available backends.

    Returns:
        BackendInfo object with details about available backends.

    Example:
        >>> info = get_backend_info()
        >>> print(f"CUDA available: {info.cuda}")
        >>> print(f"Metal available: {info.metal}")
    """
    lib = get_lib()
    ffi = get_ffi()

    system_info_ptr = lib.llama_print_system_info()
    system_info = ffi.string(system_info_ptr).decode("utf-8")

    backends = _check_backend_from_system_info(system_info)
    packaged_backends = _check_bundled_backend_libraries()
    for backend, available in packaged_backends.items():
        backends[backend] = backends[backend] or available

    return BackendInfo(
        cuda=backends["cuda"],
        metal=backends["metal"],
        vulkan=backends["vulkan"],
        rocm=backends["rocm"],
        blas=backends["blas"],
        gpu_offload=lib.llama_supports_gpu_offload(),
        mmap=lib.llama_supports_mmap(),
        mlock=lib.llama_supports_mlock(),
        max_devices=lib.llama_max_devices(),
        system_info=system_info,
    )


def get_available_backends() -> list[str]:
    """
    Get a list of available backend names.

    Returns:
        List of backend name strings (e.g., ["cuda", "blas"]).

    Example:
        >>> backends = get_available_backends()
        >>> print(backends)
        ['cuda', 'blas']
    """
    info = get_backend_info()
    backends = []

    if info.cuda:
        backends.append("cuda")
    if info.metal:
        backends.append("metal")
    if info.vulkan:
        backends.append("vulkan")
    if info.rocm:
        backends.append("rocm")
    if info.blas:
        backends.append("blas")

    if not backends:
        backends.append("cpu")

    return backends


def is_cuda_available() -> bool:
    """Check if CUDA backend is available."""
    return get_backend_info().cuda


def is_metal_available() -> bool:
    """Check if Metal backend is available (macOS only)."""
    return get_backend_info().metal


def is_vulkan_available() -> bool:
    """Check if Vulkan backend is available."""
    return get_backend_info().vulkan


def is_rocm_available() -> bool:
    """Check if ROCm backend is available (AMD GPUs)."""
    return get_backend_info().rocm


def is_blas_available() -> bool:
    """Check if BLAS backend is available."""
    return get_backend_info().blas


def is_gpu_available() -> bool:
    """Check if any GPU backend is available."""
    info = get_backend_info()
    return info.cuda or info.metal or info.vulkan or info.rocm


def get_recommended_gpu_layers(model_size_gb: float) -> int:
    """
    Get recommended number of GPU layers based on model size.

    This is a rough heuristic and may need adjustment based on
    actual GPU memory available.

    Args:
        model_size_gb: Approximate model size in gigabytes.

    Returns:
        Recommended number of GPU layers.
    """
    if not is_gpu_available():
        return 0

    get_backend_info()

    if model_size_gb <= 4:
        return 35
    elif model_size_gb <= 8:
        return 28
    elif model_size_gb <= 16:
        return 20
    else:
        return 10


def print_backend_info():
    """Print formatted backend information to stdout."""
    info = get_backend_info()

    print("=" * 60)
    print("llama-cpp-py-sync Backend Information")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()
    print("Available Backends:")
    print(f"  CUDA:     {'✓' if info.cuda else '✗'}")
    print(f"  Metal:    {'✓' if info.metal else '✗'}")
    print(f"  Vulkan:   {'✓' if info.vulkan else '✗'}")
    print(f"  ROCm:     {'✓' if info.rocm else '✗'}")
    print(f"  BLAS:     {'✓' if info.blas else '✗'}")
    print()
    print("Capabilities:")
    print(f"  GPU Offload:  {'✓' if info.gpu_offload else '✗'}")
    print(f"  Memory Map:   {'✓' if info.mmap else '✗'}")
    print(f"  Memory Lock:  {'✓' if info.mlock else '✗'}")
    print(f"  Max Devices:  {info.max_devices}")
    print()
    print("System Info:")
    print(f"  {info.system_info}")
    print("=" * 60)
