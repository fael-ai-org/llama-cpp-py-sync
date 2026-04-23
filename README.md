# llama-cpp-py-sync

**Auto-synchronized Python bindings for llama.cpp**

## Supported Backends

| Backend | Platform(s)         | Notes                |
|---------|---------------------|----------------------|
| CPU     | All                 | Default PyPI wheel   |
| CUDA    | Linux, Windows      | CUDA 12.4            |
| Metal   | macOS (Apple Silicon) | MPS/Metal support |
| Vulkan  | All                 | Experimental         |

CPU wheels are published to PyPI. Backend-specific wheels (CUDA, Metal, Vulkan) are published to GitHub Releases.

[![Build Wheels](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/build.yml/badge.svg)](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/build.yml)
[![Sync Upstream](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/sync.yml/badge.svg)](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/sync.yml)
[![Tests](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/test.yml/badge.svg)](https://github.com/FarisZahrani/llama-cpp-py-sync/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/llama-cpp-py-sync.svg)](https://pypi.org/project/llama-cpp-py-sync/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**llama-cpp-py-sync** provides Python bindings for `llama.cpp` that are kept up-to-date automatically. It generates bindings from upstream headers using **CFFI ABI mode**, and ships prebuilt wheels.

### Key Features

- Automatic upstream sync and binding regeneration
- Prebuilt wheels built by CI
- CPU wheels published to PyPI
- Backend-specific wheels (CUDA / Vulkan / Metal) published to GitHub Releases
- CI checks that the generated CFFI surface matches the upstream C API (functions, structs, enums, and signatures)
- A small, explicit Python API (`Llama.generate`, `tokenize`, `get_embeddings`, etc.)

### What You Get (and What You Don’t)

- This project binds to the **public C API** that llama.cpp exposes in `llama.h`.
- It does **not** attempt to bind llama.cpp’s internal C++ implementation such as private headers, C++ classes/templates, or functions that never appear in `llama.h`.
- We use **CFFI ABI mode**: Python loads a prebuilt shared library at runtime (no compiled Python extension module for the bindings).
- Because of that, you still need a compatible llama.cpp shared library available, either bundled in the wheel or via `LLAMA_CPP_LIB`.
- You get a small high-level API (`llama_cpp_py_sync.Llama`) for common tasks, and an “escape hatch” to call the low-level C functions directly via CFFI when needed.

### High-level vs Low-level APIs

- High-level API: `llama_cpp_py_sync.Llama` is the recommended entry point for typical usage such as generation, tokenization, and embeddings.

```python
import llama_cpp_py_sync as llama

with llama.Llama("path/to/model.gguf", n_ctx=2048, n_gpu_layers=0) as llm:
    print(llm.generate("Hello", max_tokens=64))
```

- Low-level API: `llama_cpp_py_sync._cffi_bindings` exposes CFFI access to the underlying llama.cpp C API for advanced use.

```python
from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib

ffi = get_ffi()
lib = get_lib()

print(ffi.string(lib.llama_print_system_info()).decode("utf-8", errors="replace"))
```

## Installation

This project supports **Python 3.8 through 3.14**. CI builds wheels with **Python 3.13.13** for reproducibility; the published wheels are intended to work across supported Python versions.

### From PyPI (Recommended)

```bash
pip install llama-cpp-py-sync
```

This installs the **CPU** wheel.

Note: depending on CI configuration and platform support, additional wheels may also be published to PyPI.

### Quick Chat (Recommended)

After installing from PyPI, you can start an interactive chat session with:

```bash
python -m llama_cpp_py_sync chat
```

If you do not pass `--model` (and `LLAMA_MODEL` is not set), the CLI will prompt before downloading a default GGUF model and cache it locally for future runs.

To auto-download without prompting, pass `--yes`.

One-shot prompt:

```bash
python -m llama_cpp_py_sync chat --prompt "Say 'ok'." --max-tokens 32
```

Use a specific local model:

```bash
python -m llama_cpp_py_sync chat --model path/to/model.gguf
```

### From GitHub Releases (Wheel)

Download the wheel for your platform/backend from GitHub Releases and install the `.whl`:

```bash
pip install path/to/llama_cpp_py_sync-*.whl
```

### From Source

```bash
git clone https://github.com/FarisZahrani/llama-cpp-py-sync.git
cd llama-cpp-py-sync

# Sync upstream llama.cpp
python scripts/sync_upstream.py

# Regenerate CFFI bindings from the synced llama.cpp headers
# (Optional) record the exact llama.cpp commit SHA in the generated file.
python scripts/gen_bindings.py --commit-sha "$(python scripts/sync_upstream.py --sha)"

# Build the shared library
python scripts/build_llama_cpp.py

# Install the package
pip install -e .
```

`vendor/llama.cpp` is cloned locally by `scripts/sync_upstream.py` (and in CI during builds) and is not committed to this repository.

## Quick Start

```python
import llama_cpp_py_sync as llama

# Load a model
llm = llama.Llama("path/to/model.gguf", n_ctx=2048, n_gpu_layers=35)

# Generate text
response = llm.generate("Hello, world!", max_tokens=100)
print(response)

# Streaming generation
for token in llm.generate("Write a poem:", max_tokens=100, stream=True):
    print(token, end="", flush=True)

# Clean up
llm.close()
```

### Using Context Manager

```python
with llama.Llama("model.gguf", n_gpu_layers=35) as llm:
    print(llm.generate("Once upon a time"))
```

### Embeddings

```python
# Load an embedding model
with llama.Llama("embed-model.gguf", embedding=True) as llm:
    emb = llm.get_embeddings("Hello, world!")
    print(f"Embedding dimension: {len(emb)}")
```

### Check Available Backends

```python
from llama_cpp_py_sync import get_available_backends, get_backend_info

print(get_available_backends())  # ['cuda', 'blas'] or similar

info = get_backend_info()
print(f"CUDA available: {info.cuda}")
print(f"Metal available: {info.metal}")
```

<details>
<summary>Full API (click to expand)</summary>

```python
import llama_cpp_py_sync as llama

# Versions
llama.__version__
llama.__llama_cpp_commit__

# Main class
llm = llama.Llama(
    model_path="path/to/model.gguf",
    n_ctx=512,
    n_batch=512,
    n_threads=None,
    n_gpu_layers=0,
    seed=-1,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    embedding=False,
)

text = llm.generate(
    "Hello",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    repeat_penalty=1.1,
    stop_sequences=None,
    stream=False,
)

stream = llm.generate(
    "Hello",
    max_tokens=256,
    stream=True,
)

tokens = llm.tokenize("Hello")
text = llm.detokenize(tokens)
piece = llm.token_to_piece(tokens[0])

llm.get_model_desc()
llm.get_model_size()
llm.get_model_n_params()

# Embeddings (requires embedding=True)
emb = llm.get_embeddings("Hello")

llm.close()

# Module-level embeddings helpers
llama.get_embeddings("path/to/model.gguf", "Hello")
llama.get_embeddings_batch("path/to/model.gguf", ["Hello", "World"])

# Backend helpers
llama.get_available_backends()
llama.get_backend_info()
llama.is_cuda_available()
llama.is_metal_available()
llama.is_vulkan_available()
llama.is_rocm_available()
llama.is_blas_available()
```

</details>

## How It Works

### Automatic Synchronization

1. **Scheduled Checks**: GitHub Actions checks upstream llama.cpp on a schedule
2. **Tag Mirroring**: When an upstream tag exists, the workflow can mirror it into this repository
3. **Wheel Building**: CI builds wheels for all platforms/backends
4. **Release Publishing**: GitHub Releases are created only for tags that exist upstream
5. **PyPI Publishing**: CPU-only wheels are published to PyPI for upstream tags (if configured)

### Bindings Validation (API Surface)

To keep the Python bindings aligned with upstream, CI runs a validation step that compares upstream `llama.h` to the generated CFFI `cdef`.

It checks:

- Public function coverage (missing/extra)
- Struct and enum coverage (missing fields/members)
- Function signatures (return + parameter types)

Local run (after syncing upstream headers):

```bash
python scripts/sync_upstream.py
python scripts/gen_bindings.py --commit-sha "$(python scripts/sync_upstream.py --sha)"
python scripts/validate_cffi_surface.py --check-structs --check-enums --check-signatures
```

### CFFI ABI Mode

Unlike pybind11 or manual ctypes, CFFI ABI mode:

- Reads C declarations directly (no compilation needed for bindings)
- Loads the shared library at runtime via `ffi.dlopen()`
- Automatically handles type conversions
- Works across platforms without modification

### Version Tracking

Check which llama.cpp version you're running:

```python
import llama_cpp_py_sync as llama

print(f"Package version: {llama.__version__}")
print(f"llama.cpp commit: {llama.__llama_cpp_commit__}")
print(f"llama.cpp tag: {getattr(llama, '__llama_cpp_tag__', '')}")
```

## GPU Backend Selection

### Build-time Detection

The build system automatically detects available backends:

| Backend | Platform | Detection |
|---------|----------|-----------|
| CUDA | Linux, Windows | `CUDA_HOME` or `/usr/local/cuda` |
| ROCm | Linux | `ROCM_PATH` or `/opt/rocm` |
| Metal | macOS | Xcode SDK |
| Vulkan | All | `VULKAN_SDK` environment variable |
| BLAS | All | OpenBLAS, MKL, or Accelerate |

### Runtime Configuration

```python
# Use GPU acceleration
llm = llama.Llama("model.gguf", n_gpu_layers=35)

# CPU only (no GPU offload)
llm = llama.Llama("model.gguf", n_gpu_layers=0)

# Full GPU offload (all layers)
llm = llama.Llama("model.gguf", n_gpu_layers=-1)
```

## API Reference

### Llama Class

```python
class Llama:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,           # Context window size
        n_batch: int = 512,         # Batch size for prompt processing
        n_threads: int = None,      # CPU threads (auto-detect if None)
        n_gpu_layers: int = 0,      # Layers to offload to GPU
        seed: int = -1,             # Random seed (-1 for random)
        use_mmap: bool = True,      # Memory map model file
        use_mlock: bool = False,    # Lock model in RAM
        verbose: bool = False,      # Print loading info
        embedding: bool = False,    # Enable embedding mode
    ): ...
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        stop_sequences: List[str] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]: ...
    
    def tokenize(self, text: str, add_special: bool = True) -> List[int]: ...
    def detokenize(self, tokens: List[int]) -> str: ...
    def get_embeddings(self, text: str) -> List[float]: ...
    def close(self): ...
```

### Backend Functions

```python
def get_available_backends() -> List[str]: ...
def get_backend_info() -> BackendInfo: ...
def is_cuda_available() -> bool: ...
def is_metal_available() -> bool: ...
def is_vulkan_available() -> bool: ...
def is_rocm_available() -> bool: ...
def is_blas_available() -> bool: ...
```

### Embedding Functions

```python
def get_embeddings(model: Union[str, Llama], text: str) -> List[float]: ...
def get_embeddings_batch(model: Union[str, Llama], texts: List[str]) -> List[List[float]]: ...
def cosine_similarity(a: List[float], b: List[float]) -> float: ...
```

## Examples

See the `examples/` directory:

- `basic_generation.py` - Simple text generation
- `streaming_generation.py` - Real-time token streaming
- `embeddings_example.py` - Generate and compare embeddings
- `backend_info.py` - Check available GPU backends
- `benchmark.py` - Measure token throughput

## Smoke Test / Chat CLI

This repository includes an interactive smoke test that can run either as a one-shot prompt (CI-friendly) or as a back-and-forth chat.

```bash
# Interactive chat (Ctrl+C or blank line to exit)
python -m llama_cpp_py_sync chat

# One-shot prompt
python -m llama_cpp_py_sync chat --prompt "Say 'ok'." --max-tokens 16

# Use a specific model
python -m llama_cpp_py_sync chat --model path/to/model.gguf
```

By default it uses `LLAMA_MODEL` if set. Otherwise it downloads a default GGUF model and caches it locally.

If the default model is missing, the CLI will prompt before downloading it. To auto-download without prompting, pass `--yes`.

Model cache location:

- **Windows**: `%LOCALAPPDATA%\llama-cpp-py-sync\models\`
- **Linux/macOS**: `~/.cache/llama-cpp-py-sync/models/`

## Building from Source

### Prerequisites

- Python 3.8+
- Ninja
- CMake (configure step)
- C/C++ compiler (GCC, Clang, MSVC)
- Git

### Build Commands

```bash
# Clone repository
git clone https://github.com/FarisZahrani/llama-cpp-py-sync.git
cd llama-cpp-py-sync

# Sync upstream llama.cpp
python scripts/sync_upstream.py

# Regenerate bindings from the synced llama.cpp headers
# (Optional) record the exact llama.cpp commit SHA in the generated file.
python scripts/gen_bindings.py --commit-sha "$(python scripts/sync_upstream.py --sha)"

# Build with auto-detected backends
python scripts/build_llama_cpp.py

# Build a specific backend
python scripts/build_llama_cpp.py --backend cuda
python scripts/build_llama_cpp.py --backend vulkan
python scripts/build_llama_cpp.py --backend cpu

# On Windows, the build script bundles required runtime DLLs (MSVC/OpenMP and backend runtimes)
# next to the built library by default. You can disable this behavior with:
python scripts/build_llama_cpp.py --no-bundle-runtime-dlls

# Detect available backends without building
python scripts/build_llama_cpp.py --detect-only

# Build wheel
pip install build
python -m build --wheel
```

### Low-level C API access (advanced)

If you need direct access to the underlying C API (beyond the high-level `Llama` wrapper), you can use the generated CFFI bindings:

```python
from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib

ffi = get_ffi()
lib = get_lib()

print(ffi.string(lib.llama_print_system_info()).decode("utf-8", errors="replace"))
```

## Project Structure

```
llama-cpp-py-sync/
├── src/llama_cpp_py_sync/      # Python package
│   ├── __init__.py             # Public API
│   ├── _cffi_bindings.py       # Auto-generated CFFI bindings
│   ├── _version.py             # Version info
│   ├── llama.py                # High-level Llama class
│   ├── embeddings.py           # Embedding utilities
│   └── backends.py             # Backend detection
├── scripts/                     # Build and sync scripts
│   ├── sync_upstream.py        # Sync upstream llama.cpp
│   ├── gen_bindings.py         # Generate CFFI bindings
│   ├── build_llama_cpp.py      # Build shared library
│   └── auto_version.py         # Version generation
├── examples/                    # Example scripts
├── vendor/llama.cpp/           # Upstream source (cloned at build time)
├── .github/workflows/          # CI/CD pipelines
├── pyproject.toml              # Package metadata
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run checks:

```bash
python scripts/run_tests.py
```

Optionally also verify wheel packaging locally:

```bash
python scripts/run_tests.py
```

5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

This project uses llama.cpp which is also MIT licensed.

Third-party license notices are included in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Acknowledgments

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - The upstream C/C++ implementation
- [CFFI](https://cffi.readthedocs.io/) - C Foreign Function Interface for Python
