# llama-cpp-py-sync

**Auto-synchronized Python bindings for llama.cpp**

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
- Backend-specific wheels published to GitHub Releases: one broad CUDA 12.8 wheel per Linux/Windows platform, Linux and Windows Vulkan, macOS Apple Silicon Metal, and macOS Intel Vulkan (MoltenVK)
- CI checks that the generated CFFI surface matches the upstream C API (functions, structs, enums, and signatures)
- A small, explicit Python API (`Llama.generate`, `tokenize`, `get_embeddings`, etc.)
- Upstream `mtmd` multimodal support for vision-language GGUF models

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

## Multimodal vision and audio

Vision-language inference uses the current upstream `mtmd` C API. The language
model and projector remain separate files; pass the projector explicitly or
let the package discover the first matching file beside the model in this
order: `{model-stem}-mmproj.gguf`, `{model-stem}.mmproj.gguf`,
`mmproj-{model-stem}.gguf`, `mmproj.gguf`.

```python
from llama_cpp_py_sync import Llama
from llama_cpp_py_sync.multimodal import MultimodalContext

image_bytes = open("photo.png", "rb").read()

with Llama("model.gguf", n_ctx=4096, n_gpu_layers=35) as model:
    with MultimodalContext(model, projector_path="mmproj.gguf") as multimodal:
        print(multimodal.capabilities)
        response = model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "input_image", "mime_type": "image/png", "data": image_bytes},
                ],
            }],
            multimodal_context=multimodal,
        )
        print(response["choices"][0]["message"]["content"])
```

`image_url` accepts only validated local data URLs such as
`data:image/png;base64,...`; HTTP and HTTPS URLs are rejected and must be
downloaded and validated by the caller. Image count, encoded bytes, dimensions,
and total pixels are bounded by `MultimodalLimits`. Text and images are
evaluated in their original order, and streamed calls return the same
OpenAI-compatible chunk shape as `create_chat_completion(..., stream=True)`.

Multimodal contexts expose `capabilities` with `multimodal`, `modalities`,
`multiple_images`, `projector_path`, `projector_type`, and decoder properties.
Projectors are validated by `mtmd_init_from_file` against the loaded language
model before a request begins; incompatible or corrupt projectors raise
`ProjectorCompatibilityError`.

The generated binding banner records the exact llama.cpp revision
`a94d563ed801d1da1b8c2432946de07d0231bb3d` and the CFFI surface includes
`tools/mtmd/mtmd.h` and `mtmd-helper.h`. Wheels contain `libllama`, `libmtmd`,
and their `libggml-*` dependencies (or the platform equivalents); a wheel
missing the required `mtmd` ABI fails closed when `MultimodalContext` is
created. Existing text-only `Llama` requests do not require a projector.

For existing `Llama` users, `generate()` and its string streaming iterator are
unchanged. Chat users can continue using text-only `create_chat_completion()`;
adding images requires an explicit `MultimodalContext`, so invalid or
incompatible multimodal inputs cannot silently fall back to text inference.
The bundled ABI and native-library inventory are recorded in
`src/llama_cpp_py_sync/native_manifest.json` and the third-party notices.

Audio-only projectors are supported as well as combined vision/audio
projectors. Audio paths must name readable local files; encoded WAV, MP3, and
FLAC bytes can also be passed in memory. Codec detection and preprocessing are
performed by the synchronized `mtmd` implementation.

```python
from pathlib import Path

with llama.Llama("speech-model.gguf", n_ctx=4096) as model:
    print(model.get_capabilities(projector_path="speech-mmproj.gguf"))

    text = model.transcribe(
        "recording.wav",
        projector_path="speech-mmproj.gguf",
        language="en",
        structured=True,  # returns text/partial
    )

    generated = model.generate_audio(
        "Hello from llama.cpp.",
        projector_path="tts-mmproj.gguf",
        language="en",
        speaker_reference="speaker.wav",  # optional when supported
        seed=42,
    )
    Path("output.wav").write_bytes(generated.data)
```

`get_capabilities()` combines GGUF metadata with native llama.cpp and `mtmd`
queries. It reports audio input/output, sample rate, languages discoverable
from the native vocabulary, speaker-reference support, model variants, and
accepted generation options when those features are provided. Explicit
projector paths are checked first; otherwise the existing beside-model naming
conventions are used.

Transcription and generation accept `cancel_callback`; cancellation raises a
clear exception and releases native prompts, bitmaps, projector contexts, and
audio-generation helpers deterministically. Missing, corrupt, or incompatible
companion artifacts fail without falling back to another modality.

Actual embedding, ASR, and TTS model-family support is defined by the pinned
upstream llama.cpp revision. This package does not keep a Python model-name
registry and never launches `llama-server`, `llama-tts`, or another CLI.

The synchronized mtmd C API supports language selection and speaker-reference
audio for TTS, plus step-wise generation and the stateless
`mtmd_gen_audio_process()` frame API. The Python layer exposes the available
helper step API as PCM frame deltas and cleans native ASR wrapper tags while
yielding partial text.

### Embeddings

```python
# Load an embedding model. ``pooling_type="none"`` enables per-token output.
with llama.Llama(
    "embed-model.gguf",
    embedding=True,
    pooling_type="mean",
    n_seq_max=8,
    n_gpu_layers=-1,
) as llm:
    emb = llm.get_embeddings("Hello, world!", normalize="l2")
    batch = llm.get_embeddings_batch(["Hello", "world"], normalize="none")
    print(f"Embedding dimension: {len(emb)}")

with llama.Llama("embed-model.gguf", embedding=True, pooling_type="none") as llm:
    token_vectors = llm.get_embeddings("Hello", per_token=True)
```

Pooling, batching, per-token output, and offload settings are passed to the
native llama.cpp API. Normalization is an explicit Python post-processing step
over the returned vectors.

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
    n_ubatch=None,
    n_threads_batch=None,
    seed=-1,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    embedding=False,
    flash_attn_type=None,
    pooling_type=None,
    n_seq_max=1,
    op_offload=None,
)

text = llm.generate(
    "Hello",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    repeat_penalty=1.1,
    repeat_last_n=64,
    stop_sequences=None,
    stream=False,
    seed=None,
)

stream = llm.generate(
    "Hello",
    max_tokens=256,
    stream=True,
)

tokens = llm.tokenize("Hello", add_special=True, parse_special=False)
text = llm.detokenize(tokens, remove_special=False, unparse_special=True)
piece = llm.token_to_piece(tokens[0])

llm.get_model_desc()
llm.get_model_size()
llm.get_model_n_params()

# Properties
llm.n_vocab
llm.n_ctx
llm.n_embd
llm.n_layer
llm.bos_token
llm.eos_token

# Embeddings (requires embedding=True)
emb = llm.get_embeddings("Hello", normalize="l2")
batch = llm.get_embeddings_batch(["Hello", "World"], normalize="none")
token_vectors = llm.get_embeddings("Hello", per_token=True)  # pooling_type="none"

# Structured ASR and complete TTS
result = llm.transcribe("audio.wav", structured=True)
generated_audio = llm.generate_audio("Hello")

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
| Vulkan | Linux, Windows, macOS (Intel and Apple Silicon) | CI uses the pinned Vulkan SDK 1.4.335.0; local builds can use `VULKAN_SDK`, Homebrew, or system headers |
| BLAS | All | OpenBLAS, MKL, or Accelerate |

CUDA release wheels use CUDA 12.8 and compile native targets for Maxwell through
Blackwell (`50;52;60;61;70;75;80;86;89;90;100;120`). This is the single modern
CUDA package line; users do not need a matching CUDA toolkit installed, but do
need an NVIDIA driver compatible with the bundled CUDA runtime.

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
        n_ctx: int = 512,                   # Context window size
        n_batch: int = 512,                 # Logical max batch size for prompt processing
        n_threads: int = None,              # CPU threads (auto-detect if None)
        n_gpu_layers: int = 0,              # Layers to offload to GPU
        n_ubatch: int = None,               # Physical microbatch size (defaults to n_batch)
        n_threads_batch: int = None,        # Threads for batch processing (defaults to n_threads)
        seed: int = -1,                     # Random seed (-1 for random)
        use_mmap: bool = True,              # Memory map model file
        use_mlock: bool = False,            # Lock model in RAM
        verbose: bool = False,              # Print loading info
        embedding: bool = False,            # Enable embedding mode
        flash_attn_type: int = None,        # Flash attention type (None = use env var)
        offload_kqv: bool = None,            # Offload KV-cache operations
        op_offload: bool = None,             # Offload host tensor operations
        pooling_type: int | str = None,      # none, mean, cls, last, or rank
        n_seq_max: int = 1,                  # Native sequence capacity
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
        repeat_last_n: int = 64,
        stop_sequences: List[str] = None,
        stream: bool = False,
        seed: int = None,
    ) -> Union[str, Iterator[str]]: ...

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = False) -> List[int]: ...
    def detokenize(self, tokens: List[int], remove_special: bool = False, unparse_special: bool = True) -> str: ...
    def token_to_piece(self, token: int) -> str: ...
    def get_embeddings(self, text: str, normalize=None, per_token: bool = False): ...
    def get_embeddings_batch(self, texts: Sequence[str], normalize=None, per_token: bool = False): ...
    def get_model_desc(self) -> str: ...
    def get_model_size(self) -> int: ...
    def get_model_n_params(self) -> int: ...
    def close(self): ...

    # Properties
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_layer: int
    bos_token: int
    eos_token: int
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
def get_embeddings(model: Union[str, Llama], text: str, normalize=True, pooling_type=None, per_token=False, offload_kqv=None, op_offload=None): ...
def get_embeddings_batch(model: Union[str, Llama], texts: List[str], normalize=True, pooling_type=None, per_token=False, offload_kqv=None, op_offload=None): ...
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
