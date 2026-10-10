import subprocess
from pathlib import Path

import pytest

from scripts.apply_native_patches import apply
from scripts.gen_bindings import generate_cdef


def test_real_regeneration_uses_patched_vendor_headers(tmp_path):
    import sys
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor/llama.cpp"
    if not (vendor / "include/llama.h").exists():
        pytest.skip("Vendored headers required for actual regeneration")
    output = tmp_path / "src/llama_cpp_py_sync/_cffi_bindings.py"
    subprocess.run([sys.executable, str(root / "scripts/gen_bindings.py"),
        "--vendor-path", str(vendor), "--output", str(output),
        "--project-root", str(tmp_path)], check=True, capture_output=True)
    generated = output.read_text(encoding="utf-8")
    compile(generated, str(output), "exec")
    assert "ggml_backend_rpc_serve_stream" in generated
    assert "retain_openmp_runtime" in generated


def test_binding_regeneration_retains_stream_layout_and_rpc_limits():
    root = Path(__file__).resolve().parents[1]
    header = root / "vendor/llama.cpp/ggml/include/ggml-rpc.h"
    if not header.exists():
        pytest.skip("Vendored headers required for regeneration check")
    generated = generate_cdef({"ggml-rpc.h": header})
    assert "typedef struct ggml_rpc_stream" in generated
    assert "ggml_backend_rpc_add_stream" in generated
    assert "ggml_backend_rpc_serve_stream" in generated
    assert "ggml_backend_rpc_remove_stream" in generated
    assert "#define GGML_RPC_MAX_SERVERS" in generated


def test_native_patch_applies_once_and_rejects_conflicting_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor/llama.cpp"
    if not (vendor / ".git").exists():
        pytest.skip("Vendored Git source required for patch acceptance")
    paths = ["ggml/include/ggml-rpc.h", "ggml/src/ggml-rpc/transport.h",
        "ggml/src/ggml-rpc/transport.cpp", "ggml/src/ggml-rpc/ggml-rpc.cpp"]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", "HEAD:" + path], cwd=vendor))
    apply(tmp_path)
    after = {path: (tmp_path / path).read_bytes() for path in paths}
    apply(tmp_path)
    assert {path: (tmp_path / path).read_bytes() for path in paths} == after
    header = tmp_path / paths[0]
    header.write_text(header.read_text().replace("ggml_rpc_stream", "conflicting_rpc_stream"))
    with pytest.raises(subprocess.CalledProcessError):
        apply(tmp_path)
