import json
import subprocess
from pathlib import Path

import pytest

from llama_cpp_py_sync._cffi_bindings import MTMD_REQUIRED_SYMBOLS, ffi, get_binding_health
from llama_cpp_py_sync.llama import Llama
from llama_cpp_py_sync.multimodal import (
    AudioValidationError,
    MultimodalCancelledError,
    MultimodalContext,
    MultimodalPrompt,
    _check_cancelled,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generated_audio_cffi_surface_is_complete():
    assert ffi.sizeof("struct mtmd_gen_audio_info") > 0
    assert ffi.sizeof("struct mtmd_helper_gen_audio_inp") > 0
    bindings = (ROOT / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py").read_text()
    assert "int32_t mtmd_helper_gen_audio_step_gen(" in bindings
    assert "mtmd_helper_gen_audio_get_output" in MTMD_REQUIRED_SYMBOLS


@pytest.mark.native
def test_packaged_native_mtmd_matches_generated_abi():
    assert get_binding_health(require_mtmd=True) == {
        "llama": True,
        "mtmd": True,
        "missing_mtmd_symbols": [],
    }


def test_source_manifest_bindings_and_version_use_one_commit():
    vendor_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "vendor" / "llama.cpp", text=True
    ).strip()
    manifest = json.loads(
        (ROOT / "src" / "llama_cpp_py_sync" / "native_manifest.json").read_text()
    )
    bindings = (ROOT / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py").read_text()
    version = (ROOT / "src" / "llama_cpp_py_sync" / "_version.py").read_text()
    assert manifest["llama_cpp_commit"] == vendor_sha
    assert vendor_sha in bindings
    assert vendor_sha[:7] in version


def test_audio_only_projector_is_accepted(monkeypatch, tmp_path):
    projector = tmp_path / "audio-mmproj.gguf"
    projector.write_bytes(b"gguf")

    class Caps:
        inp_vision = False
        inp_audio = True

    class Params:
        use_gpu = False
        print_timings = False
        n_threads = 1
        flash_attn_type = 0
        warmup = False
        image_min_tokens = -1
        image_max_tokens = -1
        batch_max_tokens = 1024
        progress_callback = None
        progress_callback_user_data = None

    class Info:
        type = 0
        sample_rate = 0
        model_variant = None

    class Lib:
        def mtmd_get_cap_from_file(self, _path):
            return Caps()

        def mtmd_context_params_default(self):
            return Params()

        def mtmd_init_from_file(self, *_args):
            return object()

        def mtmd_support_vision(self, _ctx):
            return False

        def mtmd_support_audio(self, _ctx):
            return True

        def mtmd_gen_audio_get_info(self, _ctx):
            return Info()

        def mtmd_get_audio_sample_rate(self, _ctx):
            return 16000

        def mtmd_get_marker(self, _ctx):
            return b"<__media__>"

        def mtmd_decode_use_mrope(self, _ctx):
            return False

        def mtmd_decode_use_non_causal(self, *_args):
            return False

        def mtmd_free(self, _ctx):
            pass

    class FFI:
        NULL = None

        @staticmethod
        def string(value):
            return value

    model = type(
        "Model", (), {"_model": object(), "_ctx": object(), "model_path": "model.gguf"}
    )()
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.get_ffi", lambda: FFI())
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.get_mtmd_lib", lambda: Lib())
    with MultimodalContext(model, projector, warmup=False) as context:
        assert context.supports_audio
        assert not context.supports_vision
        assert context.capabilities["projector_type"] == "audio"


def test_public_audio_methods_and_capability_api_exist():
    assert callable(Llama.get_embeddings)
    assert callable(Llama.transcribe)
    assert callable(Llama.generate_audio)
    assert callable(Llama.get_capabilities)


def test_runtime_does_not_branch_on_model_names():
    sources = "\n".join(
        (ROOT / "src" / "llama_cpp_py_sync" / name).read_text().lower()
        for name in ("llama.py", "multimodal.py")
    )
    for model_name in ("qwen", "whisper", "parakeet", "pockettts", "granite"):
        assert model_name not in sources


def test_missing_explicit_projector_is_clear(tmp_path):
    model = type("Model", (), {"model_path": str(tmp_path / "model.gguf")})()
    with pytest.raises(FileNotFoundError, match="Projector file not found"):
        MultimodalContext._resolve_projector_path(model, tmp_path / "missing.gguf", True)


def test_audio_cancellation_is_deterministic():
    with pytest.raises(MultimodalCancelledError, match="cancelled"):
        _check_cancelled(lambda: True)


def test_empty_in_memory_audio_is_rejected_before_native_decode():
    context = object.__new__(MultimodalContext)
    context._closed = False
    context._ctx = object()
    context._ffi = type("FFI", (), {"NULL": None})()
    context.limits = type("Limits", (), {"max_audio_bytes": 1024})()
    with pytest.raises(AudioValidationError, match="must not be empty"):
        context.create_audio_bitmap(b"")


def test_audio_prompt_cleanup_is_idempotent():
    freed = []

    class Lib:
        def mtmd_input_chunks_free(self, value):
            freed.append(("chunks", value))

        def mtmd_bitmap_free(self, value):
            freed.append(("bitmap", value))

    context = type("Context", (), {"_ffi": type("FFI", (), {"NULL": None})(), "_lib": Lib()})()
    prompt = MultimodalPrompt(context, "chunks", ["audio"])
    prompt.close()
    prompt.close()
    assert freed == [("chunks", "chunks"), ("bitmap", "audio")]
