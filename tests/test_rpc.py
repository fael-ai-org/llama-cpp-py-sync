from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from llama_cpp_py_sync import _cffi_bindings as bindings
from llama_cpp_py_sync import rpc
from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib


@pytest.mark.parametrize("endpoint", [
    "", "host", "http://host:123", "host:0", "host:65536", "host:-1",
    "[::1]:123", "host:1junk", "host:123\0", "host name:123",
])
def test_rejects_invalid_endpoints(endpoint):
    with pytest.raises(ValueError):
        rpc._validate_endpoint(endpoint)


def test_endpoint_normalizes_port():
    assert rpc._validate_endpoint("127.0.0.1:05052") == "127.0.0.1:5052"


@pytest.mark.parametrize("endpoints", ["host:123", [], ["host:123", "host:00123"]])
def test_invalid_server_selection_never_loads_native(monkeypatch, endpoints):
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native code must not run"))
    with pytest.raises(ValueError):
        rpc.rpc_devices(endpoints)


def test_disabled_backend_fails_before_loading_rpc(monkeypatch):
    monkeypatch.setattr(rpc, "get_lib", lambda: SimpleNamespace(llama_supports_rpc=lambda: False))
    monkeypatch.setattr(rpc, "get_rpc_lib", lambda: pytest.fail("RPC library must not load"))
    with pytest.raises(RuntimeError, match="without RPC"):
        rpc._require_rpc()


@pytest.mark.parametrize("system,name", [
    ("Windows", "ggml-rpc.dll"), ("Darwin", "libggml-rpc.dylib"),
    ("Linux", "libggml-rpc.so.0"),
])
def test_rpc_loader_uses_selected_library_directory(monkeypatch, tmp_path, system, name):
    library = tmp_path / name
    library.touch()
    loaded = []
    monkeypatch.setattr(bindings, "_ggml_libraries", {})
    monkeypatch.setattr(bindings, "_find_library", lambda: str(tmp_path / "llama"))
    monkeypatch.setattr(bindings.platform, "system", lambda: system)
    monkeypatch.setattr(bindings, "_configure_runtime_for_library", lambda path: None)
    monkeypatch.setattr(bindings, "ffi", SimpleNamespace(dlopen=lambda path: loaded.append(path) or path))
    assert bindings.get_rpc_lib() == str(library)
    assert bindings.get_rpc_lib() == str(library)
    assert loaded == [str(library)]


def test_missing_rpc_library_does_not_use_system_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(bindings, "_ggml_libraries", {})
    monkeypatch.setattr(bindings, "_find_library", lambda: str(tmp_path / "llama"))
    with pytest.raises(RuntimeError, match="Missing bundled native library"):
        bindings.get_rpc_lib()


def test_selects_only_requested_remote_devices_and_local_accelerators(monkeypatch):
    ffi = get_ffi()
    cpu, gpu, stale, first, second = [ffi.cast("void *", i) for i in range(1, 6)]
    calls = []
    local = [cpu, gpu, stale]
    backend = SimpleNamespace(
        ggml_backend_dev_count=lambda: len(local),
        ggml_backend_dev_get=lambda index: local[index],
        ggml_backend_register=lambda reg: calls.append(reg),
    )
    names = {cpu: b"CPU", gpu: b"CUDA0", stale: b"RPC0"}
    base = SimpleNamespace(
        ggml_backend_reg_dev_count=lambda reg: 1,
        ggml_backend_reg_dev_get=lambda reg, index: reg,
        ggml_backend_dev_name=lambda device: ffi.new("char[]", names[device]),
        ggml_backend_dev_type=lambda device: 0 if device == cpu else 1,
        GGML_BACKEND_DEVICE_TYPE_GPU=1, GGML_BACKEND_DEVICE_TYPE_IGPU=2,
    )
    remote = SimpleNamespace(
        GGML_RPC_MAX_SERVERS=16,
        ggml_backend_rpc_add_server=lambda endpoint: first if endpoint == b"a:1" else second,
    )
    monkeypatch.setattr(rpc, "_require_rpc", lambda: (ffi, backend, remote))
    monkeypatch.setattr(rpc, "get_backend_base_lib", lambda: base)
    monkeypatch.setattr(rpc, "get_lib", lambda: SimpleNamespace(llama_max_devices=lambda: 4))
    assert rpc.rpc_devices(["a:1", "b:2"]) == [gpu, first, second]
    assert calls == [first, second]
    assert rpc.rpc_devices(["b:2"], include_local=False) == [second]


@pytest.mark.parametrize("devices,threads", [([], 1), ("CPU", 1), (["CPU", "CPU"], 1), (["CPU\0"], 1), (["CPU"], 0), (["CPU"], True), (["CPU"], 2**32)])
def test_invalid_worker_configuration_never_loads_native(monkeypatch, devices, threads):
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native code must not run"))
    with pytest.raises(ValueError):
        rpc.start_rpc_server(devices=devices, n_threads=threads)


@pytest.mark.integration
def test_real_generation_across_two_rpc_workers(tmp_path):
    if os.environ.get("LLAMA_TEST_RPC") != "1":
        pytest.skip("Set LLAMA_TEST_RPC=1 and LLAMA_TEST_MODEL to test native RPC inference")
    model_path = os.environ.get("LLAMA_TEST_MODEL")
    assert model_path, "LLAMA_TEST_MODEL is required for RPC acceptance"
    assert get_lib().llama_supports_rpc(), "RPC acceptance requires an RPC-enabled native library"
    workers = []
    logs = []
    endpoints = []
    try:
        for index in range(2):
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            endpoint = f"127.0.0.1:{port}"
            log = (tmp_path / f"worker-{index}.log").open("w+")
            logs.append(log)
            worker = subprocess.Popen(
                [sys.executable, "-c", "import sys; from llama_cpp_py_sync.rpc import start_rpc_server; start_rpc_server(sys.argv[1], devices=['CPU'], n_threads=2)", endpoint],
                stdout=log, stderr=log,
            )
            workers.append(worker)
            deadline = time.monotonic() + 20
            while True:
                if worker.poll() is not None:
                    log.seek(0)
                    pytest.fail("RPC worker exited: " + log.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    assert time.monotonic() < deadline, "RPC worker startup timed out"
                    time.sleep(0.05)
            endpoints.append(endpoint)
        client = subprocess.run(
            [sys.executable, "-c", """
import sys
from llama_cpp_py_sync import Llama
from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib
ffi = get_ffi()
messages = []
@ffi.callback('void(enum ggml_log_level, const char *, void *)')
def capture(level, message, data):
    text = ffi.string(message).decode('utf-8', errors='replace')
    if 'model buffer size' in text or 'offloaded' in text:
        messages.append(text)
get_lib().llama_log_set(capture, ffi.NULL)
with Llama(sys.argv[1], rpc_servers=sys.argv[2:], n_gpu_layers=-1,
           split_mode='layer', tensor_split=[1, 1], n_ctx=256, n_batch=32,
           n_threads=2) as model:
    assert len(model._devices_buffer) == 3
    assert model._devices_buffer[2] == ffi.NULL
    output = model.generate('Say hello.', max_tokens=4, temperature=0.0, seed=123)
    assert output.strip(), 'Generation was empty'
text = ''.join(messages)
for endpoint in sys.argv[2:]:
    assert any(endpoint in line and 'model buffer size' in line for line in text.splitlines()), text
print('rpc-generation-ok')
for line in text.splitlines():
    if 'model buffer size' in line or 'offloaded' in line:
        print(line)
""", model_path, *endpoints],
            capture_output=True, text=True, timeout=180,
        )
        assert client.returncode == 0, client.stdout + client.stderr
        assert "rpc-generation-ok" in client.stdout
        print(client.stdout)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=15)
        for log in logs:
            log.close()
