from __future__ import annotations

import os
import shutil
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llama_cpp_py_sync import rpc
from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib


def transport(**changes):
    fields = {"sendall": Mock(), "recv_into": Mock(), "close": Mock()}
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_requires_stream_contract_before_native_use():
    with pytest.raises(ValueError):
        rpc.RPCStream(object())


@pytest.mark.parametrize("cache_kind", ["disabled", "string", "path"])
def test_optional_stream_cache_directory_is_forwarded(monkeypatch, tmp_path, cache_kind):
    ffi = get_ffi()
    handle = ffi.cast("ggml_backend_dev_t", 1)
    native = SimpleNamespace(ggml_backend_rpc_serve_stream=Mock(return_value=True))
    backend = SimpleNamespace(ggml_backend_dev_by_name=Mock(return_value=handle))
    monkeypatch.setattr(rpc, "_require_rpc", lambda: (ffi, backend, native))
    wire = transport()
    owner = rpc.RPCStream(wire)
    cache_dir = {"disabled": None, "string": str(tmp_path), "path": tmp_path}[cache_kind]
    owner.serve(devices=["CPU"], cache_dir=cache_dir)
    expected = ffi.NULL if cache_dir is None else os.fsencode(tmp_path)
    assert native.ggml_backend_rpc_serve_stream.call_args.args[1] == expected
    wire.close.assert_called_once()


def test_missing_stream_cache_directory_is_rejected_before_native(monkeypatch, tmp_path):
    owner = rpc.RPCStream(transport())
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native serving must not run"))
    with pytest.raises(ValueError):
        owner.serve(devices=["CPU"], cache_dir=tmp_path / "missing")
    owner.close()


@pytest.mark.parametrize("cache_dir", ["", "bad\0path", b"bytes-path"])
def test_invalid_stream_cache_directory_is_rejected_before_native(monkeypatch, cache_dir):
    owner = rpc.RPCStream(transport())
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native serving must not run"))
    with pytest.raises(ValueError):
        owner.serve(devices=["CPU"], cache_dir=cache_dir)
    owner.close()


def test_stream_cache_file_is_rejected_before_native(monkeypatch, tmp_path):
    cache_file = tmp_path / "file"
    cache_file.write_bytes(b"not a directory")
    owner = rpc.RPCStream(transport())
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native serving must not run"))
    with pytest.raises(ValueError):
        owner.serve(devices=["CPU"], cache_dir=cache_file)
    owner.close()


def test_exact_transfer_handles_short_reads_and_zero_length():
    ffi = get_ffi()
    parts = iter([b"ab", b"c", b"def"])
    def read(buffer):
        part = next(parts)
        buffer[:len(part)] = part
        return len(part)
    wire = transport(recv_into=read)
    owner = rpc.RPCStream(wire)
    data = ffi.new("char[6]")
    assert owner._native.recv(ffi.NULL, data, 6)
    assert bytes(ffi.buffer(data, 6)) == b"abcdef"
    assert owner._native.send(ffi.NULL, data, 6)
    assert bytes(wire.sendall.call_args.args[0]) == b"abcdef"
    assert owner._native.recv(ffi.NULL, data, 0)
    owner.close()
    owner.close()
    wire.close.assert_called_once()


@pytest.mark.parametrize("result", [0, -1, 7, True, None])
def test_invalid_reads_fail_closed(result):
    ffi = get_ffi()
    owner = rpc.RPCStream(transport(recv_into=lambda _: result))
    assert not owner._native.recv(ffi.NULL, ffi.new("char[6]"), 6)
    owner.close()


def test_callback_exceptions_never_cross_cffi():
    ffi = get_ffi()
    failure = OSError("closed")
    owner = rpc.RPCStream(transport(sendall=Mock(side_effect=failure)))
    assert not owner._native.send(ffi.NULL, ffi.new("char[1]"), 1)
    assert owner.error is failure
    owner.close()


@pytest.mark.parametrize("streams", [[], "endpoint:123", [object()]])
def test_invalid_stream_selection_never_loads_native(monkeypatch, streams):
    monkeypatch.setattr(rpc, "_require_rpc", lambda: pytest.fail("Native code must not run"))
    with pytest.raises(ValueError):
        rpc.rpc_stream_devices(streams)


def test_retains_callbacks_reuses_registration_and_releases_after_native(monkeypatch):
    ffi = get_ffi()
    handle = ffi.cast("void *", 1)
    calls = []
    wire = transport()
    def remove(_):
        calls.append("removed")
        return True
    native = SimpleNamespace(ggml_backend_rpc_add_stream=Mock(return_value=handle),
        ggml_backend_rpc_remove_stream=remove)
    backend = SimpleNamespace(ggml_backend_register=Mock())
    base = SimpleNamespace(ggml_backend_reg_dev_count=lambda _: 1,
        ggml_backend_reg_dev_get=lambda *_: handle)
    monkeypatch.setattr(rpc, "_require_rpc", lambda: (ffi, backend, native))
    monkeypatch.setattr(rpc, "get_rpc_lib", lambda: native)
    monkeypatch.setattr(rpc, "get_backend_base_lib", lambda: base)
    owner = rpc.RPCStream(wire)
    wire.close.side_effect = lambda: calls.append("closed")
    assert owner.devices() == [handle]
    assert owner.devices() == [handle]
    assert rpc._registered_streams[owner.name] is owner
    native.ggml_backend_rpc_add_stream.assert_called_once()
    backend.ggml_backend_register.assert_not_called()
    owner.close()
    assert calls == ["removed", "closed"]
    assert owner.name not in rpc._registered_streams
    with pytest.raises(RuntimeError):
        owner.devices()


def test_failed_native_release_retains_owner_and_transport(monkeypatch):
    wire = transport()
    owner = rpc.RPCStream(wire)
    owner._mode = "registered"
    rpc._registered_streams[owner.name] = owner
    native = SimpleNamespace(ggml_backend_rpc_remove_stream=Mock(return_value=False))
    monkeypatch.setattr(rpc, "get_rpc_lib", lambda: native)
    with pytest.raises(RuntimeError, match="consumers"):
        owner.close()
    assert rpc._registered_streams[owner.name] is owner
    wire.close.assert_not_called()
    native.ggml_backend_rpc_remove_stream.return_value = True
    owner.close()


@pytest.fixture
def tls_workers(tmp_path):
    if os.environ.get("LLAMA_TEST_RPC_STREAM") != "1":
        pytest.skip("Set LLAMA_TEST_RPC_STREAM=1 for native TLS stream acceptance")
    openssl = shutil.which("openssl")
    if not openssl and sys.platform == "win32":
        candidate = Path("C:/Program Files/Git/usr/bin/openssl.exe")
        openssl = str(candidate) if candidate.exists() else None
    if not openssl:
        pytest.skip("OpenSSL CLI is required to create ephemeral test certificates")
    certificate, key = tmp_path / "certificate.pem", tmp_path / "private.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(certificate), "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"],
        capture_output=True, check=True)
    workers, owners, logs = [], [], []
    def connect(*, reject_unauthenticated=False, cache_dir=None):
        log = (tmp_path / f"worker-{len(workers)}.log").open("w+")
        logs.append(log)
        command = [sys.executable, str(Path(__file__).with_name("rpc_stream_worker.py")),
            str(certificate), str(key)]
        if cache_dir is not None:
            command.append(str(cache_dir))
        worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
        workers.append(worker)
        port = int(worker.stdout.readline().strip())
        context = ssl.create_default_context(cafile=str(certificate))
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        if reject_unauthenticated:
            with context.wrap_socket(socket.create_connection(("127.0.0.1", port)), server_hostname="localhost") as denied:
                denied.settimeout(5)
                try:
                    denied.sendall(b"unauthorized")
                    assert denied.recv(1) == b""
                except (ssl.SSLError, ConnectionResetError):
                    pass
        context.load_cert_chain(str(certificate), str(key))
        secured = context.wrap_socket(socket.create_connection(("127.0.0.1", port)), server_hostname="localhost")
        secured.settimeout(180)
        assert secured.version() == "TLSv1.3"
        owner = rpc.RPCStream(secured)
        owners.append(owner)
        return owner
    yield connect, workers, logs
    for owner in owners:
        owner.close()
    for worker in workers:
        if worker.poll() is None:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.terminate()
        worker.wait(timeout=10)
        worker.stdout.close()
    for log in logs:
        log.close()


@pytest.mark.native
def test_native_registration_over_mutual_tls_rejects_unauthenticated_peer(tls_workers):
    connect, workers, logs = tls_workers
    owner = connect(reject_unauthenticated=True)
    devices = owner.devices()
    assert len(devices) == 1
    base = rpc.get_backend_base_lib()
    ffi = get_ffi()
    assert ffi.string(base.ggml_backend_dev_name(devices[0])).startswith(b"RPC")
    if sys.platform == "win32":
        connections = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True)
        assert not any("LISTENING" in row and row.split()[-1] == str(workers[0].pid)
                       for row in connections.splitlines() if row.strip())
    owner.close()
    assert workers[0].wait(timeout=10) == 0
    logs[0].flush()
    logs[0].seek(0)
    assert "unauthenticated-peer-rejected" in logs[0].read()


@pytest.mark.integration
def test_generation_over_two_caller_owned_tls_streams(tls_workers):
    model_path = os.environ.get("LLAMA_TEST_MODEL")
    if not model_path:
        pytest.skip("LLAMA_TEST_MODEL is required for generation")
    from llama_cpp_py_sync import Llama
    connect, workers, _ = tls_workers
    owners = [connect(), connect()]
    ffi = get_ffi()
    messages = []
    @ffi.callback("void(enum ggml_log_level, const char *, void *)")
    def capture(_level, message, _context):
        text = ffi.string(message).decode("utf-8", errors="replace")
        if "model buffer size" in text:
            messages.append(text)
    get_lib().llama_log_set(capture, ffi.NULL)
    try:
        with Llama(model_path, rpc_streams=owners, n_gpu_layers=-1, split_mode="layer",
                tensor_split=[1, 1], n_ctx=256, n_batch=32, n_threads=2) as model:
            assert len(model._devices_buffer) == 3
            with pytest.raises(RuntimeError, match="consumers"):
                owners[0].close()
            assert model.generate("Say hello.", max_tokens=4, temperature=0.0, seed=123).strip()
        for owner in owners:
            assert any("stream:" + owner.name in message for message in messages), messages
            owner.close()
        assert all(worker.wait(timeout=10) == 0 for worker in workers)
    finally:
        get_lib().llama_log_set(ffi.NULL, ffi.NULL)


@pytest.mark.integration
def test_native_tensor_cache_populates_reuses_and_clears(tls_workers, tmp_path):
    model_path = os.environ.get("LLAMA_TEST_MODEL")
    if not model_path:
        pytest.skip("LLAMA_TEST_MODEL is required for cache generation")
    from llama_cpp_py_sync import Llama

    connect, workers, _ = tls_workers
    cache_dir = tmp_path / "tensor-cache"
    cache_dir.mkdir()
    sent = []
    generated = []
    cached_bytes = 0
    for phase in ("populate", "reuse", "clear"):
        if phase == "clear":
            for entry in cache_dir.iterdir():
                assert entry.is_file() and not entry.is_symlink()
                entry.unlink()
        owner = connect(cache_dir=cache_dir)
        wire = owner.transport
        sendall = wire.sendall
        byte_count = [0]

        def measure(data, sendall=sendall, byte_count=byte_count):
            sendall(data)
            byte_count[0] += len(data)

        wire.sendall = measure
        with Llama(model_path, rpc_streams=[owner], n_gpu_layers=-1, split_mode="layer",
                tensor_split=[1], n_ctx=256, n_batch=32, n_threads=2) as model:
            sent.append(byte_count[0])
            generated.append(model.generate("Say hello.", max_tokens=4, temperature=0.0, seed=123))
            assert generated[-1].strip()
        owner.close()
        assert workers[-1].wait(timeout=15) == 0
        entries = list(cache_dir.iterdir())
        if phase == "populate":
            if not entries:
                pytest.skip("Model has no cache-eligible weight tensors larger than 10 MiB")
            cached_bytes = sum(entry.stat().st_size for entry in entries)
        assert entries, "Native tensor cache was not populated"
    assert sent[0] - sent[1] >= cached_bytes * 0.95
    assert sent[2] == sent[0]
    assert generated[0] == generated[1] == generated[2]
