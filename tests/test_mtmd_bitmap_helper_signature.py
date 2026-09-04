from __future__ import annotations

from types import SimpleNamespace

from llama_cpp_py_sync.multimodal import MultimodalContext


class _NativeFn:
    def __init__(self, expected_argc: int) -> None:
        self.expected_argc = expected_argc
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> SimpleNamespace:
        if len(args) != self.expected_argc:
            raise TypeError(f"expects {self.expected_argc} arguments, got {len(args)}")
        self.calls.append(args)
        return SimpleNamespace(bitmap="ok")


class _Ffi:
    def typeof(self, fn: _NativeFn) -> SimpleNamespace:
        return SimpleNamespace(args=[None] * fn.expected_argc)


def test_bitmap_init_from_buf_passes_default_opt_for_five_arg_native_signature():
    native = _NativeFn(5)
    default_opt = object()
    ctx = object()
    owner = SimpleNamespace(
        _ctx=ctx,
        _ffi=_Ffi(),
        _lib=SimpleNamespace(
            mtmd_helper_bitmap_init_from_buf=native,
            mtmd_helper_init_opt_default=lambda: default_opt,
        ),
    )
    owner._mtmd_helper_init_opt = lambda: MultimodalContext._mtmd_helper_init_opt(owner)

    result = MultimodalContext._mtmd_helper_bitmap_init_from_buf(owner, b"wav", 3, False)

    assert result.bitmap == "ok"
    assert native.calls == [(ctx, b"wav", 3, False, default_opt)]


def test_bitmap_init_from_buf_keeps_four_arg_legacy_signature():
    native = _NativeFn(4)
    ctx = object()
    owner = SimpleNamespace(
        _ctx=ctx,
        _ffi=_Ffi(),
        _lib=SimpleNamespace(mtmd_helper_bitmap_init_from_buf=native),
    )

    result = MultimodalContext._mtmd_helper_bitmap_init_from_buf(owner, b"wav", 3, False)

    assert result.bitmap == "ok"
    assert native.calls == [(ctx, b"wav", 3, False)]
