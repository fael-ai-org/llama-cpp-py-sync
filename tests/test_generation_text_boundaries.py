"""Generation regressions independent of a platform-specific native library."""
from types import SimpleNamespace

import pytest

from llama_cpp_py_sync.llama import Llama


def fake_model(pieces):
    model = object.__new__(Llama)
    values = iter(range(len(pieces) + 1))
    model._sample_token = lambda: next(values)
    model._lib = SimpleNamespace(llama_vocab_is_eog=lambda _, token: token == len(pieces))
    model._vocab = object()
    model._token_to_bytes = lambda token: pieces[token]
    model._eval_tokens = lambda _, count: count + 1
    return model


@pytest.mark.parametrize("text", ["اهلا بالعالم", "مرحبا 🙂 中文", "café e\u0301"])
def test_utf8_characters_split_at_every_byte_remain_intact(text):
    data = text.encode("utf-8")
    model = fake_model([bytes([value]) for value in data])
    chunks = list(model._generate_from_n_past(0, len(data) + 1, None))
    assert "".join(chunks) == text
    assert all("\ufffd" not in chunk for chunk in chunks)


@pytest.mark.parametrize("pieces", [
    [b"hello<END>discarded"],
    [b"hello<", b"EN", b"D>discarded"],
    [b"he", b"llo", b"<END>", b"unreachable"],
])
def test_stop_marker_and_following_text_never_escape_in_stream_chunks(pieces):
    model = fake_model(pieces)
    chunks = list(model._generate_from_n_past(0, len(pieces) + 1, ["<END>"]))
    assert "".join(chunks) == "hello"
    assert all("<" not in chunk for chunk in chunks)


def test_earliest_stop_wins_regardless_of_configuration_order():
    model = fake_model([b"before<FIRST>middle<LAST>after"])
    assert "".join(model._generate_from_n_past(0, 2, ["<LAST>", "<FIRST>"])) == "before"


def test_false_partial_stop_prefix_is_released_and_terminal_partial_is_flushed():
    model = fake_model([b"a<EN", b"x", b"<EN"])
    assert "".join(model._generate_from_n_past(0, 4, ["<END>"])) == "a<ENx<EN"


def test_unicode_stop_can_span_byte_tokens():
    data = "قبلنهايةبعد".encode("utf-8")
    model = fake_model([bytes([value]) for value in data])
    assert "".join(model._generate_from_n_past(0, len(data) + 1, ["نهاية"])) == "قبل"


def test_closing_stream_does_not_flush_buffered_stop_or_incomplete_character():
    model = fake_model([b"ok<EN", b"D>"])
    stream = model._generate_from_n_past(0, 3, ["<END>"], cancel_callback=lambda: False)
    assert next(stream) == "ok"
    stream.close()


def test_cancellation_aborts_without_flushing_partial_stop():
    model = fake_model([b"ok<EN", b"D>"])
    cancelled = False
    stream = model._generate_from_n_past(0, 3, ["<END>"], cancel_callback=lambda: cancelled)
    assert next(stream) == "ok"
    cancelled = True
    with pytest.raises(RuntimeError, match="cancelled"):
        next(stream)


def test_long_token_piece_resizes_native_buffer_without_data_loss():
    from cffi import FFI
    ffi = FFI()
    data = ("سلام" * 100).encode("utf-8")
    sizes = []

    def read_piece(_vocab, _token, buffer, size, *_):
        sizes.append(size)
        if size < len(data):
            return -len(data)
        ffi.buffer(buffer, len(data))[:] = data
        return len(data)

    model = object.__new__(Llama)
    model._ffi = ffi
    model._lib = SimpleNamespace(llama_token_to_piece=read_piece)
    model._vocab = object()
    assert model._token_to_bytes(1) == data
    assert sizes == [128, len(data)]


def test_empty_stop_is_rejected():
    model = fake_model([b"hello"])
    with pytest.raises(ValueError, match="nonempty"):
        list(model._generate_from_n_past(0, 2, [""]))
