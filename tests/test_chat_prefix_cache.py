"""Live prefix lifecycle tests without loading a native model."""
from types import SimpleNamespace

import pytest

from llama_cpp_py_sync.llama import Llama


def model_fixture(*, rewind=True):
    model = object.__new__(Llama)
    model._ctx = None
    model._sampler = None
    model._n_ctx = 1000
    model._ffi = SimpleNamespace(NULL=None)
    events = []
    model._lib = SimpleNamespace(
        llama_get_memory=lambda _: object(),
        llama_memory_clear=lambda *_: events.append('clear'),
        llama_memory_seq_rm=lambda _, seq, start, end: events.append(('trim', start)) or rewind,
    )
    model._format_chat_prompt = lambda messages, _: (messages[0]['content'], [])
    model.tokenize = lambda text, **_: list(text.encode())
    model._configure_generation_sampler = lambda *_: events.append('sampler')
    model._set_abort_callback = lambda _: None

    def evaluate(tokens, past):
        events.append(('eval', list(tokens), past))
        if getattr(model, '_chat_cache_tokens', None) is not None:
            assert len(model._chat_cache_tokens) == past
            model._chat_cache_tokens.extend(tokens)
        return past + len(tokens)

    def generate(past, *_):
        yield 'x'
        evaluate([ord('x')], past)

    model._eval_tokens = evaluate
    model._generate_from_n_past = generate
    return model, events


def complete(model, prompt, key='a', **kwargs):
    return model.create_chat_completion([{'role': 'user', 'content': prompt}],
                                        max_tokens=8, cache_key=key, **kwargs)


def test_append_only_reuses_decoded_output_and_prefills_only_suffix():
    model, events = model_fixture()
    stats = []
    complete(model, 'abc', on_stats=stats.append)
    events.clear()
    answer = complete(model, 'abcx observation', on_stats=stats.append)
    assert answer['choices'][0]['message']['content'] == 'x'
    assert 'clear' not in events
    assert ('eval', list(b' observation'), 4) in events
    assert stats[-1]['reused_tokens'] == 4
    assert stats[-1]['prefilled_tokens'] == len(' observation')
    assert stats[-1]['cache_status'] == 'prefix_hit'
    assert stats[-1]['completed']
    assert 0 <= stats[-1]['prefill_seconds'] <= stats[-1]['total_seconds']
    assert 0 <= stats[-1]['ttft_seconds'] <= stats[-1]['total_seconds']
    assert 'prompt' not in stats[-1]


@pytest.mark.parametrize('key', [None, 'another'])
def test_helpers_and_session_changes_clear_native_memory(key):
    model, events = model_fixture()
    complete(model, 'abc')
    events.clear()
    complete(model, 'abcx observation', key)
    assert 'clear' in events
    assert ('eval', list(b'abcx observation'), 0) in events


@pytest.mark.parametrize('rewind', [True, False])
def test_prefix_edits_require_successful_native_rewind(rewind):
    model, events = model_fixture(rewind=rewind)
    complete(model, 'abc')
    events.clear()
    stats = []
    complete(model, 'ab changed', on_stats=stats.append)
    assert ('trim', 2) in events
    assert ('clear' in events) is not rewind
    assert stats[-1]['reused_tokens'] == (2 if rewind else 0)
    assert stats[-1]['cache_status'] == ('prefix_trimmed' if rewind else 'rewind_unsupported')


def test_same_prompt_refreshes_logits_and_never_reuses_stale_output():
    model, events = model_fixture()
    complete(model, 'abc')
    events.clear()
    complete(model, 'abc')
    assert ('trim', 2) in events
    assert ('eval', [ord('c')], 2) in events


@pytest.mark.parametrize('failure', ['decode', 'cancel', 'close'])
def test_interrupted_or_failed_generation_invalidates_cache(failure):
    model, events = model_fixture()
    stats = []
    if failure == 'decode':
        def broken(*_):
            yield 'x'
            raise RuntimeError('decode failed')
        model._generate_from_n_past = broken
    stream = complete(model, 'abc', stream=True, on_stats=stats.append,
                      cancel_callback=lambda: failure == 'cancel')
    assert next(stream)['choices'][0]['delta']['content'] == 'x'
    if failure == 'close':
        stream.close()
    else:
        with pytest.raises(RuntimeError):
            list(stream)
    assert model._chat_cache_key is None
    assert model._chat_cache_tokens is None
    assert not stats[-1]['completed']


def test_non_chat_context_clear_invalidates_previous_session():
    model, _ = model_fixture()
    complete(model, 'abc')
    model._clear_context_state()
    stats = []
    complete(model, 'abcx observation', on_stats=stats.append)
    assert stats[-1]['reused_tokens'] == 0


def test_opt_in_rejects_empty_key_and_multimodal():
    model, _ = model_fixture()
    with pytest.raises(ValueError, match='cache_key'):
        complete(model, 'abc', '')
    with pytest.raises(ValueError, match='text chat only'):
        complete(model, 'abc', multimodal_context=SimpleNamespace(model=model))


def test_native_tracking_counts_only_successfully_decoded_chunks():
    model = object.__new__(Llama)
    model._ctx = None
    model._sampler = None
    model._n_batch = 2
    model._chat_cache_key = 'one'
    model._chat_cache_tokens = []
    calls = []

    def batch(size, *_):
        return SimpleNamespace(token=[0]*size, pos=[0]*size, n_seq_id=[0]*size,
                               seq_id=[[0] for _ in range(size)], logits=[0]*size)

    def decode(_, value):
        calls.append(value.token)
        return 0 if len(calls) == 1 else -1

    model._lib = SimpleNamespace(llama_batch_init=batch, llama_decode=decode,
                                 llama_batch_free=lambda _: None)
    with pytest.raises(RuntimeError, match='llama_decode failed'):
        model._eval_tokens([10, 20, 30], 0)
    assert model._chat_cache_tokens == [10, 20]
    calls.clear()
    model._eval_tokens([40], 2)
    assert model._chat_cache_tokens == [10, 20, 40]
    # An external evaluation at a different position cannot be claimed as a
    # contiguous live prefix by a later chat request.
    calls.clear()
    model._eval_tokens([50], 0)
    assert model._chat_cache_tokens is None
    assert model._chat_cache_key is None
