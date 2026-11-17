"""Model lifecycle and sync validation without removed upstream aliases."""
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llama_cpp_py_sync.llama import Llama

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(('property_name', 'symbol', 'value'), [
    ('n_embd', 'llama_model_n_embd', 4096),
    ('n_layer', 'llama_model_n_layer', 32),
])
def test_model_dimensions_use_current_api_without_legacy_aliases(property_name, symbol, value):
    model = object.__new__(Llama)
    pointer = object()
    model._model = pointer
    calls = []
    def query(actual):
        calls.append(actual)
        return value
    model._lib = SimpleNamespace(**{symbol: query})
    try:
        assert getattr(model, property_name) == value
        assert calls == [pointer]
    finally:
        model._model = None


def test_close_releases_context_then_model_with_current_api_once():
    model = object.__new__(Llama)
    context, pointer = object(), object()
    model._ctx, model._model = context, pointer
    calls = []
    model._lib = SimpleNamespace(
        llama_free=lambda p: calls.append(('context', p)),
        llama_model_free=lambda p: calls.append(('model', p)),
    )
    model.close()
    model.close()
    assert calls == [('context', context), ('model', pointer)]
    assert model._ctx is None and model._model is None


def test_sync_validator_accepts_bindings_without_removed_model_aliases(tmp_path):
    source = (ROOT / 'src/llama_cpp_py_sync/_cffi_bindings.py').read_text(encoding='utf-8')
    for symbol in ('llama_free_model', 'llama_load_model_from_file', 'llama_n_embd',
                   'llama_n_layer', 'llama_new_context_with_model'):
        source = re.sub(r'(?m)^[^\n;{}]*\b' + symbol + r'\s*\([^;{}]*\);', '', source)
        assert not re.search(r'\b' + symbol + r'\s*\(', source)
    bindings = tmp_path / '_cffi_bindings.py'
    bindings.write_text(source, encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_high_level_api.py'),
                             '--bindings', str(bindings)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
