import pytest

from llama_cpp_py_sync._cffi_bindings import ffi
from llama_cpp_py_sync.constraints import (
    JSON_OBJECT_GRAMMAR,
    resolve_grammar,
    response_format_to_grammar,
)
from llama_cpp_py_sync.llama import Llama


def test_json_object_response_format_uses_generic_json_grammar():
    grammar = response_format_to_grammar({"type": "json_object"})

    assert grammar == JSON_OBJECT_GRAMMAR
    assert "root ::= object" in grammar


def test_json_schema_response_format_constrains_common_object_shape():
    grammar = response_format_to_grammar(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    )

    assert 'root-name-kv ::= "\\\"name\\\"" space ":" space string' in grammar
    assert 'root-age-kv ::= "\\\"age\\\"" space ":" space integral' in grammar
    assert 'root ::= "{" space root-name-kv' in grammar


def test_json_schema_optional_properties_are_independently_optional():
    grammar = response_format_to_grammar(
        {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "first": {"type": "string"},
                    "second": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        }
    )

    assert "root-first-kv root-first-rest" in grammar
    assert "root-first-rest ::=" in grammar
    assert "| root-second-kv" in grammar


def test_empty_schema_allows_any_json_value():
    grammar = response_format_to_grammar(
        {"type": "json_schema", "schema": {}}
    )

    assert "root ::= value" in grammar


def test_prefix_items_requires_explicitly_closed_tuple():
    with pytest.raises(ValueError, match="items=false"):
        response_format_to_grammar(
            {
                "type": "json_schema",
                "schema": {"type": "array", "prefixItems": [{"type": "string"}]},
            }
        )


def test_invalid_array_bounds_fail_closed():
    with pytest.raises(ValueError, match="bounds are invalid"):
        response_format_to_grammar(
            {
                "type": "json_schema",
                "schema": {"type": "array", "items": {}, "minItems": 1, "maxItems": 0},
            }
        )


def test_unsupported_schema_constraints_fail_closed():
    with pytest.raises(ValueError, match="minimum"):
        response_format_to_grammar(
            {"type": "json_schema", "schema": {"type": "integer", "minimum": 1}}
        )


def test_grammar_and_constrained_response_format_cannot_be_combined():
    with pytest.raises(ValueError, match="cannot be used together"):
        resolve_grammar("root ::= \"ok\"", "root", {"type": "json_object"})


class _FakeLib:
    def __init__(self):
        self.calls = []

    def llama_sampler_chain_default_params(self):
        return object()

    def llama_sampler_chain_init(self, _params):
        return "chain"

    def llama_sampler_chain_add(self, _chain, sampler):
        self.calls.append(("add", sampler))

    def llama_sampler_init_penalties(self, *_args):
        return "penalties"

    def llama_sampler_init_top_k(self, _value):
        return "top_k"

    def llama_sampler_init_top_p(self, _value, _keep):
        return "top_p"

    def llama_sampler_init_min_p(self, _value, _keep):
        return "min_p"

    def llama_sampler_init_temp(self, _value):
        return "temp"

    def llama_sampler_init_dist(self, _seed):
        return "dist"

    def llama_sampler_init_grammar(self, _vocab, grammar, root):
        self.calls.append(("grammar", ffi.string(grammar), ffi.string(root)))
        return "grammar"

    def llama_vocab_n_tokens(self, _vocab):
        return 128

    def llama_sampler_free(self, _sampler):
        pass


def test_native_sampler_chain_adds_grammar_before_distribution():
    model = object.__new__(Llama)
    model._lib = _FakeLib()
    model._ffi = ffi
    model._vocab = object()
    model._sampler = None

    model._configure_generation_sampler(
        0.0,
        40,
        0.95,
        0.05,
        1.1,
        64,
        123,
        'root ::= "ok"',
        "root",
    )

    assert ("grammar", b'root ::= "ok"', b"root") in model._lib.calls
    added = [call[1] for call in model._lib.calls if call[0] == "add"]
    assert added.index("grammar") < added.index("dist")
