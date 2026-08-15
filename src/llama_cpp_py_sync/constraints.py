"""Helpers for llama.cpp grammar and response-format constraints.

The public llama.cpp API accepts GBNF through ``llama_sampler_init_grammar``.
OpenAI-style response formats are a higher-level convenience, so this module
maps the supported response-format shapes to GBNF before the native sampler is
created.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

GENERIC_JSON_GRAMMAR = r'''
root ::= value
value ::= object | array | string | number | boolean | null
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
array ::= "[" space ( value ("," space value)* )? space "]"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
number ::= ("-"? integral) ("." [0-9]+)? ([eE] [-+]? integral)?
integral ::= "0" | [1-9] [0-9]*
boolean ::= "true" | "false"
null ::= "null"
space ::= " " | "\n"{1,2} [ \t]{0,20}
'''.strip()

JSON_OBJECT_GRAMMAR = GENERIC_JSON_GRAMMAR.replace("root ::= value", "root ::= object", 1)


def _grammar_literal(value: Any) -> str:
    """Encode one JSON value as a GBNF literal containing its JSON spelling."""
    json_text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    escaped = json_text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _rule_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-") or "rule"
    if name[0].isdigit():
        name = f"rule-{name}"
    return name


class _JsonSchemaGrammar:
    """Small, strict JSON Schema to GBNF converter.

    This intentionally implements the structural subset used by the upstream
    server's common response-format examples. Unsupported constraints raise a
    clear error instead of being ignored and producing weaker output.
    """

    def __init__(self) -> None:
        self.rules: dict[str, str] = {
            "space": '" " | "\\n"{1,2} [ \\t]{0,20}',
            "char": r'[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})',
            "string": r'"\"" char* "\""',
            "number": r'("-"? integral) ("." [0-9]+)? ([eE] [-+]? integral)?',
            "integral": '"0" | [1-9] [0-9]*',
            "boolean": '"true" | "false"',
            "null": '"null"',
            "value": "object | array | string | number | boolean | null",
            "object": '"{" space ( string ":" space value ("," space string ":" space value)* )? space "}"',
            "array": '"[" space ( value ("," space value)* )? space "]"',
        }
        self._used_names = set(self.rules)

    def _unique_name(self, requested: str) -> str:
        base = _rule_name(requested)
        name = base
        suffix = 1
        while name in self._used_names:
            name = f"{base}-{suffix}"
            suffix += 1
        self._used_names.add(name)
        return name

    def _add(self, requested: str, expression: str) -> str:
        name = self._unique_name(requested)
        self.rules[name] = expression
        return name

    def _unsupported(self, schema: Mapping[str, Any], keys: set[str]) -> None:
        unsupported = sorted(key for key in schema if key in keys)
        if unsupported:
            raise ValueError(
                "Unsupported JSON Schema constraint(s) for llama.cpp GBNF: "
                + ", ".join(unsupported)
            )

    def _scalar_rule(self, name: str, value: Any) -> str:
        return self._add(name, _grammar_literal(value))

    def _repeat(self, item: str, minimum: int, maximum: int | None) -> str:
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("JSON Schema repetition bounds are invalid")
        if maximum == 0:
            return ""
        if maximum is None:
            return f"{item}{{{minimum},}}"
        if minimum == maximum:
            return f"{item}{{{minimum}}}"
        return f"{item}{{{minimum},{maximum}}}"

    def _object(self, schema: Mapping[str, Any], name: str) -> str:
        self._unsupported(
            schema,
            {
                "minProperties",
                "maxProperties",
                "patternProperties",
                "propertyNames",
                "dependentRequired",
                "dependentSchemas",
                "unevaluatedProperties",
            },
        )
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError("JSON Schema 'properties' must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("JSON Schema 'required' must be a list of property names")
        unknown_required = sorted(set(required) - set(properties))
        if unknown_required:
            raise ValueError("JSON Schema requires unknown properties: " + ", ".join(unknown_required))

        additional = schema.get("additionalProperties", False)
        if additional not in (False, None) and properties:
            raise ValueError(
                "JSON Schema additionalProperties with declared properties is not supported; "
                "use an explicit property schema or a custom GBNF grammar"
            )
        if not properties and isinstance(additional, Mapping):
            raise ValueError(
                "JSON Schema additionalProperties schemas are not supported; "
                "use a custom GBNF grammar"
            )
        if not properties and additional is True:
            return "object"

        property_rules: dict[str, str] = {}
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str) or not isinstance(property_schema, Mapping):
                raise ValueError("JSON Schema properties must map names to schema objects")
            value_rule = self._visit(property_schema, f"{name}-{property_name}")
            property_rules[property_name] = self._add(
                f"{name}-{property_name}-kv",
                f"{_grammar_literal(property_name)} space \":\" space {value_rule}",
            )

        required_names = [key for key in properties if key in required]
        optional_names = [key for key in properties if key not in required]

        def sequence(keys: list[str]) -> str:
            if not keys:
                return ""
            return ' "," space '.join(property_rules[key] for key in keys)

        required_part = sequence(required_names)
        body = required_part
        if optional_names:
            # Match upstream's ordered-object strategy while allowing every
            # subset of optional properties.  A single optional wrapper around
            # ``a, b, c`` would incorrectly allow only all three or none.
            def optional_tail(keys: list[str], first: bool) -> str:
                key, *remaining = keys
                item = property_rules[key]
                expression = item if first else f'( "," space {item} )?'
                if remaining:
                    tail_rule = self._add(
                        f"{name}-{key}-rest",
                        optional_tail(remaining, False),
                    )
                    expression = f"{expression} {tail_rule}"
                return expression

            alternatives = " | ".join(
                optional_tail(optional_names[index:], True)
                for index in range(len(optional_names))
            )
            if required_part:
                body = f'{required_part} ( "," space ( {alternatives} ) )?'
            else:
                body = f"( {alternatives} )?"
        return self._add(name, f'"{{" space {body} space "}}"')

    def _array(self, schema: Mapping[str, Any], name: str) -> str:
        self._unsupported(schema, {"contains", "minContains", "maxContains"})
        if schema.get("uniqueItems"):
            raise ValueError("JSON Schema uniqueItems is not supported by GBNF")
        if "prefixItems" in schema:
            self._unsupported(schema, {"minItems", "maxItems"})
            prefix = schema["prefixItems"]
            if not isinstance(prefix, list):
                raise ValueError("JSON Schema prefixItems must be a list")
            if schema.get("items") is not False:
                raise ValueError(
                    "JSON Schema prefixItems is supported only with items=false; "
                    "trailing items require a custom GBNF grammar"
                )
            item_rules = [self._visit(item, f"{name}-item-{index}") for index, item in enumerate(prefix)]
            return self._add(name, '"[" space ' + ' "," space '.join(item_rules) + ' space "]"')

        item_schema = schema.get("items", {})
        if item_schema is False:
            return self._add(name, '"[" space "]"')
        if item_schema is True:
            item_rule = "value"
        elif isinstance(item_schema, Mapping):
            item_rule = self._visit(item_schema, f"{name}-item")
        else:
            raise ValueError("JSON Schema 'items' must be a schema object or boolean")
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or (
                maximum is not None
                and (not isinstance(maximum, int) or isinstance(maximum, bool))
            )
        ):
            raise ValueError("JSON Schema array bounds must be integers")
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("JSON Schema array bounds are invalid")
        separator = f'( "," space {item_rule} )'
        if maximum == 0:
            repetition = ""
        else:
            tail_maximum = None if maximum is None else maximum - 1
            tail = self._repeat(separator, max(0, minimum - 1), tail_maximum)
            repetition = f"{item_rule} {tail}"
            if minimum == 0:
                repetition = f"( {repetition} )?"
        return self._add(name, f'"[" space {repetition} space "]"')

    def _visit(self, schema: Mapping[str, Any], name: str) -> str:
        if not isinstance(schema, Mapping):
            raise ValueError("Each JSON Schema node must be an object")
        if not schema:
            return "value"
        schema_type = schema.get("type")
        if schema_type != "string":
            self._unsupported(schema, {"minLength", "maxLength"})
        self._unsupported(
            schema,
            {
                "$ref",
                "allOf",
                "not",
                "if",
                "then",
                "else",
                "format",
                "pattern",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "minimum",
                "maximum",
                "multipleOf",
            },
        )
        if "const" in schema:
            return self._scalar_rule(name, schema["const"])
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise ValueError("JSON Schema enum must be a non-empty list")
            return self._add(name, " | ".join(_grammar_literal(value) for value in values))
        alternatives = schema.get("oneOf", schema.get("anyOf"))
        if alternatives is not None:
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError("JSON Schema oneOf/anyOf must be a non-empty list")
            return self._add(
                name,
                " | ".join(self._visit(item, f"{name}-alternative-{index}") for index, item in enumerate(alternatives)),
            )
        if isinstance(schema_type, list):
            return self._visit(
                {"anyOf": [{**dict(schema), "type": item} for item in schema_type]},
                name,
            )
        if schema_type in (None, "object") and ("properties" in schema or "additionalProperties" in schema):
            return self._object(schema, name)
        if schema_type in (None, "array") and ("items" in schema or "prefixItems" in schema):
            return self._array(schema, name)
        if schema_type == "string":
            minimum = schema.get("minLength", 0)
            maximum = schema.get("maxLength")
            if (
                not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or (
                    maximum is not None
                    and (not isinstance(maximum, int) or isinstance(maximum, bool))
                )
            ):
                raise ValueError("JSON Schema string bounds must be integers")
            if minimum < 0 or (maximum is not None and maximum < minimum):
                raise ValueError("JSON Schema string bounds are invalid")
            if minimum or maximum is not None:
                return self._add(name, f'"\\\"" char{{{minimum},{"" if maximum is None else maximum}}} "\\\""')
            return "string"
        if schema_type == "number":
            return "number"
        if schema_type == "integer":
            return "integral"
        if schema_type == "boolean":
            return "boolean"
        if schema_type == "null":
            return "null"
        if schema_type is None:
            return "value"
        if schema_type == "object":
            return "object"
        if schema_type == "array":
            return "array"
        raise ValueError(f"Unsupported JSON Schema type: {schema_type!r}")

    def convert(self, schema: Mapping[str, Any]) -> str:
        root = self._visit(schema, "root")
        if root != "root":
            self.rules["root"] = root
        return "\n".join(f"{name} ::= {expression}" for name, expression in self.rules.items())


def response_format_to_grammar(response_format: Any) -> str | None:
    """Convert an OpenAI-style response format to a GBNF grammar.

    Supported values are ``{"type": "text"}``, ``{"type": "json_object"}``,
    and ``{"type": "json_schema", "json_schema": {"schema": ...}}``.
    ``json_object`` may also contain a direct ``schema`` field, matching the
    upstream server's compatibility form.
    """
    if response_format is None:
        return None
    if isinstance(response_format, str):
        response_format = {"type": response_format}
    if not isinstance(response_format, Mapping):
        raise TypeError("response_format must be a mapping, a format name, or None")
    response_type = response_format.get("type")
    if response_type == "text" or response_type is None:
        if response_type is None:
            raise ValueError("response_format requires a 'type' field")
        return None
    if response_type == "json_object":
        schema = response_format.get("schema")
        if schema is None:
            return JSON_OBJECT_GRAMMAR
    elif response_type == "json_schema":
        wrapper = response_format.get("json_schema", response_format)
        if not isinstance(wrapper, Mapping) or "schema" not in wrapper:
            raise ValueError("json_schema response_format requires json_schema.schema")
        schema = wrapper["schema"]
    else:
        raise ValueError(
            "response_format type must be one of 'text', 'json_object', or 'json_schema'"
        )
    if not isinstance(schema, Mapping):
        raise TypeError("response_format schema must be a mapping")
    return _JsonSchemaGrammar().convert(schema)


def resolve_grammar(grammar: str | None, grammar_root: str, response_format: Any) -> tuple[str | None, str]:
    """Validate and combine explicit GBNF and response-format constraints."""
    if not isinstance(grammar_root, str) or not grammar_root.strip():
        raise ValueError("grammar_root must be a non-empty string")
    response_grammar = response_format_to_grammar(response_format)
    if response_grammar is not None and grammar_root.strip() != "root":
        raise ValueError("grammar_root is only applicable to an explicit grammar")
    if grammar is not None:
        if not isinstance(grammar, str) or not grammar.strip():
            raise ValueError("grammar must be a non-empty GBNF string or None")
        if response_grammar is not None:
            raise ValueError("grammar and a constrained response_format cannot be used together")
        return grammar, grammar_root.strip()
    return response_grammar, grammar_root.strip()
