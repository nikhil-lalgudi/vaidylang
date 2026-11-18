"""Deterministic answer checks for the DeepSeek workload."""

import json

from third_party.deepseek_v41 import encoding


def _strict_json(text):
    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate JSON key")
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant {value}")

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def score_answer(expected, checker, text, *, thinking=False):
    """Score one generated continuation, preserving its DeepSeek protocol markers."""
    kind = checker["type"] if checker else "exact"

    def result(passed, reason):
        return {"checker": kind, "passed": passed, "reason": reason}

    try:
        if not text.endswith(encoding.eos_token):
            text += encoding.eos_token
        message = encoding.parse_message_from_completion_text(text, "thinking" if thinking else "chat")
        content = message["content"].strip()
        calls = message["tool_calls"]
    except (AssertionError, ValueError):
        return result(False, "invalid_completion_protocol")

    if kind == "exact":
        matched = not calls and content == expected
        return result(matched, "match" if matched else "wrong_answer")

    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    schema = checker["schema"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"Invalid answer checker schema: {error.message}") from error
    try:
        if kind == "tool":
            if content or len(calls) != 1:
                return result(False, "expected_one_tool_call")
            call = calls[0]["function"]
            name = call["name"]
            if "namespace" in calls[0]:
                name = calls[0]["namespace"] + "::" + name
            actual = {"name": name, "arguments": _strict_json(call["arguments"])}
        else:
            if calls:
                return result(False, "unexpected_tool_call")
            actual = _strict_json(content)
    except ValueError:
        return result(False, "invalid_json")

    expected_value = _strict_json(expected)
    validator = Draft202012Validator(schema)
    if not validator.is_valid(expected_value):
        raise ValueError("Expected answer does not satisfy its checker schema")
    if not validator.is_valid(actual):
        return result(False, "schema_mismatch")
    matched = actual == expected_value
    return result(matched, "match" if matched else "wrong_answer")
