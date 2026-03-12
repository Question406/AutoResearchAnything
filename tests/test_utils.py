import json

import pytest

from aura.utils.parsing import extract_json, render_prompt


def test_extract_json_direct():
    assert extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_markdown_fence():
    text = 'Here is the result:\n```json\n[{"a": 1}]\n```\nDone.'
    assert extract_json(text) == [{"a": 1}]


def test_extract_json_fence_no_lang():
    text = 'Result:\n```\n{"key": "value"}\n```'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_embedded():
    text = 'Some text before [{"a": 1}, {"b": 2}] and after'
    assert extract_json(text) == [{"a": 1}, {"b": 2}]


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json("no json here at all")


def test_extract_json_nested():
    data = {"outer": {"inner": [1, 2, 3]}}
    assert extract_json(f"prefix {json.dumps(data)} suffix") == data


def test_render_prompt_basic():
    result = render_prompt("Hello {{ name }}, you are {{ age }}", name="Alice", age=30)
    assert result == "Hello Alice, you are 30"


def test_render_prompt_missing_var():
    result = render_prompt("Hello {{ name }}, your {{ role }}", name="Bob")
    assert result == "Hello Bob, your "


def test_render_prompt_list():
    result = render_prompt(
        "{% for item in items %}- {{ item }}\n{% endfor %}", items=["a", "b", "c"]
    )
    assert "- a" in result and "- b" in result and "- c" in result


def test_render_prompt_dict():
    result = render_prompt("Config: {{ config | tojson }}", config={"lr": 0.001})
    assert "0.001" in result


def test_render_prompt_conditional():
    template = "{% if insights %}Insights: {{ insights }}{% else %}No insights yet{% endif %}"
    assert render_prompt(template, insights="") == "No insights yet"
    assert render_prompt(template, insights="lr works") == "Insights: lr works"


def test_render_prompt_json_in_template():
    """Jinja2 {{ }} syntax doesn't conflict with JSON curly braces."""
    result = render_prompt('Respond as JSON: {"score": {{ max_score }}}', max_score=1.0)
    assert result == 'Respond as JSON: {"score": 1.0}'
