from __future__ import annotations

import json
import re


def extract_json(text: str):
    """Extract JSON from text, handling markdown code fences.

    Tries direct JSON parse first, then looks for ```json blocks.
    Returns the parsed Python object (dict, list, etc).
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try markdown code fence
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    # Try finding first { or [ and matching to end (try object before array)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not extract JSON from text: {text[:200]}")


def render_prompt(template: str, **kwargs) -> str:
    """Render a prompt template using Jinja2.

    Supports {{ variable }} syntax, filters, loops, and conditionals.
    Undefined variables render as empty strings.

    Examples:
        render_prompt("Hello {{ name }}", name="Alice")
        render_prompt("{% for item in items %}- {{ item }}\n{% endfor %}", items=["a", "b"])
        render_prompt("Score: {{ score | round(2) }}", score=0.856)
    """
    from jinja2 import Environment, BaseLoader, Undefined

    class _SilentUndefined(Undefined):
        """Render undefined variables as empty string instead of raising."""
        def __str__(self):
            return ""
        def __iter__(self):
            return iter([])
        def __bool__(self):
            return False

    env = Environment(
        loader=BaseLoader(),
        undefined=_SilentUndefined,
        keep_trailing_newline=True,
    )
    # Add json filter for serializing dicts/lists in prompts
    env.filters["tojson"] = lambda v, indent=2: json.dumps(v, indent=indent, default=str)

    return env.from_string(template).render(**kwargs)
