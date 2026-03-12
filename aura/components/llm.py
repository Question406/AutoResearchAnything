from __future__ import annotations

from typing import Callable

# The universal LLM interface: prompt string in, response string out
LLMCallable = Callable[[str], str]


def anthropic_llm(model: str = "claude-sonnet-4-20250514", max_tokens: int = 2000, **kwargs) -> LLMCallable:
    """Create an LLM callable using the Anthropic SDK.

    Requires `anthropic` package. Set ANTHROPIC_API_KEY env var.
    """
    def call(prompt: str) -> str:
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text
    return call


def openai_llm(model: str = "gpt-4o", max_tokens: int = 2000, **kwargs) -> LLMCallable:
    """Create an LLM callable using the OpenAI SDK.

    Requires `openai` package. Set OPENAI_API_KEY env var.
    """
    def call(prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
    return call


def command_llm(cmd_template: str = "claude -p '{prompt}' --output-format text") -> LLMCallable:
    """Create an LLM callable that shells out to a CLI command.

    Use {prompt} placeholder in the command template.
    Works with claude CLI, codex, aider, etc.
    """
    import subprocess
    import shlex

    def call(prompt: str) -> str:
        # Escape the prompt for shell safety
        escaped = prompt.replace("'", "'\\''")
        cmd = cmd_template.replace("{prompt}", escaped)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout.strip()
    return call
