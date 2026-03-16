from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING

from aura.interfaces import Runner
from aura.utils.parsing import render_prompt

if TYPE_CHECKING:
    from aura.components.llm import LLMCallable
    from aura.workspace import Workspace


class LLMRunner(Runner):
    """Single LLM API call."""

    def __init__(self, llm: LLMCallable):
        self.llm = llm

    def run(self, prompt_template: str, context: dict) -> dict:
        prompt = render_prompt(prompt_template, **context)
        return {"content": self.llm(prompt)}


class CommandRunner(Runner):
    """CLI agent (claude, codex, aider, etc.)."""

    def __init__(self, command: list[str], timeout: int = 600, output_format: str = "text"):
        self.command = command
        self.timeout = timeout
        self.output_format = output_format

    def setup(self, workspace: Workspace) -> None:
        subprocess.run([self.command[0], "--version"], capture_output=True, check=True)

    def run(self, prompt_template: str, context: dict) -> dict:
        prompt = render_prompt(prompt_template, **context)
        workdir = context.get("workspace_root")
        result = subprocess.run(
            self.command + [prompt],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {result.stderr[-500:]}"
            )
        content = result.stdout.strip()
        if self.output_format == "json":
            envelope = json.loads(content)
            return {"content": envelope.get("result", content), "raw": envelope}
        return {"content": content}


class FunctionRunner(Runner):
    """Wraps a Python callable."""

    def __init__(self, fn: Callable[[str, dict], dict | str]):
        self.fn = fn

    def run(self, prompt_template: str, context: dict) -> dict:
        prompt = render_prompt(prompt_template, **context)
        result = self.fn(prompt, context)
        if isinstance(result, str):
            return {"content": result}
        return result


def as_runner(llm: LLMCallable | Runner) -> Runner:
    """Normalize LLMCallable or Runner into a Runner."""
    if isinstance(llm, Runner):
        return llm
    return LLMRunner(llm)
