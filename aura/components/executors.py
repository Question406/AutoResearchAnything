from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from aura.components.llm import LLMCallable
from aura.interfaces import Experimenter
from aura.types import Experiment, ExperimentStep, Hypothesis
from aura.workspace import Workspace


class ScriptExperimenter(Experimenter):
    """Execute tasks by running a shell command.

    The command_template uses {field} placeholders that are filled from task.spec.
    Stdout is parsed as JSON for the trajectory output.

    Example:
        ScriptExperimenter("python train.py --lr {lr} --epochs {epochs}")
    """

    def __init__(self, command_template: str, timeout: int = 300, parse_json: bool = True):
        self.command_template = command_template
        self.timeout = timeout
        self.parse_json = parse_json

    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        ts = datetime.now(UTC).isoformat()

        # Build command from template + task spec
        cmd = self.command_template.format(**task.spec)

        # Constraints override constructor timeout
        timeout = (
            workspace.constraints().get("time_budget", self.timeout) if workspace else self.timeout
        )

        steps = [ExperimentStep(step=0, data={"type": "command", "cmd": cmd}, timestamp=ts)]

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

        steps.append(
            ExperimentStep(
                step=1,
                data={
                    "type": "result",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        if result.returncode != 0:
            return Experiment(
                task_id=task.id, status="failed", steps=steps, output=None, error=result.stderr
            )

        output: Any = result.stdout.strip()
        if self.parse_json:
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass

        return Experiment(
            task_id=task.id,
            status="completed",
            steps=steps,
            output=output,
            metadata={"spec": task.spec},
        )


class FunctionExperimenter(Experimenter):
    """Execute tasks by calling a Python function.

    The function receives task.spec as keyword arguments and returns the output.
    Respects constraints.time_budget as a timeout (seconds).

    Example:
        FunctionExperimenter(lambda lr, epochs, **kw: train(lr=lr, epochs=epochs))
    """

    def __init__(self, fn: Callable[..., Any], timeout: int | None = None):
        self.fn = fn
        self.timeout = timeout

    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        ts = datetime.now(UTC).isoformat()
        steps = [ExperimentStep(step=0, data={"type": "call", "spec": task.spec}, timestamp=ts)]

        timeout = self.timeout
        if workspace:
            timeout = workspace.constraints().get("time_budget", timeout)

        if timeout:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.fn, **task.spec)
                try:
                    result = future.result(timeout=timeout)
                except FuturesTimeout:
                    return Experiment(
                        task_id=task.id,
                        status="failed",
                        steps=steps,
                        output=None,
                        error=f"Timed out after {timeout}s",
                    )
        else:
            result = self.fn(**task.spec)

        steps.append(
            ExperimentStep(
                step=1,
                data={
                    "type": "result",
                    "output": result
                    if isinstance(result, (dict, list, str, int, float, bool))
                    else str(result),
                },
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        output = (
            result
            if isinstance(result, (dict, list, str, int, float, bool, type(None)))
            else str(result)
        )
        return Experiment(task_id=task.id, status="completed", steps=steps, output=output)


class LLMExperimenter(Experimenter):
    """Execute tasks by sending them to an LLM.

    The prompt_template uses {{ field }} placeholders filled from task.spec.
    """

    def __init__(self, llm: LLMCallable, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or "Complete this task:\n\n{{ query }}"

    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        from aura.utils.parsing import render_prompt

        ts = datetime.now(UTC).isoformat()

        constraints = workspace.constraints() if workspace else {}
        prompt = render_prompt(self.prompt_template, **task.spec, constraints=constraints)
        steps = [ExperimentStep(step=0, data={"type": "prompt", "prompt": prompt}, timestamp=ts)]

        response = self.llm(prompt)

        steps.append(
            ExperimentStep(
                step=1,
                data={"type": "response", "content": response},
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        return Experiment(
            task_id=task.id, status="completed", steps=steps, output={"response": response}
        )
