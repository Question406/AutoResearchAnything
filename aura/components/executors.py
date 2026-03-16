from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from aura.components.aggregators import LastTrialAggregator
from aura.components.backends.collector_backends import StdoutCollector
from aura.components.backends.executor_backends import ScriptExecutor
from aura.components.llm import LLMCallable
from aura.interfaces import SingleTrialExperimenter
from aura.types import Hypothesis, Trial, TrialStep
from aura.workspace import Workspace


class ScriptExperimenter(SingleTrialExperimenter):
    """Execute tasks by running a shell command.

    The command_template uses {field} placeholders that are filled from task.spec.
    Stdout is parsed as JSON for the experiment output.

    Example::

        ScriptExperimenter("python train.py --lr {lr} --epochs {epochs}")
    """

    def __init__(self, command_template: str, timeout: int = 300, parse_json: bool = True):
        super().__init__(aggregator=LastTrialAggregator())
        self.command_template = command_template
        self.timeout = timeout
        self.parse_json = parse_json

    def execute(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any:
        return ScriptExecutor(self.command_template, self.timeout).run(task, context, workspace)

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        return StdoutCollector(self.parse_json).collect(task, raw, context, workspace)


class FunctionExperimenter(SingleTrialExperimenter):
    """Execute tasks by calling a Python function.

    The function receives task.spec as keyword arguments and returns the output.
    Respects constraints.time_budget as a timeout (seconds).

    Example::

        FunctionExperimenter(lambda lr, epochs, **kw: train(lr=lr, epochs=epochs))
    """

    def __init__(self, fn: Callable[..., Any], timeout: int | None = None):
        super().__init__(aggregator=LastTrialAggregator())
        self.fn = fn
        self.timeout = timeout

    def execute(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        timeout = self.timeout
        if workspace:
            timeout = workspace.constraints().get("time_budget", timeout)

        if timeout:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.fn, **task.spec)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeout:
                    raise RuntimeError(f"Timed out after {timeout}s")
        else:
            return self.fn(**task.spec)

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        ts = datetime.now(UTC).isoformat()
        output = (
            raw
            if isinstance(raw, (dict, list, str, int, float, bool, type(None)))
            else str(raw)
        )
        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[
                TrialStep(step=0, data={"type": "call", "spec": task.spec}, timestamp=ts),
                TrialStep(
                    step=1,
                    data={"type": "result", "output": output},
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            ],
            output=output,
        )


class LLMExperimenter(SingleTrialExperimenter):
    """Execute tasks by sending them to a Runner or LLM.

    The prompt_template uses {{ field }} placeholders filled from task.spec.
    Accepts either a Runner instance or an LLMCallable.
    """

    def __init__(
        self, llm: LLMCallable | None = None, prompt_template: str | None = None, runner=None
    ):
        super().__init__(aggregator=LastTrialAggregator())
        self.prompt_template = prompt_template or "Complete this task:\n\n{{ query }}"
        if runner is not None:
            from aura.components.runners import as_runner

            self._runner = as_runner(runner)
            self.llm = None
        elif llm is not None:
            self._runner = None
            self.llm = llm
        else:
            raise TypeError("Either 'llm' or 'runner' must be provided")

    def execute(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any:
        from aura.utils.parsing import render_prompt

        if self._runner is not None:
            ctx = {
                **task.spec,
                "role": "experimenter",
                "workspace_root": str(workspace.root) if workspace else ".",
                "trial_dir": str(context.get("trial_dir", ".")),
                "constraints": workspace.constraints() if workspace else {},
            }
            response = self._runner.run(self.prompt_template, ctx)
            return {"prompt": self.prompt_template, "response": response["content"]}

        constraints = workspace.constraints() if workspace else {}
        prompt = render_prompt(self.prompt_template, **task.spec, constraints=constraints)
        assert self.llm is not None
        response = self.llm(prompt)
        return {"prompt": prompt, "response": response}

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        ts = datetime.now(UTC).isoformat()
        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[
                TrialStep(step=0, data={"type": "prompt", "prompt": raw["prompt"]}, timestamp=ts),
                TrialStep(
                    step=1,
                    data={"type": "response", "content": raw["response"]},
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            ],
            output={"response": raw["response"]},
        )
