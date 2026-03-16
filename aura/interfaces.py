from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aura.types import Evaluation, Experiment, Hypothesis, Insight, Trial
    from aura.workspace import Workspace


class Runner(ABC):
    """AURA-agnostic execution backend.

    Receives a prompt template + plain context dict.
    Handles prompt rendering and execution.
    Returns a result dict.
    """

    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def run(self, prompt_template: str, context: dict) -> dict:
        """Render and execute a prompt.

        Args:
            prompt_template: Jinja2 template string.
            context: Plain strings/ints/dicts — no AURA types.

        Returns:
            Dict with at least "content" (str). Optional:
            - "structured" (dict|list) — pre-parsed, bypasses extract_json
            - "steps" (list[dict]) — intermediate steps
            - "files" (dict[str,str]) — files created/modified
            - "metadata" (dict) — token usage, timing
        """
        ...


class Researcher(ABC):
    def __init__(self, runner: Runner | None = None, prompt_template: str | None = None, **config):
        self.runner = None
        self.prompt_template = prompt_template
        self.config = config
        if runner is not None:
            from aura.components.runners import as_runner

            self.runner = as_runner(runner)

    def setup(self, workspace: Workspace) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.setup(workspace)

    def teardown(self) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.teardown()

    def hypothesize(self, insights: list[Insight], workspace: Workspace) -> list[Hypothesis]:
        if getattr(self, "runner", None) is None:
            raise NotImplementedError("Provide a runner or subclass and override hypothesize()")

        from aura.types import Hypothesis
        from aura.utils.parsing import extract_json

        context = {
            "insights": (
                "\n".join(f"- {i.content}" for i in insights)
                if insights
                else "None yet (first iteration)"
            ),
            "inputs": self._read_inputs(workspace),
            "iteration": workspace.current_iteration(),
            "workspace_root": str(workspace.root),
            "role": "researcher",
            **self.config,
        }

        response = self.runner.run(self.prompt_template, context)

        items = response.get("structured") or extract_json(response["content"])
        if isinstance(items, dict):
            items = [items]
        return [
            Hypothesis(
                id=item.pop("id", str(uuid.uuid4())[:8]),
                spec=item,
                metadata=item.pop("metadata", {}),
            )
            for item in items
        ]

    def _read_inputs(self, workspace: Workspace) -> str:
        """Read all files from inputs/ directory as context."""
        inputs_dir = workspace.inputs_dir()
        parts = []
        for f in sorted(inputs_dir.rglob("*")):
            if f.is_file():
                try:
                    content = f.read_text()
                    rel_path = f.relative_to(inputs_dir)
                    parts.append(f"--- {rel_path} ---\n{content}")
                except (UnicodeDecodeError, OSError):
                    continue
        return "\n\n".join(parts) if parts else "No input files."


class Experimenter(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment: ...


class SingleTrialExperimenter(Experimenter):
    """Base for experimenters that run exactly one trial per hypothesis.

    Override prepare / execute / collect / cleanup for lifecycle composition.
    run_experiment() is provided — no need to override it.
    """

    def __init__(self, aggregator=None, trial_inputs: list[Path] | None = None):
        from aura.components.aggregators import LastTrialAggregator

        self.aggregator = aggregator or LastTrialAggregator()
        self.trial_inputs = [Path(p) for p in (trial_inputs or [])]

    def prepare(self, task: Hypothesis, workspace: Workspace) -> dict:
        """Create a per-trial working directory and return it in context."""
        trial_dir = workspace.trial_dir(task.id)
        return {"trial_dir": trial_dir, "trial_inputs": self.trial_inputs}

    @abstractmethod
    def execute(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any: ...

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        """Convert raw executor output into a Trial. Override for custom collection."""
        from datetime import UTC, datetime

        from aura.types import Trial

        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[],
            output=raw if isinstance(raw, (dict, list, str, int, float, bool, type(None))) else str(raw),
        )

    def cleanup(self, task: Hypothesis, context: dict, workspace: Workspace) -> None:
        """Tear down resources created in prepare()."""
        pass

    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        from aura.types import Experiment, Trial

        context: dict = {}
        trial: Trial | None = None
        try:
            context = self.prepare(task, workspace)
            raw = self.execute(task, context, workspace)
            trial = self.collect(task, raw, context, workspace)
        except Exception as exc:
            trial = Trial(
                id=f"{task.id}-{str(uuid.uuid4())[:4]}",
                spec=task.spec,
                status="failed",
                steps=[],
                output=None,
                error=str(exc),
            )
        finally:
            try:
                self.cleanup(task, context, workspace)
            except Exception:
                pass

        trials = [trial] if trial is not None else []
        summary = self.aggregator.aggregate(trials)
        status = (
            "completed"
            if trials and all(t.status == "completed" for t in trials)
            else "failed"
        )
        first = trials[0] if trials else None
        return Experiment(
            hypothesis_id=task.id,
            status=status,
            trials=trials,
            summary=summary,
            # Populate legacy fields for backward compatibility
            steps=first.steps if first else [],
            output=first.output if first else None,
            error=first.error if first else None,
        )


# --- Sub-ABCs for composable backends ---


class Environment(ABC):
    """Manages the execution environment for experiments (conda, docker, venv, etc.)."""

    def setup(self, task: Hypothesis, workspace: Workspace) -> dict:
        """Set up the environment. Return a context dict."""
        return {}

    def teardown(self, context: dict, workspace: Workspace) -> None:
        """Tear down any resources created during setup."""
        pass


class Executor(ABC):
    """Runs a hypothesis inside a given environment."""

    @abstractmethod
    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any: ...


class Collector(ABC):
    """Converts raw executor output into a Trial."""

    @abstractmethod
    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial: ...


class Evaluator(ABC):
    def __init__(self, runner: Runner | None = None, prompt_template: str | None = None, **config):
        self.runner = None
        self.prompt_template = prompt_template or (
            "Evaluate this result.\n\n"
            "Task: {{ task }}\n\n"
            "Output: {{ output }}\n\n"
            'Rate on 0.0-1.0. Respond in JSON: {"score": float, "passed": bool, "reason": string}'
        )
        self.config = config
        if runner is not None:
            from aura.components.runners import as_runner

            self.runner = as_runner(runner)

    def setup(self, workspace: Workspace) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.setup(workspace)

    def teardown(self) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.teardown()

    def evaluate(
        self, task: Hypothesis, experiment: Experiment, workspace: Workspace
    ) -> Evaluation:
        if getattr(self, "runner", None) is None:
            raise NotImplementedError("Provide a runner or subclass and override evaluate()")

        from aura.types import Evaluation
        from aura.utils.parsing import extract_json

        if experiment.status == "failed":
            return Evaluation(
                task_id=task.id,
                score=0.0,
                passed=False,
                details={"reason": "execution_failed", "error": experiment.error},
            )

        context = {
            "task": task.spec,
            "output": experiment.summary,
            "trajectory": [s.data for s in experiment.steps],
            "workspace_root": str(workspace.root),
            "role": "evaluator",
            **self.config,
        }

        response = self.runner.run(self.prompt_template, context)
        result = response.get("structured") or extract_json(response["content"])
        return Evaluation(
            task_id=task.id,
            score=float(result.get("score", 0.0)),
            passed=bool(result.get("passed", False)),
            details={"reason": result.get("reason", "")},
        )


class Reviewer(ABC):
    def __init__(self, runner: Runner | None = None, prompt_template: str | None = None, **config):
        self.runner = None
        self.prompt_template = prompt_template or (
            "Analyze these experiment results:\n\n"
            "{{ results }}\n\n"
            "Extract 2-3 insights. What worked? What failed? What to try next?\n\n"
            'Respond as JSON list: [{"finding": string, "recommendation": string}]'
        )
        self.config = config
        if runner is not None:
            from aura.components.runners import as_runner

            self.runner = as_runner(runner)

    def setup(self, workspace: Workspace) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.setup(workspace)

    def teardown(self) -> None:
        runner = getattr(self, "runner", None)
        if runner:
            runner.teardown()

    def review(
        self,
        tasks: list[Hypothesis],
        experiments: list[Experiment],
        evaluations: list[Evaluation],
        workspace: Workspace,
    ) -> list[Insight]:
        if getattr(self, "runner", None) is None:
            raise NotImplementedError("Provide a runner or subclass and override review()")

        from aura.types import Insight
        from aura.utils.parsing import extract_json

        eval_by_id = {e.task_id: e for e in evaluations}
        exp_by_id = {e.task_id: e for e in experiments}

        lines = []
        for task in tasks:
            ev = eval_by_id.get(task.id)
            exp = exp_by_id.get(task.id)
            output_summary = exp.summary if exp and exp.summary else "N/A"
            if isinstance(output_summary, dict):
                output_summary = ", ".join(f"{k}={v}" for k, v in output_summary.items())
            lines.append(
                f"- Task {task.id} (spec: {task.spec}): "
                f"output={output_summary}, "
                f"score={ev.score if ev else 'N/A'}, "
                f"passed={ev.passed if ev else 'N/A'}"
            )

        context = {
            "results": "\n".join(lines) if lines else "No results.",
            "iteration": workspace.current_iteration(),
            "workspace_root": str(workspace.root),
            "role": "reviewer",
            **self.config,
        }

        response = self.runner.run(self.prompt_template, context)
        items = response.get("structured") or extract_json(response["content"])
        if isinstance(items, dict):
            items = [items]
        return [
            Insight(
                id=str(uuid.uuid4())[:8],
                source_iteration=workspace.current_iteration(),
                content=item,
            )
            for item in items
        ]
