from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aura.types import Evaluation, Experiment, Hypothesis, Insight, Trial
    from aura.workspace import Workspace


class Researcher(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def hypothesize(self, insights: list[Insight], workspace: Workspace) -> list[Hypothesis]: ...


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
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def evaluate(
        self, task: Hypothesis, experiment: Experiment, workspace: Workspace
    ) -> Evaluation: ...


class Reviewer(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def review(
        self,
        tasks: list[Hypothesis],
        experiments: list[Experiment],
        evaluations: list[Evaluation],
        workspace: Workspace,
    ) -> list[Insight]: ...
