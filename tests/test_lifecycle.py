"""Tests for SingleTrialExperimenter lifecycle (prepare/execute/collect/cleanup)."""

from pathlib import Path
from typing import Any

import pytest

from aura.interfaces import SingleTrialExperimenter
from aura.types import Hypothesis, Trial
from aura.workspace import Workspace


class EchoExperimenter(SingleTrialExperimenter):
    """Returns task.spec as the output — minimal concrete subclass."""

    def execute(self, task: Hypothesis, context: dict, workspace: Workspace) -> Any:
        return task.spec


class LifecycleTracker(SingleTrialExperimenter):
    """Records which lifecycle methods were called."""

    def __init__(self):
        super().__init__()
        self.log: list[str] = []

    def prepare(self, task, workspace):
        self.log.append("prepare")
        return {"prepared": True}

    def execute(self, task, context, workspace):
        assert context.get("prepared"), "context from prepare must be passed to execute"
        self.log.append("execute")
        return {"result": 42}

    def collect(self, task, raw, context, workspace):
        self.log.append("collect")
        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[],
            output=raw,
        )

    def cleanup(self, task, context, workspace):
        self.log.append("cleanup")


class FailingExecuteExperimenter(SingleTrialExperimenter):
    def __init__(self):
        super().__init__()
        self.cleanup_called = False

    def execute(self, task, context, workspace):
        raise ValueError("execution error")

    def cleanup(self, task, context, workspace):
        self.cleanup_called = True


# --- Basic run_experiment ---


def test_echo_experimenter(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    exp_impl = EchoExperimenter()
    task = Hypothesis(id="t1", spec={"x": 1})
    result = exp_impl.run_experiment(task, ws)

    assert result.status == "completed"
    assert result.hypothesis_id == "t1"
    assert result.task_id == "t1"  # backward-compat property
    assert result.summary == {"x": 1}
    assert result.output == {"x": 1}  # legacy field also populated
    assert len(result.trials) == 1
    assert result.trials[0].status == "completed"


# --- Lifecycle order ---


def test_lifecycle_order(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    tracker = LifecycleTracker()
    task = Hypothesis(id="t1", spec={})

    result = tracker.run_experiment(task, ws)

    assert tracker.log == ["prepare", "execute", "collect", "cleanup"]
    assert result.status == "completed"
    assert result.summary == {"result": 42}


# --- Cleanup called even when execute raises ---


def test_cleanup_called_on_execute_failure(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    exp_impl = FailingExecuteExperimenter()
    task = Hypothesis(id="t1", spec={})

    result = exp_impl.run_experiment(task, ws)

    assert result.status == "failed"
    assert "execution error" in result.error
    assert exp_impl.cleanup_called


# --- Failed trial creates failed Experiment ---


def test_failed_experiment_has_error(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    exp_impl = FailingExecuteExperimenter()
    task = Hypothesis(id="t1", spec={})

    result = exp_impl.run_experiment(task, ws)

    assert result.error == "execution error"
    assert len(result.trials) == 1
    assert result.trials[0].status == "failed"


# --- Backward compat: steps and output from trial ---


def test_legacy_fields_populated_from_trial(tmp_path: Path):
    """steps and output on Experiment are populated from the trial for legacy code."""

    class StepfulExperimenter(SingleTrialExperimenter):
        def execute(self, task, context, workspace):
            return None

        def collect(self, task, raw, context, workspace):
            from aura.types import TrialStep

            return Trial(
                id=f"{task.id}-0",
                spec=task.spec,
                status="completed",
                steps=[
                    TrialStep(step=0, data={"x": 1}, timestamp="t"),
                    TrialStep(step=1, data={"x": 2}, timestamp="t"),
                ],
                output={"answer": 99},
            )

    ws = Workspace.create(tmp_path / "run")
    exp_impl = StepfulExperimenter()
    task = Hypothesis(id="t1", spec={})

    result = exp_impl.run_experiment(task, ws)

    assert result.output == {"answer": 99}
    assert len(result.steps) == 2
    assert result.steps[0].data == {"x": 1}
