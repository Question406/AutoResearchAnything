"""Tests for composed experimenter patterns (type compat, multi-trial, new type hierarchy)."""

from pathlib import Path
from typing import Any

import pytest

from aura.components.aggregators import BestTrialAggregator, LastTrialAggregator
from aura.interfaces import Experimenter, SingleTrialExperimenter
from aura.types import Experiment, Hypothesis, Trial, TrialStep
from aura.workspace import Workspace


# --- Type hierarchy ---


def test_trial_construction():
    trial = Trial(
        id="t1-0",
        spec={"lr": 0.01},
        status="completed",
        steps=[TrialStep(step=0, data={}, timestamp="t")],
        output={"acc": 0.9},
    )
    assert trial.id == "t1-0"
    assert trial.status == "completed"
    assert trial.output == {"acc": 0.9}


def test_trial_roundtrip():
    original = Trial(
        id="t1-0",
        spec={"lr": 0.01},
        status="completed",
        steps=[TrialStep(step=0, data={"x": 1}, timestamp="ts")],
        output={"acc": 0.9},
        metadata={"seed": 42},
    )
    restored = Trial.model_validate_json(original.model_dump_json())
    assert original == restored


def test_experiment_new_shape():
    trial = Trial(id="t1-0", spec={}, status="completed", steps=[], output={"acc": 0.8})
    exp = Experiment(
        hypothesis_id="t1",
        status="completed",
        trials=[trial],
        summary={"acc": 0.8},
    )
    assert exp.hypothesis_id == "t1"
    assert exp.task_id == "t1"  # backward-compat property
    assert exp.summary == {"acc": 0.8}
    assert len(exp.trials) == 1


def test_experiment_backward_compat_construction():
    """Old-style Experiment(task_id=...) still works."""
    exp = Experiment(
        task_id="t1",
        status="completed",
        steps=[TrialStep(step=0, data={}, timestamp="t")],
        output={"result": 7},
    )
    assert exp.task_id == "t1"
    assert exp.hypothesis_id == "t1"
    assert exp.output == {"result": 7}
    assert exp.summary == {"result": 7}  # auto-synced


def test_experiment_roundtrip_new_shape():
    trial = Trial(id="t1-0", spec={"lr": 0.01}, status="completed", steps=[], output={"acc": 0.9})
    original = Experiment(
        hypothesis_id="t1",
        status="completed",
        trials=[trial],
        summary={"acc": 0.9},
    )
    restored = Experiment.model_validate_json(original.model_dump_json())
    assert original == restored
    assert restored.task_id == "t1"


# --- Multi-trial custom Experimenter ---


class GridSearchExperimenter(Experimenter):
    """Runs the same task across multiple seeds."""

    def __init__(self, seeds: list[int] | None = None):
        self.seeds = seeds or [42, 123, 456]

    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        trials = []
        for i, seed in enumerate(self.seeds):
            acc = task.spec.get("lr", 0.01) * 10 + seed / 1000
            trials.append(
                Trial(
                    id=f"{task.id}-seed{seed}",
                    spec={**task.spec, "seed": seed},
                    status="completed",
                    steps=[],
                    output={"acc": round(acc, 4)},
                )
            )

        agg = BestTrialAggregator(metric="acc")
        summary = agg.aggregate(trials)
        return Experiment(
            hypothesis_id=task.id,
            status="completed",
            trials=trials,
            summary=summary,
            # populate legacy output from best trial
            output=summary,
        )


def test_multi_trial_grid_search(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    exp_impl = GridSearchExperimenter(seeds=[10, 20, 30])
    task = Hypothesis(id="t1", spec={"lr": 0.1})
    result = exp_impl.run_experiment(task, ws)

    assert result.status == "completed"
    assert len(result.trials) == 3
    assert all(t.status == "completed" for t in result.trials)
    # Summary should be best trial's output
    assert result.summary is not None
    assert isinstance(result.summary, dict)
    assert "acc" in result.summary


def test_multi_trial_workspace_persistence(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    exp_impl = GridSearchExperimenter(seeds=[1, 2])
    task = Hypothesis(id="t1", spec={"lr": 0.05})
    result = exp_impl.run_experiment(task, ws)

    ws.save_experiment(result, iteration=1)
    loaded = ws.load_experiment("t1", iteration=1)

    assert loaded.hypothesis_id == "t1"
    assert len(loaded.trials) == 2
    assert loaded.summary == result.summary


# --- SingleTrialExperimenter with custom aggregator ---


def test_single_trial_with_best_aggregator(tmp_path: Path):
    class AccExperimenter(SingleTrialExperimenter):
        def execute(self, task, context, workspace):
            return {"acc": task.spec.get("acc", 0.5)}

        def collect(self, task, raw, context, workspace):
            return Trial(id=f"{task.id}-0", spec=task.spec, status="completed", steps=[], output=raw)

    ws = Workspace.create(tmp_path / "run")
    exp_impl = AccExperimenter(aggregator=BestTrialAggregator(metric="acc"))
    task = Hypothesis(id="t1", spec={"acc": 0.85})
    result = exp_impl.run_experiment(task, ws)

    assert result.summary == {"acc": 0.85}


# --- ExperimentStep backward compat ---


def test_experiment_step_alias():
    from aura.types import ExperimentStep, TrialStep

    assert ExperimentStep is TrialStep

    step = ExperimentStep(step=0, data={"x": 1}, timestamp="t")
    assert isinstance(step, TrialStep)


# --- Evaluator with new experiment shape ---


def test_metric_evaluator_reads_summary(tmp_path: Path):
    from aura.components.evaluators import MetricEvaluator

    task = Hypothesis(id="t1", spec={})
    trial = Trial(id="t1-0", spec={}, status="completed", steps=[], output={"accuracy": 0.85})
    exp = Experiment(
        hypothesis_id="t1",
        status="completed",
        trials=[trial],
        summary={"accuracy": 0.85},
    )

    ev = MetricEvaluator(metric="accuracy", baseline=0.5, max_improvement=0.5)
    result = ev.evaluate(task, exp, None)

    assert result.passed is True
    assert result.score == pytest.approx(0.7)  # (0.85 - 0.5) / 0.5


def test_metric_evaluator_legacy_output_still_works():
    """Old-style Experiment(task_id=..., output={...}) still evaluates correctly."""
    from aura.components.evaluators import MetricEvaluator

    task = Hypothesis(id="t1", spec={})
    # Old-style construction — summary auto-synced from output
    exp = Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.75})

    ev = MetricEvaluator(metric="accuracy", baseline=0.5, max_improvement=0.5)
    result = ev.evaluate(task, exp, None)

    assert result.passed is True
    assert result.score == pytest.approx(0.5)
