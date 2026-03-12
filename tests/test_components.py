import json
import pytest
from pathlib import Path

from aura.types import Hypothesis, Experiment, ExperimentStep, Evaluation, Insight
from aura.workspace import Workspace
from aura.components.llm import LLMCallable
from aura.components.researchers import LLMResearcher
from aura.components.executors import ScriptExperimenter, FunctionExperimenter, LLMExperimenter
from aura.components.evaluators import MetricEvaluator, LLMJudgeEvaluator
from aura.components.reviewers import LLMReviewer


# --- Mock LLM ---

def mock_llm_researcher(prompt: str) -> str:
    """Returns a JSON list of experiment specs."""
    return json.dumps([
        {"id": "exp_001", "lr": 0.001, "epochs": 10, "batch_size": 32, "rationale": "baseline"},
        {"id": "exp_002", "lr": 0.0003, "epochs": 50, "batch_size": 16, "rationale": "low lr"},
    ])


def mock_llm_judge(prompt: str) -> str:
    return json.dumps({"score": 0.85, "passed": True, "reason": "good result"})


def mock_llm_reviewer(prompt: str) -> str:
    return json.dumps([
        {"finding": "low lr works better", "recommendation": "try lr=0.0002"},
    ])


def mock_llm_experimenter(prompt: str) -> str:
    return "This is the LLM response to the task."


# --- LLMResearcher ---

def test_llm_researcher(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)
    # Write some input
    (ws.inputs_dir() / "config.txt").write_text("goal: maximize accuracy")

    gen = LLMResearcher(llm=mock_llm_researcher, prompt_template="Generate {{ num_tasks }} tasks.\nInputs: {{ inputs }}\nInsights: {{ insights }}")
    tasks = gen.hypothesize([], ws)

    assert len(tasks) == 2
    assert tasks[0].id == "exp_001"
    assert tasks[0].spec["lr"] == 0.001
    assert tasks[1].metadata["rationale"] == "low lr"


def test_llm_researcher_with_insights(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(2)
    insights = [Insight(id="i1", source_iteration=1, content={"finding": "lr=0.001 is good"})]

    gen = LLMResearcher(llm=mock_llm_researcher, prompt_template="Insights: {{ insights }}\nGenerate tasks.")
    tasks = gen.hypothesize(insights, ws)
    assert len(tasks) == 2


# --- ScriptExperimenter ---

def test_script_experimenter(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    # Create a simple script
    script = tmp_path / "test_script.py"
    script.write_text('import json, sys; print(json.dumps({"result": float(sys.argv[1]) * 2}))')

    experimenter = ScriptExperimenter(command_template=f"python {script} {{value}}")
    task = Hypothesis(id="t1", spec={"value": 5})
    traj = experimenter.run_experiment(task, ws)

    assert traj.status == "completed"
    assert traj.output["result"] == 10.0
    assert len(traj.steps) == 2


def test_script_experimenter_failure(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    experimenter = ScriptExperimenter(command_template="python -c 'import sys; sys.exit(1)'")
    task = Hypothesis(id="t1", spec={})
    traj = experimenter.run_experiment(task, ws)

    assert traj.status == "failed"


# --- FunctionExperimenter ---

def test_function_experimenter(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    def my_fn(lr, epochs, **kwargs):
        return {"accuracy": 0.5 + lr * epochs}

    experimenter = FunctionExperimenter(fn=my_fn)
    task = Hypothesis(id="t1", spec={"lr": 0.01, "epochs": 10})
    traj = experimenter.run_experiment(task, ws)

    assert traj.status == "completed"
    assert traj.output["accuracy"] == 0.6


# --- LLMExperimenter ---

def test_llm_experimenter(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    experimenter = LLMExperimenter(llm=mock_llm_experimenter, prompt_template="Solve: {query}")
    task = Hypothesis(id="t1", spec={"query": "What is 2+2?"})
    traj = experimenter.run_experiment(task, ws)

    assert traj.status == "completed"
    assert "LLM response" in traj.output["response"]


# --- MetricEvaluator ---

def test_metric_evaluator_above_baseline():
    task = Hypothesis(id="t1", spec={})
    traj = Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.8})

    ev = MetricEvaluator(metric="accuracy", baseline=0.5, max_improvement=0.5)
    result = ev.evaluate(task, traj, None)

    assert result.passed is True
    assert result.score == 0.6  # (0.8 - 0.5) / 0.5


def test_metric_evaluator_below_baseline():
    task = Hypothesis(id="t1", spec={})
    traj = Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.3})

    ev = MetricEvaluator(metric="accuracy", baseline=0.5)
    result = ev.evaluate(task, traj, None)

    assert result.passed is False
    assert result.score == 0.0


def test_metric_evaluator_failed_trajectory():
    task = Hypothesis(id="t1", spec={})
    traj = Experiment(task_id="t1", status="failed", steps=[], output=None, error="boom")

    ev = MetricEvaluator(metric="accuracy", baseline=0.5)
    result = ev.evaluate(task, traj, None)

    assert result.score == 0.0
    assert result.passed is False


def test_metric_evaluator_lower_is_better():
    task = Hypothesis(id="t1", spec={})
    traj = Experiment(task_id="t1", status="completed", steps=[], output={"loss": 0.2})

    ev = MetricEvaluator(metric="loss", baseline=0.5, higher_is_better=False, max_improvement=0.5)
    result = ev.evaluate(task, traj, None)

    assert result.passed is True
    assert result.score == 0.6  # (0.5 - 0.2) / 0.5


# --- LLMJudgeEvaluator ---

def test_llm_judge_evaluator(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = Hypothesis(id="t1", spec={"query": "test"})
    traj = Experiment(task_id="t1", status="completed", steps=[], output={"answer": "42"})

    ev = LLMJudgeEvaluator(llm=mock_llm_judge)
    result = ev.evaluate(task, traj, ws)

    assert result.score == 0.85
    assert result.passed is True
    assert result.details["reason"] == "good result"


# --- LLMReviewer ---

def test_llm_reviewer(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    ws.set_current_iteration(1)

    tasks = [Hypothesis(id="t1", spec={"lr": 0.001})]
    trajs = [Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.8})]
    evals = [Evaluation(task_id="t1", score=0.6, passed=True)]

    reviewer = LLMReviewer(llm=mock_llm_reviewer)
    insights = reviewer.review(tasks, trajs, evals, ws)

    assert len(insights) == 1
    assert "low lr" in insights[0].content["finding"]
