"""Tests for Runner ABC and built-in runner implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.components.runners import FunctionRunner, LLMRunner, as_runner
from aura.interfaces import Evaluator, Researcher, Reviewer, Runner
from aura.types import Evaluation, Experiment, Hypothesis, Insight
from aura.workspace import Workspace

# --- Mock LLM ---


def mock_llm(prompt: str) -> str:
    return json.dumps([{"id": "t1", "lr": 0.01, "epochs": 10}])


def mock_llm_judge(prompt: str) -> str:
    return json.dumps({"score": 0.9, "passed": True, "reason": "great"})


def mock_llm_reviewer(prompt: str) -> str:
    return json.dumps([{"finding": "lr works", "recommendation": "try lower"}])


# --- LLMRunner ---


class TestLLMRunner:
    def test_basic(self):
        runner = LLMRunner(mock_llm)
        result = runner.run("Generate tasks for {{ role }}", {"role": "researcher"})
        assert "content" in result
        parsed = json.loads(result["content"])
        assert isinstance(parsed, list)

    def test_template_rendering(self):
        captured = {}

        def capturing_llm(prompt):
            captured["prompt"] = prompt
            return '{"ok": true}'

        runner = LLMRunner(capturing_llm)
        runner.run("Hello {{ name }}, iteration {{ iteration }}", {"name": "world", "iteration": 3})
        assert "Hello world" in captured["prompt"]
        assert "iteration 3" in captured["prompt"]


# --- FunctionRunner ---


class TestFunctionRunner:
    def test_returns_str(self):
        def fn(prompt, context):
            return f"result for {context.get('role')}"

        runner = FunctionRunner(fn)
        result = runner.run("template", {"role": "evaluator"})
        assert result["content"] == "result for evaluator"

    def test_returns_dict(self):
        def fn(prompt, context):
            return {"content": "hello", "structured": {"score": 0.9}}

        runner = FunctionRunner(fn)
        result = runner.run("template", {})
        assert result["content"] == "hello"
        assert result["structured"]["score"] == 0.9


# --- as_runner ---


class TestAsRunner:
    def test_wraps_callable(self):
        runner = as_runner(mock_llm)
        assert isinstance(runner, LLMRunner)

    def test_passthrough_runner(self):
        original = LLMRunner(mock_llm)
        runner = as_runner(original)
        assert runner is original

    def test_wraps_lambda(self):
        runner = as_runner(lambda p: "response")
        assert isinstance(runner, LLMRunner)
        result = runner.run("test", {})
        assert result["content"] == "response"


# --- Researcher with Runner ---


class TestResearcherWithRunner:
    def test_researcher_with_runner(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        researcher = Researcher(
            runner=mock_llm,
            prompt_template="Generate {{ num_tasks }} tasks. Insights: {{ insights }}",
            num_tasks=3,
        )
        tasks = researcher.hypothesize([], ws)

        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].spec["lr"] == 0.01

    def test_researcher_with_insights(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(2)
        insights = [Insight(id="i1", source_iteration=1, content={"finding": "lr=0.001 is good"})]

        researcher = Researcher(runner=mock_llm, prompt_template="Insights: {{ insights }}")
        tasks = researcher.hypothesize(insights, ws)
        assert len(tasks) == 1

    def test_researcher_with_structured_bypass(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        def structured_fn(prompt, context):
            return {
                "content": "raw text",
                "structured": [{"id": "s1", "param": "value"}],
            }

        researcher = Researcher(runner=FunctionRunner(structured_fn), prompt_template="template")
        tasks = researcher.hypothesize([], ws)
        assert len(tasks) == 1
        assert tasks[0].id == "s1"
        assert tasks[0].spec["param"] == "value"

    def test_researcher_no_runner_raises(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        researcher = Researcher()
        with pytest.raises(NotImplementedError, match="Provide a runner"):
            researcher.hypothesize([], ws)


# --- Evaluator with Runner ---


class TestEvaluatorWithRunner:
    def test_evaluator_with_runner(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        task = Hypothesis(id="t1", spec={"query": "test"})
        exp = Experiment(task_id="t1", status="completed", steps=[], output={"answer": "42"})

        evaluator = Evaluator(runner=mock_llm_judge)
        result = evaluator.evaluate(task, exp, ws)

        assert result.score == 0.9
        assert result.passed is True
        assert result.details["reason"] == "great"

    def test_evaluator_failed_experiment(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        task = Hypothesis(id="t1", spec={})
        exp = Experiment(task_id="t1", status="failed", steps=[], output=None, error="boom")

        evaluator = Evaluator(runner=mock_llm_judge)
        result = evaluator.evaluate(task, exp, ws)

        assert result.score == 0.0
        assert result.passed is False
        assert result.details["reason"] == "execution_failed"

    def test_evaluator_with_structured_bypass(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        task = Hypothesis(id="t1", spec={})
        exp = Experiment(task_id="t1", status="completed", steps=[], output={"answer": "42"})

        def structured_fn(prompt, context):
            return {
                "content": "raw",
                "structured": {"score": 0.75, "passed": True, "reason": "structured bypass"},
            }

        evaluator = Evaluator(runner=FunctionRunner(structured_fn))
        result = evaluator.evaluate(task, exp, ws)
        assert result.score == 0.75
        assert result.details["reason"] == "structured bypass"

    def test_evaluator_no_runner_raises(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        task = Hypothesis(id="t1", spec={})
        exp = Experiment(task_id="t1", status="completed", steps=[], output={})

        evaluator = Evaluator()
        with pytest.raises(NotImplementedError, match="Provide a runner"):
            evaluator.evaluate(task, exp, ws)


# --- Reviewer with Runner ---


class TestReviewerWithRunner:
    def test_reviewer_with_runner(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        tasks = [Hypothesis(id="t1", spec={"lr": 0.001})]
        exps = [Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.8})]
        evals = [Evaluation(task_id="t1", score=0.6, passed=True)]

        reviewer = Reviewer(runner=mock_llm_reviewer)
        insights = reviewer.review(tasks, exps, evals, ws)

        assert len(insights) == 1
        assert "lr works" in insights[0].content["finding"]

    def test_reviewer_no_runner_raises(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        reviewer = Reviewer()
        with pytest.raises(NotImplementedError, match="Provide a runner"):
            reviewer.review([], [], [], ws)

    def test_reviewer_with_structured_bypass(self, tmp_path: Path):
        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        def structured_fn(prompt, context):
            return {
                "content": "raw",
                "structured": [{"finding": "A", "recommendation": "B"}],
            }

        reviewer = Reviewer(runner=FunctionRunner(structured_fn))
        tasks = [Hypothesis(id="t1", spec={})]
        exps = [Experiment(task_id="t1", status="completed", steps=[], output={})]
        evals = [Evaluation(task_id="t1", score=0.5, passed=True)]

        insights = reviewer.review(tasks, exps, evals, ws)
        assert len(insights) == 1
        assert insights[0].content["finding"] == "A"


# --- Backward compatibility ---


class TestBackwardCompat:
    def test_llm_researcher_deprecated_alias(self):
        from aura.components.researchers import LLMResearcher

        assert issubclass(LLMResearcher, Researcher)

    def test_llm_judge_evaluator_deprecated_alias(self):
        from aura.components.evaluators import LLMJudgeEvaluator

        assert issubclass(LLMJudgeEvaluator, Evaluator)

    def test_llm_reviewer_deprecated_alias(self):
        from aura.components.reviewers import LLMReviewer

        assert issubclass(LLMReviewer, Reviewer)

    def test_cli_runner_alias(self):
        from aura.runner import CliRunner
        from aura.runner import Runner as CliRunnerAlias

        assert CliRunnerAlias is CliRunner

    def test_runner_abc_is_new_runner(self):
        """aura.Runner is now the execution backend, not the CLI runner."""
        import aura

        assert aura.Runner is Runner  # interfaces.Runner
        assert hasattr(aura, "CliRunner")  # old CLI runner still available


# --- Custom Runner subclass ---


class TestCustomRunner:
    def test_custom_runner_subclass(self, tmp_path: Path):
        class EchoRunner(Runner):
            def run(self, prompt_template, context):
                return {
                    "content": json.dumps([{"id": "echo", "input": context.get("role", "unknown")}])
                }

        ws = Workspace.create(tmp_path / "run")
        ws.set_current_iteration(1)

        researcher = Researcher(runner=EchoRunner(), prompt_template="ignored")
        tasks = researcher.hypothesize([], ws)
        assert tasks[0].id == "echo"
        assert tasks[0].spec["input"] == "researcher"
