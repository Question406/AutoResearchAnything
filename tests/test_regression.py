"""Regression tests for AURA — catch behavioral and API regressions without real LLMs."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aura
from aura import (
    Evaluation,
    Experiment,
    Hypothesis,
    Insight,
    LLMResearcher,
    LLMReviewer,
    MetricEvaluator,
    Pipeline,
    ScriptExperimenter,
    Workspace,
)

# ---------------------------------------------------------------------------
# Path to the self-contained training script used by regression tests
# ---------------------------------------------------------------------------
MOCK_TRAIN = str(Path(__file__).parent / "scripts" / "mock_train_reg.py")

# ---------------------------------------------------------------------------
# Deterministic LLM stubs
# ---------------------------------------------------------------------------

RESEARCHER_RESPONSE = json.dumps(
    [
        {"id": "reg_001", "lr": 3.162e-4, "epochs": 50, "batch_size": 32},
        {"id": "reg_002", "lr": 1e-3, "epochs": 30, "batch_size": 32},
    ]
)

REVIEWER_RESPONSE = json.dumps(
    [
        {
            "finding": "lr=3.162e-4 outperforms baseline",
            "recommendation": "narrow search around 3e-4",
        }
    ]
)


class DeterministicLLM:
    """Detects researcher vs reviewer calls by inspecting prompt content,
    returns the appropriate JSON structure without any network call."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        if "Propose" in prompt or "num_tasks" in prompt:  # researcher
            return RESEARCHER_RESPONSE
        # reviewer (default template contains "Analyze")
        return REVIEWER_RESPONSE


class SpyLLM:
    """Like DeterministicLLM but records every prompt it receives."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Propose" in prompt or "num_tasks" in prompt:
            return RESEARCHER_RESPONSE
        return REVIEWER_RESPONSE

    @property
    def researcher_prompts(self) -> list[str]:
        return [p for p in self.prompts if "Propose" in p or "num_tasks" in p]


# ---------------------------------------------------------------------------
# Shared pipeline fixture (module-scoped — runs once for all e2e tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run a 2-iteration pipeline with DeterministicLLM; return (workspace, llm)."""
    run_dir = tmp_path_factory.mktemp("regression_run")
    ws = Workspace.create(run_dir)
    llm = DeterministicLLM()

    pipeline = Pipeline(
        researcher=LLMResearcher(
            llm=llm,
            prompt_template=(
                "Propose {{ num_tasks }} experiments.\n\n"
                "Inputs: {{ inputs }}\n"
                "Previous insights:\n{{ insights }}\n\n"
                "Each must have: id, lr (float), epochs (int), batch_size (int).\n"
                "Respond as JSON list."
            ),
            num_tasks=2,
        ),
        experimenter=ScriptExperimenter(
            command_template=(
                f"{sys.executable} {MOCK_TRAIN}"
                " --lr {lr} --epochs {epochs} --batch-size {batch_size}"
            ),
        ),
        evaluator=MetricEvaluator(metric="accuracy", baseline=0.5),
        reviewer=LLMReviewer(llm=llm),
        workspace=ws,
        max_iterations=2,
    )
    pipeline.run()
    return ws, llm


@pytest.fixture(scope="module")
def spy_run(tmp_path_factory):
    """Run a 2-iteration pipeline with SpyLLM; return (workspace, spy)."""
    run_dir = tmp_path_factory.mktemp("regression_spy")
    ws = Workspace.create(run_dir)
    spy = SpyLLM()

    pipeline = Pipeline(
        researcher=LLMResearcher(
            llm=spy,
            prompt_template=(
                "Propose {{ num_tasks }} experiments.\n\n"
                "Inputs: {{ inputs }}\n"
                "Previous insights:\n{{ insights }}\n\n"
                "Each must have: id, lr (float), epochs (int), batch_size (int).\n"
                "Respond as JSON list."
            ),
            num_tasks=2,
        ),
        experimenter=ScriptExperimenter(
            command_template=(
                f"{sys.executable} {MOCK_TRAIN}"
                " --lr {lr} --epochs {epochs} --batch-size {batch_size}"
            ),
        ),
        evaluator=MetricEvaluator(metric="accuracy", baseline=0.5),
        reviewer=LLMReviewer(llm=spy),
        workspace=ws,
        max_iterations=2,
    )
    pipeline.run()
    return ws, spy


# ===========================================================================
# 3a. API Surface
# ===========================================================================

EXPECTED_SYMBOLS = [
    "Hypothesis",
    "ExperimentStep",
    "Experiment",
    "Evaluation",
    "Insight",
    "JsonValue",
    "Researcher",
    "Experimenter",
    "Evaluator",
    "Reviewer",
    "as_researcher",
    "as_experimenter",
    "as_evaluator",
    "as_reviewer",
    "Artifact",
    "FileArtifact",
    "DirectoryArtifact",
    "Workspace",
    "Pipeline",
    "Runner",
    "extract_json",
    "render_prompt",
    "setup_logging",
    "LLMCallable",
    "LLMResearcher",
    "ScriptExperimenter",
    "FunctionExperimenter",
    "LLMExperimenter",
    "MetricEvaluator",
    "LLMJudgeEvaluator",
    "LLMReviewer",
    "anthropic_llm",
    "openai_llm",
    "command_llm",
]


@pytest.mark.parametrize("symbol", EXPECTED_SYMBOLS)
def test_public_symbol_importable(symbol):
    assert hasattr(aura, symbol), f"aura.{symbol} missing from public API"


def test_all_exports_complete():
    missing = [s for s in EXPECTED_SYMBOLS if s not in aura.__all__]
    assert missing == [], f"Symbols missing from aura.__all__: {missing}"


def test_pipeline_accepts_all_constructor_params():
    sig = inspect.signature(Pipeline.__init__)
    expected_params = {
        "researcher",
        "experimenter",
        "evaluator",
        "reviewer",
        "workspace",
        "max_retries",
        "max_iterations",
        "parallel_tasks",
        "insight_window",
        "prepare",
        "artifacts",
        "rollback",
        "constraints",
    }
    actual_params = set(sig.parameters.keys()) - {"self"}
    missing = expected_params - actual_params
    assert missing == set(), f"Pipeline.__init__ missing params: {missing}"


# ===========================================================================
# 3b. End-to-end pipeline
# ===========================================================================


def test_regression_pipeline_completes(pipeline_run):
    ws, _ = pipeline_run
    manifest = ws.manifest()
    assert manifest["status"] == "completed"
    assert manifest["iterations_completed"] == 2


def test_regression_iteration_count(pipeline_run):
    ws, _ = pipeline_run
    assert (ws.root / "iteration_001").exists()
    assert (ws.root / "iteration_002").exists()
    assert not (ws.root / "iteration_003").exists()


def test_regression_workspace_structure(pipeline_run):
    ws, _ = pipeline_run
    for it_num in (1, 2):
        it_dir = ws.root / f"iteration_{it_num:03d}"
        assert (it_dir / "tasks").is_dir()
        assert (it_dir / "trajectories").is_dir()
        assert (it_dir / "evaluations").is_dir()
        assert (it_dir / "insights.json").is_file()

        tasks = list((it_dir / "tasks").glob("*.json"))
        trajs = list((it_dir / "trajectories").glob("*.json"))
        evals = list((it_dir / "evaluations").glob("*.json"))
        assert len(tasks) == 2, f"iteration {it_num}: expected 2 tasks, got {len(tasks)}"
        assert len(trajs) == 2, f"iteration {it_num}: expected 2 trajectories, got {len(trajs)}"
        assert len(evals) == 2, f"iteration {it_num}: expected 2 evaluations, got {len(evals)}"


def test_regression_evaluation_scores_in_range(pipeline_run):
    ws, _ = pipeline_run
    for it_num in (1, 2):
        for ev in ws.load_evaluations(it_num):
            assert 0.0 <= ev.score <= 1.0, f"score {ev.score} out of [0, 1] range"


def test_regression_insights_generated(pipeline_run):
    ws, _ = pipeline_run
    for it_num in (1, 2):
        insights = ws.load_insights(it_num)
        assert len(insights) >= 1, f"iteration {it_num}: expected ≥1 insight"
        for ins in insights:
            assert ins.id, "insight missing id"
            assert ins.source_iteration == it_num
            assert ins.content, "insight missing content"


def test_regression_insights_flow_to_next_iteration(spy_run):
    """Iteration-2 researcher prompt must contain insight text from iteration 1."""
    _, spy = spy_run
    researcher_prompts = spy.researcher_prompts
    assert len(researcher_prompts) >= 2, "Expected at least 2 researcher prompts"

    iter2_prompt = researcher_prompts[1]
    # The iteration-1 insight finding is injected into the iteration-2 researcher prompt
    assert "outperforms baseline" in iter2_prompt, (
        "Iteration-2 researcher prompt does not contain iteration-1 insight text — "
        "insight propagation regression detected"
    )


def test_regression_llm_called_expected_times(pipeline_run):
    """Exactly 4 LLM calls for 2 iterations (1 researcher + 1 reviewer each)."""
    _, llm = pipeline_run
    assert llm.call_count == 4, f"Expected 4 LLM calls, got {llm.call_count}"


# ===========================================================================
# 3c. CLI subprocess — golden output
# ===========================================================================


def _write_tmp(tmp_path: Path, code: str, name: str = "experiment.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(code))
    return p


def test_regression_cli_validate_golden_output(tmp_path):
    """validate JSON output must have the stable schema: valid, entry_pattern, errors, warnings."""
    f = _write_tmp(
        tmp_path,
        """\
        def main():
            pass
    """,
    )
    result = subprocess.run(
        [sys.executable, "-m", "aura", "validate", str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert set(data.keys()) == {"valid", "entry_pattern", "errors", "warnings"}
    assert data["valid"] is True
    assert data["entry_pattern"] == "main"
    assert data["errors"] == []


def test_regression_cli_validate_invalid_exits_nonzero(tmp_path):
    f = _write_tmp(tmp_path, "x = 1\n")
    result = subprocess.run(
        [sys.executable, "-m", "aura", "validate", str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["valid"] is False


def test_regression_cli_validate_all_three_patterns(tmp_path):
    patterns = {
        "main": "def main():\n    pass\n",
        "runner": "class MyRunner(Runner):\n    pass\n",
        "globals": "workspace = 1\npipeline = 2\n",
    }
    for expected_pattern, code in patterns.items():
        f = _write_tmp(tmp_path, code, name=f"exp_{expected_pattern}.py")
        result = subprocess.run(
            [sys.executable, "-m", "aura", "validate", str(f)],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        assert data["entry_pattern"] == expected_pattern, (
            f"Expected pattern '{expected_pattern}', got '{data['entry_pattern']}'"
        )


# ===========================================================================
# 3d. Pydantic schema regression
# ===========================================================================


def test_regression_pydantic_hypothesis_schema():
    h = Hypothesis(id="t1", spec={"lr": 1e-3})
    assert h.id == "t1"
    assert h.spec == {"lr": 1e-3}
    assert isinstance(h.metadata, dict)


def test_regression_pydantic_experiment_schema():
    e = Experiment(task_id="t1", status="completed", steps=[], output={"accuracy": 0.9})
    assert e.task_id == "t1"
    assert e.status == "completed"
    assert e.output == {"accuracy": 0.9}
    assert e.error is None


def test_regression_pydantic_evaluation_schema():
    ev = Evaluation(task_id="t1", score=0.75, passed=True)
    assert ev.task_id == "t1"
    assert ev.score == 0.75
    assert ev.passed is True
    assert isinstance(ev.details, dict)


def test_regression_pydantic_insight_schema():
    ins = Insight(id="abc", source_iteration=1, content={"finding": "x"})
    assert ins.id == "abc"
    assert ins.source_iteration == 1
    assert ins.content == {"finding": "x"}
    assert isinstance(ins.metadata, dict)
