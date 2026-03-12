from pathlib import Path

from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer
from aura.pipeline import Pipeline
from aura.runner import Runner
from aura.types import Evaluation, Experiment, Hypothesis


class StubResearcher(Researcher):
    def hypothesize(self, insights, workspace):
        return [Hypothesis(id="t0", spec={})]


class StubExperimenter(Experimenter):
    def run_experiment(self, task, workspace):
        return Experiment(task_id=task.id, status="completed", steps=[], output=None)


class StubEvaluator(Evaluator):
    def evaluate(self, task, trajectory, workspace):
        return Evaluation(task_id=task.id, score=1.0, passed=True)


class StubReviewer(Reviewer):
    def review(self, tasks, trajectories, evaluations, workspace):
        return []


class TestRunner(Runner):
    def setup_inputs(self, workspace):
        (workspace.inputs_dir() / "config.json").write_text('{"test": true}')

    def build_pipeline(self, workspace):
        return Pipeline(
            researcher=StubResearcher(),
            experimenter=StubExperimenter(),
            evaluator=StubEvaluator(),
            reviewer=StubReviewer(),
            workspace=workspace,
            max_iterations=1,
        )


def test_runner_end_to_end(tmp_path: Path):
    runner = TestRunner()
    ws = runner.run(run_dir=tmp_path / "run_001")
    assert ws.root.exists()
    assert (ws.inputs_dir() / "config.json").exists()
    assert ws.manifest()["status"] == "completed"
    assert ws.manifest()["iterations_completed"] == 1
    assert len(ws.load_tasks(1)) == 1
