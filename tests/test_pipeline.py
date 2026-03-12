import time
from pathlib import Path

from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer
from aura.pipeline import Pipeline
from aura.types import Evaluation, Experiment, ExperimentStep, Hypothesis, Insight
from aura.workspace import Workspace


class MockResearcher(Researcher):
    def __init__(self, num_tasks: int = 3):
        self.num_tasks = num_tasks
        self.call_count = 0

    def hypothesize(self, insights, workspace):
        self.call_count += 1
        return [Hypothesis(id=f"task_{i:03d}", spec={"n": i}) for i in range(self.num_tasks)]


class MockExperimenter(Experimenter):
    def run_experiment(self, task, workspace):
        step = ExperimentStep(step=0, data={"echo": task.spec}, timestamp="t")
        return Experiment(
            task_id=task.id,
            status="completed",
            steps=[step],
            output={"done": True},
            metadata={},
        )


class MockEvaluator(Evaluator):
    def evaluate(self, task, trajectory, workspace):
        return Evaluation(
            task_id=task.id,
            score=0.8,
            passed=True,
            details={"auto": True},
        )


class MockReviewer(Reviewer):
    def review(self, tasks, trajectories, evaluations, workspace):
        return [
            Insight(
                id=f"insight_{workspace.current_iteration()}",
                source_iteration=workspace.current_iteration(),
                content={"num_tasks": len(tasks), "avg_score": 0.8},
            )
        ]


class FailingExperimenter(Experimenter):
    def run_experiment(self, task, workspace):
        raise RuntimeError("boom")


class EmptyResearcher(Researcher):
    def hypothesize(self, insights, workspace):
        return []


def test_pipeline_basic_2_iterations(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    pipeline = Pipeline(
        researcher=MockResearcher(num_tasks=3),
        experimenter=MockExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=2,
    )
    pipeline.run()

    assert (ws.root / "iteration_001").exists()
    assert (ws.root / "iteration_002").exists()

    for it in [1, 2]:
        assert len(ws.load_tasks(it)) == 3
        assert len(ws.load_trajectories(it)) == 3
        assert len(ws.load_evaluations(it)) == 3
        assert len(ws.load_insights(it)) == 1

    m = ws.manifest()
    assert m["iterations_completed"] == 2
    assert m["status"] == "completed"


def test_pipeline_executor_failure_retries(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    pipeline = Pipeline(
        researcher=MockResearcher(num_tasks=2),
        experimenter=FailingExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=1,
        max_retries=2,
    )
    pipeline.run()

    trajs = ws.load_trajectories(1)
    assert len(trajs) == 2
    for t in trajs:
        assert t.status == "failed"
        assert "boom" in t.error


def test_pipeline_empty_task_list(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    pipeline = Pipeline(
        researcher=EmptyResearcher(),
        experimenter=MockExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=1,
    )
    pipeline.run()

    assert ws.load_tasks(1) == []
    m = ws.manifest()
    assert m["iterations_completed"] == 1


def test_pipeline_resumability(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")

    task = Hypothesis(id="task_000", spec={"n": 0})
    ws.save_task(task, iteration=1)
    step = ExperimentStep(step=0, data={}, timestamp="t")
    traj = Experiment(task_id="task_000", status="completed", steps=[step], output=None)
    ws.save_trajectory(traj, iteration=1)
    ev = Evaluation(task_id="task_000", score=1.0, passed=True)
    ws.save_evaluation(ev, iteration=1)
    ws.save_insights([Insight(id="i0", source_iteration=1, content={})], iteration=1)
    ws.update_manifest(iterations_completed=1, status="in_progress")

    counter_gen = MockResearcher(num_tasks=2)
    pipeline = Pipeline(
        researcher=counter_gen,
        experimenter=MockExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=2,
    )
    pipeline.run()

    assert counter_gen.call_count == 1
    assert ws.manifest()["iterations_completed"] == 2


def test_pipeline_insight_window(tmp_path: Path):
    received_insights = []

    class SpyResearcher(Researcher):
        def hypothesize(self, insights, workspace):
            received_insights.append(list(insights))
            return [Hypothesis(id="t0", spec={})]

    ws = Workspace.create(tmp_path / "run")
    pipeline = Pipeline(
        researcher=SpyResearcher(),
        experimenter=MockExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=3,
        insight_window=1,
    )
    pipeline.run()

    assert len(received_insights[0]) == 0
    assert len(received_insights[1]) == 1
    assert len(received_insights[2]) == 1


def test_pipeline_parallel_tasks(tmp_path: Path):
    class SlowExperimenter(Experimenter):
        def run_experiment(self, task, workspace):
            time.sleep(0.1)
            return Experiment(
                task_id=task.id,
                status="completed",
                steps=[],
                output=None,
            )

    ws = Workspace.create(tmp_path / "run")
    start = time.time()
    pipeline = Pipeline(
        researcher=MockResearcher(num_tasks=4),
        experimenter=SlowExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=1,
        parallel_tasks=4,
    )
    pipeline.run()
    elapsed = time.time() - start

    assert elapsed < 0.35
    assert len(ws.load_trajectories(1)) == 4


# --- Artifact and rollback tests ---

from aura.artifacts import FileArtifact  # noqa: E402


def test_pipeline_with_artifact_snapshot(tmp_path: Path):
    """Artifacts are snapshotted each iteration."""
    ws = Workspace.create(tmp_path / "run")
    artifact_file = ws.artifacts_dir() / "code.py"
    artifact_file.write_text("initial")
    artifact = FileArtifact(artifact_file)

    class ModifyingResearcher(Researcher):
        def hypothesize(self, insights, workspace):
            art = workspace.artifacts["code.py"]
            current = art.read()
            art.write(current + f"\n# iteration {workspace.current_iteration()}")
            return [Hypothesis(id="h0", spec={})]

    pipeline = Pipeline(
        researcher=ModifyingResearcher(),
        experimenter=MockExperimenter(),
        evaluator=MockEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=2,
        artifacts=[artifact],
    )
    pipeline.run()

    # Check snapshots exist
    snap1 = (ws.iteration_dir(1) / "artifacts" / "code.py").read_text()
    assert snap1 == "initial"  # snapshot taken BEFORE modification
    snap2 = (ws.iteration_dir(2) / "artifacts" / "code.py").read_text()
    assert "iteration 1" in snap2  # snapshot of state after iter 1 modification


def test_pipeline_rollback_best(tmp_path: Path):
    """With rollback='best', artifacts revert when score doesn't improve."""
    ws = Workspace.create(tmp_path / "run")
    artifact_file = ws.artifacts_dir() / "code.py"
    artifact_file.write_text("initial")
    artifact = FileArtifact(artifact_file)

    call_count = [0]

    class ModifyingResearcher(Researcher):
        def hypothesize(self, insights, workspace):
            call_count[0] += 1
            art = workspace.artifacts["code.py"]
            art.write(f"version {call_count[0]}")
            return [Hypothesis(id="h0", spec={})]

    scores = [0.8, 0.5, 0.3]  # iter 1 best, iter 2 worse, iter 3 worse
    score_idx = [0]

    class DecliningEvaluator(Evaluator):
        def evaluate(self, task, trajectory, workspace):
            s = scores[score_idx[0]]
            score_idx[0] += 1
            return Evaluation(task_id=task.id, score=s, passed=True)

    pipeline = Pipeline(
        researcher=ModifyingResearcher(),
        experimenter=MockExperimenter(),
        evaluator=DecliningEvaluator(),
        reviewer=MockReviewer(),
        workspace=ws,
        max_iterations=3,
        artifacts=[artifact],
        rollback="best",
    )
    pipeline.run()

    # Artifact should be rolled back to iteration 1's version (best score)
    assert artifact.read() == "version 1"
