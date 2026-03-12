from pathlib import Path
from aura.types import Hypothesis, ExperimentStep, Experiment, Evaluation, Insight
from aura.workspace import Workspace


def test_create_workspace(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run_001")
    assert ws.root.exists()
    assert ws.inputs_dir().exists()
    assert (ws.root / "manifest.json").exists()


def test_manifest_initial(tmp_workspace: Workspace):
    m = tmp_workspace.manifest()
    assert m["status"] == "created"
    assert m["iterations_completed"] == 0


def test_current_iteration_initial(tmp_workspace: Workspace):
    assert tmp_workspace.current_iteration() == 0


def test_iteration_dir_created(tmp_workspace: Workspace):
    d = tmp_workspace.iteration_dir(1)
    assert d.exists()
    assert (d / "tasks").exists()
    assert (d / "trajectories").exists()
    assert (d / "evaluations").exists()


def test_save_and_load_task(tmp_workspace: Workspace):
    task = Hypothesis(id="task_001", spec={"prompt": "hello"}, metadata={"tag": "test"})
    tmp_workspace.save_task(task, iteration=1)
    loaded = tmp_workspace.load_tasks(1)
    assert len(loaded) == 1
    assert loaded[0] == task


def test_save_and_load_trajectory(tmp_workspace: Workspace):
    step0 = ExperimentStep(step=0, data={"action": "think"}, timestamp="2026-03-11T00:00:00Z")
    step1 = ExperimentStep(step=1, data={"action": "act"}, timestamp="2026-03-11T00:00:01Z")
    traj = Experiment(
        task_id="task_001", status="completed", steps=[step0, step1],
        output={"result": 42}, metadata={"duration_ms": 100},
    )
    tmp_workspace.save_trajectory(traj, iteration=1)

    loaded = tmp_workspace.load_trajectories(1)
    assert len(loaded) == 1
    assert loaded[0] == traj


def test_has_trajectory(tmp_workspace: Workspace):
    assert tmp_workspace.has_trajectory("task_001", iteration=1) is False
    traj = Experiment(task_id="task_001", status="completed", steps=[], output=None)
    tmp_workspace.save_trajectory(traj, iteration=1)
    assert tmp_workspace.has_trajectory("task_001", iteration=1) is True


def test_save_and_load_evaluation(tmp_workspace: Workspace):
    ev = Evaluation(task_id="task_001", score=0.9, passed=True, details={"ok": True})
    tmp_workspace.save_evaluation(ev, iteration=1)
    loaded = tmp_workspace.load_evaluations(1)
    assert len(loaded) == 1
    assert loaded[0] == ev


def test_has_evaluation(tmp_workspace: Workspace):
    assert tmp_workspace.has_evaluation("task_001", iteration=1) is False
    ev = Evaluation(task_id="task_001", score=0.5, passed=False)
    tmp_workspace.save_evaluation(ev, iteration=1)
    assert tmp_workspace.has_evaluation("task_001", iteration=1) is True


def test_save_and_load_insights(tmp_workspace: Workspace):
    insights = [
        Insight(id="i1", source_iteration=1, content={"note": "good"}),
        Insight(id="i2", source_iteration=1, content={"note": "bad"}),
    ]
    tmp_workspace.save_insights(insights, iteration=1)
    loaded = tmp_workspace.load_insights(1)
    assert loaded == insights


def test_load_insights_empty(tmp_workspace: Workspace):
    loaded = tmp_workspace.load_insights(1)
    assert loaded == []


def test_multiple_tasks(tmp_workspace: Workspace):
    t1 = Hypothesis(id="task_001", spec={"a": 1})
    t2 = Hypothesis(id="task_002", spec={"b": 2})
    tmp_workspace.save_task(t1, iteration=1)
    tmp_workspace.save_task(t2, iteration=1)
    loaded = tmp_workspace.load_tasks(1)
    assert len(loaded) == 2
    ids = {t.id for t in loaded}
    assert ids == {"task_001", "task_002"}


def test_summary(tmp_workspace: Workspace):
    t1 = Hypothesis(id="task_001", spec={"a": 1})
    t2 = Hypothesis(id="task_002", spec={"b": 2})
    tmp_workspace.save_task(t1, iteration=1)
    tmp_workspace.save_task(t2, iteration=1)

    step = ExperimentStep(step=0, data={}, timestamp="t")
    for tid in ["task_001", "task_002"]:
        traj = Experiment(task_id=tid, status="completed", steps=[step], output=None)
        tmp_workspace.save_trajectory(traj, iteration=1)

    ev1 = Evaluation(task_id="task_001", score=0.8, passed=True)
    ev2 = Evaluation(task_id="task_002", score=0.6, passed=True)
    tmp_workspace.save_evaluation(ev1, iteration=1)
    tmp_workspace.save_evaluation(ev2, iteration=1)

    tmp_workspace.save_insights([], iteration=1)
    tmp_workspace.update_manifest(iterations_completed=1, status="completed")

    s = tmp_workspace.summary()
    assert s["iterations_completed"] == 1
    assert s["total_tasks"] == 2
    assert s["best_score"] == 0.8
    assert s["best_task_id"] == "task_001"
    assert s["total_passed"] == 2
    assert len(s["iterations"]) == 1
    assert s["iterations"][0]["avg_score"] == 0.7
