import pytest

from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer
from aura.types import Evaluation, Experiment, Hypothesis, Insight


def test_researcher_without_runner_raises_on_call():
    r = Researcher()
    with pytest.raises(NotImplementedError, match="Provide a runner"):
        r.hypothesize([], None)


def test_cannot_instantiate_abstract_experimenter():
    with pytest.raises(TypeError):
        Experimenter()


def test_evaluator_without_runner_raises_on_call():
    e = Evaluator()
    with pytest.raises(NotImplementedError, match="Provide a runner"):
        e.evaluate(
            Hypothesis(id="t1", spec={}),
            Experiment(task_id="t1", status="completed", steps=[], output={}),
            None,
        )


def test_reviewer_without_runner_raises_on_call():
    r = Reviewer()
    with pytest.raises(NotImplementedError, match="Provide a runner"):
        r.review([], [], [], None)


def test_concrete_researcher():
    class MyGen(Researcher):
        def hypothesize(self, insights, workspace):
            return [Hypothesis(id="t1", spec={"q": "test"})]

    gen = MyGen()
    tasks = gen.hypothesize([], None)
    assert len(tasks) == 1
    assert tasks[0].id == "t1"


def test_concrete_experimenter():
    class MyExec(Experimenter):
        def run_experiment(self, task, workspace):
            return Experiment(task_id=task.id, status="completed", steps=[], output="done")

    exc = MyExec()
    traj = exc.run_experiment(Hypothesis(id="t1", spec={}), None)
    assert traj.status == "completed"


def test_concrete_evaluator():
    class MyEval(Evaluator):
        def evaluate(self, task, trajectory, workspace):
            return Evaluation(task_id=task.id, score=1.0, passed=True)

    ev = MyEval()
    result = ev.evaluate(
        Hypothesis(id="t1", spec={}),
        Experiment(task_id="t1", status="completed", steps=[], output=None),
        None,
    )
    assert result.passed is True


def test_concrete_reviewer():
    class MyExt(Reviewer):
        def review(self, tasks, trajectories, evaluations, workspace):
            return [Insight(id="i1", source_iteration=1, content={"note": "ok"})]

    ext = MyExt()
    insights = ext.review([], [], [], None)
    assert len(insights) == 1


def test_setup_teardown_defaults():
    class MyGen(Researcher):
        def hypothesize(self, insights, workspace):
            return []

    gen = MyGen()
    gen.setup(None)
    gen.teardown()
