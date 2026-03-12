import pytest
from aura.interfaces import Researcher, Experimenter, Evaluator, Reviewer
from aura.types import Hypothesis, Experiment, ExperimentStep, Evaluation, Insight


def test_cannot_instantiate_abstract_researcher():
    with pytest.raises(TypeError):
        Researcher()


def test_cannot_instantiate_abstract_experimenter():
    with pytest.raises(TypeError):
        Experimenter()


def test_cannot_instantiate_abstract_evaluator():
    with pytest.raises(TypeError):
        Evaluator()


def test_cannot_instantiate_abstract_reviewer():
    with pytest.raises(TypeError):
        Reviewer()


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
            return Experiment(
                task_id=task.id, status="completed", steps=[], output="done"
            )

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
