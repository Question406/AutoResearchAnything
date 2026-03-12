from aura.decorators import as_evaluator, as_experimenter, as_researcher, as_reviewer
from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer
from aura.types import Evaluation, Experiment, Hypothesis, Insight


def test_as_researcher():
    @as_researcher
    def my_gen(insights, workspace):
        return [Hypothesis(id="t1", spec={"q": "test"})]

    assert isinstance(my_gen, Researcher)
    tasks = my_gen.hypothesize([], None)
    assert len(tasks) == 1
    assert tasks[0].id == "t1"


def test_as_experimenter():
    @as_experimenter
    def my_exec(task, workspace):
        return Experiment(task_id=task.id, status="completed", steps=[], output="ok")

    assert isinstance(my_exec, Experimenter)
    traj = my_exec.run_experiment(Hypothesis(id="t1", spec={}), None)
    assert traj.status == "completed"


def test_as_evaluator():
    @as_evaluator
    def my_eval(task, trajectory, workspace):
        return Evaluation(task_id=task.id, score=0.9, passed=True)

    assert isinstance(my_eval, Evaluator)
    result = my_eval.evaluate(
        Hypothesis(id="t1", spec={}),
        Experiment(task_id="t1", status="completed", steps=[], output=None),
        None,
    )
    assert result.score == 0.9


def test_as_reviewer():
    @as_reviewer
    def my_ext(tasks, trajectories, evaluations, workspace):
        return [Insight(id="i1", source_iteration=0, content={})]

    assert isinstance(my_ext, Reviewer)
    insights = my_ext.review([], [], [], None)
    assert len(insights) == 1
