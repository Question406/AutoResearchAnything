from __future__ import annotations

from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer


def as_researcher(fn):
    class _FnResearcher(Researcher):
        def hypothesize(self, insights, workspace):
            return fn(insights, workspace)

    return _FnResearcher()


def as_experimenter(fn):
    """Wrap a function ``fn(task, workspace) -> Experiment`` as an Experimenter."""

    class _FnExperimenter(Experimenter):
        def run_experiment(self, task, workspace):
            return fn(task, workspace)

    return _FnExperimenter()


def as_evaluator(fn):
    """Wrap a function ``fn(task, experiment, workspace) -> Evaluation`` as an Evaluator.

    The function may use either positional args or keyword args named ``task``,
    ``experiment`` (new) or ``trajectory`` (old) for the second parameter.
    """

    class _FnEvaluator(Evaluator):
        def evaluate(self, task, experiment, workspace):
            return fn(task, experiment, workspace)

    return _FnEvaluator()


def as_reviewer(fn):
    class _FnReviewer(Reviewer):
        def review(self, tasks, experiments, evaluations, workspace):
            return fn(tasks, experiments, evaluations, workspace)

    return _FnReviewer()
