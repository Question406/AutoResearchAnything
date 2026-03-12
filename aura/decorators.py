from __future__ import annotations

from aura.interfaces import Evaluator, Experimenter, Researcher, Reviewer


def as_researcher(fn):
    class _FnResearcher(Researcher):
        def hypothesize(self, insights, workspace):
            return fn(insights, workspace)

    return _FnResearcher()


def as_experimenter(fn):
    class _FnExperimenter(Experimenter):
        def run_experiment(self, task, workspace):
            return fn(task, workspace)

    return _FnExperimenter()


def as_evaluator(fn):
    class _FnEvaluator(Evaluator):
        def evaluate(self, task, trajectory, workspace):
            return fn(task, trajectory, workspace)

    return _FnEvaluator()


def as_reviewer(fn):
    class _FnReviewer(Reviewer):
        def review(self, tasks, trajectories, evaluations, workspace):
            return fn(tasks, trajectories, evaluations, workspace)

    return _FnReviewer()
