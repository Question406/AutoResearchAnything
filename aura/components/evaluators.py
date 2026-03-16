from __future__ import annotations

from aura.components.llm import LLMCallable
from aura.interfaces import Evaluator
from aura.types import Evaluation, Experiment, Hypothesis
from aura.utils.parsing import extract_json, render_prompt
from aura.workspace import Workspace


class MetricEvaluator(Evaluator):
    """Evaluate by comparing a numeric metric against a baseline.

    Extracts ``metric`` from ``experiment.summary`` and scores based on improvement
    over baseline. Score is normalized to 0-1.

    Example::

        MetricEvaluator(metric="accuracy", baseline=0.5)
    """

    def __init__(
        self,
        metric: str,
        baseline: float = 0.0,
        higher_is_better: bool = True,
        max_improvement: float = 0.5,
    ):
        # Don't call super().__init__ with runner — MetricEvaluator doesn't use one
        super().__init__()
        self.metric = metric
        self.baseline = baseline
        self.higher_is_better = higher_is_better
        self.max_improvement = max_improvement

    def evaluate(
        self, task: Hypothesis, experiment: Experiment, workspace: Workspace
    ) -> Evaluation:
        if experiment.status == "failed":
            return Evaluation(
                task_id=task.id,
                score=0.0,
                passed=False,
                details={"reason": "execution_failed", "error": experiment.error},
            )

        # Read from summary (for multi-trial) with fallback to output (legacy)
        summary = experiment.summary
        output = summary if isinstance(summary, dict) else {}
        value = output.get(self.metric, self.baseline)

        if self.higher_is_better:
            improvement = value - self.baseline
            passed = value > self.baseline
        else:
            improvement = self.baseline - value
            passed = value < self.baseline

        score = (
            min(1.0, max(0.0, improvement / self.max_improvement))
            if self.max_improvement != 0
            else (1.0 if passed else 0.0)
        )

        return Evaluation(
            task_id=task.id,
            score=round(score, 4),
            passed=passed,
            details={
                self.metric: value,
                "baseline": self.baseline,
                "improvement": round(improvement, 4),
            },
        )


class LLMJudgeEvaluator(Evaluator):
    """Evaluate by asking an LLM to judge the experiment.

    .. deprecated::
        Use ``Evaluator(runner=llm, prompt_template=...)`` instead.

    The prompt_template receives:

    - ``{{ task }}`` — task spec as formatted string
    - ``{{ output }}`` — experiment summary/output
    - ``{{ trajectory }}`` — full experiment steps (legacy name kept for templates)

    LLM must return JSON: ``{"score": float, "passed": bool, "reason": string}``
    """

    def __init__(self, llm: LLMCallable, prompt_template: str | None = None):
        super().__init__(prompt_template=prompt_template)
        self.llm = llm

    def evaluate(
        self, task: Hypothesis, experiment: Experiment, workspace: Workspace
    ) -> Evaluation:
        if experiment.status == "failed":
            return Evaluation(
                task_id=task.id,
                score=0.0,
                passed=False,
                details={"reason": "execution_failed", "error": experiment.error},
            )

        prompt = render_prompt(
            self.prompt_template,
            task=task.spec,
            output=experiment.summary,
            # "trajectory" kept as template var name for backward compat with user templates
            trajectory=[s.data for s in experiment.steps],
        )

        response = self.llm(prompt)
        result = extract_json(response)

        return Evaluation(
            task_id=task.id,
            score=float(result.get("score", 0.0)),
            passed=bool(result.get("passed", False)),
            details={"reason": result.get("reason", "")},
        )
