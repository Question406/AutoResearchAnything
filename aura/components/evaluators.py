from __future__ import annotations

from aura.interfaces import Evaluator
from aura.types import Hypothesis, Experiment, Evaluation
from aura.workspace import Workspace
from aura.utils.parsing import extract_json, render_prompt
from aura.components.llm import LLMCallable


class MetricEvaluator(Evaluator):
    """Evaluate by comparing a numeric metric against a baseline.

    Extracts `metric` from trajectory.output and scores based on improvement
    over baseline. Score is normalized to 0-1.

    Example:
        MetricEvaluator(metric="accuracy", baseline=0.5)
    """

    def __init__(self, metric: str, baseline: float = 0.0, higher_is_better: bool = True, max_improvement: float = 0.5):
        self.metric = metric
        self.baseline = baseline
        self.higher_is_better = higher_is_better
        self.max_improvement = max_improvement

    def evaluate(self, task: Hypothesis, trajectory: Experiment, workspace: Workspace) -> Evaluation:
        if trajectory.status == "failed":
            return Evaluation(task_id=task.id, score=0.0, passed=False, details={"reason": "execution_failed", "error": trajectory.error})

        output = trajectory.output if isinstance(trajectory.output, dict) else {}
        value = output.get(self.metric, self.baseline)

        if self.higher_is_better:
            improvement = value - self.baseline
            passed = value > self.baseline
        else:
            improvement = self.baseline - value
            passed = value < self.baseline

        score = min(1.0, max(0.0, improvement / self.max_improvement)) if self.max_improvement != 0 else (1.0 if passed else 0.0)

        return Evaluation(
            task_id=task.id,
            score=round(score, 4),
            passed=passed,
            details={self.metric: value, "baseline": self.baseline, "improvement": round(improvement, 4)},
        )


class LLMJudgeEvaluator(Evaluator):
    """Evaluate by asking an LLM to judge the trajectory.

    The prompt_template receives:
    - {{ task }} — task spec as formatted string
    - {{ output }} — trajectory output
    - {{ trajectory }} — full trajectory steps

    LLM must return JSON: {"score": float, "passed": bool, "reason": string}
    """

    def __init__(self, llm: LLMCallable, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or (
            "Evaluate this result.\n\n"
            "Task: {{ task }}\n\n"
            "Output: {{ output }}\n\n"
            'Rate on 0.0-1.0. Respond in JSON: {"score": float, "passed": bool, "reason": string}'
        )

    def evaluate(self, task: Hypothesis, trajectory: Experiment, workspace: Workspace) -> Evaluation:
        if trajectory.status == "failed":
            return Evaluation(task_id=task.id, score=0.0, passed=False, details={"reason": "execution_failed", "error": trajectory.error})

        prompt = render_prompt(
            self.prompt_template,
            task=task.spec,
            output=trajectory.output,
            trajectory=[s.data for s in trajectory.steps],
        )

        response = self.llm(prompt)
        result = extract_json(response)

        return Evaluation(
            task_id=task.id,
            score=float(result.get("score", 0.0)),
            passed=bool(result.get("passed", False)),
            details={"reason": result.get("reason", "")},
        )
