from __future__ import annotations

import uuid

from aura.components.llm import LLMCallable
from aura.interfaces import Reviewer
from aura.types import Insight
from aura.utils.parsing import extract_json, render_prompt


class LLMReviewer(Reviewer):
    """Extract insights by asking an LLM to analyze results.

    The prompt_template receives:

    - ``{{ results }}`` — formatted summary of task results
    - ``{{ iteration }}`` — current iteration number

    LLM must return JSON list: ``[{"finding": string, "recommendation": string}]``
    """

    def __init__(self, llm: LLMCallable, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or (
            "Analyze these experiment results:\n\n"
            "{{ results }}\n\n"
            "Extract 2-3 insights. What worked? What failed? What to try next?\n\n"
            'Respond as JSON list: [{"finding": string, "recommendation": string}]'
        )

    def review(self, tasks, experiments, evaluations, workspace) -> list[Insight]:
        eval_by_id = {e.task_id: e for e in evaluations}
        exp_by_id = {e.task_id: e for e in experiments}

        lines = []
        for task in tasks:
            ev = eval_by_id.get(task.id)
            exp = exp_by_id.get(task.id)
            output_summary = exp.summary if exp and exp.summary else "N/A"
            if isinstance(output_summary, dict):
                output_summary = ", ".join(f"{k}={v}" for k, v in output_summary.items())
            lines.append(
                f"- Task {task.id} (spec: {task.spec}): "
                f"output={output_summary}, "
                f"score={ev.score if ev else 'N/A'}, "
                f"passed={ev.passed if ev else 'N/A'}"
            )

        results_text = "\n".join(lines) if lines else "No results."

        prompt = render_prompt(
            self.prompt_template,
            results=results_text,
            iteration=workspace.current_iteration(),
        )

        response = self.llm(prompt)
        items = extract_json(response)

        if isinstance(items, dict):
            items = [items]

        return [
            Insight(
                id=str(uuid.uuid4())[:8],
                source_iteration=workspace.current_iteration(),
                content=item,
            )
            for item in items
        ]
