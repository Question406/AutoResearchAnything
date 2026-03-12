from __future__ import annotations

import re
import uuid

from aura.components.llm import LLMCallable
from aura.interfaces import Researcher
from aura.types import Hypothesis, Insight
from aura.utils.parsing import extract_json, render_prompt
from aura.workspace import Workspace


class LLMResearcher(Researcher):
    """Generate tasks by prompting an LLM.

    The prompt_template should contain Jinja2 placeholders:
    - {{ inputs }} — contents of workspace inputs/ directory context
    - {{ insights }} — formatted insights from previous iterations
    - {{ iteration }} — current iteration number
    - {{ num_tasks }} — number of tasks to generate

    In artifact mode (when artifact parameter is set), additional placeholders:
    - {{ artifact_content }} — current content of the artifact
    - {{ artifact_name }} — name of the artifact
    - {{ artifact_diff }} — diff from previous iteration

    The LLM must return JSON: a list of objects. Each object becomes a Hypothesis.
    The 'id' field is used as task ID. All other fields go into spec.
    """

    def __init__(
        self,
        llm: LLMCallable,
        prompt_template: str,
        num_tasks: int = 3,
        artifact: str | None = None,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.num_tasks = num_tasks
        self.artifact = artifact  # name of artifact to modify, or None

    def hypothesize(self, insights: list[Insight], workspace: Workspace) -> list[Hypothesis]:
        # Build inputs context from workspace
        inputs_context = self._read_inputs(workspace)

        # Format insights
        if insights:
            insights_text = "\n".join(f"- {i.content}" for i in insights)
        else:
            insights_text = "None yet (first iteration)"

        # Artifact mode
        if self.artifact and workspace.artifacts:
            artifact = workspace.artifacts[self.artifact]
            artifact_content = artifact.read()

            # Get diff from previous iteration
            it = workspace.current_iteration()
            artifact_diff = None
            if it > 1:
                prev_snapshot = workspace.iteration_dir(it - 1) / "artifacts"
                artifact_diff = artifact.diff(prev_snapshot)

            prompt = render_prompt(
                self.prompt_template,
                inputs=inputs_context,
                insights=insights_text,
                artifact_content=artifact_content,
                artifact_name=self.artifact,
                artifact_diff=artifact_diff or "First iteration — no previous changes",
                iteration=it,
                num_tasks=self.num_tasks,
            )

            response = self.llm(prompt)

            # Extract code from response (handle markdown fences)
            new_content = self._extract_code(response)
            artifact.write(new_content)

            return [
                Hypothesis(
                    id=f"iter_{it:03d}",
                    spec={"artifact": self.artifact},
                    metadata={"change_summary": (artifact_diff or "initial")[:500]},
                )
            ]

        # Normal mode (generate experiments)
        prompt = render_prompt(
            self.prompt_template,
            inputs=inputs_context,
            insights=insights_text,
            iteration=workspace.current_iteration(),
            num_tasks=self.num_tasks,
        )

        response = self.llm(prompt)
        items = extract_json(response)

        if isinstance(items, dict):
            items = [items]

        tasks = []
        for item in items:
            task_id = item.pop("id", str(uuid.uuid4())[:8])
            metadata = {}
            if "metadata" in item:
                metadata = item.pop("metadata")
            if "rationale" in item:
                metadata["rationale"] = item.pop("rationale")
            if "difficulty" in item:
                metadata["difficulty"] = item.pop("difficulty")
            tasks.append(Hypothesis(id=task_id, spec=item, metadata=metadata))

        return tasks

    def _read_inputs(self, workspace: Workspace) -> str:
        """Read all files from inputs/ directory as context."""
        inputs_dir = workspace.inputs_dir()
        parts = []
        for f in sorted(inputs_dir.rglob("*")):
            if f.is_file():
                try:
                    content = f.read_text()
                    rel_path = f.relative_to(inputs_dir)
                    parts.append(f"--- {rel_path} ---\n{content}")
                except (UnicodeDecodeError, OSError):
                    continue
        return "\n\n".join(parts) if parts else "No input files."

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response, handling markdown fences."""
        # Try to find code blocks
        match = re.search(r"```(?:\w+)?\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
