from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aura.interfaces import Collector
from aura.types import Trial, TrialStep

if TYPE_CHECKING:
    from aura.types import Hypothesis
    from aura.workspace import Workspace


class StdoutCollector(Collector):
    """Collect from a ScriptExecutor result dict (stdout / returncode).

    Sets trial status to "failed" if returncode != 0.
    """

    def __init__(self, parse_json: bool = True):
        self.parse_json = parse_json

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        ts = datetime.now(UTC).isoformat()
        steps = [
            TrialStep(
                step=0,
                data={"type": "command", "cmd": raw.get("cmd", "")},
                timestamp=raw.get("ts_start", ts),
            ),
            TrialStep(
                step=1,
                data={
                    "type": "result",
                    "stdout": raw.get("stdout", ""),
                    "stderr": raw.get("stderr", ""),
                    "returncode": raw.get("returncode", -1),
                },
                timestamp=raw.get("ts_end", ts),
            ),
        ]

        if raw.get("returncode", -1) != 0:
            return Trial(
                id=f"{task.id}-0",
                spec=task.spec,
                status="failed",
                steps=steps,
                output=None,
                error=raw.get("stderr", ""),
            )

        stdout = raw.get("stdout", "").strip()
        output: Any = stdout
        if self.parse_json:
            try:
                output = json.loads(stdout)
            except json.JSONDecodeError:
                pass

        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=steps,
            output=output,
            metadata={"spec": task.spec},
        )


class JSONFileCollector(Collector):
    """Collect from a JSON file written by the executor.

    ``output_path_template`` supports ``{task_id}`` interpolation.
    """

    def __init__(self, output_path_template: str):
        self.output_path_template = output_path_template

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        ts = datetime.now(UTC).isoformat()
        output_path = Path(self.output_path_template.format(task_id=task.id, **task.spec))

        if not output_path.exists():
            return Trial(
                id=f"{task.id}-0",
                spec=task.spec,
                status="failed",
                steps=[],
                output=None,
                error=f"Output file not found: {output_path}",
            )

        output = json.loads(output_path.read_text())
        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[TrialStep(step=0, data={"output_path": str(output_path)}, timestamp=ts)],
            output=output,
        )


class LogParserCollector(Collector):
    """Extract metrics from stdout using named regex groups.

    ``patterns`` maps metric names to regex patterns with a ``(?P<value>...)`` group.

    Example::

        LogParserCollector({"val_acc": r"val_acc=(?P<value>[\\d.]+)"})
    """

    def __init__(self, patterns: dict[str, str]):
        self.patterns = {name: re.compile(pat) for name, pat in patterns.items()}

    def collect(self, task: Hypothesis, raw: Any, context: dict, workspace: Workspace) -> Trial:
        ts = datetime.now(UTC).isoformat()
        stdout = raw.get("stdout", "") if isinstance(raw, dict) else str(raw)

        extracted: dict = {}
        for metric, pattern in self.patterns.items():
            m = pattern.search(stdout)
            if m:
                try:
                    extracted[metric] = float(m.group("value"))
                except (ValueError, IndexError):
                    extracted[metric] = m.group("value")

        if raw.get("returncode", 0) != 0 and not extracted:
            return Trial(
                id=f"{task.id}-0",
                spec=task.spec,
                status="failed",
                steps=[TrialStep(step=0, data={"stdout": stdout}, timestamp=ts)],
                output=None,
                error=raw.get("stderr", ""),
            )

        return Trial(
            id=f"{task.id}-0",
            spec=task.spec,
            status="completed",
            steps=[TrialStep(step=0, data={"stdout": stdout}, timestamp=ts)],
            output=extracted,
        )
