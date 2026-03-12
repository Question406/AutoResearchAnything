from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aura.types import Evaluation, Experiment, Hypothesis, Insight

if TYPE_CHECKING:
    from aura.artifacts import Artifact


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._current_iteration = 0
        self.artifacts: dict[str, Artifact] = {}

    @classmethod
    def create(cls, run_dir: Path) -> Workspace:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "inputs").mkdir(exist_ok=True)

        manifest = {
            "run_id": run_dir.name,
            "created_at": datetime.now(UTC).isoformat(),
            "components": {},
            "config": {},
            "iterations_completed": 0,
            "status": "created",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return cls(run_dir)

    # --- Directory access ---

    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    def artifacts_dir(self) -> Path:
        d = self.root / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def iteration_dir(self, iteration: int) -> Path:
        d = self.root / f"iteration_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tasks").mkdir(exist_ok=True)
        (d / "trajectories").mkdir(exist_ok=True)
        (d / "evaluations").mkdir(exist_ok=True)
        (d / "artifacts").mkdir(exist_ok=True)
        return d

    def current_iteration(self) -> int:
        return self._current_iteration

    def current_iteration_dir(self) -> Path:
        return self.iteration_dir(self._current_iteration)

    def set_current_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    # --- Save helpers ---

    def save_task(self, task: Hypothesis, iteration: int | None = None) -> None:
        it = iteration if iteration is not None else self._current_iteration
        path = self.iteration_dir(it) / "tasks" / f"{task.id}.json"
        path.write_text(task.model_dump_json(indent=2))

    def save_trajectory(self, trajectory: Experiment, iteration: int | None = None) -> None:
        it = iteration if iteration is not None else self._current_iteration
        path = self.iteration_dir(it) / "trajectories" / f"{trajectory.task_id}.json"
        path.write_text(trajectory.model_dump_json(indent=2))

    def save_evaluation(self, result: Evaluation, iteration: int | None = None) -> None:
        it = iteration if iteration is not None else self._current_iteration
        path = self.iteration_dir(it) / "evaluations" / f"{result.task_id}.json"
        path.write_text(result.model_dump_json(indent=2))

    def save_insights(self, insights: list[Insight], iteration: int | None = None) -> None:
        it = iteration if iteration is not None else self._current_iteration
        path = self.iteration_dir(it) / "insights.json"
        path.write_text(json.dumps([i.model_dump() for i in insights], indent=2))

    # --- Load helpers ---

    def load_tasks(self, iteration: int) -> list[Hypothesis]:
        tasks_dir = self.iteration_dir(iteration) / "tasks"
        return [
            Hypothesis.model_validate_json(f.read_text()) for f in sorted(tasks_dir.glob("*.json"))
        ]

    def load_trajectory(self, task_id: str, iteration: int) -> Experiment:
        path = self.iteration_dir(iteration) / "trajectories" / f"{task_id}.json"
        return Experiment.model_validate_json(path.read_text())

    def load_trajectories(self, iteration: int) -> list[Experiment]:
        traj_dir = self.iteration_dir(iteration) / "trajectories"
        return [
            Experiment.model_validate_json(f.read_text()) for f in sorted(traj_dir.glob("*.json"))
        ]

    def load_evaluation(self, task_id: str, iteration: int) -> Evaluation:
        path = self.iteration_dir(iteration) / "evaluations" / f"{task_id}.json"
        return Evaluation.model_validate_json(path.read_text())

    def load_evaluations(self, iteration: int) -> list[Evaluation]:
        eval_dir = self.iteration_dir(iteration) / "evaluations"
        return [
            Evaluation.model_validate_json(f.read_text()) for f in sorted(eval_dir.glob("*.json"))
        ]

    def load_insights(self, iteration: int) -> list[Insight]:
        path = self.iteration_dir(iteration) / "insights.json"
        if not path.exists():
            return []
        return [Insight.model_validate(d) for d in json.loads(path.read_text())]

    # --- State queries ---

    def has_trajectory(self, task_id: str, iteration: int | None = None) -> bool:
        it = iteration if iteration is not None else self._current_iteration
        return (self.iteration_dir(it) / "trajectories" / f"{task_id}.json").exists()

    def has_evaluation(self, task_id: str, iteration: int | None = None) -> bool:
        it = iteration if iteration is not None else self._current_iteration
        return (self.iteration_dir(it) / "evaluations" / f"{task_id}.json").exists()

    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    def update_manifest(self, **kwargs) -> None:
        m = self.manifest()
        m.update(kwargs)
        (self.root / "manifest.json").write_text(json.dumps(m, indent=2))

    def constraints(self) -> dict:
        return self.manifest().get("constraints", {})

    def set_constraints(self, constraints: dict) -> None:
        self.update_manifest(constraints=constraints)

    def summary(self) -> dict:
        """Return structured summary of the run."""
        m = self.manifest()
        iterations_completed = m.get("iterations_completed", 0)

        iteration_summaries = []
        best_score = 0.0
        best_task_id = None
        total_tasks = 0
        total_passed = 0
        total_failed_exec = 0

        for it in range(1, iterations_completed + 1):
            tasks = self.load_tasks(it)
            evals = self.load_evaluations(it)
            trajectories = self.load_trajectories(it)

            scores = [e.score for e in evals]
            passed = sum(1 for e in evals if e.passed)
            failed_exec = sum(1 for t in trajectories if t.status == "failed")

            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0

            for e in evals:
                if e.score > best_score:
                    best_score = e.score
                    best_task_id = e.task_id

            total_tasks += len(tasks)
            total_passed += passed
            total_failed_exec += failed_exec

            iteration_summaries.append(
                {
                    "iteration": it,
                    "num_tasks": len(tasks),
                    "avg_score": round(avg_score, 4),
                    "max_score": round(max_score, 4),
                    "passed": passed,
                    "failed_exec": failed_exec,
                }
            )

        return {
            "run_id": m.get("run_id"),
            "status": m.get("status"),
            "iterations_completed": iterations_completed,
            "total_tasks": total_tasks,
            "total_passed": total_passed,
            "total_failed_exec": total_failed_exec,
            "best_score": round(best_score, 4),
            "best_task_id": best_task_id,
            "iterations": iteration_summaries,
        }
