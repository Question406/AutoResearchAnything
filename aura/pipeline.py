from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

from aura.interfaces import Researcher, Experimenter, Evaluator, Reviewer
from aura.types import Hypothesis, Experiment, Evaluation, Insight
from aura.workspace import Workspace

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        researcher: Researcher,
        experimenter: Experimenter,
        evaluator: Evaluator,
        reviewer: Reviewer,
        workspace: Workspace,
        max_retries: int = 3,
        max_iterations: int | None = None,
        parallel_tasks: int = 1,
        insight_window: int | None = None,
        prepare: Callable[[Workspace], None] | None = None,
        artifacts: list | None = None,
        rollback: str = "none",
        constraints: dict | None = None,
    ) -> None:
        self.researcher = researcher
        self.experimenter = experimenter
        self.evaluator = evaluator
        self.reviewer = reviewer
        self.workspace = workspace
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.parallel_tasks = parallel_tasks
        self.insight_window = insight_window
        self.prepare = prepare
        self.artifacts = artifacts or []
        self.rollback = rollback
        self.constraints = constraints
        self._best_score = 0.0
        self._best_iteration = 0

    def run(self) -> None:
        # Save constraints to manifest if provided
        if self.constraints:
            self.workspace.set_constraints(self.constraints)

        # Call prepare hook before setup
        if self.prepare is not None:
            self.prepare(self.workspace)

        # Register artifacts on workspace
        for artifact in self.artifacts:
            self.workspace.artifacts[artifact.name] = artifact

        self._call_setup()
        logger.info(f"Pipeline starting (max_iterations={self.max_iterations})")
        try:
            start = self.workspace.manifest().get("iterations_completed", 0) + 1
            iteration = start

            while not self._should_stop(iteration):
                self.workspace.set_current_iteration(iteration)
                self._run_iteration(iteration)
                self.workspace.update_manifest(
                    iterations_completed=iteration, status="in_progress"
                )
                iteration += 1

            self.workspace.update_manifest(status="completed")
            logger.info("Pipeline completed")
        finally:
            self._call_teardown()

    def _should_stop(self, iteration: int) -> bool:
        if self.max_iterations is not None and iteration > self.max_iterations:
            return True
        return False

    def _run_iteration(self, iteration: int) -> None:
        logger.info(f"Starting iteration {iteration}")

        # Snapshot artifacts before this iteration modifies them
        if self.artifacts:
            self._snapshot_artifacts(iteration)

        insights = self._gather_insights(iteration)
        tasks = self.researcher.hypothesize(insights, self.workspace)
        for task in tasks:
            self.workspace.save_task(task, iteration=iteration)
        logger.info(f"Iteration {iteration}: generated {len(tasks)} tasks")

        trajectories = self._execute_tasks(tasks, iteration)
        failed = sum(1 for t in trajectories if t.status == "failed")
        logger.info(f"Iteration {iteration}: executed {len(trajectories)} tasks ({failed} failed)")

        evaluations = self._evaluate(tasks, trajectories, iteration)
        if evaluations:
            avg_score = sum(e.score for e in evaluations) / len(evaluations)
            best_this = max(e.score for e in evaluations)
            logger.info(f"Iteration {iteration}: avg score {avg_score:.4f}")

            # Track best and maybe rollback
            if self.artifacts and self.rollback == "best":
                if best_this > self._best_score:
                    self._best_score = best_this
                    self._best_iteration = iteration
                    self._save_best_artifacts()
                    logger.info(f"Iteration {iteration}: new best score {best_this:.4f}")
                else:
                    self._rollback_artifacts()
                    logger.info(f"Iteration {iteration}: no improvement, rolled back to iteration {self._best_iteration}")

        new_insights = self.reviewer.review(
            tasks, trajectories, evaluations, self.workspace
        )
        self.workspace.save_insights(new_insights, iteration=iteration)

    def _snapshot_artifacts(self, iteration: int) -> None:
        dest = self.workspace.iteration_dir(iteration) / "artifacts"
        for artifact in self.artifacts:
            artifact.snapshot(dest)

    def _save_best_artifacts(self) -> None:
        """Snapshot current artifact state as the best known state."""
        dest = self.workspace.root / "_best_artifacts"
        dest.mkdir(parents=True, exist_ok=True)
        for artifact in self.artifacts:
            artifact.snapshot(dest)

    def _rollback_artifacts(self) -> None:
        src = self.workspace.root / "_best_artifacts"
        for artifact in self.artifacts:
            artifact.restore(src)

    def _gather_insights(self, current_iteration: int) -> list[Insight]:
        insights = []
        if current_iteration <= 1:
            return insights

        if self.insight_window is None:
            start = 1
        else:
            start = max(1, current_iteration - self.insight_window)

        for it in range(start, current_iteration):
            insights.extend(self.workspace.load_insights(it))
        return insights

    def _execute_tasks(self, tasks: list[Hypothesis], iteration: int) -> list[Experiment]:
        if not tasks:
            return []

        if self.parallel_tasks <= 1:
            return [self._execute_single(task, iteration) for task in tasks]

        trajectories = []
        with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
            futures = {
                pool.submit(self._execute_single, task, iteration): task
                for task in tasks
            }
            for future in as_completed(futures):
                trajectories.append(future.result())
        return trajectories

    def _execute_single(self, task: Hypothesis, iteration: int) -> Experiment:
        if self.workspace.has_trajectory(task.id, iteration=iteration):
            logger.info(f"Task {task.id} already has trajectory, skipping")
            return self.workspace.load_trajectory(task.id, iteration=iteration)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                trajectory = self.experimenter.run_experiment(task, self.workspace)
                self.workspace.save_trajectory(trajectory, iteration=iteration)
                return trajectory
            except Exception as e:
                last_error = e
                logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {e}")

        logger.error(f"Task {task.id} failed after {self.max_retries} attempts")
        failed = Experiment(
            task_id=task.id,
            status="failed",
            steps=[],
            output=None,
            error=str(last_error),
            metadata={"retries": self.max_retries},
        )
        self.workspace.save_trajectory(failed, iteration=iteration)
        return failed

    def _evaluate(
        self, tasks: list[Hypothesis], trajectories: list[Experiment], iteration: int
    ) -> list[Evaluation]:
        traj_by_id = {t.task_id: t for t in trajectories}
        results = []
        for task in tasks:
            traj = traj_by_id.get(task.id)
            if traj is None:
                continue
            if self.workspace.has_evaluation(task.id, iteration=iteration):
                logger.info(f"Task {task.id} already has evaluation, skipping")
                existing = self.workspace.load_evaluation(task.id, iteration=iteration)
                results.append(existing)
                continue
            result = self.evaluator.evaluate(task, traj, self.workspace)
            self.workspace.save_evaluation(result, iteration=iteration)
            results.append(result)
        return results

    def _call_setup(self) -> None:
        self.researcher.setup(self.workspace)
        self.experimenter.setup(self.workspace)
        self.evaluator.setup(self.workspace)
        self.reviewer.setup(self.workspace)

    def _call_teardown(self) -> None:
        self.researcher.teardown()
        self.experimenter.teardown()
        self.evaluator.teardown()
        self.reviewer.teardown()
