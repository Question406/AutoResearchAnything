from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.types import Hypothesis, Experiment, Evaluation, Insight
    from aura.workspace import Workspace


class Researcher(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def hypothesize(self, insights: list[Insight], workspace: Workspace) -> list[Hypothesis]:
        ...


class Experimenter(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def run_experiment(self, task: Hypothesis, workspace: Workspace) -> Experiment:
        ...


class Evaluator(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, task: Hypothesis, trajectory: Experiment, workspace: Workspace) -> Evaluation:
        ...


class Reviewer(ABC):
    def setup(self, workspace: Workspace) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def review(
        self,
        tasks: list[Hypothesis],
        trajectories: list[Experiment],
        evaluations: list[Evaluation],
        workspace: Workspace,
    ) -> list[Insight]:
        ...
