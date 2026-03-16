from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.types import JsonValue, Trial


class Aggregator(ABC):
    """Reduce a list of trials into a single summary value."""

    @abstractmethod
    def aggregate(self, trials: list[Trial]) -> JsonValue: ...


class LastTrialAggregator(Aggregator):
    """Return the output of the last trial. Default for single-trial experimenters."""

    def aggregate(self, trials: list[Trial]) -> JsonValue:
        return trials[-1].output if trials else None


class BestTrialAggregator(Aggregator):
    """Return the output of the trial with the highest (or lowest) metric value."""

    def __init__(self, metric: str, higher_is_better: bool = True):
        self.metric = metric
        self.higher_is_better = higher_is_better

    def aggregate(self, trials: list[Trial]) -> JsonValue:
        if not trials:
            return None
        completed = [t for t in trials if t.status == "completed" and isinstance(t.output, dict)]
        if not completed:
            return trials[-1].output
        default = float("-inf") if self.higher_is_better else float("inf")
        key = lambda t: t.output.get(self.metric, default)  # noqa: E731
        best = max(completed, key=key) if self.higher_is_better else min(completed, key=key)
        return best.output


class AllTrialsAggregator(Aggregator):
    """Return all trial outputs as a list."""

    def aggregate(self, trials: list[Trial]) -> JsonValue:
        return [t.output for t in trials]
