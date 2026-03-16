from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

JsonValue = dict | list | str | float | int | bool | None


class Hypothesis(BaseModel):
    id: str
    spec: dict
    metadata: dict = Field(default_factory=dict)


class TrialStep(BaseModel):
    step: int
    data: dict
    timestamp: str


# Backward-compat alias — existing code using ExperimentStep continues to work
ExperimentStep = TrialStep


class Trial(BaseModel):
    """One specific run of a hypothesis."""

    id: str
    spec: dict  # what was actually run (may vary per trial)
    status: str  # "completed" | "failed"
    steps: list[TrialStep]
    output: JsonValue
    error: str | None = None
    metadata: dict = Field(default_factory=dict)


class Experiment(BaseModel):
    """Result of investigating a hypothesis (one or more trials)."""

    hypothesis_id: str  # primary identifier; accepts task_id= kwarg via validator
    status: str  # "completed" | "partial" | "failed"

    # New multi-trial fields
    trials: list[Trial] = Field(default_factory=list)
    summary: JsonValue = None  # aggregated result across trials

    # Legacy single-trial fields — kept for backward compatibility
    steps: list[TrialStep] = Field(default_factory=list)
    output: JsonValue = None
    error: str | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_ids(cls, data: object) -> object:
        """Accept task_id= as an alias for hypothesis_id=."""
        if isinstance(data, dict) and "task_id" in data:
            d = dict(data)
            if not d.get("hypothesis_id"):
                d["hypothesis_id"] = d["task_id"]
            del d["task_id"]
            return d
        return data

    @model_validator(mode="after")
    def _sync_output_summary(self) -> Experiment:
        """Keep output and summary in sync for single-trial experiments."""
        if self.output is not None and self.summary is None:
            self.summary = self.output
        elif self.summary is not None and self.output is None:
            self.output = self.summary
        return self

    @property
    def task_id(self) -> str:
        """Backward-compat alias for hypothesis_id."""
        return self.hypothesis_id


class Evaluation(BaseModel):
    task_id: str
    score: float  # normalized 0-1
    passed: bool
    details: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class Insight(BaseModel):
    id: str
    source_iteration: int
    content: dict
    metadata: dict = Field(default_factory=dict)
