from __future__ import annotations
from typing import Union
from pydantic import BaseModel, Field

JsonValue = Union[dict, list, str, float, int, bool, None]


class Hypothesis(BaseModel):
    id: str
    spec: dict
    metadata: dict = Field(default_factory=dict)


class ExperimentStep(BaseModel):
    step: int
    data: dict
    timestamp: str


class Experiment(BaseModel):
    task_id: str
    status: str  # "completed" | "failed"
    steps: list[ExperimentStep]
    output: JsonValue
    error: str | None = None
    metadata: dict = Field(default_factory=dict)


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
