from aura.artifacts import Artifact, DirectoryArtifact, FileArtifact
from aura.components import (
    AllTrialsAggregator,
    BestTrialAggregator,
    CommandRunner,
    CondaEnvironment,
    DockerEnvironment,
    FunctionRunner,
    UvEnvironment,
    FunctionExperimenter,
    FunctionExecutor,
    JSONFileCollector,
    LLMCallable,
    LLMExperimenter,
    LLMExecutor,
    LLMJudgeEvaluator,
    LLMResearcher,
    LLMReviewer,
    LLMRunner,
    LastTrialAggregator,
    LogParserCollector,
    MetricEvaluator,
    ScriptExperimenter,
    BindfsExecutor,
    BwrapExecutor,
    ScriptExecutor,
    SlurmExecutor,
    StdoutCollector,
    VenvEnvironment,
    as_runner,
)
from aura.components.llm import anthropic_llm, command_llm, openai_llm
from aura.decorators import as_evaluator, as_experimenter, as_researcher, as_reviewer
from aura.components.aggregators import Aggregator
from aura.interfaces import (
    Collector,
    Environment,
    Evaluator,
    Executor,
    Experimenter,
    Researcher,
    Reviewer,
    Runner,
    SingleTrialExperimenter,
)
from aura.pipeline import Pipeline
from aura.runner import CliRunner
from aura.types import (
    Evaluation,
    Experiment,
    ExperimentStep,
    Hypothesis,
    Insight,
    JsonValue,
    Trial,
    TrialStep,
)
from aura.utils.logging import setup_logging
from aura.utils.parsing import extract_json, render_prompt
from aura.workspace import Workspace

__all__ = [
    # Types
    "Hypothesis",
    "TrialStep",
    "ExperimentStep",  # backward-compat alias for TrialStep
    "Trial",
    "Experiment",
    "Evaluation",
    "Insight",
    "JsonValue",
    # Interfaces
    "Runner",
    "Researcher",
    "Experimenter",
    "SingleTrialExperimenter",
    "Evaluator",
    "Reviewer",
    "Environment",
    "Executor",
    "Collector",
    "Aggregator",
    # Decorators
    "as_researcher",
    "as_experimenter",
    "as_evaluator",
    "as_reviewer",
    # Core
    "Artifact",
    "FileArtifact",
    "DirectoryArtifact",
    "Workspace",
    "Pipeline",
    "CliRunner",
    # Utilities
    "extract_json",
    "render_prompt",
    "setup_logging",
    # LLM
    "LLMCallable",
    "anthropic_llm",
    "openai_llm",
    "command_llm",
    # Runners
    "LLMRunner",
    "CommandRunner",
    "FunctionRunner",
    "as_runner",
    # Built-in components
    "LLMResearcher",
    "ScriptExperimenter",
    "FunctionExperimenter",
    "LLMExperimenter",
    "MetricEvaluator",
    "LLMJudgeEvaluator",
    "LLMReviewer",
    # Aggregators
    "BestTrialAggregator",
    "LastTrialAggregator",
    "AllTrialsAggregator",
    # Environments
    "CondaEnvironment",
    "VenvEnvironment",
    "UvEnvironment",
    "DockerEnvironment",
    # Executor backends
    "ScriptExecutor",
    "BwrapExecutor",
    "BindfsExecutor",
    "FunctionExecutor",
    "LLMExecutor",
    "SlurmExecutor",
    # Collector backends
    "StdoutCollector",
    "JSONFileCollector",
    "LogParserCollector",
]
