# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_pipeline.py -v

# Run tests matching a pattern
uv run pytest -k test_pipeline_basic

# Validate a user file for CLI compatibility
aura validate examples/mock-autonas-cli/run.py

# Run an experiment via CLI
aura run examples/mock-autonas-cli/run.py --run-dir ./runs/my_experiment
```

## Architecture

AURA (Auto Research Anything) is a domain-agnostic framework for self-improving research loops. An LLM proposes hypotheses, a runner executes them, an evaluator scores results, and a reviewer distills insights — then the loop repeats.

### Three-Layer Design

```
User Code (custom Runner subclass or script with globals/main pattern)
    ↓
Built-in Components (LLMResearcher, ScriptExperimenter, MetricEvaluator, LLMReviewer)
    ↓
Core Framework (Pipeline, Workspace, Interfaces, Types)
```

### Core Loop (Pipeline.run)

`Pipeline` orchestrates the four-stage iteration: **Hypothesize → Experiment → Evaluate → Review**. Each stage delegates to a pluggable component implementing one of four ABCs defined in `aura/interfaces.py`: `Researcher`, `Experimenter`, `Evaluator`, `Reviewer`. All cross-component data flows through Pydantic models in `aura/types.py` (`Hypothesis`, `Experiment`, `Evaluation`, `Insight`).

### Persistence (Workspace)

`aura/workspace.py` manages all run state as human-readable JSON under a run directory. Structure: `manifest.json` at root, then `iteration_NNN/` subdirectories each containing `tasks/`, `trajectories/`, `evaluations/`, and `insights.json`. Pipeline checks for existing files before executing (enables mid-iteration crash recovery).

### Artifact System

`aura/artifacts.py` provides `FileArtifact` and `DirectoryArtifact` for tracking files that evolve across iterations. Pipeline snapshots artifacts at iteration start and optionally rolls back to best-scoring state (`rollback="best"`).

### CLI Entry Detection

`aura/cli.py` uses AST analysis to detect three user file patterns:
- **"main"** — file defines `def main()`
- **"runner"** — file defines a `Runner` subclass
- **"globals"** — file has module-level `workspace` and `pipeline` variables (executed via `runpy.run_path`)

### LLM Interface

`LLMCallable = Callable[[str], str]` — a single callable. Three factory functions in `aura/components/llm.py`: `anthropic_llm()`, `openai_llm()`, `command_llm()`. Components that use LLMs accept this callable, keeping them provider-agnostic.

### Decorators vs ABCs

`aura/decorators.py` provides `as_researcher`, `as_experimenter`, `as_evaluator`, `as_reviewer` for wrapping plain functions as component instances. Use ABCs (`aura/interfaces.py`) for stateful components that need `setup()`/`teardown()` lifecycle methods.

## Key Conventions

- Python >= 3.11, managed with `uv` and built with Hatchling
- Only two core dependencies: `pydantic` (validation/serialization) and `jinja2` (prompt templates)
- All public API is re-exported from `aura/__init__.py`
- Prompt templates use Jinja2 syntax with placeholders like `{{ inputs }}`, `{{ insights }}`, `{{ iteration }}`
- Built-in components that need timeout respect `workspace.constraints()["time_budget"]`
- Tests use `tmp_path` fixtures; `conftest.py` provides `tmp_workspace` fixture
