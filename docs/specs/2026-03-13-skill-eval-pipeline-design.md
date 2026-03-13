# Skill Eval Pipeline Design

**Date**: 2026-03-13
**Status**: Draft
**Topic**: Automatic skill-specific task query and evaluation judge generation using AURA

---

## Overview

An AURA pipeline that takes a skill `.md` file as input and iteratively generates and improves an **eval suite** — a set of task queries and a hybrid judge — for that skill. The eval suite is a `FileArtifact` that evolves across iterations, with `rollback="best"` ensuring the highest-quality version is always preserved.

The pipeline addresses the challenge of evaluating whether Claude correctly invokes a skill: given a user message, does Claude call the `Skill` tool with the right skill name, and does its response follow the skill's defined workflow?

---

## Input Tiers

The pipeline handles three levels of seed data gracefully:

| Tier | Inputs | Behavior |
|------|--------|----------|
| A | `skill.md` only | Bootstraps from skill text; first iteration generates synthetic calibration pairs |
| B | `skill.md` + `seeds.json` | Seeds anchor judge calibration from iteration 1 |
| C | `skill.md` + `seeds.json` + `existing_evals/` | Builds on prior work; deduplicates against existing queries |

**Inputs directory layout** (`<run-dir>/inputs/`, created by `Workspace.create()`):
```
<run-dir>/inputs/
  skill.md            # required — the skill being evaluated
  seeds.json          # optional — manually curated query/response pairs
  existing_evals/     # optional — prior eval files to build on
```

Seed entries in `seeds.json` are never modified or deleted by the pipeline. They serve as ground truth for judge calibration. Items with `expected: "trigger"` and `seed: true` form the calibration positive set; items with `expected: "no_trigger"` and `seed: true` form the negative set. If seeds contain only one polarity, `calibration_score` falls back to discrimination accuracy on those seeds alone (no pairwise ranking).

---

## Artifact: EvalSuite

The central artifact is `eval_suite.json`, a single file that evolves across iterations. It is located at `<run-dir>/eval_suite.json` so `FileArtifact("eval_suite.json")` resolves it relative to the working directory when `run.py` is invoked.

```json
{
  "skill_name": "aura",
  "queries": [
    {
      "id": "q001",
      "input": "I want to automate my ML experiments",
      "expected": "trigger",
      "tags": ["clear_trigger", "research_context"],
      "seed": true
    },
    {
      "id": "q002",
      "input": "What's the weather today?",
      "expected": "no_trigger",
      "tags": ["off_topic"],
      "seed": false
    }
  ],
  "judge": [
    {
      "id": "j001",
      "type": "code",
      "description": "Skill tool was called with correct skill name",
      "check": "skill_tool_called(response, 'aura')",
      "weight": 0.4
    },
    {
      "id": "j002",
      "type": "llm",
      "description": "Response followed the skill's workflow",
      "rubric": "Did the agent invoke the aura skill and follow its defined workflow steps?",
      "weight": 0.3
    },
    {
      "id": "j003",
      "type": "code",
      "description": "No premature implementation before skill invocation",
      "check": "no_implementation_before_skill(response)",
      "weight": 0.3
    }
  ]
}
```

**Schema invariants**:
- `seed: true` items are never deleted or modified by the pipeline
- Judge item weights sum to 1.0
- `expected` is one of `"trigger"` or `"no_trigger"`
- Code checks are string expressions naming a function from `judge_stdlib.py`; the pipeline may add new stdlib functions but not redefine existing ones
- Because the pipeline produces exactly one `Hypothesis` per iteration (artifact mode), rollback compares `iteration_score` directly across iterations

---

## Architecture

### Approach: Artifact-based co-evolution

The eval suite (queries + hybrid judge) is co-evolved in a single AURA loop. The Researcher mutates the suite as a whole each iteration, guided by insights about what is missing or miscalibrated. This is better than two sequential pipelines because:

- Insights about judge blind spots feed back into query generation immediately
- Insights about query gaps feed back into judge refinement immediately
- AURA's artifact + rollback system tracks the best-scoring suite automatically

### AURA Loop Mapping

```
Hypothesize  →  Experiment  →  Evaluate  →  Review
    │               │              │            │
 Mutate         Run queries    Score suite   Distill
 eval_suite     through        on two axes   what's
 (queries +     executor;      (discrimin-   missing or
 judge items)   calibrate      ation +       miscalibrated
                on seeds       calibration)
```

---

## Components

### Researcher

Uses `LLMResearcher` in artifact mode. Each iteration reads:
- `skill.md` from `inputs/` (arrives in `{{ inputs }}` — `LLMResearcher._read_inputs` concatenates all files under `inputs/`)
- Current `eval_suite.json` (in `{{ artifact_content }}`)
- Diff from the previous iteration (in `{{ artifact_diff }}`)
- Reviewer insights from previous iterations (in `{{ insights }}`)

Proposes mutations to the suite: add queries, remove redundant ones, add/refine judge items, adjust weights. Outputs an updated `eval_suite.json`. One hypothesis per iteration — the whole suite is the unit of improvement.

**Prompt template placeholders** (provided by `LLMResearcher` in artifact mode):
`{{ inputs }}`, `{{ artifact_content }}`, `{{ artifact_diff }}`, `{{ insights }}`, `{{ iteration }}`

### Experimenter

Pluggable via an abstract `SkillExecutor` interface. `ExecutionResult` is a Pydantic model so it serializes cleanly into `Experiment.output`:

```python
class SkillExecutor(ABC):
    @abstractmethod
    def execute(self, query: str, skill_content: str) -> ExecutionResult: ...

class ExecutionResult(BaseModel):
    skill_invoked: bool
    skill_name: str | None
    response: str
    raw: dict
```

**Built-in executors**:

| Executor | Description | Cost |
|----------|-------------|------|
| `ClaudeAPIExecutor` | Real API call; checks if `Skill` tool was called | High |
| `LLMPredictExecutor` | Asks an LLM to predict invocation (proxy) | Medium |
| `MockExecutor` | Deterministic; useful for testing the pipeline | Zero |

Each iteration the Experimenter:
1. Samples `sample_size` queries from the suite (configurable)
2. Always runs all seed examples regardless of `sample_size` (for calibration stability)
3. Runs each query through the executor and records `ExecutionResult`
4. For each judge item of type `"code"`, calls the corresponding `judge_stdlib` function with the response string; result is `1.0` if `True`, `0.0` if `False`
5. For each judge item of type `"llm"`, calls the LLM judge with the rubric; result is a float `[0,1]`

The trajectory (`Experiment`) stores all results as structured JSON.

### Judge Stdlib

Code check functions live in `judge_stdlib.py` and have the signature `(response: str) -> bool`. They are called by name — the `check` field in the judge item is a plain function name (no `eval`/`exec`): the Experimenter looks up the name in the stdlib module's namespace. The pipeline can only extend the stdlib with new functions; existing functions are frozen once written to prevent drift.

Example functions: `skill_tool_called(response, skill_name)`, `no_implementation_before_skill(response)`.

### Evaluator

Scores the suite on two axes, producing a single `score ∈ [0,1]`:

```
discrimination_score = balanced_accuracy(sampled_queries)
                     = 0.5 * (trigger_recall + no_trigger_recall)
calibration_score    = judge_items_passing_pairwise_ranking / total_judge_items
iteration_score      = 0.5 * discrimination_score + 0.5 * calibration_score
```

Balanced accuracy guards against suites that are heavily skewed toward one `expected` polarity.

**Pairwise calibration**: for each judge item, for each `(seed_positive, seed_negative)` pair, check that `judge_score(positive) > judge_score(negative)`. For `"code"` items, scores are `1.0` (True) or `0.0` (False); a pair passes if the positive scores `1.0` and the negative scores `0.0`. A judge item passes calibration if it ranks all seed pairs correctly.

**Single-polarity seeds**: if seeds contain only positives or only negatives, pairwise ranking is undefined. In this case `calibration_score` falls back to per-item accuracy on the available seeds (how often the judge item gives the right answer for the known polarity).

**Tier A bootstrap**: when no seeds exist, the Experimenter generates one synthetic positive and one synthetic negative in the first iteration to seed calibration. These are flagged `synthetic: true` and can be replaced by real seeds later.

### Reviewer

`LLMReviewer` reads tasks, trajectories, and evaluations and produces insights such as:
- "Judge item j002 scores seed negatives too high — the rubric needs tightening"
- "No queries cover adversarial near-miss cases (requests that sound like the skill but shouldn't trigger it)"
- "Query q007 and q012 are near-duplicates — one should be replaced"

These insights are fed to the Researcher in the next iteration.

---

## Scoring & Stopping

**Rollback**: `rollback="best"` — the workspace preserves the `eval_suite.json` with the highest `iteration_score`. Because the pipeline produces exactly one `Hypothesis` per iteration, `best_this == avg_score` and the rollback logic behaves as expected.

**Stopping**: `Pipeline._should_stop` only checks `max_iterations`. Early stopping based on score plateau is handled by `EvalSuiteEvaluator`: when `iteration_score` has not improved by `> 0.01` for `convergence_patience` consecutive iterations (default: 3), it raises a `StopIteration` sentinel that the `EvalSuiteExperimenter` catches to gracefully exit the loop.

---

## Usage

```python
# run.py
from aura import Pipeline, Workspace, anthropic_llm
from aura.components import LLMResearcher, LLMReviewer
from aura.artifacts import FileArtifact
from skill_eval import EvalSuiteExperimenter, EvalSuiteEvaluator, ClaudeAPIExecutor

def main():
    llm = anthropic_llm()
    workspace = Workspace.create("./runs/aura-skill-eval")
    # FileArtifact takes a path; name is derived from filename ("eval_suite.json")
    artifact = FileArtifact("eval_suite.json")

    pipeline = Pipeline(
        researcher=LLMResearcher(
            llm=llm,
            prompt_template=RESEARCHER_PROMPT,
            artifact="eval_suite.json",  # must match artifact.name
        ),
        experimenter=EvalSuiteExperimenter(
            executor=ClaudeAPIExecutor(),
            sample_size=10,
        ),
        evaluator=EvalSuiteEvaluator(convergence_patience=3),
        reviewer=LLMReviewer(llm=llm, prompt_template=REVIEWER_PROMPT),
        workspace=workspace,
        artifacts=[artifact],
        rollback="best",
        max_iterations=10,
    )
    pipeline.run()
```

Or via AURA CLI:
```bash
aura run skill_eval/run.py --run-dir ./runs/aura-skill-eval
```

---

## File Layout

New code lives under `examples/skill-eval/` to keep it self-contained:

```
examples/skill-eval/
  run.py                    # pipeline entry point (main pattern)
  skill_eval/
    __init__.py
    experimenter.py         # EvalSuiteExperimenter
    evaluator.py            # EvalSuiteEvaluator (with convergence_patience)
    schema.py               # EvalSuite, Query, JudgeItem Pydantic models
    executors/
      __init__.py
      base.py               # SkillExecutor ABC + ExecutionResult (Pydantic)
      claude_api.py         # ClaudeAPIExecutor
      llm_predict.py        # LLMPredictExecutor
      mock.py               # MockExecutor
    judge_stdlib.py         # built-in code check functions (frozen interface)
  prompts/
    researcher.md           # Jinja2 prompt template
    reviewer.md             # Jinja2 prompt template
```

Inputs are placed in `<run-dir>/inputs/` (created by `Workspace.create()`):
```
<run-dir>/inputs/
  skill.md
  seeds.json          # optional
  existing_evals/     # optional
```

---

## Key Constraints

- Python >= 3.11, managed with `uv`
- Only new dependency: `anthropic` (for `ClaudeAPIExecutor`), already present in mock-autonas example
- `skill_eval/` is importable as a package; `run.py` uses the `main()` CLI pattern
- All prompt templates use Jinja2 with AURA's `render_prompt` utility
- `ExecutionResult` is a Pydantic `BaseModel` for serialization consistency with `Experiment.output`
- Judge code checks are looked up by name in `judge_stdlib` module namespace — no `eval`/`exec` of LLM-generated strings
