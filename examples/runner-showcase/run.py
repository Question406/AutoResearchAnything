"""Showcase: Runner abstraction with different backends.

This example demonstrates the three ways to provide execution backends to AURA
stage ABCs, all producing identical pipeline behavior:

1. LLMCallable (auto-wrapped into LLMRunner)
2. FunctionRunner (custom Python logic)
3. Custom Runner subclass

Run with:
    python examples/runner-showcase/run.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from aura import (
    Evaluation,
    Evaluator,
    Experiment,
    FunctionExperimenter,
    Hypothesis,
    Insight,
    Pipeline,
    Researcher,
    Reviewer,
    Workspace,
    setup_logging,
)
from aura.components.runners import FunctionRunner
from aura.interfaces import Runner


# =============================================================================
# Mock functions (no real LLM needed)
# =============================================================================

def mock_researcher_llm(prompt: str) -> str:
    """Simulate an LLM that proposes hyperparameter experiments."""
    return json.dumps([
        {"id": "exp_001", "lr": 0.001, "epochs": 20, "batch_size": 32},
        {"id": "exp_002", "lr": 0.0003, "epochs": 50, "batch_size": 16},
    ])


def mock_evaluator_fn(prompt: str, context: dict) -> dict:
    """Simulate an evaluator that returns structured output directly."""
    task_spec = context.get("task", {})
    # Score based on how close lr is to the "optimal" 0.0003
    lr = task_spec.get("lr", 0.001) if isinstance(task_spec, dict) else 0.001
    quality = math.exp(-abs(math.log10(lr) - math.log10(0.0003)))
    return {
        "content": "evaluation complete",
        "structured": {
            "score": round(quality, 3),
            "passed": quality > 0.5,
            "reason": f"lr={lr} quality={quality:.3f}",
        },
    }


class EchoReviewerRunner(Runner):
    """Custom Runner that generates insights without calling any external service."""

    def run(self, prompt_template: str, context: dict) -> dict:
        iteration = context.get("iteration", 0)
        return {
            "content": json.dumps([
                {
                    "finding": f"Iteration {iteration} complete",
                    "recommendation": "Continue exploring lower learning rates",
                },
            ]),
        }


def train_fn(lr: float, epochs: int, batch_size: int = 32, **kwargs) -> dict:
    """Mock training function."""
    lr_factor = math.exp(-((math.log10(lr) + 3.5) ** 2) / 0.5)
    epoch_factor = 1 - math.exp(-epochs / 20)
    accuracy = 0.5 + 0.45 * lr_factor * epoch_factor
    return {"accuracy": round(accuracy, 4), "loss": round(1.0 - accuracy + 0.05, 4)}


# =============================================================================
# Pipeline wiring
# =============================================================================

def main():
    setup_logging()
    workspace = Workspace.create(Path("./runs/runner-showcase"))

    (workspace.inputs_dir() / "goal.txt").write_text(
        "Maximize classification accuracy by tuning lr, epochs, batch_size."
    )

    pipeline = Pipeline(
        # --- Approach 1: LLMCallable passed directly (auto-wrapped into LLMRunner) ---
        researcher=Researcher(
            runner=mock_researcher_llm,
            prompt_template=(
                "Research goal: {{ inputs }}\n"
                "Insights: {{ insights }}\n"
                "Propose experiments as JSON list with id, lr, epochs, batch_size."
            ),
        ),

        # --- Experimenter uses FunctionExperimenter (unchanged, no Runner needed) ---
        experimenter=FunctionExperimenter(fn=train_fn),

        # --- Approach 2: FunctionRunner for structured output bypass ---
        evaluator=Evaluator(runner=FunctionRunner(mock_evaluator_fn)),

        # --- Approach 3: Custom Runner subclass ---
        reviewer=Reviewer(runner=EchoReviewerRunner()),

        workspace=workspace,
        max_iterations=2,
    )

    pipeline.run()

    # Print results
    summary = workspace.summary()
    print(f"\nBest score: {summary['best_score']} (task: {summary['best_task_id']})")
    for it in summary["iterations"]:
        print(
            f"  Iteration {it['iteration']}: avg={it['avg_score']:.4f}, "
            f"max={it['max_score']:.4f}, tasks={it['num_tasks']}"
        )


if __name__ == "__main__":
    main()
