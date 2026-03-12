"""Autoresearch example designed for `aura run`.

Usage:
    aura validate examples/autoresearch-cli/run.py
    aura run examples/autoresearch-cli/run.py
"""
import json
from pathlib import Path

from aura import (
    Pipeline, Workspace, setup_logging,
    LLMResearcher, ScriptExperimenter, MetricEvaluator, LLMReviewer,
    anthropic_llm,
)

MOCK_TRAIN = str(Path(__file__).parent / "mock_train.py")


def main():
    setup_logging()

    llm = anthropic_llm(model="claude-sonnet-4-20250514")
    workspace = Workspace.create("./runs/autoresearch-cli")

    # Write experiment config as pipeline input
    (workspace.inputs_dir() / "config.json").write_text(json.dumps({
        "goal": "Maximize classification accuracy on the dataset",
        "search_space": {
            "lr": "[1e-5, 1e-1]",
            "epochs": "[5, 100]",
            "batch_size": [8, 16, 32, 64, 128],
        },
    }, indent=2))

    pipeline = Pipeline(
        researcher=LLMResearcher(
            llm=llm,
            prompt_template=(
                "You are a research experiment proposer.\n\n"
                "Research goal: {{ inputs }}\n\n"
                "Previous insights:\n{{ insights }}\n\n"
                "Propose {{ num_tasks }} experiments. Each must have: "
                "id, lr (float), epochs (int), batch_size (int), rationale.\n\n"
                "Respond as JSON list."
            ),
            num_tasks=3,
        ),
        experimenter=ScriptExperimenter(
            command_template=f"python {MOCK_TRAIN} --lr {{lr}} --epochs {{epochs}} --batch-size {{batch_size}}",
        ),
        evaluator=MetricEvaluator(metric="accuracy", baseline=0.5),
        reviewer=LLMReviewer(llm=llm),
        workspace=workspace,
        max_iterations=3,
    )

    pipeline.run()

    # Print results
    summary = workspace.summary()
    print(f"\nBest score: {summary['best_score']} (task: {summary['best_task_id']})")
    for it in summary["iterations"]:
        print(f"  Iteration {it['iteration']}: avg={it['avg_score']:.4f}, "
              f"max={it['max_score']:.4f}, tasks={it['num_tasks']}")


if __name__ == "__main__":
    main()
