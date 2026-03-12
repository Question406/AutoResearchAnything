"""Autoresearch example — minimal version using aura built-in components."""
import json
from pathlib import Path

from aura import (
    Pipeline, Workspace, setup_logging,
    LLMResearcher, ScriptExperimenter, MetricEvaluator, LLMReviewer,
    anthropic_llm,
)

setup_logging()

# --- Config ---
RESEARCH_GOAL = "Maximize classification accuracy on the dataset"
SEARCH_SPACE = {
    "lr": "[1e-5, 1e-1]",
    "epochs": "[5, 100]",
    "batch_size": [8, 16, 32, 64, 128],
}
MOCK_TRAIN = str(Path(__file__).parent / "mock_train.py")

# --- LLM ---
llm = anthropic_llm(model="claude-sonnet-4-20250514")

# --- Workspace ---
workspace = Workspace.create("./runs/autoresearch")
(workspace.inputs_dir() / "config.json").write_text(json.dumps({
    "goal": RESEARCH_GOAL,
    "search_space": SEARCH_SPACE,
}, indent=2))

# --- Pipeline ---
pipeline = Pipeline(
    researcher=LLMResearcher(
        llm=llm,
        prompt_template=(
            "You are a research experiment proposer.\n\n"
            "Research goal: {{ inputs }}\n\n"
            "Previous insights:\n{{ insights }}\n\n"
            "Propose {{ num_tasks }} experiments. Each must have: id, lr (float), epochs (int), batch_size (int), rationale.\n\n"
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

# --- Results ---
summary = workspace.summary()
print(f"\nBest score: {summary['best_score']} (task: {summary['best_task_id']})")
for it in summary["iterations"]:
    print(f"  Iteration {it['iteration']}: avg={it['avg_score']:.4f}, max={it['max_score']:.4f}, tasks={it['num_tasks']}")
