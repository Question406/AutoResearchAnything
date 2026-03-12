import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aura.cli import detect_entry_pattern, run_file, validate_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, code: str, name: str = "experiment.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(code))
    return p


# ===========================================================================
# detect_entry_pattern
# ===========================================================================


class TestDetectEntryPattern:
    def test_main_pattern(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                pass
        """,
        )
        assert detect_entry_pattern(f) == "main"

    def test_runner_pattern(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            class MyRunner(Runner):
                pass
        """,
        )
        assert detect_entry_pattern(f) == "runner"

    def test_globals_pattern(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            workspace = create_workspace()
            pipeline = build_pipeline()
        """,
        )
        assert detect_entry_pattern(f) == "globals"

    def test_priority_main_over_runner(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                pass
            class MyRunner(Runner):
                pass
        """,
        )
        assert detect_entry_pattern(f) == "main"

    def test_priority_runner_before_main(self, tmp_path: Path):
        """main() takes priority even when Runner class appears first in file."""
        f = _write(
            tmp_path,
            """\
            class MyRunner(Runner):
                pass
            def main():
                pass
        """,
        )
        assert detect_entry_pattern(f) == "main"

    def test_no_pattern(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            x = 1
            print(x)
        """,
        )
        assert detect_entry_pattern(f) is None

    def test_syntax_error(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def broken(
        """,
        )
        assert detect_entry_pattern(f) is None

    def test_attribute_base_runner(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            class MyRunner(aura.Runner):
                pass
        """,
        )
        assert detect_entry_pattern(f) == "runner"


# ===========================================================================
# validate_file
# ===========================================================================


class TestValidateFile:
    def test_valid_main(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                pass
        """,
        )
        result = validate_file(f)
        assert result["valid"] is True
        assert result["entry_pattern"] == "main"
        assert result["errors"] == []

    def test_valid_runner(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            class Exp(Runner):
                pass
        """,
        )
        result = validate_file(f)
        assert result["valid"] is True
        assert result["entry_pattern"] == "runner"

    def test_valid_globals(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            workspace = 1
            pipeline = 2
        """,
        )
        result = validate_file(f)
        assert result["valid"] is True
        assert result["entry_pattern"] == "globals"

    def test_syntax_error(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def broken(
        """,
        )
        result = validate_file(f)
        assert result["valid"] is False
        assert any("Syntax error" in e for e in result["errors"])

    def test_no_pattern(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            x = 1
        """,
        )
        result = validate_file(f)
        assert result["valid"] is False
        assert result["entry_pattern"] is None
        assert any("entry pattern" in e.lower() for e in result["errors"])

    def test_nonexistent_file(self, tmp_path: Path):
        f = tmp_path / "missing.py"
        result = validate_file(f)
        assert result["valid"] is False
        assert any("not found" in e.lower() for e in result["errors"])


# ===========================================================================
# run_file
# ===========================================================================


class TestRunFile:
    def test_main_pattern(self, tmp_path: Path):
        marker = tmp_path / "marker.txt"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            def main():
                Path("{str(marker)}").write_text("executed")
        """,
        )
        run_file(f)
        assert marker.read_text() == "executed"

    def test_runner_pattern(self, tmp_path: Path):
        marker = tmp_path / "marker.txt"
        run_dir = tmp_path / "my_run"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            from aura.runner import Runner
            from aura.pipeline import Pipeline
            from aura.interfaces import Researcher, Experimenter, Evaluator, Reviewer
            from aura.types import Hypothesis, Experiment, Evaluation, Insight

            class StubGen(Researcher):
                def hypothesize(self, insights, workspace):
                    return [Hypothesis(id="t0", spec={{}})]

            class StubExec(Experimenter):
                def run_experiment(self, task, workspace):
                    return Experiment(task_id=task.id, status="completed", steps=[], output=None)

            class StubEval(Evaluator):
                def evaluate(self, task, trajectory, workspace):
                    return Evaluation(task_id=task.id, score=1.0, passed=True)

            class StubExtract(Reviewer):
                def review(self, tasks, trajectories, evaluations, workspace):
                    return []

            class MyRunner(Runner):
                def setup_inputs(self, workspace):
                    Path("{str(marker)}").write_text("runner_ran")

                def build_pipeline(self, workspace):
                    return Pipeline(
                        researcher=StubGen(),
                        experimenter=StubExec(),
                        evaluator=StubEval(),
                        reviewer=StubExtract(),
                        workspace=workspace,
                        max_iterations=1,
                    )
        """,
        )
        run_file(f, run_dir=run_dir)
        assert marker.read_text() == "runner_ran"
        assert run_dir.exists()

    def test_globals_pattern(self, tmp_path: Path):
        marker = tmp_path / "marker.txt"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            workspace = "ws"
            pipeline = "pipe"
            Path("{str(marker)}").write_text("globals_ran")
        """,
        )
        run_file(f)
        assert marker.read_text() == "globals_ran"

    def test_exception_in_main(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                raise ValueError("boom")
        """,
        )
        with pytest.raises(ValueError, match="boom"):
            run_file(f)

    def test_no_pattern_exits(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            x = 1
        """,
        )
        with pytest.raises(SystemExit):
            run_file(f)

    def test_run_dir_passthrough(self, tmp_path: Path):
        """Ensure --run-dir reaches the Runner."""
        marker = tmp_path / "marker.txt"
        run_dir = tmp_path / "custom_run"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            from aura.runner import Runner
            from aura.pipeline import Pipeline
            from aura.interfaces import Researcher, Experimenter, Evaluator, Reviewer
            from aura.types import Hypothesis, Experiment, Evaluation, Insight

            class StubGen(Researcher):
                def hypothesize(self, insights, workspace):
                    return [Hypothesis(id="t0", spec={{}})]

            class StubExec(Experimenter):
                def run_experiment(self, task, workspace):
                    return Experiment(task_id=task.id, status="completed", steps=[], output=None)

            class StubEval(Evaluator):
                def evaluate(self, task, trajectory, workspace):
                    return Evaluation(task_id=task.id, score=1.0, passed=True)

            class StubExtract(Reviewer):
                def review(self, tasks, trajectories, evaluations, workspace):
                    return []

            class DirRunner(Runner):
                def setup_inputs(self, workspace):
                    Path("{str(marker)}").write_text(str(workspace.root))

                def build_pipeline(self, workspace):
                    return Pipeline(
                        researcher=StubGen(),
                        experimenter=StubExec(),
                        evaluator=StubEval(),
                        reviewer=StubExtract(),
                        workspace=workspace,
                        max_iterations=1,
                    )
        """,
        )
        run_file(f, run_dir=run_dir)
        assert marker.read_text() == str(run_dir)


# ===========================================================================
# CLI integration (subprocess)
# ===========================================================================


class TestCLIIntegration:
    def test_validate_json_stdout(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                pass
        """,
        )
        result = subprocess.run(
            [sys.executable, "-m", "aura", "validate", str(f)],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        assert data["valid"] is True
        assert data["entry_pattern"] == "main"

    def test_validate_invalid_file(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            x = 1
        """,
        )
        result = subprocess.run(
            [sys.executable, "-m", "aura", "validate", str(f)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert data["valid"] is False

    def test_run_main(self, tmp_path: Path):
        marker = tmp_path / "marker.txt"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            def main():
                Path("{str(marker)}").write_text("cli_ran")
        """,
        )
        result = subprocess.run(
            [sys.executable, "-m", "aura", "run", str(f)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert marker.read_text() == "cli_ran"

    def test_run_failure_exits_nonzero(self, tmp_path: Path):
        f = _write(
            tmp_path,
            """\
            def main():
                raise RuntimeError("fail")
        """,
        )
        result = subprocess.run(
            [sys.executable, "-m", "aura", "run", str(f)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "RuntimeError" in result.stderr

    def test_validate_no_side_effects(self, tmp_path: Path):
        marker = tmp_path / "side_effect.txt"
        f = _write(
            tmp_path,
            f"""\
            from pathlib import Path
            Path("{str(marker)}").write_text("should not exist")
            def main():
                pass
        """,
        )
        subprocess.run(
            [sys.executable, "-m", "aura", "validate", str(f)],
            capture_output=True,
            text=True,
        )
        assert not marker.exists(), "validate should not execute the file"
