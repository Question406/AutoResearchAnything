"""Tests for executor and collector backends."""

import json
from pathlib import Path

import pytest

from aura.components.backends.collector_backends import (
    JSONFileCollector,
    LogParserCollector,
    StdoutCollector,
)
from aura.components.backends.executor_backends import FunctionExecutor, ScriptExecutor
from aura.types import Hypothesis
from aura.workspace import Workspace


def _task(spec: dict | None = None) -> Hypothesis:
    return Hypothesis(id="t1", spec=spec or {})


# --- ScriptExecutor ---


def test_script_executor_success(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task({"value": 7})

    script = tmp_path / "echo.py"
    script.write_text('import json, sys; print(json.dumps({"v": int(sys.argv[1])}))')

    executor = ScriptExecutor(f"python {script} {{value}}")
    raw = executor.run(task, context={}, workspace=ws)

    assert raw["returncode"] == 0
    assert '"v": 7' in raw["stdout"]


def test_script_executor_failure(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    executor = ScriptExecutor("python -c 'import sys; sys.exit(2)'")
    raw = executor.run(task, context={}, workspace=ws)

    assert raw["returncode"] == 2


# --- FunctionExecutor ---


def test_function_executor_success(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task({"x": 5, "y": 3})

    executor = FunctionExecutor(lambda x, y, **kw: {"sum": x + y})
    raw = executor.run(task, context={}, workspace=ws)

    assert raw["output"] == {"sum": 8}
    assert raw["error"] is None


def test_function_executor_timeout(tmp_path: Path):
    import time

    ws = Workspace.create(tmp_path / "run")
    task = _task()

    executor = FunctionExecutor(lambda **kw: time.sleep(10), timeout=1)
    raw = executor.run(task, context={}, workspace=ws)

    assert raw["error"] is not None
    assert "Timed out" in raw["error"]


# --- StdoutCollector ---


def test_stdout_collector_success(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task({"lr": 0.01})

    raw = {
        "cmd": "python train.py",
        "stdout": json.dumps({"accuracy": 0.95}),
        "stderr": "",
        "returncode": 0,
        "ts_start": "2026-01-01T00:00:00Z",
        "ts_end": "2026-01-01T00:00:01Z",
    }

    trial = StdoutCollector(parse_json=True).collect(task, raw, {}, ws)

    assert trial.status == "completed"
    assert trial.output == {"accuracy": 0.95}
    assert len(trial.steps) == 2


def test_stdout_collector_failure(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    raw = {
        "cmd": "bad_cmd",
        "stdout": "",
        "stderr": "command not found",
        "returncode": 127,
        "ts_start": "t",
        "ts_end": "t",
    }

    trial = StdoutCollector().collect(task, raw, {}, ws)

    assert trial.status == "failed"
    assert "command not found" in trial.error


def test_stdout_collector_no_json_parse(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    raw = {
        "cmd": "echo hello",
        "stdout": "hello world",
        "stderr": "",
        "returncode": 0,
        "ts_start": "t",
        "ts_end": "t",
    }

    trial = StdoutCollector(parse_json=False).collect(task, raw, {}, ws)

    assert trial.status == "completed"
    assert trial.output == "hello world"


# --- JSONFileCollector ---


def test_json_file_collector_success(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    output_file = tmp_path / "results" / "t1.json"
    output_file.parent.mkdir()
    output_file.write_text(json.dumps({"loss": 0.3}))

    collector = JSONFileCollector(str(tmp_path / "results" / "{task_id}.json"))
    trial = collector.collect(task, {}, {}, ws)

    assert trial.status == "completed"
    assert trial.output == {"loss": 0.3}


def test_json_file_collector_missing_file(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    collector = JSONFileCollector(str(tmp_path / "missing" / "{task_id}.json"))
    trial = collector.collect(task, {}, {}, ws)

    assert trial.status == "failed"
    assert "not found" in trial.error


# --- LogParserCollector ---


def test_log_parser_collector_extracts_metric(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    raw = {
        "stdout": "Epoch 10/10 - val_acc=0.923 - loss=0.21",
        "stderr": "",
        "returncode": 0,
    }

    collector = LogParserCollector(
        {
            "val_acc": r"val_acc=(?P<value>[\d.]+)",
            "loss": r"loss=(?P<value>[\d.]+)",
        }
    )
    trial = collector.collect(task, raw, {}, ws)

    assert trial.status == "completed"
    assert trial.output["val_acc"] == pytest.approx(0.923)
    assert trial.output["loss"] == pytest.approx(0.21)


def test_log_parser_collector_no_match(tmp_path: Path):
    ws = Workspace.create(tmp_path / "run")
    task = _task()

    raw = {"stdout": "training done", "stderr": "", "returncode": 0}
    collector = LogParserCollector({"acc": r"acc=(?P<value>[\d.]+)"})
    trial = collector.collect(task, raw, {}, ws)

    # No match — output is empty dict, status still completed
    assert trial.status == "completed"
    assert trial.output == {}
