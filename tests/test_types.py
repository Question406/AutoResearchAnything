from aura.types import Evaluation, Experiment, ExperimentStep, Hypothesis, Insight


def test_hypothesis_construction():
    t = Hypothesis(id="task_001", spec={"prompt": "hello"}, metadata={})
    assert t.id == "task_001"
    assert t.spec == {"prompt": "hello"}
    assert t.metadata == {}


def test_experiment_step_construction():
    s = ExperimentStep(step=0, data={"action": "think"}, timestamp="2026-03-11T00:00:00Z")
    assert s.step == 0
    assert s.data == {"action": "think"}


def test_experiment_construction_completed():
    t = Experiment(
        task_id="task_001",
        status="completed",
        steps=[],
        output={"result": 42},
        error=None,
        metadata={"duration_ms": 100},
    )
    assert t.status == "completed"
    assert t.output == {"result": 42}
    assert t.error is None


def test_experiment_construction_failed():
    t = Experiment(
        task_id="task_001",
        status="failed",
        steps=[],
        output=None,
        error="RuntimeError: boom",
        metadata={},
    )
    assert t.status == "failed"
    assert t.error == "RuntimeError: boom"


def test_evaluation_construction():
    e = Evaluation(
        task_id="task_001", score=0.85, passed=True, details={"reason": "good"}, metadata={}
    )
    assert 0.0 <= e.score <= 1.0
    assert e.passed is True


def test_insight_construction():
    i = Insight(
        id="insight_001",
        source_iteration=1,
        content={"finding": "skill A works well"},
        metadata={},
    )
    assert i.id == "insight_001"
    assert i.source_iteration == 1


def test_hypothesis_roundtrip():
    original = Hypothesis(id="t1", spec={"key": "value"}, metadata={"tag": "test"})
    restored = Hypothesis.model_validate_json(original.model_dump_json())
    assert original == restored


def test_experiment_roundtrip():
    steps = [
        ExperimentStep(step=0, data={"action": "think"}, timestamp="2026-03-11T00:00:00Z"),
        ExperimentStep(step=1, data={"action": "act"}, timestamp="2026-03-11T00:00:01Z"),
    ]
    original = Experiment(
        task_id="t1",
        status="completed",
        steps=steps,
        output={"answer": 42},
        error=None,
        metadata={"duration_ms": 200},
    )
    restored = Experiment.model_validate_json(original.model_dump_json())
    assert original == restored


def test_experiment_roundtrip_failed():
    original = Experiment(
        task_id="t1", status="failed", steps=[], output=None, error="boom", metadata={}
    )
    restored = Experiment.model_validate_json(original.model_dump_json())
    assert original == restored


def test_evaluation_roundtrip():
    original = Evaluation(
        task_id="t1", score=0.75, passed=True, details={"reason": "ok"}, metadata={}
    )
    restored = Evaluation.model_validate_json(original.model_dump_json())
    assert original == restored


def test_insights_roundtrip():
    import json

    originals = [
        Insight(id="i1", source_iteration=1, content={"finding": "good"}, metadata={}),
        Insight(id="i2", source_iteration=1, content={"finding": "bad"}, metadata={}),
    ]
    json_str = json.dumps([i.model_dump() for i in originals])
    restored = [Insight.model_validate(d) for d in json.loads(json_str)]
    assert originals == restored


def test_roundtrip_empty_dicts():
    original = Hypothesis(id="t1", spec={}, metadata={})
    assert Hypothesis.model_validate_json(original.model_dump_json()) == original


def test_roundtrip_none_output():
    original = Experiment(task_id="t1", status="completed", steps=[], output=None, metadata={})
    assert Experiment.model_validate_json(original.model_dump_json()) == original


def test_roundtrip_unicode():
    original = Hypothesis(id="t1", spec={"text": "hello \u4e16\u754c"}, metadata={})
    assert Hypothesis.model_validate_json(original.model_dump_json()) == original
