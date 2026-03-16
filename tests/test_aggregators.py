"""Tests for Aggregator implementations."""

from aura.components.aggregators import AllTrialsAggregator, BestTrialAggregator, LastTrialAggregator
from aura.types import Trial


def _trial(id: str, output, status: str = "completed") -> Trial:
    return Trial(id=id, spec={}, status=status, steps=[], output=output)


# --- LastTrialAggregator ---


def test_last_trial_empty():
    assert LastTrialAggregator().aggregate([]) is None


def test_last_trial_single():
    t = _trial("t1", {"acc": 0.9})
    assert LastTrialAggregator().aggregate([t]) == {"acc": 0.9}


def test_last_trial_multiple():
    trials = [_trial("t1", {"acc": 0.7}), _trial("t2", {"acc": 0.9})]
    assert LastTrialAggregator().aggregate(trials) == {"acc": 0.9}


# --- BestTrialAggregator ---


def test_best_trial_higher_is_better():
    trials = [
        _trial("t1", {"val_acc": 0.7}),
        _trial("t2", {"val_acc": 0.9}),
        _trial("t3", {"val_acc": 0.8}),
    ]
    agg = BestTrialAggregator(metric="val_acc", higher_is_better=True)
    assert agg.aggregate(trials) == {"val_acc": 0.9}


def test_best_trial_lower_is_better():
    trials = [
        _trial("t1", {"loss": 0.5}),
        _trial("t2", {"loss": 0.2}),
        _trial("t3", {"loss": 0.8}),
    ]
    agg = BestTrialAggregator(metric="loss", higher_is_better=False)
    assert agg.aggregate(trials) == {"loss": 0.2}


def test_best_trial_skips_failed():
    trials = [
        _trial("t1", {"val_acc": 0.9}),
        _trial("t2", None, status="failed"),
        _trial("t3", {"val_acc": 0.7}),
    ]
    agg = BestTrialAggregator(metric="val_acc")
    result = agg.aggregate(trials)
    assert result == {"val_acc": 0.9}


def test_best_trial_empty():
    assert BestTrialAggregator(metric="acc").aggregate([]) is None


def test_best_trial_all_failed_returns_last():
    trials = [_trial("t1", None, status="failed"), _trial("t2", None, status="failed")]
    result = BestTrialAggregator(metric="acc").aggregate(trials)
    assert result is None  # last trial's output


# --- AllTrialsAggregator ---


def test_all_trials_empty():
    assert AllTrialsAggregator().aggregate([]) == []


def test_all_trials_multiple():
    trials = [_trial("t1", {"acc": 0.7}), _trial("t2", {"acc": 0.9})]
    result = AllTrialsAggregator().aggregate(trials)
    assert result == [{"acc": 0.7}, {"acc": 0.9}]
