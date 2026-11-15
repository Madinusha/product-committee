"""Метрики retrieval и калибровка порога (чистые функции - без torch)."""
import pytest

from backend.eval.calibration import calibrate_threshold
from backend.eval.retrieval_metrics import evaluate_channel, mrr, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], "b", 2) is True
    assert recall_at_k(["a", "b", "c"], "c", 2) is False
    assert recall_at_k(["a"], "a", 1) is True


def test_mrr_positions():
    assert mrr(["x", "y", "z"], "y") == 0.5
    assert mrr(["x", "y", "z"], "x") == 1.0
    assert mrr(["x", "y"], "z") == 0.0


def test_evaluate_channel_aggregates():
    rankings = [["a", "b"], ["b", "a"], ["c", "b"]]
    relevant = ["a", "a", "b"]
    result = evaluate_channel(rankings, relevant, k_values=(1,))
    # recall@1: 'a' первым только в 1-м запросе; mrr: 1 + 0.5 + 0.5 = 2/3
    assert result["recall@1"] == round(1 / 3, 4)
    assert result["mrr"] == round(2 / 3, 4)
    assert result["n_queries"] == 3


def test_evaluate_channel_validates_lengths():
    with pytest.raises(ValueError):
        evaluate_channel([["a"]], ["a", "b"])


def test_calibration_perfect_separation():
    used = [0.9, 0.85, 0.8]
    unused = [0.3, 0.2, 0.1]
    report = calibrate_threshold(used, unused)
    assert report["best_error_rate"] == 0.0
    assert 0.3 <= report["suggested_threshold"] <= 0.8


def test_calibration_overlapping_distributions():
    used = [0.7, 0.5, 0.3]
    unused = [0.6, 0.4, 0.2]
    report = calibrate_threshold(used, unused, current=0.7)
    assert 0 < report["best_error_rate"] < 1
    assert report["current_error_rate"] >= report["best_error_rate"]


def test_calibration_requires_both_classes():
    with pytest.raises(ValueError, match="обоих классов"):
        calibrate_threshold([0.9], [])
