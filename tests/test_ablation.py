"""Ablation-агрегация: summarize, bootstrap против baseline (чистые функции)."""
import pytest

from backend.core.state import StateKeys
from backend.eval.ablation import bootstrap_against_baseline, summarize


def _state(score=None, validation=True, stopped="success", iterations=0):
    state = {
        StateKeys.CRITIC_SCORE: score,
        StateKeys.VALIDATION_PASSED: validation,
        StateKeys.SUPERVISOR_STOPPED_REASON: stopped,
        StateKeys.ITERATIONS: iterations,
    }
    return state


def test_summarize_basic_metrics():
    runs = [
        _state(score=8.0, validation=True, stopped="success", iterations=1),
        _state(score=None, validation=True, stopped="budget_exhausted", iterations=2),
        None,  # упавший прогон
    ]
    summary = summarize(runs)
    assert summary["n"] == 3
    assert summary["errors"] == 1
    assert summary["avg_score"] == 8.0
    assert summary["score_none_count"] == 1
    assert summary["validation_rate"] == 0.67  # round(2/3, 2)
    assert summary["success_rate"] == 0.33
    assert summary["avg_iterations"] == 1.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="Нет прогонов"):
        summarize([])


def test_summarize_all_scores_none():
    summary = summarize([_state(score=None), _state(score=None)])
    assert summary["avg_score"] is None
    assert summary["score_none_count"] == 2


def test_bootstrap_against_baseline_pairs_common_scores():
    runs_by_mode = {
        "baseline": [_state(score=5.0), _state(score=5.5), _state(score=6.0)],
        "full": [_state(score=8.0), _state(score=None), _state(score=7.5)],
        # agents_only нет в прогонах - должен быть пропущен
    }
    reports = bootstrap_against_baseline(runs_by_mode, n_iterations=500)
    names = [name for name, _ in reports]
    assert "full vs baseline" in names
    # пары - позиции, где у ОБОИХ режимов оценка не None: 0 и 2 (позиция 1 - full None)
    report = dict(reports)["full vs baseline"]
    assert report["n_pairs"] == 2


def test_bootstrap_skips_mode_with_too_few_pairs():
    runs_by_mode = {
        "baseline": [_state(score=5.0), _state(score=5.5)],
        "glossary": [_state(score=None), _state(score=None)],
    }
    assert bootstrap_against_baseline(runs_by_mode, n_iterations=100) == []


def test_bootstrap_without_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        bootstrap_against_baseline({"full": [_state(score=8.0)]})
