"""Логика ablation-эксперимента: прогон ситуаций через режимы и агрегация.

Чистые функции без LLM - сам прогон делает scripts/run_ablation.py.
Результат - таблицы для раздела 3 отчёта и paired bootstrap для раздела 4.
"""
from typing import Any

from backend.core.constants import EVAL_MODES
from backend.core.state import StateKeys
from backend.eval.bootstrap import paired_bootstrap


def summarize(runs: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Агрегация прогонов одного режима: score, валидация, успех, итерации.

    runs - state'ы pipeline (None = прогон упал). Оценки None (Critic не смог)
    исключаются из среднего, но учитываются в score_none_count.
    """
    if not runs:
        raise ValueError("Нет прогонов для агрегации")
    valid = [r for r in runs if r is not None]
    scores = [
        r[StateKeys.CRITIC_SCORE]
        for r in valid
        if r.get(StateKeys.CRITIC_SCORE) is not None
    ]
    return {
        "n": len(runs),
        "errors": len(runs) - len(valid),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "score_none_count": len(valid) - len(scores),
        "validation_rate": round(
            sum(1 for r in valid if r.get(StateKeys.VALIDATION_PASSED)) / len(runs), 2
        ),
        "success_rate": round(
            sum(
                1
                for r in valid
                if str(r.get(StateKeys.SUPERVISOR_STOPPED_REASON)) == "success"
            )
            / len(runs),
            2,
        ),
        "avg_iterations": round(
            sum(r.get(StateKeys.ITERATIONS) or 0 for r in valid) / len(runs), 2
        ),
    }


def bootstrap_against_baseline(
    runs_by_mode: dict[str, list[dict[str, Any] | None]],
    n_iterations: int = 10_000,
    seed: int = 42,
) -> list[tuple[str, dict[str, Any]]]:
    """Paired bootstrap каждого режима против baseline.

    Парный подход: сравниваем оценки на ОДНИХ И ТЕХ ЖЕ ситуациях (перебутсимулируем
    ситуации, а не оценки). Пары, где у любого режима оценка None, исключаются.
    """
    baseline = runs_by_mode.get("baseline")
    if baseline is None:
        raise ValueError("baseline должен быть среди прогнанных режимов")

    reports = []
    for mode in EVAL_MODES:
        if mode == "baseline" or mode not in runs_by_mode:
            continue
        pairs = [
            (r_mode[StateKeys.CRITIC_SCORE], r_base[StateKeys.CRITIC_SCORE])
            for r_mode, r_base in zip(runs_by_mode[mode], baseline, strict=True)
            if r_mode is not None
            and r_base is not None
            and r_mode.get(StateKeys.CRITIC_SCORE) is not None
            and r_base.get(StateKeys.CRITIC_SCORE) is not None
        ]
        if len(pairs) < 2:
            continue  # мало общих оценок - bootstrap недостоверен
        scores_mode, scores_base = zip(*pairs, strict=True)
        report = paired_bootstrap(
            list(scores_mode), list(scores_base), n_iterations=n_iterations, seed=seed
        )
        report["n_pairs"] = len(pairs)
        reports.append((f"{mode} vs baseline", report))
    return reports
