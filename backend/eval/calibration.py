"""Калибровка порога similarity для проверки глоссария в Validator'е.

Порог 0.7 взят из контракта как стартовая гипотеза. Калибровка отвечает на
вопрос "верен ли он": строим распределения косинусной близости для пар
"термин использован в решении" и "термин не использован" и ищем границу
с минимальной суммой ошибок (ближайшую к идеальному разделителю).
"""
from typing import Any


def calibrate_threshold(
    used_scores: list[float], unused_scores: list[float], current: float = 0.7
) -> dict[str, Any]:
    """Подбор порога по двум распределениям сходств.

    Порог-кандидаты - середины между отсортированными значениями обеих
    выборок; лучший минимизирует долю ошибок классификации.
    """
    if not used_scores or not unused_scores:
        raise ValueError("Нужны примеры обоих классов (used и unused)")

    def errors(threshold: float) -> int:
        false_negative = sum(1 for s in used_scores if s < threshold)
        false_positive = sum(1 for s in unused_scores if s >= threshold)
        return false_negative + false_positive

    candidates = sorted(set(used_scores + unused_scores))
    # strict=False осознанно: пары соседних кандидатов, последний остаётся без пары
    boundaries = [candidates[0] - 0.01] + [
        (a + b) / 2 for a, b in zip(candidates, candidates[1:], strict=False)
    ] + [candidates[-1] + 0.01]
    best = min(boundaries, key=errors)

    total = len(used_scores) + len(unused_scores)
    return {
        "suggested_threshold": round(best, 4),
        "current_threshold": current,
        "current_error_rate": round(errors(current) / total, 4),
        "best_error_rate": round(errors(best) / total, 4),
        "min_used": round(min(used_scores), 4),
        "max_unused": round(max(unused_scores), 4),
        "n": total,
    }
