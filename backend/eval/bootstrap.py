"""Paired bootstrap: значимость разницы между двумя режимами.

Парный - потому что режимы прогоняются на ОДНОМ наборе ситуаций:
перебутсимулируем ситуации (строки), а не отдельные оценки.
"""
import random


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_iterations: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Доверительный интервал и p-value для mean(scores_a - scores_b).

    scores_a/scores_b - метрики (например critic_score) двух режимов
    на одном наборе ситуаций, len(a) == len(b).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired bootstrap требует одинаковое число наблюдений")
    if not scores_a:
        raise ValueError("Нет данных для bootstrap")

    pairs = list(zip(scores_a, scores_b, strict=True))
    diffs = [a - b for a, b in pairs]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    boot_means: list[float] = []
    for _ in range(n_iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo = boot_means[int((alpha / 2) * n_iterations)]
    hi = boot_means[min(n_iterations - 1, int((1 - alpha / 2) * n_iterations))]
    # p-value: как часто разница в бутстрэпе оказывается ещё сильнее против H0
    p_value = (
        2 * min(
            sum(1 for m in boot_means if m <= 0) / n_iterations,
            sum(1 for m in boot_means if m >= 0) / n_iterations,
        )
    )

    return {
        "mean_diff": round(observed, 4),
        "ci95": (round(lo, 4), round(hi, 4)),
        "p_value": round(min(p_value, 1.0), 4),
        "significant": not (lo <= 0 <= hi),
        "n": n,
        "n_iterations": n_iterations,
    }
