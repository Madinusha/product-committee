"""RRF - слияние ранжированных списков (BM25 + векторный поиск).

Почему RRF, а не взвешенная сумма метрик: BM25-скоры и cosine лежат в
несравнимых шкалах (0..10+ и 0..1), калибровать их нестабильно. RRF
работает только с рангами, поэтому шкалы не важны. k=60 гасит вклад
верхних позиций, чтобы один список не доминировал.
"""

from backend.core.config import get_settings


def rrf_merge(
    rankings: list[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    """Слить несколько ранжированных списков id с дедупликацией.

    score(d) = sum_i 1 / (k + rank_i(d)), где rank начинается с 1;
    документ, отсутствующий в списке, вклада не даёт.
    Возвращает [(doc_id, rrf_score)] по убыванию.
    """
    if k is None:
        k = get_settings().RRF_K
    # без k=60 первый канал почти всегда доминирует: 1/(1+1) против
    # 1/(k+1) у второго списка - при малом k разница в рангах слишком резкая
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: -pair[1])
