"""Метрики качества retrieval: Recall@k и MRR.

Зачем: на защите вопрос "почему RRF?" должен опираться на цифры, а не на
аргументы. Эксперимент прогоняет одинаковые запросы через три канала
(BM25 / векторный / гибридный RRF) и сравнивает метрики на парах
запрос -> эталонный кейс (см. scripts/dataset_generator.build_retrieval_pairs).
"""
from typing import Any


def recall_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> bool:
    """Эталон попал в первые k результатов."""
    return relevant_id in ranked_ids[:k]


def mrr(ranked_ids: list[str], relevant_id: str) -> float:
    """Mean Reciprocal Rank для одного запроса: 1/позиция эталона, 0 если не найден."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == relevant_id:
            return 1.0 / position
    return 0.0


def evaluate_channel(
    rankings: list[list[str]], relevant_ids: list[str], k_values: tuple[int, ...] = (1, 5)
) -> dict[str, Any]:
    """Метрики канала на наборе запросов: recall@k и MRR, округлённые до 4 знаков."""
    if len(rankings) != len(relevant_ids):
        raise ValueError("Число ранжирований должно совпадать с числом эталонов")
    if not rankings:
        raise ValueError("Нет данных для оценки")

    result: dict[str, Any] = {}
    for k in k_values:
        hits = sum(
            recall_at_k(ranked, relevant, k)
            for ranked, relevant in zip(rankings, relevant_ids, strict=True)
        )
        result[f"recall@{k}"] = round(hits / len(rankings), 4)
    result["mrr"] = round(
        sum(mrr(ranked, relevant) for ranked, relevant in zip(rankings, relevant_ids, strict=True))
        / len(rankings),
        4,
    )
    result["n_queries"] = len(rankings)
    return result
