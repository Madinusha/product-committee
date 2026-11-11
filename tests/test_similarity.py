"""Similarity-хук: substring-путь и работа без эмбеддинг-модели (dev-среда без torch)."""
from backend.retrieval.similarity import term_similarity


def test_substring_gives_full_score():
    assert term_similarity("Снизить CAC на 15% за квартал", "CAC") == 1.0


def test_missing_term_without_embeddings_is_zero():
    # в venv без torch sentence-transformers не загрузится -> fallback 0.0
    assert term_similarity("Снизить отток на 3 п.п.", "Churn Rate") == 0.0


def test_whitespace_in_term_is_normalized():
    assert term_similarity("используем RICE для приоритизации", "  rice  ") == 1.0
