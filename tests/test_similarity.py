"""Similarity-хук: substring-путь и работа без эмбеддинг-модели (dev-среда без torch)."""
from backend.retrieval.similarity import term_similarity


def test_substring_gives_full_score():
    assert term_similarity("Снизить CAC на 15% за квартал", "CAC") == 1.0


def test_missing_term_without_embeddings_is_zero(monkeypatch):
    # сначала я проверял это на venv без torch (реальный 0.0 от ImportError),
    # но тест падал при полном poetry install - эмбеддинги работали и давали
    # честный cosine. Теперь fallback-ветку проверяю детерминированно:
    # эмулирую недоступность сервиса эмбеддингов, что эквивалентно среде без torch.
    import backend.retrieval.embeddings as embeddings

    def _no_embeddings():
        raise RuntimeError("no torch in this env")

    monkeypatch.setattr(embeddings, "get_embedding_service", _no_embeddings)
    assert term_similarity("Снизить отток на 3 п.п.", "Churn Rate") == 0.0


def test_whitespace_in_term_is_normalized():
    assert term_similarity("используем RICE для приоритизации", "  rice  ") == 1.0
