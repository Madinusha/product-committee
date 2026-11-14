"""Эмбеддинг-сервис (синглтон) на sentence-transformers.

Модель загружается лениво при первом encode - импорт модуля не тянет torch.
Размер вектора: 384 (paraphrase-multilingual-MiniLM-L12-v2).
"""
import logging
from functools import lru_cache
from typing import Any

from backend.core.config import get_settings

logger = logging.getLogger("backend.retrieval")

VECTOR_SIZE = 384


class EmbeddingService:
    """Обёртка над sentence-transformers: один экземпляр модели на процесс."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().EMBEDDING_MODEL_NAME
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # тяжёлый импорт - только здесь

            logger.info("Загрузка эмбеддинг-модели %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Векторы для списка текстов (нормированные - работает cosine)."""
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        """Cosine-близость без numpy (векторы уже нормированы моделью)."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return dot / norm if norm else 0.0


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
