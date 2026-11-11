"""BM25-индекс (rank_bm25.BM25Okapi) с dirty-флагом и ленивой пересборкой.

Индекс держит корпус в памяти; после изменения Decision Memory вызывается
mark_dirty(), и на следующем поиске корпус пересобирается один раз.
"""
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("backend.retrieval")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Токены для BM25: нижний регистр, буквы/цифры (кириллица поддерживается)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Ленивый BM25-индекс над корпусом документов.

    corpus_fetcher - async-функция без аргументов, возвращающая
    list[(doc_id, text)]: так индекс не знает про БД и в тестах
    подставляется фейковый корпус.
    """

    def __init__(self, corpus_fetcher: Callable[[], Awaitable[list[tuple[str, str]]]]) -> None:
        self._corpus_fetcher = corpus_fetcher
        self._ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: Any = None
        self._dirty = True

    def mark_dirty(self) -> None:
        """Данные изменились - следующая выдача требует пересборки."""
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._ids)

    async def _ensure_index(self) -> Any:
        # корпус 1100+ кейсов пересобирать на каждый запрос нельзя, поэтому
        # только по dirty-флагу после изменения данных
        if not self._dirty and self._bm25 is not None:
            return self._bm25
        from rank_bm25 import BM25Okapi  # лёгкий импорт, но держим рядом с использованием

        docs = await self._corpus_fetcher()
        self._ids = [doc_id for doc_id, _ in docs]
        self._corpus = [tokenize(text) for _, text in docs]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None
        self._dirty = False
        logger.debug("BM25-индекс пересобран: %d документов", len(self._ids))
        return self._bm25

    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Лучшие top_k документов: [(doc_id, score)] по убыванию релевантности."""
        bm25 = await self._ensure_index()
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda pair: -pair[1])
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k] if score > 0]


__all__ = ["BM25Index", "tokenize"]
