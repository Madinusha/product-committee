"""RetrievalService - единственная точка, где state получает RAG-контекст.

Заполняет StateKeys.GLOSSARY_USED и StateKeys.RETRIEVED_CONTEXT.
Стратегия: гибрид (вектор Qdrant + BM25 -> RRF), при недоступности Qdrant -
деградация до BM25 (RETRIEVAL_BACKEND='bm25'). Работает и вовсе без контекста:
пустые списки - валидное состояние (агенты это учитывают).
"""
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.state import StateKeys
from backend.data.repository import CaseEntry, GlossaryEntry, get_cases, get_glossary
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.embeddings import EmbeddingService
from backend.retrieval.hybrid import rrf_merge
from backend.retrieval.qdrant import QdrantService

logger = logging.getLogger("backend.retrieval")


def _case_text(case: CaseEntry) -> str:
    return f"{case.situation} {case.outcome}"


def _glossary_text(entry: GlossaryEntry) -> str:
    return f"{entry.term} {entry.definition}"


class RetrievalService:
    """Гибридный поиск по глоссарию и кейсам для состояния агентов.

    corpus_fetcher'ы передаются извне (read-репозитории); в тестах
    подставляются фейковые - сервис не зависит от способа хранения.
    """

    def __init__(
        self,
        cases_fetcher: Any,
        glossary_fetcher: Any,
        embeddings: EmbeddingService,
        qdrant: QdrantService | None = None,
    ) -> None:
        self._cases_fetcher = cases_fetcher
        self._glossary_fetcher = glossary_fetcher
        self._embeddings = embeddings
        self._qdrant = qdrant
        top_k = get_settings().RETRIEVAL_TOP_K
        self._cases_index = BM25Index(self._fetch_case_corpus)
        self._glossary_index = BM25Index(self._fetch_glossary_corpus)
        self.top_k = top_k
        self.last_backend = "bm25"  # backend последнего поиска (для state и /health)

    # корпуса для BM25-индексов

    async def _fetch_case_corpus(self) -> list[tuple[str, str]]:
        cases = await self._cases_fetcher()
        return [(c.id, _case_text(c)) for c in cases]

    async def _fetch_glossary_corpus(self) -> list[tuple[str, str]]:
        entries = await self._glossary_fetcher()
        return [(e.id, _glossary_text(e)) for e in entries]

    async def mark_dirty(self) -> None:
        """Вызывается после изменения данных (импорт, HITL-confirm) - этап 11."""
        self._cases_index.mark_dirty()
        self._glossary_index.mark_dirty()

    # основной вход

    async def fill_context(self, session: AsyncSession, state: dict[str, Any]) -> dict[str, Any]:
        """Найти релевантные термины и кейсы для ситуации из state."""
        situation = state.get(StateKeys.SITUATION) or ""
        domain = state.get(StateKeys.DOMAIN)
        if not situation:
            self.last_backend = "none"
            state[StateKeys.GLOSSARY_USED] = []
            state[StateKeys.RETRIEVED_CONTEXT] = []
            state[StateKeys.RETRIEVAL_BACKEND] = self.last_backend
            return state

        state[StateKeys.GLOSSARY_USED] = await self.find_glossary(session, situation, domain)
        state[StateKeys.RETRIEVED_CONTEXT] = await self.find_cases(session, situation, domain)
        state[StateKeys.RETRIEVAL_BACKEND] = self.last_backend
        return state

    async def find_glossary(
        self, session: AsyncSession, situation: str, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """Термины к ситуации: BM25 по глоссарию (emb-канал для терминов избыточен)."""
        entries = await get_glossary(session, domain)
        if not entries:
            return []
        by_id = {e.id: e for e in entries}
        hits = await self._glossary_index.search(situation, top_k=self.top_k)
        return [
            {
                "term": by_id[i].term, "definition": by_id[i].definition,
                "domain": by_id[i].domain, "score": round(score, 4),
            }
            for i, score in hits
            if i in by_id
        ]

    async def find_cases(
        self, session: AsyncSession, situation: str, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """Кейсы к ситуации: гибрид (вектор + BM25 -> RRF) с деградацией до BM25."""
        cases = await get_cases(session, domain)
        if not cases:
            return []
        by_id = {c.id: c for c in cases}

        rankings: list[list[str]] = []
        backend = "bm25"
        if self._qdrant is not None and await self._qdrant.available():
            try:
                # Qdrant может отвалиться между available() и search(): тогда
                # не падаем, а деградируем на BM25 и помечаем backend в state,
                # чтобы eval потом видел, на каком канале получен результат
                vector = self._embeddings.encode_one(situation)
                vector_hits = await self._qdrant.search(vector, top_k=self.top_k)
                rankings.append([case_id for case_id, _ in vector_hits])
                backend = "hybrid"
            except Exception as exc:  # Qdrant ответил ошибкой - работаем на BM25
                logger.warning("Векторный поиск недоступен (%s), деградация до BM25", exc)
        bm25_hits = await self._cases_index.search(situation, top_k=self.top_k)
        rankings.append([case_id for case_id, _ in bm25_hits])
        self.last_backend = backend

        merged = rrf_merge(rankings)
        return [
            {
                "case_id": case_id,
                "situation": by_id[case_id].situation,
                "decision": by_id[case_id].decision,
                "outcome": by_id[case_id].outcome,
                "domain": by_id[case_id].domain,
                "score": round(score, 4),
            }
            for case_id, score in merged
            if case_id in by_id
        ][: self.top_k]

    # индексация: после импорта и HITL-confirm (этапы 6, 11)

    async def reindex(self, session: AsyncSession) -> dict[str, int]:
        """Пересобрать BM25 и перезалить векторы в Qdrant. Возвращает статистику."""
        cases = await get_cases(session)
        await self.mark_dirty()
        if self._qdrant is None:
            return {"cases": len(cases), "qdrant": 0}
        await self._qdrant.ensure_collection()
        vectors_list = self._embeddings.encode([_case_text(c) for c in cases])
        vectors = {c.id: v for c, v in zip(cases, vectors_list, strict=True)}
        payloads = {
            c.id: {"case_id": c.id, "domain": c.domain, "decision_type": c.decision_type}
            for c in cases
        }
        await self._qdrant.upsert_cases(vectors, payloads)
        return {"cases": len(cases), "qdrant": len(vectors)}
