"""Адаптер RetrievalService под Protocol Retriever из pipeline.

Расхождение контрактов после merge: pipeline зовёт fill_context(session,
state, *, hybrid=True), сервис Коли принимает (session, state) и сам решает
канал по доступности Qdrant. Адаптер навешивает недостающий флаг: при
hybrid=False временно отключает векторный канал, чтобы режим 'tm' работал
через BM25 даже при живом Qdrant. Когда сервис научится принимать hybrid
нативно, адаптер вырождается в прямой вызов.
"""
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.retrieval.search import RetrievalService


class RetrieverAdapter:
    """Делегирует сервису, форсируя канал по флагу hybrid."""

    def __init__(self, service: RetrievalService) -> None:
        self.service = service

    async def fill_context(
        self, session: AsyncSession, state: dict[str, Any], *, hybrid: bool = True
    ) -> dict[str, Any]:
        if hybrid:
            return await self.service.fill_context(session, state)
        saved = self.service._qdrant
        self.service._qdrant = None
        try:
            return await self.service.fill_context(session, state)
        finally:
            self.service._qdrant = saved
