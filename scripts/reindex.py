"""Переиндексация: перезалить векторы кейсов в Qdrant и пересобрать BM25.

Запуск: python scripts/reindex.py
Требует живой Qdrant и установленный sentence-transformers (poetry install).
После импорта данных и каждого подтверждения HITL (этап 11 делает это
автоматически через Celery).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.repository import get_cases, get_glossary
from backend.db.engine import get_session_factory, init_db
from backend.retrieval.embeddings import get_embedding_service
from backend.retrieval.qdrant import QdrantService
from backend.retrieval.search import RetrievalService


async def main() -> None:
    await init_db()
    async def cases_fetcher():
        async with get_session_factory()() as session:
            return await get_cases(session)

    async def glossary_fetcher():
        async with get_session_factory()() as session:
            return await get_glossary(session)

    service = RetrievalService(
        cases_fetcher, glossary_fetcher, get_embedding_service(), qdrant=QdrantService()
    )
    async with get_session_factory()() as session:
        stats = await service.reindex(session)
    print("Переиндексация завершена:", stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ImportError as exc:
        print(f"Нужны тяжёлые зависимости (torch/sentence-transformers): {exc}")
        print("Выполните: poetry install")
        sys.exit(1)
