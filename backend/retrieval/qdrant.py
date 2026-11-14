"""Qdrant-сервис: векторный поиск по кейсам Decision Memory.

Точки: id = uuid5 от case_id (идемпотентный upsert - повторная загрузка
того же кейса перезаписывает точку, а не создаёт дубль), метрика cosine.
При недоступности Qdrant retrieval деградирует до BM25 (см. search.py).
"""
import logging
import uuid
from typing import Any

from backend.core.config import get_settings
from backend.retrieval.embeddings import VECTOR_SIZE

logger = logging.getLogger("backend.retrieval")

# Фиксированное namespace-UUID: uuid5 одинаковый на всех машинах и в тестах
_CASE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "product-committee/decision-memory")


def case_point_id(case_id: str) -> str:
    """Стабильный UUID точки Qdrant для кейса (идемпотентный upsert)."""
    # сначала пробовал случайные id: повторная загрузка кейса плодила дубли,
    # uuid5 решает это - та же ситуация всегда даёт ту же точку
    return str(uuid.uuid5(_CASE_NAMESPACE, case_id))


class QdrantService:
    """Async-обёртка над qdrant-client. Клиент создаётся лениво."""

    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        settings = get_settings()
        self.url = url or settings.QDRANT_URL
        self.collection = collection or settings.QDRANT_COLLECTION
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # импорт рядом с использованием

            self._client = AsyncQdrantClient(url=self.url, timeout=5)
        return self._client

    async def ensure_collection(self) -> None:
        """Создать коллекцию (cosine, 384), если её нет - идемпотентно."""
        from qdrant_client import models

        client = self._ensure_client()
        if await client.collection_exists(self.collection):
            return
        await client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
        logger.info("Создана Qdrant-коллекция %s", self.collection)

    async def upsert_cases(
        self, vectors: dict[str, list[float]], payloads: dict[str, dict]
    ) -> int:
        """Загрузить кейсы: {case_id: vector} + {case_id: payload}. Возвращает число точек."""
        from qdrant_client import models

        if not vectors:
            return 0
        await self.ensure_collection()
        points = [
            models.PointStruct(
                id=case_point_id(case_id),
                vector=vector,
                payload=payloads[case_id],
            )
            for case_id, vector in vectors.items()
        ]
        await self._ensure_client().upsert(collection_name=self.collection, points=points)
        logger.info("Upsert %d точек в %s", len(points), self.collection)
        return len(points)

    async def search(self, vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Поиск по коллекции: [(payload_case_id, cosine_score)] по убыванию."""

        hits = await self._ensure_client().query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            (hit.payload["case_id"], float(hit.score))
            for hit in hits.points
            if hit.payload and "case_id" in hit.payload
        ]

    async def available(self) -> bool:
        """Проверка доступности (для /health и решения о деградации)."""
        try:
            await self._ensure_client().get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant недоступен (%s)", exc)
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
