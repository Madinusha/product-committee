"""Readiness-проверки внешних зависимостей (этап 8).

Каждая проверка возвращает "ok" | "down" и не должна бросать исключений:
/health обязан отвечать 503 с полным набором компонентов, а не падать.
"""
import logging

from backend.core.config import get_settings
from backend.db.engine import check_db

logger = logging.getLogger("backend.health")


async def check_db_component() -> str:
    return "ok" if await check_db() else "down"


async def check_ollama() -> str:
    """Ollama жива: GET /api/tags с коротким таймаутом."""
    from backend.utils.http_client import get_ollama_client

    try:
        response = await get_ollama_client().get("/api/tags", timeout=2.0)
        return "ok" if response.status_code == 200 else "down"
    except Exception as exc:
        logger.debug("Ollama недоступна: %s", exc)
        return "down"


async def check_qdrant() -> str:
    from backend.retrieval.qdrant import QdrantService

    return "ok" if await QdrantService().available() else "down"


async def check_redis() -> str:
    """Redis жив: PING (нужен Celery и кэш-инвалидация на этапах 10-11)."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            get_settings().REDIS_URL, socket_connect_timeout=2, socket_timeout=2
        )
        try:
            return "ok" if await client.ping() else "down"
        finally:
            await client.aclose()
    except Exception as exc:
        logger.debug("Redis недоступен: %s", exc)
        return "down"


async def run_all_health_checks() -> dict[str, str]:
    """Все компоненты параллельно: даже полностью лежащий стенд отвечает быстро.

    return_exceptions=True: сбой внутри любой проверки - это компонент "down",
    а не 500 на /health.
    """
    import asyncio

    names = ("db", "ollama", "qdrant", "redis")
    results = await asyncio.gather(
        check_db_component(),
        check_ollama(),
        check_qdrant(),
        check_redis(),
        return_exceptions=True,
    )
    return {
        name: "down" if isinstance(result, Exception) else result
        for name, result in zip(names, results, strict=True)
    }
