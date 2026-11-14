"""Celery-приложение (этап 11): брокер и бэкенд - Redis.

Persistent event loop: у воркера ОДИН event loop на весь процесс (создаётся
при старте, закрывается при graceful shutdown) - httpx-клиенты и
SQLAlchemy-движки привязаны к loop, пересоздавать их на каждую задачу нельзя.
"""
import asyncio
import logging

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from backend.core.config import get_settings

logger = logging.getLogger("backend.worker")

settings = get_settings()

celery_app = Celery(
    "vpc",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    imports=("backend.worker.tasks",),
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,  # задача подтверждается после выполнения, а не при получении
    worker_prefetch_multiplier=1,  # длинные LLM-задачи: не хватать несколько сразу
    # отказ брокера должен быть мгновенным (dev без Redis, тесты): без ретраев
    task_publish_retry=False,
    broker_connection_timeout=1,
    broker_transport_options={"socket_connect_timeout": 1, "socket_timeout": 1},
    # beat: ночная очистка неиспользуемого кэша
    beat_schedule={
        "cleanup-cache-nightly": {
            "task": "backend.worker.tasks.cleanup_cache",
            "schedule": 60 * 60 * 24,  # раз в сутки (crontab задаётся в tasks)
        },
    },
)

_loop: asyncio.AbstractEventLoop | None = None


def get_worker_loop() -> asyncio.AbstractEventLoop:
    """Единый event loop воркера - лениво создаётся на процесс."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        logger.info("Worker: создан persistent event loop")
    return _loop


def run_async(coro):
    """Выполнить корутину на persistent loop'е воркера (из sync-задачи Celery)."""
    return get_worker_loop().run_until_complete(coro)


@worker_ready.connect
def _on_worker_ready(**_kwargs) -> None:
    get_worker_loop()  # прогреть loop при старте, а не на первой задаче


@worker_shutdown.connect
def _on_worker_shutdown(**_kwargs) -> None:
    """Graceful shutdown: сначала клиенты (httpx, engine), затем loop."""
    global _loop
    try:
        if _loop is not None and not _loop.is_closed():
            from backend.utils.http_client import close_ollama_clients

            _loop.run_until_complete(close_ollama_clients())

            from backend.db.engine import dispose_engine

            _loop.run_until_complete(dispose_engine())
            _loop.close()
            logger.info("Worker: event loop закрыт (graceful shutdown)")
    except Exception as exc:  # shutdown не должен падать
        logger.warning("Worker: shutdown с ошибкой (%s)", exc)
    finally:
        _loop = None
