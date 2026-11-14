"""Задачи Celery (этап 11): переиндексация после HITL, batch-обработка, очистка кэша.

Все задачи - sync-обёртки над async-логикой через run_async (persistent loop).
Зависимости внутри задач резолвятся лениво, чтобы тесты могли подменять их.
"""
import logging
from datetime import UTC, datetime, timedelta

from backend.worker.celery_app import celery_app, run_async

logger = logging.getLogger("backend.worker")

# записи кэша без единого попадания старше этого срока чистятся nightly
CACHE_TTL_DAYS = 30


def _build_reindex_service():
    """RetrievalService для переиндексации (вынос для тестовых подмен)."""
    from backend.api.routes.decide import get_retriever

    return get_retriever().service


@celery_app.task(name="backend.worker.tasks.reindex_after_hitl")
def reindex_after_hitl_task() -> dict:
    """Переиндексация после подтверждения решения в HITL (active learning)."""
    from backend.db.engine import get_session_factory

    service = _build_reindex_service()

    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await service.reindex(session)

    stats = run_async(_run())
    logger.info("Переиндексация после HITL: %s", stats)
    return stats


def _broker_alive() -> bool:
    """Быстрый probe Redis (0.5s): на Windows TCP-коннект может висеть до 20с,
    поэтому решение "ставить ли задачу" принимается до вызова delay()."""
    try:
        import redis

        from backend.core.config import get_settings

        client = redis.Redis.from_url(
            get_settings().REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5
        )
        try:
            return bool(client.ping())
        finally:
            client.close()
    except Exception:
        return False


def trigger_reindex() -> bool:
    """Поставить переиндексацию в очередь; False - брокер недоступен.

    Вызывается из HITL-confirm. Брокер может лежать (dev без Redis) - это
    не ошибка подтверждения, переиндексация выполняется вручную или beat'ом.
    """
    if not _broker_alive():
        logger.warning("Celery-брокер недоступен: переиндексация отложена")
        return False
    try:
        reindex_after_hitl_task.delay()
        return True
    except Exception as exc:
        logger.warning("Celery-брокер недоступен (%s): переиндексация отложена", exc)
        return False


@celery_app.task(name="backend.worker.tasks.batch_decide")
def batch_decide_task(items: list[dict]) -> list[dict]:
    """Пакетная обработка ситуаций (для больших batch - вне HTTP-запроса)."""
    from backend.core.state import StateKeys
    from backend.db.engine import get_session_factory
    from backend.services.pipeline import safe_perform_decision

    async def _run() -> list[dict]:
        results = []
        async with get_session_factory()() as session:
            for item in items:
                state, error = await safe_perform_decision(
                    session,
                    item["situation"],
                    item.get("domain", "general"),
                    item.get("mode"),
                )
                if error is not None:
                    results.append({"status": "error", "error": error.message})
                else:
                    results.append({
                        "status": "ok",
                        "decision": state.get(StateKeys.DECISION) or "",
                        "critic_score": state.get(StateKeys.CRITIC_SCORE),
                        "eval_mode": state.get(StateKeys.EVAL_MODE),
                    })
        return results

    return run_async(_run())


@celery_app.task(name="backend.worker.tasks.cleanup_cache")
def cleanup_cache_task() -> dict:
    """Nightly-очистка: кэш-записи без единого попадания старше CACHE_TTL_DAYS."""
    from sqlalchemy import delete, func, select

    from backend.db.engine import get_session_factory
    from backend.db.models import DecisionCache

    threshold = datetime.now(UTC) - timedelta(days=CACHE_TTL_DAYS)

    async def _run() -> dict:
        async with get_session_factory()() as session:
            total = await session.scalar(select(func.count()).select_from(DecisionCache)) or 0
            result = await session.execute(
                delete(DecisionCache).where(
                    DecisionCache.hits == 0,
                    DecisionCache.created_at < threshold,
                )
            )
            await session.commit()
            return {"total_before": total, "deleted": result.rowcount or 0}

    return run_async(_run())
