"""Human-in-the-Loop (этап 10): очередь подтверждений продакт-владельцем.

Атомарный confirm: статус записи и новый кейс Decision Memory сохраняются
в ОДНОЙ транзакции - либо оба, либо ничего. После подтверждения кейс
участвует в retrieval (active learning); переиндексация - задача этапа 11.
"""
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, StateConflictError
from backend.data.importers import case_id
from backend.db.models import DecisionMemory, HitlQueue

logger = logging.getLogger("backend.hitl")

_PENDING = "pending"
_APPROVED = "approved"
_REJECTED = "rejected"


def _quality_from_score(critic_score: float | None) -> int:
    """Экспертная оценка кейса из score Critic'а (0-10, без оценки -> 5)."""
    if critic_score is None:
        return 5
    return max(0, min(10, round(critic_score)))


async def enqueue(
    session: AsyncSession,
    *,
    situation: str,
    decision: str,
    domain: str,
    decision_type: str | None = None,
    critic_score: float | None = None,
) -> HitlQueue:
    """Отправить решение в очередь на подтверждение."""
    entry = HitlQueue(
        situation=situation.strip(),
        decision=decision.strip(),
        domain=domain,
        critic_score=critic_score,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    logger.info("HITL: решение поставлено в очередь id=%d", entry.id)
    return entry


async def pending_list(session: AsyncSession, limit: int = 50) -> list[HitlQueue]:
    """Очередь в порядке поступления (старые - первыми)."""
    rows = await session.execute(
        select(HitlQueue)
        .where(HitlQueue.status == _PENDING)
        .order_by(HitlQueue.created_at.asc())
        .limit(limit)
    )
    return list(rows.scalars().all())


async def history(session: AsyncSession, limit: int = 20) -> list[HitlQueue]:
    """Последние обработанные записи (approved/rejected)."""
    rows = await session.execute(
        select(HitlQueue)
        .where(HitlQueue.status != _PENDING)
        .order_by(HitlQueue.reviewed_at.desc())
        .limit(limit)
    )
    return list(rows.scalars().all())


async def get_entry(session: AsyncSession, entry_id: int) -> HitlQueue:
    entry = await session.get(HitlQueue, entry_id)
    if entry is None:
        raise NotFoundError(f"Запись HITL {entry_id} не найдена")
    return entry


async def approve(
    session: AsyncSession, entry_id: int, reviewer_id: int, reviewer_name: str
) -> tuple[HitlQueue, bool]:
    """Подтвердить решение: статус + кейс в Decision Memory одной транзакцией.

    Возвращает (запись, создан_ли_новый_кейс). Повторное подтверждение -> 409.
    Если кейс с таким содержимым уже есть в памяти - кейс не дублируется,
    но подтверждение фиксируется.
    """
    entry = await get_entry(session, entry_id)
    if entry.status != _PENDING:
        raise StateConflictError(
            f"Запись {entry_id} уже обработана (статус: {entry.status})"
        )

    entry.status = _APPROVED
    entry.reviewer_id = reviewer_id
    entry.reviewed_at = datetime.now(UTC)

    memory_id = case_id(entry.situation, entry.decision)
    existing = await session.get(DecisionMemory, memory_id)
    created = False
    if existing is None:
        session.add(DecisionMemory(
            id=memory_id,
            situation=entry.situation,
            decision=entry.decision,
            outcome=f"Подтверждено продакт-владельцем {reviewer_name}",
            domain=entry.domain,
            decision_type="hitl",
            quality=_quality_from_score(entry.critic_score),
        ))
        created = True

    await session.commit()  # атомарность: статус и кейс либо оба, либо ничего
    if created:
        logger.info(
            "HITL: подтверждено id=%d, кейс %s добавлен в Decision Memory", entry_id, memory_id[:12]
        )
    return entry, created


async def reject(session: AsyncSession, entry_id: int, reviewer_id: int) -> HitlQueue:
    """Отклонить решение: кейс в память НЕ попадает."""
    entry = await get_entry(session, entry_id)
    if entry.status != _PENDING:
        raise StateConflictError(
            f"Запись {entry_id} уже обработана (статус: {entry.status})"
        )
    entry.status = _REJECTED
    entry.reviewer_id = reviewer_id
    entry.reviewed_at = datetime.now(UTC)
    await session.commit()
    logger.info("HITL: отклонено id=%d", entry_id)
    return entry
