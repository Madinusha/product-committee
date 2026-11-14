"""Кэш готовых решений (этап 9).

Ключ - составной UNIQUE из 6 колонок модели DecisionCache (Коля). Вместо
NULL в составе ключа - сентинел 'unknown' (NULL != NULL ломает UNIQUE).
Кэшируются только решения домена 'general' (по заданию): решения для
конкретных доменов зависят от контекста ситуации сильнее и почти не
повторяются, статистика попаданий была бы шумом.
"""
import hashlib
import logging
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import DecisionCache

logger = logging.getLogger("backend.cache")

CACHEABLE_DOMAIN = "general"
UNKNOWN = "unknown"


def situation_hash(situation: str) -> str:
    """sha256 от нормализованной ситуации: регистр и пробелы не создают 'новых' ситуаций."""
    normalized = re.sub(r"\s+", " ", situation.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_cacheable(domain: str | None) -> bool:
    return domain == CACHEABLE_DOMAIN


@dataclass(frozen=True)
class CacheHit:
    decision: str
    critic_score: float | None
    hits: int


class DecisionCacheManager:
    """Чтение/запись кэша. Идемпотентная запись: повторное сохранение того же
    ключа не создаёт дубль и не сбрасывает счётчик попаданий."""

    async def get(
        self,
        session: AsyncSession,
        *,
        situation: str,
        domain: str,
        decision_type: str | None = None,
        eval_mode: str | None = None,
        engine: str | None = None,
        model: str | None = None,
    ) -> CacheHit | None:
        if not is_cacheable(domain):
            return None
        row = await session.scalar(
            select(DecisionCache).where(
                DecisionCache.situation_hash == situation_hash(situation),
                DecisionCache.domain == domain,
                DecisionCache.decision_type == (decision_type or UNKNOWN),
                DecisionCache.eval_mode == (eval_mode or UNKNOWN),
                DecisionCache.engine == (engine or UNKNOWN),
                DecisionCache.model == (model or UNKNOWN),
            )
        )
        if row is None:
            return None
        row.hits += 1
        await session.commit()
        logger.info("Кэш-хит: situation_hash=%s, hits=%d", row.situation_hash[:12], row.hits)
        return CacheHit(decision=row.decision, critic_score=row.critic_score, hits=row.hits)

    async def save(
        self,
        session: AsyncSession,
        *,
        situation: str,
        domain: str,
        decision: str,
        decision_type: str | None = None,
        eval_mode: str | None = None,
        engine: str | None = None,
        model: str | None = None,
        critic_score: float | None = None,
    ) -> bool:
        """Сохранить решение; False - ключ уже был (дубль не создаётся)."""
        if not is_cacheable(domain):
            return False
        exists = await session.scalar(
            select(DecisionCache.id).where(
                DecisionCache.situation_hash == situation_hash(situation),
                DecisionCache.domain == domain,
                DecisionCache.decision_type == (decision_type or UNKNOWN),
                DecisionCache.eval_mode == (eval_mode or UNKNOWN),
                DecisionCache.engine == (engine or UNKNOWN),
                DecisionCache.model == (model or UNKNOWN),
            )
        )
        if exists is not None:
            return False
        session.add(DecisionCache(
            situation_hash=situation_hash(situation),
            domain=domain,
            decision_type=decision_type or UNKNOWN,
            eval_mode=eval_mode or UNKNOWN,
            engine=engine or UNKNOWN,
            model=model or UNKNOWN,
            decision=decision,
            critic_score=critic_score,
        ))
        await session.commit()
        return True
