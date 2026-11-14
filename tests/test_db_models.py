"""Модели БД: пять таблиц, детерминированные id, UNIQUE-ключ кэша с сентинелом."""
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.data.importers import case_id, glossary_id
from backend.db.models import DecisionCache, DecisionMemory, GlossaryTerm, HitlQueue


async def test_glossary_and_memory_tables(db_session_factory):
    async with db_session_factory() as session:
        session.add(GlossaryTerm(
            id=glossary_id("CAC"), term="CAC",
            definition="стоимость привлечения", domain="general",
        ))
        session.add(DecisionMemory(
            id=case_id("ситуация", "решение"),
            situation="ситуация", decision="решение", outcome="успех",
            domain="b2b_saas", decision_type="рост метрики", quality=8,
        ))
        await session.commit()

        term = await session.get(GlossaryTerm, glossary_id("cac"))  # id не зависит от регистра
        assert term is not None and term.term == "CAC"
        memory = await session.get(DecisionMemory, case_id("Ситуация ", " Решение"))
        assert memory is not None and memory.quality == 8  # id не зависит от пробелов


async def test_cache_unique_key_six_columns(db_session_factory):
    """Одинаковый ключ -> IntegrityError; отличие в одной колонке -> отдельная запись."""
    async with db_session_factory() as session:
        base = {"situation_hash": "abc", "domain": "general", "decision_type": "unknown",
                "eval_mode": "full", "engine": "ollama", "model": "qwen2.5:7b"}
        session.add(DecisionCache(**base, decision="решение v1"))
        await session.commit()

        # полный дубль ключа запрещён
        session.add(DecisionCache(**base, decision="решение v2"))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # тот же situation_hash, другой eval_mode - новая запись (sentinel 'unknown' != NULL)
        session.add(DecisionCache(**{**base, "eval_mode": "baseline"}, decision="решение v3"))
        await session.commit()
        rows = (await session.execute(select(DecisionCache).order_by(DecisionCache.id))).scalars().all()
        assert len(rows) == 2
        assert rows[0].decision == "решение v1" and rows[0].hits == 0


async def test_hitl_queue_defaults(db_session_factory):
    async with db_session_factory() as session:
        entry = HitlQueue(situation="s", decision="d", critic_score=7.5)
        session.add(entry)
        await session.commit()
        stored = await session.get(HitlQueue, entry.id)
        assert stored.status == "pending"
        assert stored.reviewer_id is None
        assert stored.critic_score == 7.5
