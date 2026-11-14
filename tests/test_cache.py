"""Кэш решений: 6-колоночный ключ, сентинел, hits, только general. Адаптер шва."""
from backend.data.importers import _normalize  # noqa: F401  (проверка согласованности нормализации)
from backend.db.models import DecisionCache
from backend.services.cache import (
    CACHEABLE_DOMAIN,
    DecisionCacheManager,
    is_cacheable,
    situation_hash,
)
from tests.test_retrieval import FakeEmbeddings, FakeQdrantUp, _fake_cases, _fake_glossary


def _manager():
    return DecisionCacheManager()


def test_situation_hash_normalized_and_stable():
    assert situation_hash("Отток 8%") == situation_hash("  отток 8%  ")
    assert situation_hash("Отток 8%") != situation_hash("Отток 9%")


def test_only_general_is_cacheable():
    assert is_cacheable("general")
    assert not is_cacheable("b2b_saas")
    assert not is_cacheable(None)


async def test_save_and_get_roundtrip(db_session_factory):
    manager = _manager()
    async with db_session_factory() as session:
        saved = await manager.save(
            session, situation="Отток 8%", domain=CACHEABLE_DOMAIN,
            decision="Снизить отток до 5%", eval_mode="full",
            engine="ollama", model="qwen2.5:7b", critic_score=8.0,
        )
        assert saved is True

        hit = await manager.get(
            session, situation="отток 8% ", domain=CACHEABLE_DOMAIN,
            decision_type=None, eval_mode="full", engine="ollama", model="qwen2.5:7b",
        )
        assert hit is not None
        assert hit.decision == "Снизить отток до 5%"
        assert hit.critic_score == 8.0
        assert hit.hits == 1  # счётчик увеличился при чтении


async def test_cache_key_components_distinguish_entries(db_session_factory):
    """Одинаковая ситуация, разные eval_mode -> разные записи (сентинел 'unknown' работает)."""
    from sqlalchemy import select

    manager = _manager()
    async with db_session_factory() as session:
        assert await manager.save(session, situation="s", domain="general", decision="v1", eval_mode="full") is True
        assert await manager.save(session, situation="s", domain="general", decision="v2", eval_mode="baseline") is True
        assert await manager.save(session, situation="s", domain="general", decision="v3") is True

        rows = (await session.execute(select(DecisionCache))).scalars().all()
        assert len(rows) == 3
        assert {r.eval_mode for r in rows} == {"full", "baseline", "unknown"}


async def test_save_is_idempotent_and_hits_preserved(db_session_factory):
    manager = _manager()
    async with db_session_factory() as session:
        await manager.save(session, situation="s", domain="general", decision="решение")
        hit = await manager.get(session, situation="s", domain="general", decision_type=None,
                               eval_mode=None, engine=None, model=None)
        assert hit.hits == 1

        assert await manager.save(session, situation="s", domain="general", decision="решение") is False
        hit = await manager.get(session, situation="s", domain="general", decision_type=None,
                               eval_mode=None, engine=None, model=None)
        assert hit.hits == 2  # дубль не сбросил счётчик


async def test_non_general_domains_are_not_cached(db_session_factory):
    manager = _manager()
    async with db_session_factory() as session:
        assert await manager.save(session, situation="s", domain="b2b_saas", decision="d") is False
        assert await manager.get(session, situation="s", domain="fintech", decision_type=None,
                                 eval_mode=None, engine=None, model=None) is None


async def test_adapter_forces_lexical_channel_when_hybrid_false(db_session_factory):
    """Адаптер: hybrid=False должен дать BM25-канал даже с живым Qdrant (режим 'tm')."""
    from backend.core.state import StateKeys, initial_state
    from backend.db.models import DecisionMemory
    from backend.retrieval.search import RetrievalService
    from backend.services.retrieval_adapter import RetrieverAdapter

    async def cases_fetcher():
        return _fake_cases()

    async def glossary_fetcher():
        return _fake_glossary()

    service = RetrievalService(
        cases_fetcher, glossary_fetcher, FakeEmbeddings(),
        qdrant=FakeQdrantUp([("case-1", 0.9)]),
    )
    adapter = RetrieverAdapter(service)

    # кейсы должны быть в БД: find_cases берёт payload-данные оттуда
    async with db_session_factory() as session:
        for c in _fake_cases():
            session.add(DecisionMemory(
                id=c.id, situation=c.situation, decision=c.decision,
                outcome=c.outcome, domain=c.domain, decision_type=c.decision_type,
                quality=c.quality,
            ))
        await session.commit()

        state = initial_state("Нужно снизить отток клиентов", "b2b_saas")
        await adapter.fill_context(session, state, hybrid=False)
        assert state[StateKeys.RETRIEVAL_BACKEND] == "bm25"

        state = initial_state("Нужно снизить отток клиентов", "b2b_saas")
        await adapter.fill_context(session, state, hybrid=True)
        assert state[StateKeys.RETRIEVAL_BACKEND] == "hybrid"

    # после форсирования BM25 векторный канал снова доступен (qdrant восстановлен)
    assert service._qdrant is not None
