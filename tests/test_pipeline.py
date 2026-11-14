"""Pipeline: 8 ablation-режимов, каналы retrieval, safe-обёртка."""
import pytest

from backend.agents import Agent
from backend.core.constants import (
    AGENTS_ONLY,
    BASELINE,
    EVAL_MODES,
    FULL,
    GLOSSARY,
    TM,
    TM_HYBRID,
    TM_HYBRID_CRITIC,
    TM_HYBRID_EDITOR,
    StopReason,
)
from backend.core.exceptions import EngineUnavailableError, UnknownEvalModeError
from backend.core.state import StateKeys
from backend.services.pipeline import (
    DecisionPipeline,
    get_mode_config,
    safe_perform_decision,
)
from tests.conftest import make_llm_mock

SITUATION = "Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц"
GOOD_DRAFT = (
    "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
    "замер - когортный анализ."
)
CRITIC_8 = (
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, '
    '"overall_score": 8.0, "suggestions": []}'
)
CRITIC_5 = (
    '{"justification": 4, "feasibility": 5, "risks": 4, "terminology": 6, '
    '"overall_score": 5.0, "suggestions": ["добавить метрику и срок"]}'
)


class FakeRetriever:
    """Стаб RAG-подсистемы: помечает канал и наполняет state-контекст."""

    def __init__(self, glossary=None, cases=None):
        self.calls = []  # значения флага hybrid по каждому вызову
        self._glossary = glossary or []
        self._cases = cases or []

    async def fill_context(self, session, state, *, hybrid=True):
        self.calls.append(hybrid)
        state[StateKeys.RETRIEVAL_BACKEND] = "hybrid" if hybrid else "bm25"
        state[StateKeys.GLOSSARY_USED] = self._glossary
        state[StateKeys.RETRIEVED_CONTEXT] = self._cases
        return state


def patch_llm(monkeypatch, responses):
    monkeypatch.setattr(Agent, "_call_llm", make_llm_mock(responses))


def test_unknown_mode_lists_supported():
    with pytest.raises(UnknownEvalModeError) as exc_info:
        get_mode_config("agentic_supermode")
    for mode in EVAL_MODES:
        assert mode in str(exc_info.value)


@pytest.mark.parametrize(
    ("mode", "glossary", "cases", "channel", "critic", "editor"),
    [
        (BASELINE, False, False, None, False, False),
        (GLOSSARY, True, False, None, False, False),
        (TM, True, True, "bm25", False, False),
        (TM_HYBRID, True, True, "hybrid", False, False),
        (AGENTS_ONLY, False, False, None, True, True),
        (TM_HYBRID_CRITIC, True, True, "hybrid", True, False),
        (TM_HYBRID_EDITOR, True, True, "hybrid", False, True),
        (FULL, True, True, "hybrid", True, True),
    ],
)
def test_mode_configs_cover_all_8_modes(mode, glossary, cases, channel, critic, editor):
    cfg = get_mode_config(mode)
    assert (cfg.use_glossary, cfg.use_cases, cfg.channel, cfg.use_critic, cfg.use_editor) == (
        glossary, cases, channel, critic, editor,
    )


async def test_baseline_returns_draft_without_agents_and_retrieval(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT})
    retriever = FakeRetriever()
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "b2b_saas", mode=BASELINE)
    assert retriever.calls == []  # baseline не ходит в retrieval
    assert state[StateKeys.DECISION] == GOOD_DRAFT
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert state[StateKeys.RETRIEVAL_BACKEND] == "none"
    assert StateKeys.CRITIC_SCORE not in state  # агенты не запускались


async def test_glossary_mode_fills_glossary_and_calls_retriever(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT})
    glossary = [{"term": "churn rate", "definition": "доля ушедших клиентов"}]
    retriever = FakeRetriever(glossary=glossary)
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "b2b_saas", mode=GLOSSARY)
    assert retriever.calls == [False]  # кейсы не нужны - лексический канал
    assert state[StateKeys.GLOSSARY_USED] == glossary
    assert state[StateKeys.DECISION] == GOOD_DRAFT


async def test_tm_mode_requests_lexical_channel(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT})
    retriever = FakeRetriever()
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        await pipeline.run(session, SITUATION, "b2b_saas", mode=TM)
    assert retriever.calls == [False]  # 'tm' обязан работать без векторного канала


async def test_tm_hybrid_mode_requests_hybrid_channel(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT})
    retriever = FakeRetriever()
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        await pipeline.run(session, SITUATION, "b2b_saas", mode=TM_HYBRID)
    assert retriever.calls == [True]


async def test_agents_only_skips_retrieval_and_runs_supervisor(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT, "critic": CRITIC_8})
    retriever = FakeRetriever()
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "b2b_saas", mode=AGENTS_ONLY)
    assert retriever.calls == []
    assert state[StateKeys.RETRIEVAL_BACKEND] == "none"
    assert state[StateKeys.CRITIC_SCORE] == 8.0
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS


async def test_full_mode_full_cycle_improves(db_session_factory, monkeypatch):
    """FULL: retrieval -> черновик -> Critic 5.0 -> Editor -> Critic 8.0 -> SUCCESS."""
    patch_llm(monkeypatch, {
        "analyst": "Улучшить продукт",  # слабый черновик
        "critic": [CRITIC_5, CRITIC_8],
        "editor": GOOD_DRAFT,
    })
    retriever = FakeRetriever(
        glossary=[{"term": "retention", "definition": "удержание пользователей"}],
        cases=[{"case_id": "c1", "situation": "s", "decision": "d", "outcome": "ok", "score": 0.9}],
    )
    pipeline = DecisionPipeline(retriever=retriever)
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "b2b_saas", mode=FULL)
    assert retriever.calls == [True]
    assert state[StateKeys.EVAL_MODE] == FULL
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert state[StateKeys.CRITIC_SCORE] == 8.0
    assert state[StateKeys.ITERATIONS] == 1
    assert state[StateKeys.DECISION] == GOOD_DRAFT


async def test_tm_hybrid_editor_mode_fixes_validation_without_critic(db_session_factory, monkeypatch):
    """Editor без Critic: судьи нет, правит по нарушениям правил Validator'а."""
    patch_llm(monkeypatch, {
        "analyst": "Улучшить продукт",  # черновик не проходит валидацию
        "editor": GOOD_DRAFT,
    })
    pipeline = DecisionPipeline(retriever=FakeRetriever())
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "b2b_saas", mode=TM_HYBRID_EDITOR)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert state[StateKeys.ITERATIONS] == 1
    assert state[StateKeys.DECISION] == GOOD_DRAFT
    assert StateKeys.CRITIC_SCORE not in state  # Critic действительно не запускался


async def test_retrieval_required_but_missing_raises(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT})
    pipeline = DecisionPipeline(retriever=None)  # режим требует retrieval, исполнитель не подключён
    async with db_session_factory() as session:
        with pytest.raises(Exception, match="требует retrieval"):
            await pipeline.run(session, SITUATION, "b2b_saas", mode=TM_HYBRID)


async def test_safe_perform_decision_returns_error_not_raise(db_session_factory, monkeypatch):
    class DownRetriever:
        async def fill_context(self, session, state, *, hybrid=True):
            raise EngineUnavailableError("Qdrant и BM25 недоступны")

    monkeypatch.setattr(Agent, "_call_llm", make_llm_mock({}))
    async with db_session_factory() as session:
        state, error = await safe_perform_decision(
            session, SITUATION, "b2b_saas", mode=TM_HYBRID, retriever=DownRetriever()
        )
    assert state is None
    assert error is not None and "недоступны" in error.message


async def test_safe_perform_decision_reraises_unknown_mode(db_session_factory):
    async with db_session_factory() as session:
        with pytest.raises(UnknownEvalModeError):
            await safe_perform_decision(session, SITUATION, "general", mode="no-such-mode")


async def test_default_mode_taken_from_settings(db_session_factory, monkeypatch):
    patch_llm(monkeypatch, {"analyst": GOOD_DRAFT, "critic": CRITIC_8})
    pipeline = DecisionPipeline(retriever=FakeRetriever())
    async with db_session_factory() as session:
        state = await pipeline.run(session, SITUATION, "general", mode=None)
    assert state[StateKeys.EVAL_MODE] == "full"  # settings.EVAL_MODE по умолчанию
