"""Pipeline принятия решений (этап 9): 8 ablation-режимов.

Шов с RAG-подсистемой - Protocol Retriever: pipeline знает только,
что есть исполнитель с fill_context(session, state, *, hybrid=True).
Реальная реализация подключается на стыке, тесты подставляют фейки.
Параметр hybrid=False требует лексический канал (BM25) - режим 'tm'
должен работать даже при живом Qdrant, иначе ablation замеряет не то.
"""
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents import Agent, BaseAgent, Supervisor
from backend.core.config import get_settings
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
from backend.core.exceptions import CommitteeException, PipelineError, UnknownEvalModeError
from backend.core.state import StateKeys, initial_state
from backend.utils.llm_processing import clean_decision

logger = logging.getLogger("backend.pipeline")

_ANALYST_SYSTEM = (
    "Ты - опытный продакт-менеджер. На основе ситуации и контекста (термины глоссария, "
    "похожие кейсы с их исходами) предложи конкретное решение. "
    "В решении обязательно: целевая метрика с числом, срок или ответственный, "
    "проверяемый эффект. Используй уместные термины из глоссария. "
    "Не повторяй ситуацию дословно и не оставляй плейсхолдеров. "
    "Верни только текст решения, без префиксов и пояснений."
)


@dataclass(frozen=True)
class ModeConfig:
    """Что включено в ablation-режиме (моделирование состава продуктовой команды)."""

    name: str
    use_glossary: bool  # термины глоссария в промпт
    use_cases: bool  # похожие кейсы в промпт
    channel: str | None  # "bm25" | "hybrid" | None (кейсы не нужны)
    use_critic: bool
    use_editor: bool


_MODE_CONFIG: dict[str, ModeConfig] = {
    BASELINE: ModeConfig(
        BASELINE, use_glossary=False, use_cases=False, channel=None,
        use_critic=False, use_editor=False,
    ),
    GLOSSARY: ModeConfig(
        GLOSSARY, use_glossary=True, use_cases=False, channel=None,
        use_critic=False, use_editor=False,
    ),
    TM: ModeConfig(
        TM, use_glossary=True, use_cases=True, channel="bm25",
        use_critic=False, use_editor=False,
    ),
    TM_HYBRID: ModeConfig(
        TM_HYBRID, use_glossary=True, use_cases=True, channel="hybrid",
        use_critic=False, use_editor=False,
    ),
    AGENTS_ONLY: ModeConfig(
        AGENTS_ONLY, use_glossary=False, use_cases=False, channel=None,
        use_critic=True, use_editor=True,
    ),
    TM_HYBRID_CRITIC: ModeConfig(
        TM_HYBRID_CRITIC, use_glossary=True, use_cases=True, channel="hybrid",
        use_critic=True, use_editor=False,
    ),
    TM_HYBRID_EDITOR: ModeConfig(
        TM_HYBRID_EDITOR, use_glossary=True, use_cases=True, channel="hybrid",
        use_critic=False, use_editor=True,
    ),
    FULL: ModeConfig(
        FULL, use_glossary=True, use_cases=True, channel="hybrid",
        use_critic=True, use_editor=True,
    ),
}


def get_mode_config(mode: str) -> ModeConfig:
    """Конфиг режима; неизвестный режим -> UnknownEvalModeError (API мапит в 400)."""
    if mode not in _MODE_CONFIG:
        raise UnknownEvalModeError(
            f"Неизвестный режим: {mode!r}. Поддерживаются: {', '.join(EVAL_MODES)}"
        )
    return _MODE_CONFIG[mode]


class Retriever(Protocol):
    """Минимальный контракт RAG-подсистемы со стороны pipeline.

    hybrid=True - все доступные каналы; hybrid=False - только лексический
    (BM25). Исполнитель обязан уважать флаг и дописать в state
    GLOSSARY_USED / RETRIEVED_CONTEXT / RETRIEVAL_BACKEND.
    """

    async def fill_context(
        self, session: AsyncSession, state: dict[str, Any], *, hybrid: bool = True
    ) -> dict[str, Any]: ...


def build_decision_prompt(state: dict[str, Any]) -> str:
    """Промпт Analyst: ситуация + глоссарий + кейсы (когда есть)."""
    situation = state.get(StateKeys.SITUATION) or ""
    parts = []

    glossary = state.get(StateKeys.GLOSSARY_USED) or []
    if glossary:
        terms = "\n".join(f"- {g['term']}: {g['definition']}" for g in glossary)
        parts.append(f"Термины глоссария к ситуации:\n{terms}")

    cases = state.get(StateKeys.RETRIEVED_CONTEXT) or []
    if cases:
        case_lines = "\n".join(
            f"- Ситуация: {c['situation']} | Решение: {c['decision']} | "
            f"Результат: {c['outcome']} | score={c['score']}"
            for c in cases
        )
        parts.append(f"Похожие кейсы из памяти решений:\n{case_lines}")

    parts.append(f"Ситуация: {situation}")
    parts.append("Предложи решение по правилам из системного сообщения.")
    return "\n\n".join(parts)


class Analyst(Agent):
    """Agent-генератор черновика решения (четвёртый агент конвейера)."""

    name = "analyst"

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        raw = await self._call_llm(build_decision_prompt(state), system=_ANALYST_SYSTEM)
        self.mark_engine_state(state)
        draft = clean_decision(raw)
        state[StateKeys.DRAFT] = draft
        state[StateKeys.DECISION] = draft
        state[StateKeys.REASONING_LOG].append(
            f"Analyst: черновик сгенерирован (engine={state.get(StateKeys.ENGINE_USED)})"
        )
        return state


class DecisionPipeline:
    """Конвейер: режим -> [retrieval] -> черновик -> [Supervisor] -> решение."""

    def __init__(self, retriever: Retriever | None = None, llm: BaseAgent | None = None) -> None:
        self.retriever = retriever
        self.analyst = llm or Analyst()

    async def generate_draft(self, session: AsyncSession, state: dict[str, Any]) -> str:
        """Черновик решения через Analyst-агента."""
        await self.analyst.process(state)
        return state[StateKeys.DECISION]

    async def run(
        self,
        session: AsyncSession,
        situation: str,
        domain: str,
        mode: str | None = None,
        decision_type: str | None = None,
    ) -> dict[str, Any]:
        """Полный проход pipeline в заданном ablation-режиме."""
        cfg = get_mode_config(mode or get_settings().EVAL_MODE)
        logger.info("Pipeline: режим %s, domain=%s", cfg.name, domain)

        state = initial_state(situation, domain)
        state[StateKeys.DECISION_TYPE] = decision_type
        state[StateKeys.EVAL_MODE] = cfg.name

        if cfg.use_glossary or cfg.use_cases:
            if self.retriever is None:
                raise PipelineError(
                    f"Режим {cfg.name!r} требует retrieval, но исполнитель не подключён"
                )
            await self.retriever.fill_context(session, state, hybrid=cfg.channel == "hybrid")
        else:
            state[StateKeys.RETRIEVAL_BACKEND] = "none"

        await self.generate_draft(session, state)

        if cfg.use_critic or cfg.use_editor:
            supervisor = Supervisor(use_critic=cfg.use_critic, use_editor=cfg.use_editor)
            await supervisor.process(state)
        else:
            # режимы без агентов: решение - черновик как есть (изоляция компонентов)
            state[StateKeys.SUPERVISOR_STOPPED_REASON] = StopReason.SUCCESS
            state[StateKeys.REASONING_LOG].append(
                f"Pipeline: режим {cfg.name} без агентов - черновик возвращается как есть"
            )
        return state


async def safe_perform_decision(
    session: AsyncSession,
    situation: str,
    domain: str,
    mode: str | None = None,
    retriever: Retriever | None = None,
    llm: BaseAgent | None = None,
) -> tuple[dict[str, Any] | None, CommitteeException | None]:
    """Обёртка для API: UnknownEvalModeError пробрасывается (мапится в 400),
    остальные ошибки подсистем возвращаются как результат вместо 500."""
    try:
        pipeline = DecisionPipeline(retriever=retriever, llm=llm)
        state = await pipeline.run(session, situation, domain, mode)
        return state, None
    except UnknownEvalModeError:
        raise
    except CommitteeException as exc:
        logger.error("Pipeline: сбой (%s: %s)", type(exc).__name__, exc.message)
        return None, exc
