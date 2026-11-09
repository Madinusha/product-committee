"""Editor — дорабатывает решение по замечаниям Critic'а (temperature 0.3).

Пропускает работу при critic_failed или отсутствии замечаний.
Правило NO-CHANGE: если править нечего, LLM возвращает маркер NO-CHANGE,
а не выдумывает изменения.
"""
import logging
from typing import Any

from backend.agents.base import Agent
from backend.core.state import StateKeys
from backend.utils.llm_processing import NO_CHANGE, clean_decision

logger = logging.getLogger("backend.editor")

_SYSTEM = (
    "Ты — продуктовый редактор. Тебе дают ситуацию, решение и список замечаний судьи. "
    "Перепиши решение так, чтобы устранить ВСЕ замечания, сохранив смысл и факты. "
    "Сохраняй требования: целевая метрика, срок/ответственный, продуктовая терминология. "
    f"Если править нечего — верни ровно {NO_CHANGE} и ничего кроме него."
)


def build_editor_prompt(state: dict[str, Any]) -> str:
    situation = state.get(StateKeys.SITUATION) or ""
    decision = state.get(StateKeys.DECISION) or state.get(StateKeys.DRAFT) or ""
    suggestions = "\n".join(f"- {s}" for s in state.get(StateKeys.CRITIC_SUGGESTIONS) or [])
    validator_errors = "\n".join(
        f"- {e.get('message')}" for e in state.get(StateKeys.VALIDATOR_ERRORS) or []
    )
    return (
        f"Ситуация: {situation}\n\nТекущее решение:\n{decision}\n\n"
        f"Замечания судьи:\n{suggestions or '- (нет)'}\n"
        + (f"Нарушения правил:\n{validator_errors}\n" if validator_errors else "")
        + "\nВерни только финальный текст решения (или NO-CHANGE)."
    )


class Editor(Agent):
    name = "editor"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("temperature", 0.3)  # низкая температура: точечная правка
        super().__init__(**kwargs)

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        suggestions = state.get(StateKeys.CRITIC_SUGGESTIONS) or []
        validator_errors = state.get(StateKeys.VALIDATOR_ERRORS) or []
        has_issues = bool(suggestions or validator_errors)

        if state.get(StateKeys.CRITIC_FAILED):
            state[StateKeys.EDITOR_CHANGED] = False
            state[StateKeys.REASONING_LOG].append("Editor: пропущен (critic_failed=True)")
            return state
        if not has_issues:
            state[StateKeys.EDITOR_CHANGED] = False
            state[StateKeys.REASONING_LOG].append("Editor: замечаний нет — правка не требуется")
            return state

        current = state.get(StateKeys.DECISION) or state.get(StateKeys.DRAFT) or ""
        raw = await self._call_llm(build_editor_prompt(state), system=_SYSTEM)
        self.mark_engine_state(state)
        cleaned = clean_decision(raw)

        if not cleaned or cleaned.upper() == NO_CHANGE or cleaned == current:
            state[StateKeys.EDITOR_CHANGED] = False
            state[StateKeys.REASONING_LOG].append("Editor: NO-CHANGE")
            return state

        state[StateKeys.DECISION] = cleaned
        state[StateKeys.EDITOR_CHANGED] = True
        state[StateKeys.REASONING_LOG].append("Editor: решение доработано")
        return state
