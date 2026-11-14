"""Supervisor — оркестратор цикла Validator -> Critic -> (Editor -> Validator -> Critic)*.

Ключевые механизмы:
- адаптивный бюджет итераций по первой оценке Critic'а:
  score < 4 -> 3, < 7 -> 2, >= 7 с замечаниями -> 1, чисто -> 0;
- snapshot/restore: отклонение НЕулучшенных итераций (rollback). Почему rollback,
  а не просто max_iterations: Editor может ухудшить решение — snapshot гарантирует
  монотонность качества; reasoning_log откату не подлежит (журнал сохраняется целиком);
- флаги use_critic / use_editor — 4 ablation-сценария;
- stopped_reason — только из фиксированного набора StopReason.
"""
import copy
import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.agents.critic import Critic
from backend.agents.editor import Editor
from backend.agents.validator import Validator
from backend.core.constants import (
    BUDGET_CLEAN,
    BUDGET_CRITICAL,
    BUDGET_POLISH,
    BUDGET_UNCERTAIN,
    SUCCESS_SCORE,
    StopReason,
)
from backend.core.state import StateKeys

logger = logging.getLogger("backend.supervisor")

# Ключи, которые не входят в snapshot и переживают rollback
_PRESERVED_KEYS = (StateKeys.REASONING_LOG, StateKeys.HISTORY)

# Эфемерные ключи: сбрасываются в начале каждого запуска
_EPHEMERAL_KEYS = (
    StateKeys.VALIDATION_PASSED,
    StateKeys.VALIDATOR_ERRORS,
    StateKeys.CRITIC_SCORE,
    StateKeys.CRITIC_DETAIL,
    StateKeys.CRITIC_SUGGESTIONS,
    StateKeys.CRITIC_FAILED,
    StateKeys.EDITOR_CHANGED,
    StateKeys.ITERATIONS,
    StateKeys.ENGINE_USED,
    StateKeys.ENGINE_FALLBACK,
)


class Supervisor(BaseAgent):
    name = "supervisor"

    def __init__(
        self,
        validator: Validator | None = None,
        critic: Critic | None = None,
        editor: Editor | None = None,
        use_critic: bool = True,
        use_editor: bool = True,
    ) -> None:
        self.validator = validator or Validator()
        self.critic = critic or Critic()
        self.editor = editor or Editor()
        self.use_critic = use_critic
        self.use_editor = use_editor

    # --- публичный конвейер ---

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        self._reset_ephemeral(state)
        log = state[StateKeys.REASONING_LOG]
        log.append(f"Supervisor: запуск (use_critic={self.use_critic}, use_editor={self.use_editor})")

        await self.validator.process(state)

        if not self.use_critic:
            # ablation: только правила - успех определяется валидацией
            passed = state[StateKeys.VALIDATION_PASSED]
            if not passed and self.use_editor:
                # режим "Editor без Critic": судьи нет, правим по нарушениям правил
                log.append("Supervisor: Editor без Critic - правка по нарушениям Validator'а")
                await self.editor.process(state)
                await self.validator.process(state)
                state[StateKeys.ITERATIONS] = int(state.get(StateKeys.ITERATIONS) or 0) + 1
                passed = state[StateKeys.VALIDATION_PASSED]
            state[StateKeys.SUPERVISOR_STOPPED_REASON] = (
                StopReason.SUCCESS if passed else StopReason.VALIDATION_FAILED
            )
            log.append(f"Supervisor: стоп {state[StateKeys.SUPERVISOR_STOPPED_REASON]} (без Critic)")
            return state

        await self.critic.process(state)
        if state.get(StateKeys.CRITIC_FAILED):
            state[StateKeys.SUPERVISOR_STOPPED_REASON] = StopReason.CRITIC_UNAVAILABLE
            log.append("Supervisor: стоп — Critic недоступен, оценить качество нельзя")
            return state

        score = state[StateKeys.CRITIC_SCORE]
        has_issues = bool(state.get(StateKeys.CRITIC_SUGGESTIONS)) or not state.get(StateKeys.VALIDATION_PASSED)
        budget = self._adaptive_budget(score, has_issues)
        log.append(f"Supervisor: budget={budget} (score={score})")

        stopped: StopReason | None = None
        for _ in range(budget):
            if not self.use_editor:
                # ablation: Critic оценивает, но править некому
                break
            snapshot = self._snapshot(state)
            await self.editor.process(state)
            state[StateKeys.ITERATIONS] = int(state.get(StateKeys.ITERATIONS) or 0) + 1
            await self.validator.process(state)
            await self.critic.process(state)

            if self._is_success(state):
                stopped = StopReason.SUCCESS
                break
            if not self._is_improvement(snapshot, state):
                self._restore(state, snapshot)
                stopped = StopReason.REGRESSION_ROLLBACK
                log.append("Supervisor: регрессия — откат к предыдущей версии решения")
                break

        if stopped is None:
            stopped = (
                StopReason.SUCCESS
                if self._is_success(state)
                else StopReason.BUDGET_EXHAUSTED
            )
        state[StateKeys.SUPERVISOR_STOPPED_REASON] = stopped
        log.append(f"Supervisor: стоп {stopped}, итераций {state.get(StateKeys.ITERATIONS, 0)}")
        return state

    # --- правила бюджета ---

    @staticmethod
    def _adaptive_budget(score: float | None, has_issues: bool) -> int:
        if score is None:
            return BUDGET_UNCERTAIN  # оценка неясна — средний запас
        if score < 4:
            return BUDGET_CRITICAL
        if score < 7:
            return BUDGET_UNCERTAIN
        return BUDGET_POLISH if has_issues else BUDGET_CLEAN

    @staticmethod
    def _is_success(state: dict[str, Any]) -> bool:
        """Критерий успеха: валидация пройдена и score >= 7.0."""
        return bool(state.get(StateKeys.VALIDATION_PASSED)) and (
            state.get(StateKeys.CRITIC_SCORE) or 0
        ) >= SUCCESS_SCORE

    @staticmethod
    def _is_improvement(before: dict[str, Any], after: dict[str, Any]) -> bool:
        """Итерация считается улучшением, только если ничего не стало хуже."""
        if after.get(StateKeys.CRITIC_FAILED):
            return False  # пропавшая оценка — не улучшение
        old_score, new_score = before.get(StateKeys.CRITIC_SCORE), after.get(StateKeys.CRITIC_SCORE)
        if old_score is not None and new_score is not None and new_score < old_score:
            return False  # регрессия по оценке
        if before.get(StateKeys.VALIDATION_PASSED) and not after.get(StateKeys.VALIDATION_PASSED):
            return False  # валидация сломалась
        return True

    # --- snapshot / restore ---

    @staticmethod
    def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in state.items() if k not in _PRESERVED_KEYS}

    @staticmethod
    def _restore(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
        preserved = {k: state[k] for k in _PRESERVED_KEYS if k in state}
        state.clear()
        state.update(snapshot)
        state.update(preserved)  # reasoning_log/history откату не подлежат

    @staticmethod
    def _reset_ephemeral(state: dict[str, Any]) -> None:
        state.setdefault(StateKeys.REASONING_LOG, [])
        state.setdefault(StateKeys.HISTORY, [])
        for key in _EPHEMERAL_KEYS:
            state.pop(key, None)
        state[StateKeys.ITERATIONS] = 0
