"""Базовые классы агентов.

BaseAgent — абстрактный интерфейс (process(state)).
Agent — промежуточный класс для агентов с LLM: выбирает движок
(auto | ollama | huggingface), умеет fallback Ollama -> HuggingFace,
пиннит seed=42 для детерминизма, повторяет неудачные вызовы через @retry.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

from backend.core.config import get_settings
from backend.core.exceptions import EngineFallbackTriggeredError, EngineUnavailableError
from backend.utils.async_utils import retry
from backend.utils.http_client import ollama_generate

logger = logging.getLogger("backend.agents")

ENGINE_OLLAMA = "ollama"
ENGINE_HF = "huggingface"
ENGINE_AUTO = "auto"
VALID_ENGINES = (ENGINE_AUTO, ENGINE_OLLAMA, ENGINE_HF)


class BaseAgent(ABC):
    """Все агенты системы. Состояние — dict (см. backend/core/state.py)."""

    name: str = "base"

    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Обработать состояние и дописать свои результаты. Возвращает state."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class Agent(BaseAgent):
    """Агент с LLM-бэкендом."""

    name = "llm-agent"

    def __init__(
        self,
        engine: str | None = None,
        strict_engine: bool = False,
        temperature: float = 0.7,
    ) -> None:
        settings = get_settings()
        self.engine = engine or settings.LLM_ENGINE
        if self.engine not in VALID_ENGINES:
            raise ValueError(f"Неизвестный движок: {self.engine} (допустимо: {VALID_ENGINES})")
        self.strict_engine = strict_engine
        self.temperature = temperature
        self.last_engine_used: str | None = None
        self.last_fallback: bool = False

    @retry(max_attempts=3, delay=0.5, backoff=2.0, exceptions=(EngineUnavailableError,))
    async def _call_ollama(self, prompt: str, system: str | None, seed: int) -> str:
        return await ollama_generate(prompt, system=system, temperature=self.temperature, seed=seed)

    async def _call_hf(self, prompt: str, system: str | None) -> str:
        from backend.utils.hf_llm import generate_hf

        return await generate_hf(prompt, system=system)

    async def _call_llm(self, prompt: str, system: str | None = None) -> str:
        """Вызов LLM с fallback и детерминированным seed=42.

        - engine=ollama: только Ollama; недоступна -> EngineUnavailableError.
        - engine=huggingface: только HF.
        - engine=auto: Ollama, при недоступности fallback на HF
          (strict_engine=True превращает fallback в EngineFallbackTriggeredError).
        """
        settings = get_settings()
        seed = settings.OLLAMA_SEED  # seed=42: детерминизм для воспроизводимости и тестов

        if self.engine == ENGINE_HF:
            self.last_engine_used, self.last_fallback = ENGINE_HF, False
            return await self._call_hf(prompt, system)

        if self.engine == ENGINE_OLLAMA:
            self.last_engine_used, self.last_fallback = ENGINE_OLLAMA, False
            return await self._call_ollama(prompt, system, seed)

        # auto: Ollama -> fallback HF
        try:
            result = await self._call_ollama(prompt, system, seed)
        except EngineUnavailableError as exc:
            if self.strict_engine:
                raise EngineFallbackTriggeredError(
                    "strict_engine=True: fallback на HuggingFace запрещён",
                    details={"cause": exc.message},
                ) from exc
            logger.warning("Ollama недоступна (%s) — fallback на HuggingFace", exc.message)
            self.last_engine_used, self.last_fallback = ENGINE_HF, True
            return await self._call_hf(prompt, system)
        self.last_engine_used, self.last_fallback = ENGINE_OLLAMA, False
        return result

    def mark_engine_state(self, state: dict[str, Any]) -> None:
        """Записать в state, чем реально вызывался LLM (для /health и eval)."""
        state["engine_used"] = self.last_engine_used
        state["engine_fallback"] = self.last_fallback


__all__ = ["BaseAgent", "Agent", "ENGINE_OLLAMA", "ENGINE_HF", "ENGINE_AUTO", "VALID_ENGINES"]
