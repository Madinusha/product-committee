"""BaseAgent/Agent: абстрактность, выбор движка, seed=42, retry, fallback."""
import json

import pytest
import respx

from backend.agents.base import Agent, BaseAgent
from backend.core.exceptions import (
    EngineFallbackTriggeredError,
    EngineUnavailableError,
)
from backend.utils.async_utils import retry

# --- BaseAgent ---

def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent()  # process() обязателен


class DummyAgent(Agent):
    """Конкретная реализация для тестов движка (Agent сам по себе — промежуточный класс)."""

    async def process(self, state):
        return state


# --- движки ---

def test_agent_resolves_engine_from_settings():
    agent = DummyAgent()  # настройки по умолчанию: auto
    assert agent.engine == "auto"


def test_agent_respects_engine_override():
    assert DummyAgent(engine="ollama").engine == "ollama"
    assert DummyAgent(engine="huggingface").engine == "huggingface"


def test_agent_unknown_engine_rejected():
    with pytest.raises(ValueError, match="Неизвестный движок"):
        DummyAgent(engine="chatgpt")


# --- seed=42 пиннится в запросе к Ollama ---

@respx.mock
async def test_ollama_call_pins_seed_42():
    route = respx.post("http://localhost:11434/api/generate").respond(
        json={"response": "ответ"}
    )
    agent = DummyAgent(engine="ollama")
    result = await agent._call_llm("привет")
    assert result == "ответ"
    payload = json.loads(route.calls.last.request.content)
    assert payload["options"]["seed"] == 42
    assert agent.last_engine_used == "ollama"
    assert agent.last_fallback is False


# --- fallback Ollama -> HuggingFace ---

async def test_auto_engine_falls_back_to_hf(monkeypatch):
    async def ollama_down(self, prompt, system, seed):
        raise EngineUnavailableError("Ollama недоступен")

    async def hf_ok(self, prompt, system=None):
        return "hf-answer"

    monkeypatch.setattr(Agent, "_call_ollama", ollama_down)
    monkeypatch.setattr(Agent, "_call_hf", hf_ok)
    agent = DummyAgent(engine="auto")
    assert await agent._call_llm("q") == "hf-answer"
    assert agent.last_fallback is True
    assert agent.last_engine_used == "huggingface"


async def test_strict_engine_forbids_fallback(monkeypatch):
    async def ollama_down(self, prompt, system, seed):
        raise EngineUnavailableError("Ollama недоступен")

    monkeypatch.setattr(Agent, "_call_ollama", ollama_down)
    agent = DummyAgent(engine="auto", strict_engine=True)
    with pytest.raises(EngineFallbackTriggeredError):
        await agent._call_llm("q")


async def test_ollama_only_engine_does_not_fallback(monkeypatch):
    async def ollama_down(self, prompt, system, seed):
        raise EngineUnavailableError("Ollama недоступен")

    async def hf_called(self, prompt, system=None):  # pragma: no cover
        raise AssertionError("HF не должен вызываться для engine=ollama")

    monkeypatch.setattr(Agent, "_call_ollama", ollama_down)
    monkeypatch.setattr(Agent, "_call_hf", hf_called)
    agent = DummyAgent(engine="ollama")
    with pytest.raises(EngineUnavailableError):
        await agent._call_llm("q")


# --- retry ---

async def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    @retry(max_attempts=3, delay=0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise EngineUnavailableError("flaky")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


async def test_retry_exhausted_raises_last_exception():
    calls = {"n": 0}

    @retry(max_attempts=2, delay=0)
    async def always_fails() -> None:
        calls["n"] += 1
        raise EngineUnavailableError(f"fail-{calls['n']}")

    with pytest.raises(EngineUnavailableError, match="fail-2"):
        await always_fails()
    assert calls["n"] == 2
