"""Editor: пропуск при critic_failed / без замечаний, правило NO-CHANGE."""

from backend.agents.editor import Editor
from backend.core.state import StateKeys, initial_state
from tests.conftest import make_llm_mock


async def test_editor_skips_when_critic_failed(monkeypatch):
    calls = []
    monkeypatch.setattr(Editor, "_call_llm", make_llm_mock({"editor": "НОВОЕ РЕШЕНИЕ"}, calls))
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state[StateKeys.CRITIC_FAILED] = True
    state[StateKeys.CRITIC_SUGGESTIONS] = ["исправить"]
    state = await Editor().process(state)
    assert state[StateKeys.EDITOR_CHANGED] is False
    assert state[StateKeys.DECISION] == "решение"
    assert calls == []  # LLM не вызывался


async def test_editor_skips_without_issues(monkeypatch):
    calls = []
    monkeypatch.setattr(Editor, "_call_llm", make_llm_mock({}, calls))
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state[StateKeys.CRITIC_SUGGESTIONS] = []
    state[StateKeys.CRITIC_FAILED] = False
    state = await Editor().process(state)
    assert state[StateKeys.EDITOR_CHANGED] is False
    assert calls == []


async def test_editor_applies_changes(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return "Снизить отток с 8% до 5% за 1 месяц (отвечает PM), retention +2.5 п.п."

    monkeypatch.setattr(Editor, "_call_llm", fake_call)
    state = initial_state("Отток 8%", "general", draft="улучшить продукт")
    state[StateKeys.DECISION] = "улучшить продукт"
    state[StateKeys.CRITIC_FAILED] = False
    state[StateKeys.CRITIC_SUGGESTIONS] = ["добавить метрику и срок"]
    state = await Editor().process(state)
    assert state[StateKeys.EDITOR_CHANGED] is True
    assert "retention" in state[StateKeys.DECISION]
    assert state[StateKeys.EDITOR_CHANGED] is True


async def test_editor_respects_no_change_marker(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return "NO-CHANGE"

    monkeypatch.setattr(Editor, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state[StateKeys.CRITIC_SUGGESTIONS] = ["какое-то замечание"]
    state[StateKeys.CRITIC_FAILED] = False
    state = await Editor().process(state)
    assert state[StateKeys.EDITOR_CHANGED] is False
    assert state[StateKeys.DECISION] == "решение"


async def test_editor_strips_fenced_blocks(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return "```markdown\nНовое решение: метрика CAC -15% за квартал, отвечает маркетинг.\n```"

    monkeypatch.setattr(Editor, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="старое")
    state[StateKeys.DECISION] = "старое"
    state[StateKeys.CRITIC_SUGGESTIONS] = ["замечание"]
    state[StateKeys.CRITIC_FAILED] = False
    state = await Editor().process(state)
    assert "```" not in state[StateKeys.DECISION]
    assert "CAC" in state[StateKeys.DECISION]
