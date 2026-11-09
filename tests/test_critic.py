"""Critic: парсер JSON с балансом скобок, fallback-правило (никогда не 5.0), few-shot."""
from backend.agents.critic import Critic, build_critic_prompt, parse_critic_response
from backend.core.state import StateKeys, initial_state

VALID_JSON = (
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, '
    '"overall_score": 8.5, "suggestions": ["уточнить срок"]}'
)


# --- парсер ---

def test_parse_clean_json():
    parsed = parse_critic_response(VALID_JSON)
    assert parsed is not None
    assert parsed["overall_score"] == 8.5


def test_parse_json_wrapped_in_text():
    parsed = parse_critic_response(f"Вот моя оценка:\n{VALID_JSON}\nНадеюсь, помог.")
    assert parsed is not None and parsed["overall_score"] == 8.5


def test_parse_json_with_braces_inside_strings():
    raw = '{"justification": 5, "overall_score": 5, "note": "пример {внутри} строки"}'
    assert parse_critic_response(raw)["overall_score"] == 5


def test_parse_garbage_returns_none():
    assert parse_critic_response("К сожалению, я не смог оценить решение.") is None


def test_parse_unbalanced_returns_none():
    assert parse_critic_response('{"overall_score": 8, "broken": [}') is None


def test_parse_object_without_score_rejected():
    # первый объект без overall_score — сканер должен взять второй, валидный
    raw = '{"foo": 1} {"overall_score": 7}'
    assert parse_critic_response(raw)["overall_score"] == 7


# --- few-shot: три тира присутствуют в промпте ---

def test_prompt_contains_three_tiers():
    state = initial_state("Отток 8%", "b2b_saas", draft="решение")
    prompt = build_critic_prompt(state)
    assert "Пример 1" in prompt          # universal
    assert "domain: продуктовые решения" in prompt  # domain
    assert "b2b_saas" in prompt          # domain-specific


def test_prompt_includes_rag_context_when_present():
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.RETRIEVED_CONTEXT] = [
        {"case_id": "1", "situation": "s", "decision": "d", "outcome": "ok", "score": 0.9}
    ]
    state[StateKeys.GLOSSARY_USED] = [{"term": "RICE", "definition": "приоритизация"}]
    prompt = build_critic_prompt(state)
    assert "Похожие кейсы" in prompt
    assert "RICE" in prompt


# --- process с мок-LLM ---

async def test_critic_sets_score_on_valid_json(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return VALID_JSON

    monkeypatch.setattr(Critic, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state = await Critic().process(state)
    assert state[StateKeys.CRITIC_FAILED] is False
    assert state[StateKeys.CRITIC_SCORE] == 8.5
    assert state[StateKeys.CRITIC_SUGGESTIONS] == ["уточнить срок"]


async def test_critic_garbage_means_failed_and_none_score(monkeypatch):
    """Сбой LLM не должен подменяться fallback-оценкой 5.0 (это запрещено заданием)."""
    async def fake_call(self, prompt, system=None):
        return "Не смог оценить"

    monkeypatch.setattr(Critic, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state = await Critic().process(state)
    assert state[StateKeys.CRITIC_FAILED] is True
    assert state[StateKeys.CRITIC_SCORE] is None


async def test_critic_non_numeric_score_is_failure(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return '{"justification": 5, "overall_score": "высокий"}'

    monkeypatch.setattr(Critic, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state = await Critic().process(state)
    assert state[StateKeys.CRITIC_FAILED] is True
    assert state[StateKeys.CRITIC_SCORE] is None


async def test_critic_out_of_range_score_is_failure(monkeypatch):
    async def fake_call(self, prompt, system=None):
        return '{"overall_score": 12}'

    monkeypatch.setattr(Critic, "_call_llm", fake_call)
    state = initial_state("ситуация", "general", draft="решение")
    state[StateKeys.DECISION] = "решение"
    state = await Critic().process(state)
    assert state[StateKeys.CRITIC_FAILED] is True
