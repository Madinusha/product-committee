"""Validator: 7 базовых проверок + SMART-правила (rule-based, без LLM)."""
import pytest

from backend.agents.validator import Validator
from backend.core.state import StateKeys, initial_state

GOOD_DECISION = (
    "Снизить отток с 8% до 5% за 1 месяц: добавить onboarding-чеклист "
    "(отвечает: продакт-менеджер), целевой эффект retention +2.5 п.п., "
    "замер — когортный анализ до/после."
)


@pytest.fixture()
def validator():
    return Validator()


async def _validate(situation: str, decision: str, **overrides):
    state = initial_state(situation, "general", draft=decision)
    state.update(overrides)
    state[StateKeys.DECISION] = decision
    return await Validator().process(state)


async def test_good_decision_passes(validator):
    state = await _validate("Отток 8%/мес, цель 5%, команда 2 человека, 1 месяц", GOOD_DECISION)
    assert state[StateKeys.VALIDATION_PASSED] is True
    assert state[StateKeys.VALIDATOR_ERRORS] == []


async def test_empty_decision_fails():
    state = await _validate("ситуация", "   ")
    assert state[StateKeys.VALIDATION_PASSED] is False
    assert state[StateKeys.VALIDATOR_ERRORS][0]["code"] == "empty_decision"


async def test_echo_detected():
    situation = "Продукт растёт медленно, надо принять решение о развитии"
    state = await _validate(situation, situation)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "echo" in codes


async def test_placeholders_detected():
    state = await _validate("ситуация", "Внедрить фичу {feature_name} до <дата>, TODO")
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "placeholder" in codes


async def test_foreign_script_detected():
    state = await _validate("ситуация", GOOD_DECISION + " テスト")
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "foreign_script" in codes


async def test_latin_flood_detected():
    decision = "We should improve the product and monitor the metrics of the users every day"
    state = await _validate("ситуация", decision)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "script_mix" in codes


async def test_repetition_detected():
    decision = GOOD_DECISION + " " + GOOD_DECISION
    state = await _validate("Отток 8%, цель 5%", decision)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "repetition" in codes


@pytest.mark.parametrize(
    "leak",
    ["Как языковая модель, я рекомендую снизить отток на 3 п.п. за месяц, отвечает PM",
     "Я — ИИ, рекомендую: метрика NPS +10 за 2 недели, отвечает владелец"],
)
async def test_prompt_leak_detected(leak):
    state = await _validate("ситуация", leak)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "prompt_leak" in codes


async def test_glossary_term_missing_detected():
    state = await _validate(
        "ситуация про монетизацию",
        GOOD_DECISION,
        glossary_used=[{"term": "CAC", "definition": "стоимость привлечения клиента"}],
    )
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "glossary_missing" in codes


async def test_glossary_term_used_passes():
    decision = GOOD_DECISION + " Ожидаем снижение CAC на 15%."
    state = await _validate(
        "ситуация",
        decision,
        glossary_used=[{"term": "CAC", "definition": "стоимость привлечения клиента"}],
    )
    assert state[StateKeys.VALIDATION_PASSED] is True


async def test_short_glossary_terms_ignored():
    state = await _validate("ситуация", GOOD_DECISION, glossary_used=["ок"])
    assert state[StateKeys.VALIDATION_PASSED] is True


async def test_smart_no_metric_detected():
    decision = "Провести исследование и обсудить результаты с командой продукта"
    state = await _validate("ситуация", decision)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "no_metric" in codes


async def test_smart_no_deadline_or_owner_detected():
    decision = "Улучшить retention на 2 п.п. за счёт онбординга"
    state = await _validate("ситуация", decision)
    codes = {e["code"] for e in state[StateKeys.VALIDATOR_ERRORS]}
    assert "no_deadline_or_owner" in codes


async def test_reasoning_log_written():
    state = await _validate("Отток 8%", GOOD_DECISION)
    assert state[StateKeys.REASONING_LOG]
