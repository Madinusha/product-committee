"""Supervisor: адаптивный бюджет 3/2/1/0, snapshot/rollback, is_improvement, ablation-флаги."""
from backend.agents.critic import Critic
from backend.agents.editor import Editor
from backend.agents.supervisor import Supervisor
from backend.core.constants import StopReason
from backend.core.state import StateKeys, initial_state

SITUATION = "Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц"
DRAFT = "Улучшить продукт"
IMPROVED = (
    "Снизить отток с 8% до 5% за 1 месяц: добавить onboarding-чеклист "
    "(отвечает: продакт-менеджер), целевой эффект retention +2.5 п.п., "
    "замер — когортный анализ до/после."
)

CRITIC_5 = (
    '{"justification": 4, "feasibility": 5, "risks": 4, "terminology": 6, '
    '"overall_score": 5.0, "suggestions": ["добавить метрику и срок"]}'
)
CRITIC_8 = (
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, '
    '"overall_score": 8.0, "suggestions": []}'
)
CRITIC_6 = '{"overall_score": 6.0, "suggestions": ["уточнить"]}'
CRITIC_3 = '{"overall_score": 3.0, "suggestions": ["плохо"]}'

# --- адаптивный бюджет ---

def test_adaptive_budget_mapping():
    budget = Supervisor._adaptive_budget
    assert budget(3.0, True) == 3    # score < 4
    assert budget(6.5, True) == 2    # score < 7
    assert budget(8.0, True) == 1    # >= 7, есть замечания
    assert budget(8.0, False) == 0   # чисто
    assert budget(None, True) == 2   # оценка неясна


def test_is_improvement_regression_rejected():
    before = {StateKeys.CRITIC_SCORE: 7.0, StateKeys.VALIDATION_PASSED: True}
    after = {StateKeys.CRITIC_SCORE: 6.0, StateKeys.VALIDATION_PASSED: True, StateKeys.CRITIC_FAILED: False}
    assert Supervisor._is_improvement(before, after) is False


def test_is_improvement_critic_failed_rejected():
    before = {StateKeys.CRITIC_SCORE: 6.0, StateKeys.VALIDATION_PASSED: True}
    after = {StateKeys.CRITIC_SCORE: None, StateKeys.VALIDATION_PASSED: True, StateKeys.CRITIC_FAILED: True}
    assert Supervisor._is_improvement(before, after) is False


def test_is_improvement_validation_downgrade_rejected():
    before = {StateKeys.CRITIC_SCORE: 6.0, StateKeys.VALIDATION_PASSED: True}
    after = {StateKeys.CRITIC_SCORE: 6.5, StateKeys.VALIDATION_PASSED: False, StateKeys.CRITIC_FAILED: False}
    assert Supervisor._is_improvement(before, after) is False


def test_is_improvement_accepts_real_improvement():
    before = {StateKeys.CRITIC_SCORE: 5.0, StateKeys.VALIDATION_PASSED: True}
    after = {StateKeys.CRITIC_SCORE: 8.0, StateKeys.VALIDATION_PASSED: True, StateKeys.CRITIC_FAILED: False}
    assert Supervisor._is_improvement(before, after) is True


# --- snapshot / restore ---

def test_restore_keeps_reasoning_log():
    state = {
        StateKeys.DECISION: "исходное решение",
        StateKeys.CRITIC_SCORE: 7.0,
        StateKeys.REASONING_LOG: ["шаг 1"],
        StateKeys.HISTORY: [{"i": 0}],
    }
    snapshot = Supervisor._snapshot(state)
    state[StateKeys.DECISION] = "испорченное решение"
    state[StateKeys.CRITIC_FAILED] = True
    state[StateKeys.REASONING_LOG].append("шаг 2 (испорченная итерация)")
    Supervisor._restore(state, snapshot)
    assert state[StateKeys.DECISION] == "исходное решение"   # решение откатилось
    assert StateKeys.CRITIC_FAILED not in state
    assert state[StateKeys.REASONING_LOG] == ["шаг 1", "шаг 2 (испорченная итерация)"]  # журнал НЕ откатился
    assert state[StateKeys.HISTORY] == [{"i": 0}]


def test_snapshot_is_deep_copy():
    state = {StateKeys.CRITIC_SUGGESTIONS: ["a"], StateKeys.REASONING_LOG: []}
    snapshot = Supervisor._snapshot(state)
    state[StateKeys.CRITIC_SUGGESTIONS].append("b")
    assert snapshot[StateKeys.CRITIC_SUGGESTIONS] == ["a"]


# --- полный цикл с мок-LLM ---

async def _run_supervisor(critic_responses: list[str], editor_responses: list[str], monkeypatch, **flags):
    calls = []
    from tests.conftest import make_llm_mock

    monkeypatch.setattr(Critic, "_call_llm", make_llm_mock({"critic": critic_responses}, calls))
    monkeypatch.setattr(Editor, "_call_llm", make_llm_mock({"editor": editor_responses}, calls))
    supervisor = Supervisor(use_critic=flags.pop("use_critic", True), use_editor=flags.pop("use_editor", True))
    state = initial_state(SITUATION, "general", draft=DRAFT)
    state[StateKeys.DECISION] = DRAFT
    await supervisor.process(state)
    return state, calls


async def test_full_loop_improves_and_succeeds(monkeypatch):
    state, _ = await _run_supervisor([CRITIC_5, CRITIC_8], [IMPROVED], monkeypatch)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert state[StateKeys.CRITIC_SCORE] == 8.0
    assert state[StateKeys.ITERATIONS] == 1
    assert state[StateKeys.DECISION] == IMPROVED
    assert state[StateKeys.VALIDATION_PASSED] is True


async def test_regression_is_rolled_back(monkeypatch):
    """Editor изменил решение, но оценка упала 6.5 -> 3.0: откат к исходной версии."""
    state, _ = await _run_supervisor(
        [CRITIC_6, CRITIC_3], ["испорченная версия решения, отвечаю за метрику 5%"], monkeypatch
    )
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.REGRESSION_ROLLBACK
    assert state[StateKeys.DECISION] == DRAFT  # откат к исходнику
    assert any("регрессия" in entry for entry in state[StateKeys.REASONING_LOG])  # журнал сохранился


async def test_budget_exhausted_without_success(monkeypatch):
    """Editor правит, но оценка не дотягивает до 7: BUDGET_EXHAUSTED, остаётся последняя версия."""
    state, _ = await _run_supervisor(
        [CRITIC_3, CRITIC_5, CRITIC_6, CRITIC_6], [IMPROVED, IMPROVED, IMPROVED], monkeypatch
    )
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.BUDGET_EXHAUSTED
    assert state[StateKeys.ITERATIONS] == 3


async def test_clean_high_score_zero_iterations(monkeypatch):
    """score 8.0, замечаний нет, валидация пройдена: budget=0, итераций нет, SUCCESS."""
    from tests.conftest import make_llm_mock

    good_decision = (
        "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
        "замер — когортный анализ."
    )
    calls = []
    monkeypatch.setattr(Critic, "_call_llm", make_llm_mock({"critic": [CRITIC_8]}, calls))
    monkeypatch.setattr(Editor, "_call_llm", make_llm_mock({}, calls))
    supervisor = Supervisor()
    state = initial_state(SITUATION, "general", draft=good_decision)
    state[StateKeys.DECISION] = good_decision
    await supervisor.process(state)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert state[StateKeys.ITERATIONS] == 0
    assert all(call["agent"] != "editor" for call in calls)


async def test_critic_unavailable_stops_pipeline(monkeypatch):
    state, _ = await _run_supervisor(["Совсем не JSON"], ["что-то"], monkeypatch)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.CRITIC_UNAVAILABLE
    assert state[StateKeys.CRITIC_SCORE] is None


# --- ablation-флаги ---

async def test_use_critic_false_validator_only_success(monkeypatch):
    from tests.conftest import make_llm_mock

    good_decision = (
        "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
        "замер — когортный анализ."
    )
    calls = []
    monkeypatch.setattr(Critic, "_call_llm", make_llm_mock({}, calls))
    supervisor = Supervisor(use_critic=False)
    state = initial_state(SITUATION, "general", draft=good_decision)
    state[StateKeys.DECISION] = good_decision
    await supervisor.process(state)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.SUCCESS
    assert StateKeys.CRITIC_SCORE not in state  # Critic не запускался
    assert all(call["agent"] != "critic" for call in calls)


async def test_use_critic_false_validation_failed(monkeypatch):
    from tests.conftest import make_llm_mock

    calls = []
    monkeypatch.setattr(Critic, "_call_llm", make_llm_mock({}, calls))
    supervisor = Supervisor(use_critic=False)
    state = initial_state(SITUATION, "general", draft=DRAFT)
    state[StateKeys.DECISION] = DRAFT  # "Улучшить продукт" — валидацию не пройдёт
    await supervisor.process(state)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.VALIDATION_FAILED
    assert all(call["agent"] != "critic" for call in calls)


async def test_use_editor_false_no_iterations(monkeypatch):
    state, calls = await _run_supervisor([CRITIC_5], [IMPROVED], monkeypatch, use_editor=False)
    assert state[StateKeys.ITERATIONS] == 0  # критик оценил, но править некому
    assert all(call["agent"] != "editor" for call in calls)
    assert state[StateKeys.SUPERVISOR_STOPPED_REASON] == StopReason.BUDGET_EXHAUSTED


async def test_ephemeral_keys_reset_between_runs(monkeypatch):
    """Повторный запуск Supervisor'а на том же state сбрасывает эфемерные ключи."""
    state, _ = await _run_supervisor([CRITIC_8], [], monkeypatch)
    assert state[StateKeys.CRITIC_SCORE] == 8.0
    state, _ = await _run_supervisor([CRITIC_5], [IMPROVED], monkeypatch)
    # после второго запуска старые значения не должны остаться
    assert state[StateKeys.CRITIC_SCORE] == 5.0
