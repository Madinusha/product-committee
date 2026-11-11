"""Контракт состояния агентов - точка стыка подсистем.

Dict `state` проходит через всех агентов (BaseAgent.process(state)).
Ключи строго из StateKeys - новые ключи добавлять только сюда.

Разделение ответственности:
  - заполнение `retrieved_context` и `glossary_used` - задача RAG-подсистемы
    (этап 7); до её появления допускаются пустые списки,
    Validator/Critic обязаны корректно работать без контекста;
  - ключи-результаты агентов перечислены в AGENT_OUTPUT_KEYS;
  - EPHEMERAL_KEYS сбрасываются Supervisor'ом в начале каждого запуска.
"""
from typing import Any


class StateKeys:
    # вход: заполняется pipeline до первого агента
    SITUATION = "situation"  # текст продуктовой ситуации
    DOMAIN = "domain"  # b2b_saas | marketplace | fintech | media | general
    DECISION_TYPE = "decision_type"  # тип решения: приоритизация / launch / growth / ...

    # RAG-контекст (этап 7)
    GLOSSARY_USED = (
        "glossary_used"  # list[dict]: {term, definition, score} - термины глоссария к ситуации
    )
    RETRIEVED_CONTEXT = (
        "retrieved_context"  # list[dict]: {case_id, situation, decision, outcome, score}
    )
    RETRIEVAL_BACKEND = "retrieval_backend"  # hybrid | bm25 | none (деградация Qdrant -> BM25)

    # черновик и результаты агентов
    DRAFT = "draft"  # черновик решения (генерируется pipeline'ом, этап 9)
    DECISION = "decision"  # рабочее/итоговое решение (пишут Editor и pipeline)
    VALIDATION_PASSED = "validation_passed"  # bool, ставит Validator
    VALIDATOR_ERRORS = "validator_errors"  # list[dict]: {code, message, term?}
    CRITIC_SCORE = "critic_score"  # float | None (None = Critic не смог оценить)
    CRITIC_DETAIL = "critic_detail"  # dict: оценки по критериям 0-10 + обоснование
    CRITIC_SUGGESTIONS = "critic_suggestions"  # list[str]: что исправить
    CRITIC_FAILED = "critic_failed"  # bool: LLM/parse сбой (fallback 5.0 запрещён)
    EDITOR_CHANGED = "editor_changed"  # bool: редактировал ли Editor
    ENGINE_USED = "engine_used"  # ollama | huggingface - чем реально вызывался LLM
    ENGINE_FALLBACK = "engine_fallback"  # bool: был ли fallback Ollama -> HF

    # Supervisor
    ITERATIONS = "iterations"  # int: сколько редактирующих итераций прошло
    SUPERVISOR_STOPPED_REASON = "supervisor_stopped_reason"  # StopReason.*
    REASONING_LOG = "reasoning_log"  # list[str] - НЕ откатывается при rollback
    HISTORY = "history"  # list[dict]: снимки состояния по итерациям


def initial_state(situation: str, domain: str, draft: str | None = None) -> dict[str, Any]:
    """Собрать валидное начальное состояние (без RAG-контекста).

    RAG-подсистема при наличии добавляет GLOSSARY_USED / RETRIEVED_CONTEXT.
    """
    return {
        StateKeys.SITUATION: situation,
        StateKeys.DOMAIN: domain,
        StateKeys.DECISION_TYPE: None,
        StateKeys.DRAFT: draft,
        StateKeys.GLOSSARY_USED: [],
        StateKeys.RETRIEVED_CONTEXT: [],
        StateKeys.REASONING_LOG: [],
        StateKeys.HISTORY: [],
    }
