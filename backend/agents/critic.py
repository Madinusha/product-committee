"""Critic — LLM-as-a-Judge: оценивает решение по критериям 0–10.

Трёхуровневый few-shot: universal (общая логика оценки) + domain
(продуктовые решения) + domain-specific (b2b_saas / marketplace / ...).
Ответ должен быть строгим JSON; парсинг — сканер с балансом скобок.
При сбое LLM или неразборчивом ответе ставится overall_score=None и
critic_failed=True; подставлять запасную оценку (например, 5.0) нельзя —
это исказило бы сравнение ablation-режимов.
"""
import json
import logging
from typing import Any

from backend.agents.base import Agent
from backend.core.constants import CRITIC_MAX_SCORE
from backend.core.exceptions import EngineUnavailableError
from backend.core.state import StateKeys

logger = logging.getLogger("backend.critic")

_SYSTEM = (
    "Ты — строгий судья продуктовых решений (LLM-as-a-Judge). Оцени решение по критериям "
    "0-10: justification (обоснованность: решение следует из ситуации и цифр), "
    "feasibility (реализуемость: ресурсы/сроки достижимы), risks (риски учтены и "
    "смягчены), terminology (корректная продуктовая терминология), overall_score "
    "(итог). Верни СТРОГО JSON без пояснений: "
    '{"justification": 0-10, "feasibility": 0-10, "risks": 0-10, "terminology": 0-10, '
    '"overall_score": 0-10, "suggestions": ["что исправить", ...]}. '
    "Если решение полностью повторяет ситуацию — ставь justification и overall_score <= 2."
)

# --- Тир 1: universal — как выглядит строгая оценка в принципе ---
_FEWSHOT_UNIVERSAL = (
    "Пример 1 (слабое решение):\n"
    "Ситуация: «Отток вырос, надо что-то делать». "
    "Решение: «Надо улучшить продукт и следить за метриками».\n"
    '{"justification": 2, "feasibility": 4, "risks": 1, "terminology": 3, "overall_score": 2, '
    '"suggestions": ["Указать конкретную метрику оттока", "Добавить срок и ответственного"]}\n'
    "Пример 2 (сильное решение):\n"
    "Ситуация: «Отток 8%/мес, цель — 5%, бюджет 2 человека, 1 месяц». "
    "Решение: «Снизить отток с 8% до 5% за 1 месяц: "
    "добавить onboarding-чеклист (отвечает: PM), целевой эффект retention +2.5 п.п., "
    "замер — cohort analysis до/после».\n"
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, "overall_score": 9, "suggestions": []}'
)

# --- Тир 2: domain — специфика продуктовой разработки ---
_FEWSHOT_DOMAIN = (
    "Пример (domain: продуктовые решения): решение без целевой метрики и срока — не решение, а намерение. "
    "Терминология: MVP, RICE, JTBD, CAC, LTV, retention — правильное употребление повышает terminology; "
    "использование терминов вне контекста — понижает."
)

# --- Тир 3: domain-specific ---
_DOMAIN_FEWSHOT: dict[str, str] = {
    "b2b_saas": (
        "Пример (b2b_saas): учитывай цикл сделки, churn по контрактам, вовлечение champions; "
        "решение с внедрением за 1 неделю для enterprise — feasibility не выше 3."
    ),
    "marketplace": (
        "Пример (marketplace): учитывай баланс спроса/предложения; "
        "снижение CAC за счёт комиссии — оценка влияния на supply."
    ),
    "fintech": (
        "Пример (fintech): учитывай регуляторные риски и доверие; "
        "решение без оценки комплаенса — risks не выше 3."
    ),
    "media": (
        "Пример (media): учитывай удержание внимания и рекламную выручку; "
        "рост охвата без retention-метрики — justification не выше 5."
    ),
    "general": "Пример (general): решение должно опираться на цифры ситуации и содержать проверяемый эффект.",
}


def build_critic_prompt(state: dict[str, Any]) -> str:
    domain = state.get(StateKeys.DOMAIN) or "general"
    situation = state.get(StateKeys.SITUATION) or ""
    decision = state.get(StateKeys.DECISION) or state.get(StateKeys.DRAFT) or ""
    glossary = ", ".join(
        (e["term"] if isinstance(e, dict) else str(e))
        for e in (state.get(StateKeys.GLOSSARY_USED) or [])
    )
    context = "\n".join(
        f"- {c.get('situation')} → {c.get('decision')} (результат: {c.get('outcome')}, score={c.get('score')})"
        for c in (state.get(StateKeys.RETRIEVED_CONTEXT) or [])
    )
    parts = [_FEWSHOT_UNIVERSAL, _FEWSHOT_DOMAIN, _DOMAIN_FEWSHOT.get(domain, _DOMAIN_FEWSHOT["general"])]
    return (
        "\n\n".join(parts)
        + f"\n\nОцени решение.\nДомен: {domain}\n"
        + (f"Глоссарий к ситуации: {glossary}\n" if glossary else "")
        + (f"Похожие кейсы:\n{context}\n" if context else "")
        + f"Ситуация: {situation}\nРешение: {decision}\n"
        + "Верни строго JSON по схеме из системного сообщения."
    )


def parse_critic_response(raw: str) -> dict[str, Any] | None:
    """Сканер с балансом скобок: извлекает первый валидный JSON-объект с overall_score."""
    depth = 0
    in_string = False
    escaped = False
    start = -1
    for i, ch in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = raw[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(parsed, dict) and "overall_score" in parsed:
                    return parsed
    return None


class Critic(Agent):
    name = "critic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(temperature=0.2, **kwargs)

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = build_critic_prompt(state)
        try:
            raw = await self._call_llm(prompt, system=_SYSTEM)
        except EngineUnavailableError as exc:
            logger.error("Critic: LLM недоступен: %s", exc.message)
            state[StateKeys.CRITIC_FAILED] = True
            state[StateKeys.CRITIC_SCORE] = None
            state[StateKeys.REASONING_LOG].append("Critic: LLM недоступен — оценка не выставлена")
            self.mark_engine_state(state)
            return state

        self.mark_engine_state(state)
        parsed = parse_critic_response(raw)
        if parsed is None:
            logger.warning("Critic: не удалось разобрать ответ (лог сохранён для отладки)")
            state[StateKeys.CRITIC_FAILED] = True
            state[StateKeys.CRITIC_SCORE] = None
            state[StateKeys.CRITIC_SUGGESTIONS] = []
            state[StateKeys.REASONING_LOG].append("Critic: ответ не разобран — failed=True, score=None")
            return state

        score = parsed.get("overall_score")
        if not isinstance(score, (int, float)) or not 0 <= score <= CRITIC_MAX_SCORE:
            logger.warning("Critic: некорректный overall_score=%r — трактуется как сбой", score)
            state[StateKeys.CRITIC_FAILED] = True
            state[StateKeys.CRITIC_SCORE] = None
            state[StateKeys.CRITIC_SUGGESTIONS] = []
            state[StateKeys.REASONING_LOG].append(f"Critic: некорректный score={score!r} — failed=True")
            return state
        suggestions = parsed.get("suggestions") or []
        if not isinstance(suggestions, list):
            suggestions = []
        state[StateKeys.CRITIC_FAILED] = False
        state[StateKeys.CRITIC_SCORE] = float(score)
        state[StateKeys.CRITIC_DETAIL] = {
            k: parsed.get(k) for k in ("justification", "feasibility", "risks", "terminology")
        }
        state[StateKeys.CRITIC_SUGGESTIONS] = [str(s) for s in suggestions]
        state[StateKeys.REASONING_LOG].append(
            f"Critic: overall={score:.1f}, замечаний: {len(suggestions)}"
        )
        return state
