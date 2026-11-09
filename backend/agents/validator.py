"""Validator — rule-based агент, БЕЗ LLM.

Семь базовых проверок + два продуктовых (SMART) правила.
Каждая ошибка: {"code": ..., "message": ..., дополнительные поля}.
Дополнительно: код глоссария готов к подключению similarity-функции
из RAG-подсистемы (этап 7): сначала точное совпадение, затем,
если backend.retrieval.similarity доступен, — эмбеддинг-сходство >= 0.7.
"""
import re
import unicodedata
from typing import Any

from backend.agents.base import BaseAgent
from backend.core.constants import GLOSSARY_MIN_LENGTH, MAX_LATIN_RATIO
from backend.core.state import StateKeys

_PLACEHOLDER_PATTERNS = (
    re.compile(r"\{\s*\w*\s*\}"),  # {value}
    re.compile(r"<[A-Za-zА-Яа-я_ ]{2,30}>"),  # <TODO>
    re.compile(r"\bTODO\b|\bFIXME\b|\bXXX\b"),
    re.compile(r"\bXX\s*%|\b\d+\s*XX\b"),
    re.compile(r"\[\s*\.{3}\s*\]|\[…\]"),
)

_LEAK_MARKERS = (
    "как языковая модель",
    "как ии",
    "я — ии",
    "я - ии",
    "языковая модель",
    "system prompt",
    "допустим, ты",
    "ты — эксперт",
    "ты - эксперт",
    "[inst]",
    "<|",
    "переведи текст",
    "translate",
)

_METRIC_MARKERS = (
    "конверсия", "retention", "удержан", "cac", "ltv", "arpu", "dau", "mau",
    "mrr", "arr", "revenue", "выручк", "метрик", "churn", "отток", "nps",
    "активность", "трафик", "доход",
)

_DEADLINE_OWNER_MARKERS = (
    "срок", "дедлайн", "до ", "недел", "месяц", "квартал", "дней", "дня",
    "спринт", "отвечает", "ответственный", "владелец", "owner", "пилот",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\wа-яё\s]", " ", text, flags=re.UNICODE)).strip()


class Validator(BaseAgent):
    name = "validator"

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        decision: str = state.get(StateKeys.DECISION) or state.get(StateKeys.DRAFT) or ""
        errors: list[dict[str, Any]] = []

        if self._check_non_empty(decision) is None:
            state[StateKeys.VALIDATION_PASSED] = False
            state[StateKeys.VALIDATOR_ERRORS] = [{"code": "empty_decision", "message": "Решение пустое"}]
            state[StateKeys.REASONING_LOG].append("Validator: решение пустое — валидация провалена")
            return state

        errors += self._check_glossary(decision, state)
        errors += self._check_echo(decision, state)
        errors += self._check_placeholders(decision)
        errors += self._check_script(decision)
        errors += self._check_repetitions(decision)
        errors += self._check_prompt_leak(decision)
        errors += self._check_smart(decision)

        state[StateKeys.VALIDATION_PASSED] = not errors
        state[StateKeys.VALIDATOR_ERRORS] = errors
        state[StateKeys.REASONING_LOG].append(
            f"Validator: {'пройдено' if not errors else f'{len(errors)} нарушений'}"
        )
        return state

    # --- 7 базовых проверок ---

    @staticmethod
    def _check_non_empty(decision: str) -> str | None:
        return decision if decision.strip() else None

    def _check_glossary(self, decision: str, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Каждый термин, выданный RAG, должен быть использован в решении."""
        errors = []
        decision_norm = _normalize(decision)
        for entry in state.get(StateKeys.GLOSSARY_USED) or []:
            term = entry["term"] if isinstance(entry, dict) else str(entry)
            if len(term) < GLOSSARY_MIN_LENGTH:
                continue
            if not self._term_present(decision, decision_norm, term):
                errors.append({
                    "code": "glossary_missing",
                    "message": f"Термин глоссария «{term}» не использован в решении",
                    "term": term,
                })
        return errors

    @staticmethod
    def _term_present(decision: str, decision_norm: str, term: str) -> bool:
        if _normalize(term) in decision_norm:
            return True
        # Хук на similarity из RAG-подсистемы (этап 7, порог 0.7).
        try:  # pragma: no cover — подключается Колей
            from backend.retrieval.similarity import term_similarity  # type: ignore[attr-defined]

            return term_similarity(decision, term) >= 0.7
        except ImportError:
            return False

    @staticmethod
    def _check_echo(decision: str, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Решение не должно дословно повторять ситуацию."""
        situation = _normalize(state.get(StateKeys.SITUATION) or "")
        decision_norm = _normalize(decision)
        if not situation or not decision_norm:
            return []
        if decision_norm == situation or decision_norm in situation:
            return [{"code": "echo", "message": "Решение дословно повторяет ситуацию"}]
        return []

    @staticmethod
    def _check_placeholders(decision: str) -> list[dict[str, Any]]:
        for pattern in _PLACEHOLDER_PATTERNS:
            match = pattern.search(decision)
            if match:
                return [{
                    "code": "placeholder",
                    "message": f"Остался плейсхолдер: {match.group(0)!r}",
                    "fragment": match.group(0),
                }]
        return []

    @staticmethod
    def _check_script(decision: str) -> list[dict[str, Any]]:
        """Решение на русском; недопустимы CJK/арабица, латиница <= MAX_LATIN_RATIO."""
        errors = []
        if any(
            unicodedata.name(ch, "").startswith(("CJK", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL"))
            for ch in decision
        ):
            errors.append({"code": "foreign_script", "message": "Обнаружены символы чужой письменности"})
        words = re.findall(r"[A-Za-z]+|[А-Яа-яЁё]+", decision)
        if words:
            latin = [w for w in words if w.isascii()]
            if len(latin) / len(words) > MAX_LATIN_RATIO:
                errors.append({
                    "code": "script_mix",
                    "message": f"Слишком много латиницы: {len(latin)}/{len(words)} слов",
                })
        return errors

    @staticmethod
    def _check_repetitions(decision: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        for sentence in _SENTENCE_SPLIT_RE.split(decision):
            norm = _normalize(sentence)
            if len(norm) < 15:
                continue
            if norm in seen:
                return [{"code": "repetition", "message": f"Предложение повторяется: «{sentence.strip()[:60]}…»"}]
            seen.add(norm)
        return []

    @staticmethod
    def _check_prompt_leak(decision: str) -> list[dict[str, Any]]:
        low = decision.lower()
        for marker in _LEAK_MARKERS:
            if marker in low:
                return [{
                    "code": "prompt_leak",
                    "message": f"Утечка промпта/служебная фраза: {marker!r}",
                    "marker": marker,
                }]
        return []

    # --- продуктовые (SMART) проверки: доменная адаптация ---

    @staticmethod
    def _check_smart(decision: str) -> list[dict[str, Any]]:
        low = decision.lower()
        errors = []
        if not any(ch.isdigit() for ch in decision) and not any(m in low for m in _METRIC_MARKERS):
            errors.append({
                "code": "no_metric",
                "message": "В решении нет измеримой метрики (число/процент/название метрики)",
            })
        if not any(m in low for m in _DEADLINE_OWNER_MARKERS):
            errors.append({
                "code": "no_deadline_or_owner",
                "message": "В решении нет срока или ответственного",
            })
        return errors
