"""Чистка сырых ответов LLM: ограждения, JSON-обёртки, служебные фразы, артефакты."""
import re

# JSON-обёртки и markdown-ограждения, в которые LLM любит заворачивать ответ
_FENCED_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)
_JSON_WRAP_RE = re.compile(r"^[\s{]*(?:\"?(?:decision|решение)\"?\s*[:=]\s*)?", re.IGNORECASE)

# Служебные префиксы, которые надо отрезать
_PREFIXES = (
    "Решение:",
    "Decision:",
    "Ответ:",
    "Вот решение:",
    "Вот доработанное решение:",
)

_META_PATTERNS = (
    re.compile(r"^\s*(?:как (?:ИИ|языковая модель|модель)|Я (?:ИИ|не могу|не умею)).*$", re.IGNORECASE | re.MULTILINE),
)

# Токенизаторные артефакты
_ARTIFACT_RE = re.compile(r"<\/?[a-z_]+>|^\s*###\s*$", re.MULTILINE)

NO_CHANGE = "NO-CHANGE"


def clean_decision(raw: str) -> str:
    """Чистит ответ LLM: ограждения, JSON-обёртки, префиксы, мета-строки, артефакты."""
    text = raw.strip()
    fenced = _FENCED_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    for prefix in _PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    for pattern in _META_PATTERNS:
        text = pattern.sub("", text)

    text = _ARTIFACT_RE.sub("", text)
    text = _JSON_WRAP_RE.sub("", text, count=1)
    return text.strip()
