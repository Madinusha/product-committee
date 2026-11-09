"""Общие константы домена и системы."""
from enum import StrEnum

# --- роли пользователей ---
ROLE_USER = "user"
ROLE_EDITOR = "editor"
ROLE_ADMIN = "admin"
ALL_ROLES = (ROLE_USER, ROLE_EDITOR, ROLE_ADMIN)

# --- требования к паролю ---
PASSWORD_MIN_LENGTH = 8
BCRYPT_ROUNDS = 12

# --- 8 ablation-режимов (этап 9): состав «команды» в эксперименте ---
GLOSSARY = "glossary"
TM = "tm"
TM_HYBRID = "tm_hybrid"
AGENTS_ONLY = "agents_only"
TM_HYBRID_CRITIC = "tm_hybrid_critic"
TM_HYBRID_EDITOR = "tm_hybrid_editor"
BASELINE = "baseline"
FULL = "full"
EVAL_MODES = (
    BASELINE,
    GLOSSARY,
    TM,
    TM_HYBRID,
    AGENTS_ONLY,
    TM_HYBRID_CRITIC,
    TM_HYBRID_EDITOR,
    FULL,
)


class Domain(StrEnum):
    """Продуктовые домены (влияют на few-shot Critic и промпты)."""

    GENERAL = "general"
    B2B_SAAS = "b2b_saas"
    MARKETPLACE = "marketplace"
    FINTECH = "fintech"
    MEDIA = "media"


KNOWN_DOMAINS = tuple(d.value for d in Domain)

# --- параметры качества решения ---
SUCCESS_SCORE = 7.0  # критерий успеха: валидация + score >= 7.0
CRITIC_MAX_SCORE = 10.0


class StopReason(StrEnum):
    """Причина остановки Supervisor'а — только из фиксированного набора."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REGRESSION_ROLLBACK = "regression_rollback"
    VALIDATION_FAILED = "validation_failed"
    CRITIC_UNAVAILABLE = "critic_unavailable"


# --- бюджет итераций Supervisor'а: (score < 4 → 3, < 7 → 2, >= 7 c issues → 1, чисто → 0) ---
BUDGET_CRITICAL = 3
BUDGET_UNCERTAIN = 2
BUDGET_POLISH = 1
BUDGET_CLEAN = 0

# --- правила Validator'а ---
MAX_LATIN_RATIO = 0.25  # доля латинских слов в решении (решения — на русском)
GLOSSARY_MIN_LENGTH = 3  # термины короче 3 символов не проверяем
