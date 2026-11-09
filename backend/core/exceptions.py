"""Иерархия исключений приложения.

Базовый класс CommitteeException; исключения группируются по подсистемам.
API-слой маппит их в HTTP-статусы централизованно (см. backend/api/main.py).
"""
from typing import Any


class CommitteeException(Exception):
    """Базовое исключение всех подсистем."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


# --- конфигурация ---
class ConfigError(CommitteeException):
    pass


# --- аутентификация / авторизация ---
class AuthError(CommitteeException):
    pass


class InvalidCredentialsError(AuthError):
    """Неверный email/пароль → 401."""


class TokenError(AuthError):
    """Отсутствующий/просроченный/некорректный токен → 401."""


class UserExistsError(AuthError):
    """Email или username уже заняты → 409."""


class WeakPasswordError(AuthError):
    """Пароль не проходит политику → 400."""


class InsufficientRoleError(AuthError):
    """Роли недостаточно → 403."""


# --- LLM / движки ---
class AgentError(CommitteeException):
    pass


class EngineUnavailableError(AgentError):
    """Бэкенд LLM недоступен (Ollama выключен → fallback на HF)."""


class EngineFallbackTriggeredError(AgentError):
    """Произошёл нежелательный fallback при strict_engine=True."""


class CriticParseError(AgentError):
    """Critic не смог разобрать ответ LLM (не прерывает пайплайн: failed=True)."""


# --- pipeline / API ---
class PipelineError(CommitteeException):
    pass


class UnknownEvalModeError(PipelineError):
    """Неизвестный ablation-режим → 400 со списком поддерживаемых."""


# --- данные ---
class DataError(CommitteeException):
    pass


class DuplicateRecordError(DataError):
    pass


# --- внешние сервисы ---
class ExternalServiceError(CommitteeException):
    pass
