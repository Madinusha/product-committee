"""Сервис аутентификации: bcrypt напрямую + JWT (HS256) — без промежуточных библиотек."""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from backend.core.config import get_settings
from backend.core.constants import BCRYPT_ROUNDS, PASSWORD_MIN_LENGTH, ROLE_USER
from backend.core.exceptions import InvalidCredentialsError, WeakPasswordError

logger = logging.getLogger("backend.auth")


def validate_password(password: str) -> None:
    """Политика: 8-72 символа (72 — лимит bcrypt), минимум одна цифра и одна заглавная."""
    if len(password) < PASSWORD_MIN_LENGTH:
        raise WeakPasswordError(
            f"Пароль должен быть не короче {PASSWORD_MIN_LENGTH} символов"
        )
    if len(password) > 72:
        raise WeakPasswordError("Пароль не может быть длиннее 72 символов (лимит bcrypt)")
    if not any(c.isdigit() for c in password):
        raise WeakPasswordError("Пароль должен содержать хотя бы одну цифру")
    if not any(c.isupper() for c in password):
        raise WeakPasswordError("Пароль должен содержать хотя бы одну заглавную букву")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: int, role: str = ROLE_USER) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_TTL_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Бросает InvalidCredentialsError при любой проблеме с токеном."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise InvalidCredentialsError("Токен истёк") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidCredentialsError("Некорректный токен") from exc
    if "sub" not in payload:
        raise InvalidCredentialsError("В токене нет субъекта")
    return payload
