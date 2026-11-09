"""Зависимости FastAPI: сессия БД, текущий пользователь, проверка ролей."""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.constants import ALL_ROLES
from backend.core.exceptions import InvalidCredentialsError
from backend.db.engine import get_session
from backend.db.models import User
from backend.services import auth as auth_service

_bearer = HTTPBearer(auto_error=False)

SessionDep = Annotated[AsyncSession, Depends(get_session)]
CredentialsDep = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


async def get_current_user(
    session: SessionDep, credentials: CredentialsDep
) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Требуется Bearer-токен")
    try:
        payload = auth_service.decode_access_token(credentials.credentials)
    except InvalidCredentialsError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, exc.message) from exc
    user = await session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Пользователь не найден или неактивен")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def require_roles(*roles: str):
    """Фабрика зависимостей: require_roles("admin") / require_roles("editor", "admin")."""
    for role in roles:
        if role not in ALL_ROLES:
            raise ValueError(f"Неизвестная роль: {role}")

    async def _checker(user: CurrentUserDep) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Недостаточно прав")
        return user

    return _checker


require_admin = require_roles("admin")
require_editor = require_roles("editor", "admin")
