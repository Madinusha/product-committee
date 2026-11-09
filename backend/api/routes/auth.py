"""Роуты аутентификации: register / login / me.

Коды: 201 регистрация, 401 неверные данные/токен, 409 дубликат, 400 слабый пароль.
Первый зарегистрированный пользователь получает роль admin (иначе некому
раздавать права; продублировано скриптом scripts/create_admin.py, этап 6).
"""
import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from backend.api.deps import CurrentUserDep, SessionDep
from backend.core.constants import ROLE_ADMIN, ROLE_USER
from backend.core.exceptions import UserExistsError
from backend.core.models import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.db.models import User
from backend.services import auth as auth_service

logger = logging.getLogger("backend.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, session: SessionDep) -> User:
    auth_service.validate_password(payload.password)
    dup = await session.execute(
        select(User).where(
            (User.email == payload.email.lower()) | (User.username == payload.username)
        )
    )
    if dup.scalar_one_or_none() is not None:
        raise UserExistsError("Email или username уже заняты")

    count_all = await session.scalar(select(User.id).limit(1))
    role = ROLE_ADMIN if count_all is None else ROLE_USER

    user = User(
        email=payload.email.lower(),
        username=payload.username,
        password_hash=auth_service.hash_password(payload.password),
        role=role,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info("Зарегистрирован пользователь id=%s role=%s", user.id, user.role)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: SessionDep) -> TokenResponse:
    user = (
        await session.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()
    if user is None or not auth_service.verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный email или пароль")
    if not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Пользователь неактивен")
    return TokenResponse(access_token=auth_service.create_access_token(user.id, user.role))


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUserDep) -> User:
    return user
