"""ORM-модели (SQLAlchemy 2.0, Mapped).

Пять таблиц по проекту: users, glossary, decision_memory, decisions_cache,
hitl_queue. Сейчас реализована users (нужна auth, этап 4). Остальные четыре —
зона Коли (этап 3): id для data-таблиц — детерминированные
(SHA-256 от содержимого), импорт идемпотентный.
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(72))  # bcrypt <= 72 байт
    role: Mapped[str] = mapped_column(String(16), default="user", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
