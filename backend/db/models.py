"""ORM-модели (SQLAlchemy 2.0, Mapped).

Пять таблиц проекта: users, glossary, decision_memory, decisions_cache,
hitl_queue. id data-таблиц - детерминированные (SHA-256 от содержимого),
поэтому повторный импорт не создаёт дублей; create_all идемпотентен.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
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


class GlossaryTerm(Base):
    """Термины продукт-менеджмента: подаются в промпт и проверяются Validator'ом."""

    __tablename__ = "glossary"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(term)
    term: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    definition: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(32), default="general", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DecisionMemory(Base):
    """Память решений (аналог translation memory): кейс «ситуация -> решение -> результат»."""

    __tablename__ = "decision_memory"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(situation + decision)
    situation: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)  # чем закончилось
    domain: Mapped[str] = mapped_column(String(32), index=True)
    decision_type: Mapped[str] = mapped_column(String(32), default="general", index=True)
    quality: Mapped[int] = mapped_column(Integer, default=5)  # экспертная 0-10
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DecisionCache(Base):
    """Кэш готовых решений. Вместо NULL в составе ключа - сентинел 'unknown'
    (NULL ломает UNIQUE-ограничение: NULL != NULL)."""

    __tablename__ = "decisions_cache"
    __table_args__ = (
        UniqueConstraint(
            "situation_hash", "domain", "decision_type", "eval_mode", "engine", "model",
            name="uq_cache_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    situation_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256(situation)
    domain: Mapped[str] = mapped_column(String(32), default="unknown")
    decision_type: Mapped[str] = mapped_column(String(32), default="unknown")
    eval_mode: Mapped[str] = mapped_column(String(32), default="unknown")
    engine: Mapped[str] = mapped_column(String(16), default="unknown")
    model: Mapped[str] = mapped_column(String(64), default="unknown")
    decision: Mapped[str] = mapped_column(Text)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class HitlQueue(Base):
    """Очередь Human-in-the-Loop: решения, ожидающие подтверждения продакт-владельца."""

    __tablename__ = "hitl_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    situation: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(32), default="general")
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|approved|rejected
    reviewer_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
