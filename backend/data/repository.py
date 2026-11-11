"""Лёгкие read-репозитории для retrieval: DTO без ORM-объектов, чтобы
индексы не держали сессии и объекты не отвязывались от сессии."""
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import DecisionMemory, GlossaryTerm


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    term: str
    definition: str
    domain: str


@dataclass(frozen=True)
class CaseEntry:
    id: str
    situation: str
    decision: str
    outcome: str
    domain: str
    decision_type: str
    quality: int


async def get_glossary(session: AsyncSession, domain: str | None = None) -> list[GlossaryEntry]:
    """Все термины (или одного домена; general-термины показываются всегда)."""
    stmt = select(GlossaryTerm)
    if domain:
        stmt = stmt.where(GlossaryTerm.domain.in_([domain, "general"]))
    rows = await session.execute(stmt)
    return [
        GlossaryEntry(r.id, r.term, r.definition, r.domain)
        for r in rows.scalars().all()
    ]


async def get_cases(session: AsyncSession, domain: str | None = None) -> list[CaseEntry]:
    """Все кейсы Decision Memory (или одного домена)."""
    stmt = select(DecisionMemory)
    if domain:
        stmt = stmt.where(DecisionMemory.domain == domain)
    rows = await session.execute(stmt)
    return [
        CaseEntry(r.id, r.situation, r.decision, r.outcome, r.domain, r.decision_type, r.quality)
        for r in rows.scalars().all()
    ]
