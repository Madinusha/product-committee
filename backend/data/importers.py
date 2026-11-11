"""Импорт данных: глоссарий и память решений.

Идемпотентность: id вычисляется как SHA-256 от нормализованного содержимого
(или uuid5 в Qdrant), повторный импорт пропускает существующие записи.
Эквивалент `ON CONFLICT DO NOTHING`: сначала один SELECT существующих id,
затем вставка только недостающего (переносимо между SQLite и PostgreSQL).
"""
import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import DecisionMemory, GlossaryTerm

logger = logging.getLogger("backend.data")


def _normalize(text: str) -> str:
    """Нормализация для детерминированных id: регистр и пробелы не влияют на id."""
    # без нормализации "CAC " и "cac" дали бы разные sha256 и дубль в глоссарии
    return re.sub(r"\s+", " ", text.strip().lower())


def glossary_id(term: str) -> str:
    return hashlib.sha256(_normalize(term).encode("utf-8")).hexdigest()


def case_id(situation: str, decision: str) -> str:
    return hashlib.sha256(
        (_normalize(situation) + "\n" + _normalize(decision)).encode("utf-8")
    ).hexdigest()


@dataclass
class ImportReport:
    created: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _validate_row(row: dict[str, Any], required: tuple[str, ...], source: str) -> str | None:
    missing = [key for key in required if not str(row.get(key) or "").strip()]
    if missing:
        return f"{source}: нет обязательных полей {missing}"
    return None


async def import_glossary(session: AsyncSession, rows: Iterable[dict[str, Any]]) -> ImportReport:
    """Импорт терминов глоссария: {term, definition, domain?}."""
    report = ImportReport()
    rows = list(rows)
    existing = set(
        await session.scalars(
            select(GlossaryTerm.id).where(
                GlossaryTerm.id.in_([glossary_id(r["term"]) for r in rows])
            )
        )
    )
    for i, row in enumerate(rows):
        if error := _validate_row(row, ("term", "definition"), f"строка {i + 1}"):
            report.errors.append(error)
            continue
        term_id = glossary_id(row["term"])
        if term_id in existing:
            report.skipped += 1
            continue
        session.add(GlossaryTerm(
            id=term_id,
            term=row["term"].strip(),
            definition=row["definition"].strip(),
            domain=str(row.get("domain") or "general").strip(),
        ))
        existing.add(term_id)
        report.created += 1
    await session.commit()
    logger.info("Глоссарий: создано %d, пропущено %d, ошибок %d", report.created, report.skipped, len(report.errors))
    return report


async def import_cases(session: AsyncSession, rows: Iterable[dict[str, Any]]) -> ImportReport:
    """Импорт кейсов Decision Memory: {situation, decision, outcome, domain?, decision_type?, quality?}."""
    report = ImportReport()
    rows = list(rows)
    existing = set(
        await session.scalars(
            select(DecisionMemory.id).where(
                DecisionMemory.id.in_([case_id(r["situation"], r["decision"]) for r in rows])
            )
        )
    )
    for i, row in enumerate(rows):
        if error := _validate_row(row, ("situation", "decision", "outcome"), f"строка {i + 1}"):
            report.errors.append(error)
            continue
        memory_id = case_id(row["situation"], row["decision"])
        if memory_id in existing:
            report.skipped += 1
            continue
        session.add(DecisionMemory(
            id=memory_id,
            situation=row["situation"].strip(),
            decision=row["decision"].strip(),
            outcome=row["outcome"].strip(),
            domain=str(row.get("domain") or "general").strip(),
            decision_type=str(row.get("decision_type") or "general").strip(),
            quality=int(row.get("quality") or 5),
        ))
        existing.add(memory_id)
        report.created += 1
    await session.commit()
    logger.info(
        "Decision Memory: создано %d, пропущено %d, ошибок %d",
        report.created, report.skipped, len(report.errors),
    )
    return report
