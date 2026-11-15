"""Admin CRUD (зона Коля): глоссарий, память кейсов, статистика кэша.

Все операции требуют роль admin. Импорт идемпотентный (см. importers):
повторная загрузка того же файла ничего не создаёт.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Query, UploadFile
from sqlalchemy import delete, func, select

from backend.api.deps import SessionDep, require_admin
from backend.core.models import (
    CacheStatsResponse,
    CasesListResponse,
    DeleteResponse,
    GlossaryListResponse,
    GlossaryTermOut,
    ImportReportResponse,
    MemoryCaseOut,
)
from backend.data.importers import import_cases, import_glossary
from backend.db.models import DecisionCache, DecisionMemory, GlossaryTerm

router = APIRouter(prefix="/admin", tags=["admin"])

AdminDep = Annotated[object, Depends(require_admin)]


def _parse_rows(content: bytes, filename: str) -> list[dict]:
    """Строки для импорта из JSON- или CSV-файла."""
    import csv
    import io
    import json

    text = content.decode("utf-8-sig")
    if filename.endswith(".json"):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("JSON должен содержать список объектов")
        return rows
    if filename.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(text)))
    raise ValueError("Поддерживаются форматы .json и .csv")


@router.post("/glossary/import", response_model=ImportReportResponse)
async def import_glossary_file(
    file: UploadFile, session: SessionDep, user: AdminDep
) -> ImportReportResponse:
    """Импорт глоссария: JSON [{term, definition, domain?}] или CSV (идемпотентно)."""
    rows = _parse_rows(await file.read(), file.filename or "")
    report = await import_glossary(session, rows)
    return ImportReportResponse(**report.__dict__)


@router.post("/cases/import", response_model=ImportReportResponse)
async def import_cases_file(
    file: UploadFile, session: SessionDep, user: AdminDep
) -> ImportReportResponse:
    """Импорт кейсов: JSON [{situation, decision, outcome, ...}] или CSV (идемпотентно)."""
    rows = _parse_rows(await file.read(), file.filename or "")
    report = await import_cases(session, rows)
    return ImportReportResponse(**report.__dict__)


@router.get("/glossary", response_model=GlossaryListResponse)
async def list_glossary(
    session: SessionDep,
    user: AdminDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> GlossaryListResponse:
    rows = await session.execute(
        select(GlossaryTerm).order_by(GlossaryTerm.term).offset(offset).limit(limit)
    )
    return GlossaryListResponse(
        entries=[GlossaryTermOut.model_validate(r) for r in rows.scalars().all()]
    )


@router.delete("/glossary/{term_id}", response_model=DeleteResponse)
async def delete_glossary_term(term_id: str, session: SessionDep, user: AdminDep) -> DeleteResponse:
    result = await session.execute(
        delete(GlossaryTerm).where(GlossaryTerm.id == term_id)
    )
    await session.commit()
    return DeleteResponse(deleted=result.rowcount or 0)


@router.get("/cases", response_model=CasesListResponse)
async def list_cases(
    session: SessionDep,
    user: AdminDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> CasesListResponse:
    rows = await session.execute(
        select(DecisionMemory).order_by(DecisionMemory.created_at.desc()).offset(offset).limit(limit)
    )
    return CasesListResponse(
        entries=[MemoryCaseOut.model_validate(r) for r in rows.scalars().all()]
    )


@router.delete("/cases/{case_id}", response_model=DeleteResponse)
async def delete_case(case_id: str, session: SessionDep, user: AdminDep) -> DeleteResponse:
    result = await session.execute(delete(DecisionMemory).where(DecisionMemory.id == case_id))
    await session.commit()
    return DeleteResponse(deleted=result.rowcount or 0)


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(session: SessionDep, user: AdminDep) -> CacheStatsResponse:
    """Статистика кэша: число записей, суммарные попадания, средний score."""
    total = await session.scalar(select(func.count()).select_from(DecisionCache)) or 0
    hits = await session.scalar(select(func.coalesce(func.sum(DecisionCache.hits), 0))) or 0
    avg_score = await session.scalar(select(func.avg(DecisionCache.critic_score)))
    return CacheStatsResponse(
        entries=total,
        total_hits=int(hits),
        avg_critic_score=round(avg_score, 2) if avg_score is not None else None,
    )


@router.delete("/cache", response_model=DeleteResponse)
async def clear_cache(session: SessionDep, user: AdminDep) -> DeleteResponse:
    """Полная очистка кэша (автоматическая инвалидация - задача Celery, этап 11)."""
    result = await session.execute(delete(DecisionCache))
    await session.commit()
    return DeleteResponse(deleted=result.rowcount or 0)
