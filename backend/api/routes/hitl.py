"""Роуты Human-in-the-Loop (этап 10): очередь подтверждений.

Отправка в очередь - любой авторизованный; просмотр очереди, подтверждение
и отклонение - editor/admin (подтверждает продакт-владелец).
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.api.deps import CurrentUserDep, SessionDep, require_editor
from backend.core.constants import KNOWN_DOMAINS
from backend.core.exceptions import UnknownDomainError
from backend.core.models import (
    HitlCreateRequest,
    HitlDecisionResponse,
    HitlEntryResponse,
    HitlListResponse,
)
from backend.db.models import HitlQueue, User
from backend.services import hitl as hitl_service

router = APIRouter(prefix="/hitl", tags=["hitl"])

EditorUserDep = Annotated[User, Depends(require_editor)]


def _validate_domain(domain: str) -> None:
    if domain not in KNOWN_DOMAINS:
        raise UnknownDomainError(
            f"Неизвестный домен: {domain!r}. Допустимо: {', '.join(KNOWN_DOMAINS)}"
        )


@router.post("", response_model=HitlEntryResponse, status_code=201)
async def create_entry(
    payload: HitlCreateRequest, session: SessionDep, user: CurrentUserDep
) -> HitlQueue:
    """Отправить решение на подтверждение продакт-владельцу."""
    _validate_domain(payload.domain)
    return await hitl_service.enqueue(
        session,
        situation=payload.situation,
        decision=payload.decision,
        domain=payload.domain,
        decision_type=payload.decision_type,
        critic_score=payload.critic_score,
    )


@router.get("/pending", response_model=HitlListResponse)
async def list_pending(
    session: SessionDep,
    user: EditorUserDep,
    limit: int = Query(default=50, ge=1, le=100),
) -> HitlListResponse:
    """Очередь ожидающих подтверждения (editor/admin)."""
    entries = await hitl_service.pending_list(session, limit=limit)
    return HitlListResponse(entries=[HitlEntryResponse.model_validate(e) for e in entries])


@router.get("/history", response_model=HitlListResponse)
async def list_history(
    session: SessionDep,
    user: EditorUserDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> HitlListResponse:
    """Последние обработанные записи (editor/admin)."""
    entries = await hitl_service.history(session, limit=limit)
    return HitlListResponse(entries=[HitlEntryResponse.model_validate(e) for e in entries])


@router.post("/{entry_id}/approve", response_model=HitlDecisionResponse)
async def approve_entry(
    entry_id: int, session: SessionDep, user: EditorUserDep
) -> HitlDecisionResponse:
    """Подтвердить решение: кейс попадает в Decision Memory (editor/admin)."""
    entry, created = await hitl_service.approve(
        session, entry_id, reviewer_id=user.id, reviewer_name=user.username
    )
    return HitlDecisionResponse(
        entry=HitlEntryResponse.model_validate(entry),
        memory_case_created=created,
    )


@router.post("/{entry_id}/reject", response_model=HitlEntryResponse)
async def reject_entry(
    entry_id: int, session: SessionDep, user: EditorUserDep
) -> HitlQueue:
    """Отклонить решение: в Decision Memory не попадает (editor/admin)."""
    return await hitl_service.reject(session, entry_id, reviewer_id=user.id)
