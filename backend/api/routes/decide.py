"""Роуты принятия решений: /decide, /decide/multi, /decide/batch, /decide/ensemble.

Все требуют аутентификации. LLM-вызовы ограничены семафором на 4 параллельных
запроса к Ollama. Кэшируются только решения домена 'general' (см. cache.py).
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from backend.api.deps import CurrentUserDep, SessionDep
from backend.core.config import get_settings
from backend.core.constants import KNOWN_DOMAINS
from backend.core.exceptions import (
    CommitteeException,
    PipelineError,
)
from backend.core.models import (
    BatchDecideRequest,
    BatchDecideResponse,
    BatchItemResult,
    DecideRequest,
    DecideResponse,
    EnsembleDecideRequest,
    EnsembleDecideResponse,
    EnsembleVariant,
    MultiDecideRequest,
    MultiDecideResponse,
)
from backend.core.state import StateKeys
from backend.data.repository import get_cases, get_glossary
from backend.db.engine import get_session_factory
from backend.retrieval.embeddings import get_embedding_service
from backend.retrieval.qdrant import QdrantService
from backend.retrieval.search import RetrievalService
from backend.services.cache import DecisionCacheManager
from backend.services.pipeline import (
    safe_perform_decision,
    select_best_variant,
)
from backend.services.retrieval_adapter import RetrieverAdapter

logger = logging.getLogger("backend.decide")

router = APIRouter(prefix="/decide", tags=["decide"])

# не более 4 одновременных вызовов LLM (qwen2.5:7b держит 1-2 параллельных запроса без деградации)
LLM_SEMAPHORE = asyncio.Semaphore(4)

_retriever: RetrieverAdapter | None = None


def get_retriever() -> RetrieverAdapter:
    """Ленивый синглтон RAG-исполнителя для pipeline (fetcher'ы открывают свои сессии)."""
    global _retriever
    if _retriever is None:
        async def cases_fetcher():
            async with get_session_factory()() as session:
                return await get_cases(session)

        async def glossary_fetcher():
            async with get_session_factory()() as session:
                return await get_glossary(session)

        service = RetrievalService(
            cases_fetcher, glossary_fetcher, get_embedding_service(), qdrant=QdrantService()
        )
        _retriever = RetrieverAdapter(service)
        logger.info("RetrievalService подключён к pipeline через адаптер")
    return _retriever


def _validate_domains(domains: list[str]) -> None:
    unknown = sorted(set(domains) - set(KNOWN_DOMAINS))
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Неизвестные домены: {', '.join(unknown)}. Допустимо: {', '.join(KNOWN_DOMAINS)}",
        )


def _state_to_response(state: dict, cached: bool = False) -> DecideResponse:
    return DecideResponse(
        decision=state.get(StateKeys.DECISION) or "",
        eval_mode=state.get(StateKeys.EVAL_MODE) or "unknown",
        domain=state.get(StateKeys.DOMAIN) or "general",
        critic_score=state.get(StateKeys.CRITIC_SCORE),
        validation_passed=state.get(StateKeys.VALIDATION_PASSED),
        stopped_reason=(
            state.get(StateKeys.SUPERVISOR_STOPPED_REASON).value
            if state.get(StateKeys.SUPERVISOR_STOPPED_REASON) is not None
            else None
        ),
        iterations=state.get(StateKeys.ITERATIONS) or 0,
        engine_used=state.get(StateKeys.ENGINE_USED),
        engine_fallback=state.get(StateKeys.ENGINE_FALLBACK),
        retrieval_backend=state.get(StateKeys.RETRIEVAL_BACKEND),
        cached=cached,
        reasoning_log=state.get(StateKeys.REASONING_LOG) or [],
    )


async def _run_single(session: SessionDep, payload: DecideRequest) -> DecideResponse:
    """Полный проход с кэшем; ошибка подсистем -> 502/503, неизвестный режим -> 400."""
    settings = get_settings()
    cache = DecisionCacheManager()
    _validate_domains([payload.domain])

    hit = await cache.get(
        session,
        situation=payload.situation,
        domain=payload.domain,
        decision_type=payload.decision_type,
        eval_mode=payload.mode or settings.EVAL_MODE,
    )
    if hit is not None:
        return DecideResponse(
            decision=hit.decision,
            eval_mode=payload.mode or settings.EVAL_MODE,
            domain=payload.domain,
            critic_score=hit.critic_score,
            cached=True,
            iterations=0,
        )

    async with LLM_SEMAPHORE:
        state, error = await safe_perform_decision(
            session,
            payload.situation,
            payload.domain,
            payload.mode,
            retriever=get_retriever(),
        )
    if error is not None:
        if isinstance(error, PipelineError):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, error.message)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, error.message)

    await cache.save(
        session,
        situation=payload.situation,
        domain=payload.domain,
        decision=state.get(StateKeys.DECISION) or "",
        decision_type=payload.decision_type,
        eval_mode=state.get(StateKeys.EVAL_MODE),
        critic_score=state.get(StateKeys.CRITIC_SCORE),
    )
    return _state_to_response(state)


@router.post("", response_model=DecideResponse)
async def decide(payload: DecideRequest, session: SessionDep, user: CurrentUserDep) -> DecideResponse:
    return await _run_single(session, payload)


@router.post("/multi", response_model=MultiDecideResponse)
async def decide_multi(
    payload: MultiDecideRequest, session: SessionDep, user: CurrentUserDep
) -> MultiDecideResponse:
    """Одна ситуация - решения с позиций нескольких доменов."""
    _validate_domains(payload.domains)
    results = []
    for domain in payload.domains:
        single = DecideRequest(
            situation=payload.situation,
            domain=domain,
            decision_type=payload.decision_type,
            mode=payload.mode,
        )
        results.append(await _run_single(session, single))
    return MultiDecideResponse(results=results)


@router.post("/batch", response_model=BatchDecideResponse)
async def decide_batch(
    payload: BatchDecideRequest, session: SessionDep, user: CurrentUserDep
) -> BatchDecideResponse:
    """Список ситуаций; ошибка отдельного элемента не валит весь batch
    (полностью асинхронная очередь появится на этапе 11 в Celery)."""
    results = []
    for index, item in enumerate(payload.items):
        try:
            response = await _run_single(session, item)
            results.append(BatchItemResult(index=index, status="ok", response=response))
        except HTTPException as exc:
            results.append(BatchItemResult(index=index, status="error", error=str(exc.detail)))
        except CommitteeException as exc:
            # UnknownEvalModeError и другие доменные ошибки не валят весь batch
            results.append(BatchItemResult(index=index, status="error", error=exc.message))
    return BatchDecideResponse(results=results)


@router.post("/ensemble", response_model=EnsembleDecideResponse)
async def decide_ensemble(
    payload: EnsembleDecideRequest, session: SessionDep, user: CurrentUserDep
) -> EnsembleDecideResponse:
    """N независимых прогонов pipeline, лучший вариант по оценке Critic'а.
    Доступен только для известных доменов (аналог KNOWN_LANGUAGES референса)."""
    _validate_domains([payload.domain])

    states = []
    for _ in range(payload.variants):
        async with LLM_SEMAPHORE:
            state, error = await safe_perform_decision(
                session,
                payload.situation,
                payload.domain,
                payload.mode,
                retriever=get_retriever(),
            )
        if error is not None:
            if isinstance(error, PipelineError):
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, error.message)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, error.message)
        states.append(state)

    best_state = select_best_variant(states)
    variants = [
        EnsembleVariant(decision=s.get(StateKeys.DECISION) or "", critic_score=s.get(StateKeys.CRITIC_SCORE))
        for s in states
    ]
    best = next(v for v in variants if v.decision == (best_state.get(StateKeys.DECISION) or ""))
    return EnsembleDecideResponse(best=best, variants=variants)
