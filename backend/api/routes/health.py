"""Health-роуты: /healthz (liveness) и /health (readiness, этап 8).

/health проверяет все обязательные компоненты (db, ollama, qdrant, redis)
и отвечает 503, если хотя бы один лежит.
"""
from fastapi import APIRouter, HTTPException, status

from backend.core.models import HealthResponse
from backend.services.health import run_all_health_checks

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Жив ли процесс (без внешних проверок)."""
    return HealthResponse(status="ok", components={"app": "ok"})


@router.get("/health", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Готов ли сервис принимать трафик: все компоненты обязательны."""
    components = {"app": "ok", **await run_all_health_checks()}
    if "down" in components.values():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            HealthResponse(status="down", components=components).model_dump(),
        )
    return HealthResponse(status="ok", components=components)
