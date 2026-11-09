"""Health-роуты: /healthz (liveness) и /health (readiness с проверкой БД).

По мере добавления сервисов (этапы 7–11) в /health добавляются Ollama,
Qdrant, Redis — статус 503 при падении любого обязательного компонента.
"""
from fastapi import APIRouter, HTTPException, status

from backend.core.models import HealthResponse
from backend.db.engine import check_db

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Жив ли процесс (без внешних проверок)."""
    return HealthResponse(status="ok", components={"app": "ok"})


@router.get("/health", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Готов ли сервис принимать трафик: БД обязательна."""
    components: dict[str, str] = {"app": "ok", "db": "ok" if await check_db() else "down"}
    if components["db"] == "down":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            HealthResponse(status="down", components=components).model_dump(),
        )
    return HealthResponse(status="ok", components=components)
