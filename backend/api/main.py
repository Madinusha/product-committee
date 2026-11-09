"""Сборка FastAPI-приложения."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.exceptions import (
    CommitteeException,
    InsufficientRoleError,
    InvalidCredentialsError,
    UnknownEvalModeError,
    UserExistsError,
    WeakPasswordError,
)
from backend.core.logging import setup_logging

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger("backend.api")

# Маппинг иерархии исключений на HTTP-статусы (централизованно, без try/except в роутах)
_STATUS_BY_EXCEPTION: dict[type[CommitteeException], int] = {
    WeakPasswordError: 400,
    UnknownEvalModeError: 400,
    InvalidCredentialsError: 401,
    InsufficientRoleError: 403,
    UserExistsError: 409,
}


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version="0.1.0")

    @app.exception_handler(CommitteeException)
    async def committee_exception_handler(request: Request, exc: CommitteeException) -> JSONResponse:
        status_code = _STATUS_BY_EXCEPTION.get(type(exc), 500)
        logger.warning("%s -> %d: %s", type(exc).__name__, status_code, exc.message)
        return JSONResponse(status_code=status_code, content={"detail": exc.message, "details": exc.details})

    from backend.api.routes import auth, health

    app.include_router(health.router)
    app.include_router(auth.router)

    # Роуты последующих этапов (регистрируются по мере появления):
    #   этап 6-9: admin.py, decide.py, eval.py
    #   этап 10:  hitl.py
    return app


app = create_app()
