"""Pydantic-модели запросов/ответов API (этапы 2–4; decide/batch/ensemble — этап 9)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=32)
    # длина пароля проверяется в services.auth.validate_password (слабый -> 400, а не 422)
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str] = {}


# --- принятие решений (этап 9) ---


class DecideRequest(BaseModel):
    situation: str = Field(min_length=10, max_length=4000)
    domain: str = "general"
    decision_type: str | None = None
    mode: str | None = None  # None -> settings.EVAL_MODE


class DecideResponse(BaseModel):
    decision: str
    eval_mode: str
    domain: str
    critic_score: float | None = None
    validation_passed: bool | None = None
    stopped_reason: str | None = None
    iterations: int = 0
    engine_used: str | None = None
    engine_fallback: bool | None = None
    retrieval_backend: str | None = None
    cached: bool = False
    reasoning_log: list[str] = []


class MultiDecideRequest(BaseModel):
    """Одна ситуация - решения с позиций нескольких доменов."""

    situation: str = Field(min_length=10, max_length=4000)
    domains: list[str] = Field(min_length=1, max_length=5)
    decision_type: str | None = None
    mode: str | None = None


class MultiDecideResponse(BaseModel):
    results: list[DecideResponse]


class BatchItemResult(BaseModel):
    index: int
    status: str  # ok | error
    response: DecideResponse | None = None
    error: str | None = None


class BatchDecideRequest(BaseModel):
    items: list[DecideRequest] = Field(min_length=1, max_length=20)


class BatchDecideResponse(BaseModel):
    results: list[BatchItemResult]


class EnsembleDecideRequest(BaseModel):
    situation: str = Field(min_length=10, max_length=4000)
    domain: str
    decision_type: str | None = None
    mode: str | None = None
    variants: int = Field(default=3, ge=2, le=5)


class EnsembleVariant(BaseModel):
    decision: str
    critic_score: float | None = None


class EnsembleDecideResponse(BaseModel):
    best: EnsembleVariant
    variants: list[EnsembleVariant]
