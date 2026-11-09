"""Конфигурация приложения (pydantic-settings).

Все настройки читаются из окружения / .env (см. .env.example).
В production (DEBUG=false) действуют жёсткие требования к JWT_SECRET.
"""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_SECRET = "dev-secret-change-me-in-production-0123456789"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- приложение ---
    APP_NAME: str = "Виртуальный продуктовый комитет"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_JSON: bool = False

    # --- БД ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./dev.db"
    DB_ECHO: bool = False

    # --- JWT ---
    JWT_SECRET: str = _DEV_SECRET
    JWT_ALGORITHM: str = "HS256"
    JWT_TTL_MINUTES: int = 60

    # --- LLM ---
    LLM_ENGINE: str = "auto"  # auto | ollama | huggingface
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_TIMEOUT: float = 120.0
    OLLAMA_SEED: int = 42
    HF_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    HF_MAX_NEW_TOKENS: int = 768

    # --- RAG (Коля) ---
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "decision_memory"
    RETRIEVAL_TOP_K: int = 5
    RRF_K: int = 60

    # --- инфраструктура ---
    REDIS_URL: str = "redis://localhost:6379/0"
    EVAL_MODE: str = "full"

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        """Жёсткая валидация для production: слабые секреты запрещены."""
        if not self.DEBUG:
            if len(self.JWT_SECRET) < 32 or self.JWT_SECRET == _DEV_SECRET:
                raise ValueError(
                    "JWT_SECRET: в production требуется >= 32 случайных символов "
                    "(и значение по умолчанию для dev запрещено)"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Синглтон настроек (пересоздаётся только в тестах через cache_clear)."""
    return Settings()
