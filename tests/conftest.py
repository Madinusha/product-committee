"""Общие фикстуры тестов.

- auth-тесты: SQLite in-memory (StaticPool) + override get_session;
- агентные тесты: LLM мокается на уровне Agent._call_llm (никаких реальных
  вызовов Ollama/HF); между тестами unload_hf_llm() не нужен, так как
  transformers импортируется лениво и в тестах не загружается.
"""
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.api.deps import get_session
from backend.api.main import app
from backend.db.models import Base


@pytest.fixture()
async def db_session_factory() -> AsyncIterator[async_sessionmaker]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture()
async def client(db_session_factory: async_sessionmaker) -> AsyncGenerator[AsyncClient, None]:
    async def override_session() -> AsyncGenerator:
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def make_llm_mock(responses: dict[str, str], calls: list | None = None):
    """Фабрика мока Agent._call_llm: responses[name] — что отвечает агент name.

    Можно передать список ответов для одного агента (расходуются по порядку).
    """

    async def fake_call_llm(self, prompt: str, system: str | None = None) -> str:
        if calls is not None:
            calls.append({"agent": self.name, "prompt": prompt})
        scripted = responses.get(self.name, "")
        if isinstance(scripted, list):
            return scripted.pop(0) if scripted else ""
        return scripted

    return fake_call_llm


@pytest.fixture()
def registered_user(client: AsyncClient) -> dict[str, Any]:
    """Создаёт пользователя и возвращает {email, username, password}."""
    return {
        "email": "test@example.com",
        "username": "tester",
        "password": "Password1",
    }
