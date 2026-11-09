"""Per-event-loop httpx.AsyncClient для Ollama.

httpx-клиенты привязаны к event loop, а у нас их несколько (FastAPI,
Celery-воркер с persistent loop). WeakKeyDictionary на текущем loop даёт
каждому loop'у свой клиент; клиент умирает вместе с loop'ом.
"""
import asyncio
import logging
import weakref

import httpx

from backend.core.config import get_settings

logger = logging.getLogger("backend.http")

_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)


def get_ollama_client() -> httpx.AsyncClient:
    """Клиент Ollama для текущего event loop (создаётся лениво)."""
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        settings = get_settings()
        client = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        _clients[loop] = client
        logger.debug("Создан httpx-клиент Ollama для loop %s", id(loop))
    return client


async def close_ollama_clients() -> None:
    """Закрыть клиент текущего loop'а (graceful shutdown воркера)."""
    loop = asyncio.get_running_loop()
    client = _clients.pop(loop, None)
    if client is not None and not client.is_closed:
        await client.aclose()


async def ollama_generate(
    prompt: str, system: str | None = None, temperature: float = 0.7, seed: int | None = None
) -> str:
    """POST /api/generate с детерминированным seed; исключения маппятся в EngineUnavailableError."""
    from backend.core.exceptions import EngineUnavailableError

    settings = get_settings()
    payload: dict = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": seed if seed is not None else settings.OLLAMA_SEED},
    }
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    try:
        client = get_ollama_client()
        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, OSError) as exc:
        raise EngineUnavailableError(
            "Ollama недоступен", details={"error": str(exc)}
        ) from exc
    return data.get("response", "")
