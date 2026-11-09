"""Асинхронные утилиты: retry-декоратор с экспоненциальной задержкой."""
import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger("backend.utils")

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Декоратор повторных попыток для async-функций.

    При исчерпании попыток поднимается последнее исключение.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts должен быть >= 1")

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    logger.warning(
                        "%s: попытка %d/%d не удалась (%s), повтор через %.1fs",
                        func.__qualname__, attempt, max_attempts, exc, current_delay,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
