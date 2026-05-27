from collections.abc import Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def http_retry(func: Callable) -> Callable:
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )(func)


def storage_retry(func: Callable) -> Callable:
    try:
        import asyncpg
        import redis.exceptions

        exc_types: tuple[type[Exception], ...] = (
            asyncpg.PostgresConnectionError,
            redis.exceptions.ConnectionError,
        )
    except ImportError:
        exc_types = (OSError,)

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(exc_types),
        reraise=True,
    )(func)


def create_retry(
    max_attempts: int,
    min_wait: float,
    max_wait: float,
    exceptions: tuple[type[Exception], ...],
) -> Callable[[Callable], Callable]:
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
