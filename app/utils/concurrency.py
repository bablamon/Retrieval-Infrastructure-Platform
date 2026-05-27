import asyncio
from collections.abc import Callable
from typing import Any


async def run_in_executor[T](func: Callable[..., T], *args: Any) -> T:
    """Run a blocking, CPU-bound function in the default thread pool so it does
    not block the event loop. Used to wrap trafilatura, sentence-transformers
    and the reranker, which are all synchronous."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)
