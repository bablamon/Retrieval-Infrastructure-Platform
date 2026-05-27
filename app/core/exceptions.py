from fastapi import Request
from fastapi.responses import JSONResponse


class RetrievalBaseException(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class CrawlError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class ExtractionError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class SearchError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class EmbeddingError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class StorageError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class RateLimitError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=429)


class URLBlockedError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=403)


class DeduplicationError(RetrievalBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


async def retrieval_exception_handler(request: Request, exc: RetrievalBaseException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": type(exc).__name__},
    )
