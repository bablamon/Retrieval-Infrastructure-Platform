import time

from fastapi import APIRouter, Depends

from app.core.dependencies import get_query_expander
from app.models.requests import QueryGenerateRequest
from app.models.responses import QueryGenerateResponse
from app.search.query_expander import QueryExpander

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/generate", response_model=QueryGenerateResponse)
async def generate_queries(
    request: QueryGenerateRequest,
    expander: QueryExpander = Depends(get_query_expander),
) -> QueryGenerateResponse:
    start = time.monotonic()
    queries = expander.expand(
        request.query,
        num_queries=request.num_queries,
        include_original=request.include_original,
    )
    return QueryGenerateResponse(
        original=request.query,
        queries=queries,
        elapsed_ms=(time.monotonic() - start) * 1000,
    )
