from fastapi import APIRouter, Depends

from app.core.dependencies import get_search_service
from app.models.requests import SearchRequest
from app.models.responses import SearchResponse
from app.search.cache import CachedSearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_service: CachedSearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search(request)
