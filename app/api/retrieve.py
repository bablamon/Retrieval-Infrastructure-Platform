from fastapi import APIRouter, Depends

from app.core.dependencies import get_retrieval_pipeline
from app.models.requests import RetrieveRequest
from app.models.responses import RetrieveResponse
from app.retrieval.pipeline import RetrievalPipeline

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> RetrieveResponse:
    return await pipeline.retrieve(request)
