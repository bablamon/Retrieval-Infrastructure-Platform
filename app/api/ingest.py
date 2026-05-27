from fastapi import APIRouter, Depends

from app.core.dependencies import get_retrieval_pipeline
from app.models.requests import IngestRequest
from app.models.responses import IngestResponse
from app.retrieval.pipeline import RetrievalPipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> IngestResponse:
    return await pipeline.ingest(request)
