from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel


class CrawlStateRecord(BaseModel):
    id: int
    url: str
    url_hash: str
    status: Literal["pending", "crawled", "failed", "skipped"]
    http_status: int | None = None
    crawled_at: datetime | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class JobRecord(BaseModel):
    id: UUID
    job_type: str
    status: Literal["queued", "running", "completed", "failed"]
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
