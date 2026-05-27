import asyncio

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import get_postgres, get_qdrant, get_redis, get_uptime
from app.models.responses import HealthStatus
from app.storage.postgres_client import PostgresClient
from app.storage.qdrant_client import QdrantStore
from app.storage.redis_client import RedisCache

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(
    redis: RedisCache = Depends(get_redis),
    postgres: PostgresClient = Depends(get_postgres),
    qdrant: QdrantStore = Depends(get_qdrant),
) -> HealthStatus:
    redis_ok, postgres_ok, qdrant_ok = await asyncio.gather(
        redis.health_check(),
        postgres.health_check(),
        qdrant.health_check(),
    )

    services = {
        "redis": redis_ok,
        "postgres": postgres_ok,
        "qdrant": qdrant_ok,
    }

    healthy_count = sum(services.values())
    if healthy_count == len(services):
        status = "healthy"
    elif healthy_count == 0:
        status = "unhealthy"
    else:
        status = "degraded"

    return HealthStatus(
        status=status,
        services=services,
        version=settings.app_version,
        uptime_seconds=get_uptime(),
    )
