import hashlib
from datetime import UTC, datetime
from typing import Any

import asyncpg

from app.models.db import CrawlStateRecord, JobRecord
from app.utils.url_utils import normalize_url


def _url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


class PostgresClient:
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresClient not connected — call connect() first")
        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        async with self._pool_or_raise().acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        async with self._pool_or_raise().acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        async with self._pool_or_raise().acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self._pool_or_raise().acquire() as conn:
            return await conn.fetchval(query, *args)

    async def upsert_crawl_state(
        self,
        url: str,
        status: str,
        http_status: int | None = None,
        error: str | None = None,
    ) -> None:
        url_hash = _url_hash(url)
        crawled_at = datetime.now(tz=UTC) if status in ("crawled", "failed") else None
        await self.execute(
            """
            INSERT INTO crawl_state (url, url_hash, status, http_status, crawled_at, error, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (url_hash) DO UPDATE SET
                status = EXCLUDED.status,
                http_status = EXCLUDED.http_status,
                crawled_at = EXCLUDED.crawled_at,
                error = EXCLUDED.error,
                updated_at = NOW()
            """,
            url,
            url_hash,
            status,
            http_status,
            crawled_at,
            error,
        )

    async def get_crawl_state(self, url: str) -> CrawlStateRecord | None:
        row = await self.fetchrow(
            "SELECT * FROM crawl_state WHERE url_hash = $1",
            _url_hash(url),
        )
        if row is None:
            return None
        return CrawlStateRecord(**dict(row))

    async def is_url_crawled(self, url: str) -> bool:
        val = await self.fetchval(
            "SELECT status FROM crawl_state WHERE url_hash = $1",
            _url_hash(url),
        )
        return val == "crawled"

    async def create_job(self, job_id: str, job_type: str, payload: dict[str, Any]) -> JobRecord:
        import json

        row = await self.fetchrow(
            """
            INSERT INTO jobs (id, job_type, status, payload)
            VALUES ($1::uuid, $2, 'queued', $3::jsonb)
            RETURNING *
            """,
            job_id,
            job_type,
            json.dumps(payload),
        )
        return JobRecord(**dict(row))

    async def update_job(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        import json

        await self.execute(
            """
            UPDATE jobs SET status = $2, result = $3::jsonb, error = $4, updated_at = NOW()
            WHERE id = $1::uuid
            """,
            job_id,
            status,
            json.dumps(result) if result else None,
            error,
        )

    async def get_job(self, job_id: str) -> JobRecord | None:
        row = await self.fetchrow("SELECT * FROM jobs WHERE id = $1::uuid", job_id)
        if row is None:
            return None
        return JobRecord(**dict(row))

    async def health_check(self) -> bool:
        try:
            val = await self.fetchval("SELECT 1")
            return val == 1
        except Exception:
            return False
