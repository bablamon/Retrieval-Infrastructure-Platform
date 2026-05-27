"""
Run once to initialize PostgreSQL tables.
Usage: python scripts/init_db.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


DDL = """
CREATE TABLE IF NOT EXISTS crawl_state (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL,
    url_hash    CHAR(64) NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    http_status SMALLINT,
    crawled_at  TIMESTAMPTZ,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_crawl_state_url_hash UNIQUE (url_hash)
);

CREATE INDEX IF NOT EXISTS idx_crawl_state_url_hash ON crawl_state(url_hash);
CREATE INDEX IF NOT EXISTS idx_crawl_state_status    ON crawl_state(status);

CREATE TABLE IF NOT EXISTS jobs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type   VARCHAR(50) NOT NULL,
    status     VARCHAR(20) NOT NULL DEFAULT 'queued',
    payload    JSONB NOT NULL DEFAULT '{}',
    result     JSONB,
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type   ON jobs(job_type);
"""


async def main() -> None:
    import asyncpg
    from dotenv import load_dotenv

    load_dotenv()
    dsn = os.getenv("POSTGRES_DSN", "postgresql://user:password@localhost:5432/retrieval_db")

    print(f"Connecting to: {dsn.split('@')[-1]}")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(DDL)
        print("Database initialized successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
