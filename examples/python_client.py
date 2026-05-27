"""
Example Python client for the Retrieval Infrastructure Platform.
Run: python examples/python_client.py
Requires: pip install httpx
"""
import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"


async def check_health(client: httpx.AsyncClient) -> None:
    print("\n=== Health Check ===")
    resp = await client.get(f"{BASE_URL}/health")
    print(json.dumps(resp.json(), indent=2))


async def generate_queries(client: httpx.AsyncClient, query: str) -> list[str]:
    print(f"\n=== Query Generation: '{query}' ===")
    resp = await client.post(
        f"{BASE_URL}/query/generate",
        json={"query": query, "num_queries": 5},
    )
    data = resp.json()
    print(f"Generated {len(data['queries'])} queries in {data['elapsed_ms']:.1f}ms:")
    for i, q in enumerate(data["queries"], 1):
        print(f"  {i}. {q}")
    return data["queries"]


async def search(client: httpx.AsyncClient, query: str) -> list[dict]:
    print(f"\n=== Search: '{query}' ===")
    resp = await client.post(
        f"{BASE_URL}/search",
        json={"query": query, "num_results": 5, "use_cache": True},
        timeout=30.0,
    )
    data = resp.json()
    print(f"Found {data['total']} results in {data['elapsed_ms']:.1f}ms (cached={data['cached']}):")
    for r in data["results"][:3]:
        print(f"  [{r['engine']}] {r['title'][:60]}")
        print(f"    {r['url'][:80]}")
    return data["results"]


async def crawl_url(client: httpx.AsyncClient, url: str) -> dict:
    print(f"\n=== Crawl: {url} ===")
    resp = await client.post(
        f"{BASE_URL}/crawl",
        json={"urls": [url], "use_playwright": False, "use_cache": True},
        timeout=30.0,
    )
    data = resp.json()
    if data["results"]:
        r = data["results"][0]
        html_len = len(r["html"] or "")
        print(f"Status: {r['status_code']}, HTML length: {html_len} chars, cached: {r['cached']}")
    else:
        print(f"Failures: {data['failed']}")
    return data


async def ingest_urls(client: httpx.AsyncClient, urls: list[str]) -> dict:
    print(f"\n=== Ingest {len(urls)} URLs ===")
    resp = await client.post(
        f"{BASE_URL}/ingest",
        json={
            "urls": urls,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "use_playwright": False,
        },
        timeout=120.0,
    )
    data = resp.json()
    print(f"Ingested: {len(data['ingested_urls'])} URLs, {data['total_chunks']} chunks")
    if data["failed_urls"]:
        print(f"Failed: {data['failed_urls']}")
    return data


async def retrieve(client: httpx.AsyncClient, query: str) -> dict:
    print(f"\n=== Retrieve: '{query}' ===")
    resp = await client.post(
        f"{BASE_URL}/retrieve",
        json={"query": query, "top_k": 20, "rerank_top_k": 5},
        timeout=30.0,
    )
    data = resp.json()
    print(f"Retrieved {data['total_retrieved']}, reranked to {data['total_reranked']} in {data['elapsed_ms']:.1f}ms")
    for i, chunk in enumerate(data["chunks"], 1):
        print(f"\n  Chunk {i} (score={chunk['score']:.3f}, rerank={chunk['rerank_score']:.3f}):")
        print(f"    URL: {chunk['metadata']['url'][:70]}")
        print(f"    Text: {chunk['text'][:150]}...")
    return data


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        await check_health(client)

        # 2. Query expansion
        queries = await generate_queries(client, "how does transformer attention work")

        # 3. Ingest a well-known page
        urls = ["https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"]
        await ingest_urls(client, urls)

        # 4. Retrieve semantically relevant chunks
        await retrieve(client, "explain multi-head self-attention")


if __name__ == "__main__":
    asyncio.run(main())
