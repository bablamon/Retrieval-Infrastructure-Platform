
from app.core.logging import get_logger
from app.storage.redis_client import RedisCache
from app.utils.concurrency import run_in_executor
from app.utils.text_utils import compute_text_hash

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        cache: RedisCache,
        cache_ttl: int = 86400,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._model = None

    def load_model(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        # Bound CPU thread usage to keep memory/CPU predictable under concurrency.
        if self._device == "cpu":
            torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))

        logger.info("loading_embedding_model", model=self._model_name, device=self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("embedding_model_loaded", model=self._model_name)

    def count_tokens(self, text: str) -> int:
        """Length of ``text`` in model tokens. Used by the chunker so chunks
        never silently overflow the encoder's context window. Falls back to
        a whitespace word count if the tokenizer isn't available (e.g. the
        model failed to load and we're degrading)."""
        if self._model is None:
            return len(text.split())
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return len(text.split())
        # ``encode`` without special tokens is the cheapest path; we don't
        # need the actual ids — just len() of the resulting list.
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return len(text.split())
        return len(ids)

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("EmbeddingService model not loaded — call load_model() first")
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        hashes = [compute_text_hash(t) for t in texts]

        # Single batched cache lookup (one MGET instead of N round-trips)
        cached_vectors: list[list[float] | None] = await self._cache.get_cached_embeddings(hashes)

        uncached_indices = [i for i, v in enumerate(cached_vectors) if v is None]
        uncached_texts = [texts[i] for i in uncached_indices]

        if uncached_texts:
            new_vectors = await run_in_executor(self._encode_batch, uncached_texts)
            to_cache: list[tuple[str, list[float]]] = []
            for idx, vec in zip(uncached_indices, new_vectors, strict=False):
                cached_vectors[idx] = vec
                to_cache.append((hashes[idx], vec))
            # Single batched write
            await self._cache.set_cached_embeddings(to_cache, self._cache_ttl)

        return [v for v in cached_vectors if v is not None]

    async def embed_query(self, query: str) -> list[float]:
        # BGE instruction-tuning: prefix queries with "query: "
        prefixed = f"query: {query}"
        results = await self.embed_texts([prefixed])
        return results[0]

    async def health_check(self) -> bool:
        return self._model is not None
