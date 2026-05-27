from app.core.logging import get_logger
from app.models.responses import RetrievedChunk
from app.utils.concurrency import run_in_executor

logger = get_logger(__name__)


class RerankerService:
    def __init__(
        self,
        model_name: str,
        use_fp16: bool = True,
        top_k: int = 5,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._default_top_k = top_k
        self._reranker = None

    def load_model(self) -> None:
        import torch
        from FlagEmbedding import FlagReranker

        # fp16 only helps on CUDA; on CPU it degrades to slow emulated ops.
        use_fp16 = self._use_fp16 and torch.cuda.is_available()
        logger.info(
            "loading_reranker_model",
            model=self._model_name,
            use_fp16=use_fp16,
            cuda=torch.cuda.is_available(),
        )
        self._reranker = FlagReranker(self._model_name, use_fp16=use_fp16)
        logger.info("reranker_model_loaded", model=self._model_name)

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        if self._reranker is None:
            raise RuntimeError("RerankerService model not loaded — call load_model() first")
        scores = self._reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            return [scores]
        return list(scores)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        effective_top_k = top_k if top_k is not None else self._default_top_k
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = await run_in_executor(self._compute_scores, pairs)

        reranked = [
            RetrievedChunk(
                text=chunk.text,
                score=chunk.score,
                rerank_score=score,
                metadata=chunk.metadata,
            )
            for chunk, score in zip(chunks, scores, strict=False)
        ]

        reranked.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        logger.info("reranked", total=len(reranked), top_k=effective_top_k)
        return reranked[:effective_top_k]

    async def health_check(self) -> bool:
        return self._reranker is not None
