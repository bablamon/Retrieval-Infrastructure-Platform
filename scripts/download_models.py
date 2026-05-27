"""
Pre-downloads embedding and reranker models.
Called during Docker build to bake models into the image.
Usage: python scripts/download_models.py
"""
import os


def download_models() -> None:
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    print(f"Downloading embedding model: {embedding_model}")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(embedding_model)
    print(f"Embedding model downloaded: {embedding_model}")

    print(f"Downloading reranker model: {reranker_model}")
    from FlagEmbedding import FlagReranker
    # use_fp16=False for CPU-safe download
    FlagReranker(reranker_model, use_fp16=False)
    print(f"Reranker model downloaded: {reranker_model}")

    print("All models downloaded successfully.")


if __name__ == "__main__":
    download_models()
