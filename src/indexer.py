import math
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.models import Chunk
from src.tokenizer import tokenize


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine_similarity(a, b); 0.0 if either vector is all zeros."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chroma_safe_metadata(metadata: dict) -> dict:
    """Chroma rejects None metadata values -- drop keys whose value is None."""
    return {k: v for k, v in metadata.items() if v is not None}


def deduplicate(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    threshold: float = 0.95,
) -> tuple[list[Chunk], list[list[float]]]:
    """Drop any chunk whose cosine similarity to an already-kept chunk in this
    same batch exceeds threshold. First occurrence wins; order is preserved.
    """
    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []

    for chunk, embedding in zip(chunks, embeddings):
        is_duplicate = any(
            _cosine_similarity(embedding, kept) > threshold for kept in kept_embeddings
        )
        if is_duplicate:
            continue
        kept_chunks.append(chunk)
        kept_embeddings.append(embedding)

    return kept_chunks, kept_embeddings


class DenseIndex:
    """Thin wrapper around a Chroma persistent collection."""

    def __init__(self, persist_dir: str, collection_name: str = "chunks"):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(collection_name)

    def reset(self) -> None:
        """Delete and recreate the collection -- used to force a full rebuild."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_chroma_safe_metadata(c.to_metadata()) for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()
    
    
    def query(self, query_embedding: list[float], k: int = 10) -> list[str]:
        if not query_embedding:
            return []

        result = self._collection.query(query_embeddings=[query_embedding], n_results=k)
        return  result["ids"][0]
        
    


class SparseIndex:
    """BM25 over the same chunks, tokenized with src.tokenizer.tokenize."""

    def __init__(self):
        self.chunk_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str) -> "SparseIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index.chunk_ids = data["chunk_ids"]
        index.bm25 = data["bm25"]
        return index
    
    def query(self, query_tokens: list[str], k: int = 10) -> list[str]:
        if self.bm25 is None:
            return []

        score = self.bm25.get_scores(query_tokens)
        paired =zip(self.chunk_ids, score)
        ranked = sorted(paired, key=lambda pair: pair[1], reverse = True)
        top_k = ranked[:k]
        result = []
        for chunk_id, score in top_k:
            result.append(chunk_id)
            
        return result
        


def build_indexes(chunks: list[Chunk], embedder, config=None) -> dict:
    """Embed all chunks once, dedup within the batch, then rebuild the dense
    and sparse indexes from scratch over the identical set of survivors.
    Returns {"indexed": n, "deduped": n}.
    """
    if not chunks:
        return {"indexed": 0, "deduped": 0}

    persist_dir = getattr(config, "chroma_persist_dir", "data/chroma")
    collection_name = getattr(config, "collection_name", "chunks")
    bm25_path = getattr(config, "bm25_path", "data/bm25.pkl")
    threshold = getattr(config, "dedup_threshold", 0.95)

    embeddings = embedder.embed([c.text for c in chunks])
    kept_chunks, kept_embeddings = deduplicate(chunks, embeddings, threshold)

    dense_index = DenseIndex(persist_dir=persist_dir, collection_name=collection_name)
    dense_index.reset()
    dense_index.add(kept_chunks, kept_embeddings)

    sparse_index = SparseIndex()
    sparse_index.build(kept_chunks)
    sparse_index.save(bm25_path)

    return {"indexed": len(kept_chunks), "deduped": len(chunks) - len(kept_chunks)}
