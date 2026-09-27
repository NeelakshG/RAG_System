import math
import os
import pickle
import shutil
import uuid
from datetime import datetime, timezone
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
    
    
    def query(
        self, query_embedding: list[float], k: int = 10, source_names: list[str] | None = None
    ) -> list[str]:
        if not query_embedding:
            return []

        where = {"source_name": {"$in": source_names}} if source_names else None
        result = self._collection.query(query_embeddings=[query_embedding], n_results=k, where=where)
        return  result["ids"][0]

    def get_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """Fetch the stored text for each chunk_id directly, no similarity search."""
        if not chunk_ids:
            return {}
        result = self._collection.get(ids=chunk_ids)
        return dict(zip(result["ids"], result["documents"]))

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Rehydrate full Chunk objects (text + metadata) for the given ids,
        in the same order as chunk_ids. section_heading defaults back to None
        since Chroma metadata drops it when it was None at insert time.
        """
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids)
        by_id = dict(zip(result["ids"], zip(result["documents"], result["metadatas"])))

        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id not in by_id:
                continue
            text, metadata = by_id[chunk_id]
            fields = {"section_heading": None, **metadata}
            chunks.append(Chunk(text=text, **fields))
        return chunks

    def list_documents(self) -> list[dict]:
        """Group every indexed chunk's metadata by source doc -- there's no
        separate document registry, so this IS the document view.
        """
        result = self._collection.get()
        docs: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            name = metadata["source_name"]
            entry = docs.setdefault(name, {"source_name": name, "chunk_count": 0, "total_tokens": 0})
            entry["chunk_count"] += 1
            entry["total_tokens"] += metadata.get("token_count", 0)
        return [docs[name] for name in sorted(docs)]



class SparseIndex:
    """BM25 over the same chunks, tokenized with src.tokenizer.tokenize."""

    def __init__(self):
        self.chunk_ids: list[str] = []
        self.chunk_sources: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_sources = [c.source_name for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "chunk_sources": self.chunk_sources, "bm25": self.bm25},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "SparseIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index.chunk_ids = data["chunk_ids"]
        index.chunk_sources = data.get("chunk_sources", [])
        index.bm25 = data["bm25"]
        return index

    def query(self, query_tokens: list[str], k: int = 10, source_names: list[str] | None = None) -> list[str]:
        if self.bm25 is None:
            return []

        score = self.bm25.get_scores(query_tokens)
        sources = self.chunk_sources or [None] * len(self.chunk_ids)
        triples = zip(self.chunk_ids, score, sources)
        if source_names:
            allowed = set(source_names)
            triples = [(cid, s, src) for cid, s, src in triples if src in allowed]
        ranked = sorted(triples, key=lambda triple: triple[1], reverse=True)
        top_k = ranked[:k]
        result = []
        for chunk_id, _, _ in top_k:
            result.append(chunk_id)

        return result
        


def build_indexes(
    chunks: list[Chunk],
    embedder,
    config=None,
    *,
    persist_dir: str | None = None,
    bm25_path: str | None = None,
) -> dict:
    """Embed all chunks once, dedup within the batch, then rebuild the dense
    and sparse indexes from scratch over the identical set of survivors.
    persist_dir/bm25_path override the config's paths (used by
    build_index_version to write somewhere other than the live index).
    Returns {"indexed": n, "deduped": n}.
    """
    if not chunks:
        return {"indexed": 0, "deduped": 0}

    persist_dir = persist_dir or getattr(config, "chroma_persist_dir", "data/chroma")
    collection_name = getattr(config, "collection_name", "chunks")
    bm25_path = bm25_path or getattr(config, "bm25_path", "data/bm25.pkl")
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


def build_index_version(chunks: list[Chunk], embedder, config) -> tuple[dict, Path | None]:
    """Build both indexes into a brand-new version directory under
    config.index_versions_dir, never touching the live one -- so queries
    running during a rebuild keep reading a complete, consistent index.
    Returns (stats, version_dir); version_dir is None when there was nothing
    to index (the live index is left as-is).
    """
    if not chunks:
        return {"indexed": 0, "deduped": 0}, None

    # Microsecond UTC timestamp prefix makes directory names sort oldest -> newest
    # (pruning relies on this); the suffix only guards against exact collisions.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    version_dir = config.index_versions_dir / f"{stamp}_{uuid.uuid4().hex[:6]}"
    stats = build_indexes(
        chunks,
        embedder,
        config,
        persist_dir=str(version_dir / "chroma"),
        bm25_path=str(version_dir / "bm25.pkl"),
    )
    return stats, version_dir


def publish_index_version(config, version_dir: Path) -> None:
    """Make version_dir the live index: rewrite the CURRENT pointer
    atomically (write a temp file, then os.replace), then prune old
    versions beyond config.index_versions_to_keep. Rolling back is just
    pointing CURRENT at a previous version's directory name.
    """
    pointer = config.current_index_pointer
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_name(pointer.name + ".tmp")
    tmp.write_text(version_dir.name)
    os.replace(tmp, pointer)
    _prune_index_versions(config, live=version_dir.name)


def _prune_index_versions(config, live: str) -> None:
    # Never fewer than 2: requests that started before the swap may still be
    # reading the previous version.
    keep = max(getattr(config, "index_versions_to_keep", 2), 2)
    versions = sorted(p for p in config.index_versions_dir.iterdir() if p.is_dir())
    for old in versions[:-keep]:
        if old.name == live:
            continue
        # ignore_errors: on Windows a still-open Chroma handle can lock files;
        # a leftover directory is harmless and gets retried on the next prune.
        shutil.rmtree(old, ignore_errors=True)
