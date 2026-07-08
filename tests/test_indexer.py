import pytest

from src.indexer import (
    _cosine_similarity,
    _chroma_safe_metadata,
    deduplicate,
    DenseIndex,
    SparseIndex,
    build_indexes,
)
from src.models import Chunk
from src.tokenizer import tokenize


def _make_chunk(chunk_id="doc1::0", text="hello world", section_heading=None) -> Chunk:
    doc_id = chunk_id.split("::")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_name=f"{doc_id}.md",
        chunk_index=0,
        text=text,
        section_heading=section_heading,
        chunking_strategy="fixed",
        char_count=len(text),
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )


class StubEmbedder:
    """Deterministic offline embedder for tests: looks up each text in a fixed mapping."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[t] for t in texts]


# --- _cosine_similarity ---

def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- _chroma_safe_metadata ---

def test_chroma_safe_metadata_drops_none_values():
    metadata = {"a": 1, "b": None, "c": "x"}
    assert _chroma_safe_metadata(metadata) == {"a": 1, "c": "x"}


# --- deduplicate ---

def test_deduplicate_drops_near_duplicate_within_same_batch():
    a = _make_chunk(chunk_id="doc1::0", text="unique text")
    b = _make_chunk(chunk_id="doc2::0", text="planted duplicate paragraph")
    c = _make_chunk(chunk_id="doc3::0", text="planted duplicate paragraph again")

    emb_a = [1.0, 0.0]
    emb_b = [0.0, 1.0]
    emb_c = [0.0, 0.999]  # cosine sim to emb_b ~ 1.0 > 0.95

    kept_chunks, kept_embeddings = deduplicate(
        chunks=[a, b, c],
        embeddings=[emb_a, emb_b, emb_c],
        threshold=0.95,
    )

    assert [ch.chunk_id for ch in kept_chunks] == ["doc1::0", "doc2::0"]
    assert kept_embeddings == [emb_a, emb_b]


def test_deduplicate_keeps_distinct_chunks():
    a = _make_chunk(chunk_id="doc1::0", text="alpha")
    b = _make_chunk(chunk_id="doc2::0", text="beta")

    kept_chunks, kept_embeddings = deduplicate(
        chunks=[a, b],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        threshold=0.95,
    )

    assert [ch.chunk_id for ch in kept_chunks] == ["doc1::0", "doc2::0"]
    assert kept_embeddings == [[1.0, 0.0], [0.0, 1.0]]


def test_deduplicate_empty_input():
    kept_chunks, kept_embeddings = deduplicate([], [], threshold=0.95)
    assert kept_chunks == []
    assert kept_embeddings == []


# --- DenseIndex ---

def test_dense_index_add_and_count(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    chunk = _make_chunk(section_heading=None)  # exercises the None-metadata gotcha

    index.add([chunk], [[0.1, 0.2, 0.3]])

    assert index.count() == 1


def test_dense_index_reset_clears_collection(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    index.add([_make_chunk()], [[0.1, 0.2, 0.3]])
    assert index.count() == 1

    index.reset()

    assert index.count() == 0


def test_dense_index_add_empty_is_noop(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    index.add([], [])
    assert index.count() == 0


def test_dense_index_query_returns_closest_first(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    index.add(
        [_make_chunk(chunk_id="a::0"), _make_chunk(chunk_id="b::0"), _make_chunk(chunk_id="c::0")],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    results = index.query([1.0, 0.0], k=2)

    assert results == ["a::0", "b::0"]


def test_dense_index_get_texts_returns_matching_text(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    index.add(
        [_make_chunk(chunk_id="a::0", text="alpha text"), _make_chunk(chunk_id="b::0", text="beta text")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    texts = index.get_texts(["b::0", "a::0"])

    assert texts == {"a::0": "alpha text", "b::0": "beta text"}


def test_dense_index_get_texts_empty_input_returns_empty_dict(tmp_path):
    index = DenseIndex(persist_dir=str(tmp_path / "chroma"))
    assert index.get_texts([]) == {}


# --- SparseIndex ---

def test_sparse_index_build_tracks_chunk_ids():
    chunks = [
        _make_chunk(chunk_id="doc1::0", text="alpha"),
        _make_chunk(chunk_id="doc2::0", text="beta"),
    ]
    index = SparseIndex()
    index.build(chunks)

    assert index.chunk_ids == ["doc1::0", "doc2::0"]
    assert index.bm25 is not None


def test_sparse_index_query_ranks_by_relevance():
    chunks = [
        _make_chunk(chunk_id="a::0", text="unrelated content about cats"),
        _make_chunk(chunk_id="b::0", text="ERR_2043 troubleshooting guide"),
        _make_chunk(chunk_id="c::0", text="something else entirely"),
    ]
    index = SparseIndex()
    index.build(chunks)

    results = index.query(tokenize("ERR_2043"), k=2)

    assert results[0] == "b::0"
    assert len(results) == 2


def test_sparse_index_save_and_load_roundtrip(tmp_path):
    chunks = [
        _make_chunk(chunk_id="doc1::0", text="alpha beta"),
        _make_chunk(chunk_id="doc2::0", text="gamma delta"),
    ]
    index = SparseIndex()
    index.build(chunks)
    path = tmp_path / "bm25.pkl"
    index.save(str(path))

    loaded = SparseIndex.load(str(path))

    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.bm25 is not None


# --- build_indexes ---

def test_build_indexes_end_to_end(tmp_path):
    a = _make_chunk(chunk_id="doc1::0", text="alpha unique text")
    b = _make_chunk(chunk_id="doc2::0", text="planted duplicate paragraph")
    c = _make_chunk(chunk_id="doc3::0", text="planted duplicate paragraph again")

    embedder = StubEmbedder({
        "alpha unique text": [1.0, 0.0],
        "planted duplicate paragraph": [0.0, 1.0],
        "planted duplicate paragraph again": [0.0, 0.999],
    })

    class Config:
        chroma_persist_dir = str(tmp_path / "chroma")
        collection_name = "chunks"
        bm25_path = str(tmp_path / "bm25.pkl")
        dedup_threshold = 0.95

    stats = build_indexes([a, b, c], embedder, config=Config())

    assert stats == {"indexed": 2, "deduped": 1}

    dense_index = DenseIndex(
        persist_dir=Config.chroma_persist_dir, collection_name=Config.collection_name
    )
    assert dense_index.count() == 2

    sparse_index = SparseIndex.load(Config.bm25_path)
    assert sparse_index.chunk_ids == ["doc1::0", "doc2::0"]


def test_build_indexes_empty_chunks_returns_zero_stats():
    stats = build_indexes([], embedder=None, config=None)
    assert stats == {"indexed": 0, "deduped": 0}
