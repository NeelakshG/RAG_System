from ingest import chunk_corpus, discover_files, ingest, load_corpus
from src.models import Document


class StubEmbedder:
    """Deterministic offline embedder: assigns each distinct text its own
    orthogonal one-hot vector, so unrelated texts never collide under
    cosine similarity (a hash-based low-dimensional vector could).
    """

    def __init__(self):
        self._seen: dict[str, int] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        for t in texts:
            if t not in self._seen:
                self._seen[t] = len(self._seen)

        dim = len(self._seen)
        vectors = []
        for t in texts:
            vec = [0.0] * dim
            vec[self._seen[t]] = 1.0
            vectors.append(vec)
        return vectors


class Config:
    chunk_size = 256
    chunk_overlap = 32
    max_tokens = 256
    chroma_persist_dir = None
    collection_name = "chunks"
    bm25_path = None
    dedup_threshold = 0.95


def _config(tmp_path):
    cfg = Config()
    cfg.chroma_persist_dir = str(tmp_path / "chroma")
    cfg.bm25_path = str(tmp_path / "bm25.pkl")
    return cfg


# --- discover_files ---

def test_discover_files_finds_supported_extensions_recursively(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world")
    (tmp_path / "c.html").write_text("<p>hi</p>")
    (tmp_path / "ignore.pdf").write_text("nope")
    (tmp_path / "ignore.json").write_text("{}")

    found = discover_files(tmp_path)

    assert sorted(p.name for p in found) == ["a.md", "b.txt", "c.html"]


# --- load_corpus ---

def test_load_corpus_returns_documents_for_every_supported_file(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\nAlpha content.")
    (tmp_path / "b.txt").write_text("Beta content.")
    (tmp_path / "ignore.json").write_text("{}")

    docs = load_corpus(tmp_path)

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert sorted(d.source_name for d in docs) == ["a.md", "b.txt"]


# --- chunk_corpus ---

def test_chunk_corpus_flattens_chunks_across_documents():
    docs = [
        Document(
            doc_id="a.md",
            source_path="/a.md",
            source_name="a.md",
            fmt="md",
            raw_text="Alpha one. Alpha two.",
            clean_text="Alpha one. Alpha two.",
            metadata={},
        ),
        Document(
            doc_id="b.md",
            source_path="/b.md",
            source_name="b.md",
            fmt="md",
            raw_text="Beta one. Beta two.",
            clean_text="Beta one. Beta two.",
            metadata={},
        ),
    ]

    chunks = chunk_corpus(docs, strategy="recursive", config=Config())

    assert len(chunks) == 2
    assert {c.doc_id for c in chunks} == {"a.md", "b.md"}


def test_chunk_corpus_chunk_ids_are_unique_across_documents():
    docs = [
        Document(
            doc_id="a.md", source_path="/a.md", source_name="a.md", fmt="md",
            raw_text="x", clean_text="x", metadata={},
        ),
        Document(
            doc_id="b.md", source_path="/b.md", source_name="b.md", fmt="md",
            raw_text="y", clean_text="y", metadata={},
        ),
    ]

    chunks = chunk_corpus(docs, strategy="recursive", config=Config())

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --- ingest (end-to-end) ---

def test_ingest_end_to_end(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# Alpha\nSome alpha content here.")
    (corpus / "b.md").write_text("# Beta\nSome beta content here.")

    stats = ingest(
        corpus,
        strategy="fixed",
        config=_config(tmp_path),
        embedder=StubEmbedder(),
    )

    assert stats["indexed"] == 2
    assert stats["deduped"] == 0
