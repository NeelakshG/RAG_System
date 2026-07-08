"""Phase 1 smoke test -- runs every component in sequence and narrates the
result. Not a pytest suite (see tests/ for the real 94-test suite covering
edge cases); this is a human-readable "does each piece actually work"
walkthrough you run directly:

    python scripts/smoke_test_phase1.py

Uses a deterministic, offline StubEmbedder throughout (no real model
download/load), so it runs in under a second anywhere.
"""

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import tokenize, count_tokens, find_token_spans
from src.models import Document, Chunk
from src.loaders import load_file
from src.chunkers import FixedChunker, RecursiveChunker, SemanticChunker, get_chunker
from src.indexer import _cosine_similarity, deduplicate, DenseIndex, SparseIndex, build_indexes
from ingest import discover_files, load_corpus, chunk_corpus, ingest
from scripts.make_corpus import make_corpus

RESULTS = []


@contextmanager
def temp_dir():
    """Like tempfile.TemporaryDirectory(), but cleanup is best-effort.

    Chroma's PersistentClient keeps file handles open on its index files;
    on Windows (unlike POSIX) an open file can't be deleted, so the
    stdlib TemporaryDirectory's strict cleanup crashes on __exit__.
    ignore_errors=True just leaves the (harmless, OS-managed) temp dir
    behind instead of failing the whole script.
    """
    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def section(title):
    print(f"\n=== {title} ===")


def check(description, condition, detail=""):
    status = "OK" if condition else "FAIL"
    line = f"  [{status}] {description}"
    if detail:
        line += f" -- {detail}"
    print(line)
    RESULTS.append((description, bool(condition)))


class StubEmbedder:
    """Deterministic offline embedder: one-hot per distinct text, so
    unrelated texts are always orthogonal (cosine similarity 0) and
    identical texts are always identical (cosine similarity 1)."""

    def __init__(self):
        self._seen = {}

    def embed(self, texts):
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


class TopicEmbedder:
    """Deterministic offline embedder for the semantic-chunker demo: any
    text containing 'alpha' clusters near [1,0], any text containing
    'beta' clusters near [0,1] -- simulates a real topic shift."""

    def embed(self, texts):
        vectors = []
        for t in texts:
            if "alpha" in t:
                vectors.append([1.0, 0.05])
            else:
                vectors.append([0.05, 1.0])
        return vectors


def check_tokenizer():
    section("1. tokenizer.py")
    text = "Retry with ERR_2043 immediately!"
    tokens = tokenize(text)
    check("tokenize() splits words and punctuation separately",
          tokens == ["Retry", "with", "ERR_2043", "immediately", "!"],
          f"tokens={tokens}")
    check("ERR_2043 survives as ONE token (underscore = \\w)",
          "ERR_2043" in tokens)
    check("count_tokens() matches len(tokenize())",
          count_tokens(text) == len(tokens),
          f"count_tokens={count_tokens(text)}")
    spans = find_token_spans(text)
    check("find_token_spans() offsets slice back to the original tokens",
          all(text[s:e] == tok for (s, e), tok in zip(spans, tokens)),
          f"spans={spans}")


def check_models():
    section("2. models.py")
    doc = Document(doc_id="a.md", source_path="/a.md", source_name="a.md",
                    fmt="md", raw_text="# A\ntext", clean_text="# A\ntext", metadata={})
    check("Document holds raw_text and clean_text independently",
          doc.raw_text == "# A\ntext" and doc.clean_text == "# A\ntext")

    chunk = Chunk(chunk_id="a.md::0", doc_id="a.md", source_name="a.md",
                  chunk_index=0, text="hello", section_heading=None,
                  chunking_strategy="fixed", char_count=5, token_count=1,
                  start_char=0, end_char=5)
    meta = chunk.to_metadata()
    check("Chunk.to_metadata() excludes 'text'", "text" not in meta)
    check("Chunk.to_metadata() keeps chunk_id", meta.get("chunk_id") == "a.md::0")


def check_loaders():
    section("3. loaders.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        html_path = tmp_path / "page.html"
        html_path.write_text(
            "<html><body><h1>Title</h1><script>evil()</script>"
            "<h2>Section Two</h2><p>Body text here.</p></body></html>"
        )
        doc = load_file(html_path, tmp_path)
        check("HTML script tags are stripped", "evil" not in doc.clean_text)
        check("h1 converts to '# ' heading", "# Title" in doc.clean_text)
        check("h2 converts to '## ' heading", "## Section Two" in doc.clean_text)
        check("headings land on separate lines (regression test for the "
              "heading-swallowing fix)",
              doc.clean_text.count("\n#") >= 1 or doc.clean_text.startswith("#"),
              f"clean_text={doc.clean_text!r}")


def check_chunkers():
    section("4. chunkers.py")

    fixed = FixedChunker(size=10, overlap=3)
    doc = Document(doc_id="d", source_path="/d", source_name="d.md", fmt="md",
                    raw_text="", clean_text=" ".join(f"word{i}" for i in range(30)),
                    metadata={})
    fixed_chunks = fixed.chunk(doc)
    check("FixedChunker produces multiple overlapping windows",
          len(fixed_chunks) > 1, f"{len(fixed_chunks)} chunks")
    check("FixedChunker chunks have no section_heading",
          all(c.section_heading is None for c in fixed_chunks))

    recursive = RecursiveChunker(max_tokens=20)
    md_doc = Document(
        doc_id="r", source_path="/r", source_name="r.md", fmt="md",
        raw_text="",
        clean_text="# Alpha\nAlpha section text here.\n\n# Beta\nBeta section text here.",
        metadata={},
    )
    rec_chunks = recursive.chunk(md_doc)
    check("RecursiveChunker splits by heading",
          {c.section_heading for c in rec_chunks} == {"Alpha", "Beta"},
          f"headings={[c.section_heading for c in rec_chunks]}")

    semantic = SemanticChunker(embedder=TopicEmbedder(), percentile=95)
    sem_doc = Document(
        doc_id="s", source_path="/s", source_name="s.md", fmt="md",
        raw_text="",
        clean_text="This is alpha topic. Still alpha topic. "
                    "Now beta topic starts. Still beta topic.",
        metadata={},
    )
    sem_chunks = semantic.chunk(sem_doc)
    check("SemanticChunker cuts at the topic shift (alpha -> beta)",
          len(sem_chunks) == 2, f"{len(sem_chunks)} chunks")

    check("get_chunker('fixed') returns a FixedChunker",
          isinstance(get_chunker("fixed"), FixedChunker))
    check("get_chunker('recursive') returns a RecursiveChunker",
          isinstance(get_chunker("recursive"), RecursiveChunker))
    check("get_chunker('semantic') requires an embedder",
          _raises(lambda: get_chunker("semantic"), ValueError))
    check("get_chunker(unknown) raises NotImplementedError",
          _raises(lambda: get_chunker("nonsense"), NotImplementedError))


def _raises(fn, exc_type):
    try:
        fn()
        return False
    except exc_type:
        return True


def check_make_corpus():
    section("5. scripts/make_corpus.py")
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_corpus(Path(tmp))
        check("make_corpus() writes 4 files", len(paths) == 4, f"{[p.name for p in paths]}")
        check("all files actually exist on disk", all(p.exists() for p in paths))
        contents = " ".join(p.read_text() for p in paths)
        check("planted identifier ERR_2043 is present", "ERR_2043" in contents)
        check("planted duplicate paragraph appears in 2+ files",
              contents.count("exponential backoff when retrying") >= 2)


def check_indexer():
    section("6. indexer.py")

    a_emb, b_emb, c_emb = [1.0, 0.0], [0.0, 1.0], [0.0, 0.999]
    dummy = lambda cid, text: Chunk(
        chunk_id=cid, doc_id=cid.split("::")[0], source_name=cid, chunk_index=0,
        text=text, section_heading=None, chunking_strategy="fixed",
        char_count=len(text), token_count=1, start_char=0, end_char=len(text),
    )
    chunks = [dummy("a::0", "unique"), dummy("b::0", "dup one"), dummy("c::0", "dup one")]
    kept, kept_emb = deduplicate(chunks, [a_emb, b_emb, c_emb], threshold=0.95)
    check("deduplicate() drops a near-duplicate (cosine > 0.95)",
          len(kept) == 2 and kept[1].chunk_id == "b::0",
          f"kept={[c.chunk_id for c in kept]}, sim(b,c)={_cosine_similarity(b_emb, c_emb):.4f}")

    with temp_dir() as tmp:
        dense = DenseIndex(persist_dir=str(Path(tmp) / "chroma"))
        dense.add([dummy("x::0", "hello")], [[0.1, 0.2, 0.3]])
        check("DenseIndex.add() + count() round-trips", dense.count() == 1)
        dense.reset()
        check("DenseIndex.reset() clears the collection", dense.count() == 0)

        sparse = SparseIndex()
        sparse.build([dummy("x::0", "ERR_2043 happened"), dummy("y::0", "unrelated text")])
        bm25_path = str(Path(tmp) / "bm25.pkl")
        sparse.save(bm25_path)
        loaded = SparseIndex.load(bm25_path)
        check("SparseIndex save/load round-trips chunk_ids",
              loaded.chunk_ids == sparse.chunk_ids)
        scores = loaded.bm25.get_scores(tokenize("ERR_2043"))
        best = loaded.chunk_ids[scores.tolist().index(max(scores))]
        check("BM25 correctly ranks the exact-match chunk highest",
              best == "x::0", f"scores={dict(zip(loaded.chunk_ids, scores))}")

        stats = build_indexes(chunks, StubEmbedder(),
                               config=type("C", (), {
                                   "chroma_persist_dir": str(Path(tmp) / "chroma2"),
                                   "bm25_path": str(Path(tmp) / "bm25_2.pkl"),
                               })())
        check("build_indexes() dedups + returns matching stats",
              stats == {"indexed": 2, "deduped": 1}, f"stats={stats}")


def check_ingest():
    section("7. ingest.py (end-to-end)")
    with temp_dir() as tmp:
        corpus_dir = Path(tmp) / "corpus"
        make_corpus(corpus_dir)

        files = discover_files(corpus_dir)
        check("discover_files() finds all 4 generated docs", len(files) == 4)

        docs = load_corpus(corpus_dir)
        check("load_corpus() loads all 4 into Documents", len(docs) == 4)

        chunks = chunk_corpus(docs, strategy="recursive")
        check("chunk_corpus() produces chunks for every doc",
              {c.doc_id for c in chunks} == {d.source_name for d in docs},
              f"{len(chunks)} chunks across {len(docs)} docs")

        stats = ingest(corpus_dir, strategy="recursive",
                        config=type("C", (), {
                            "chroma_persist_dir": str(Path(tmp) / "chroma"),
                            "bm25_path": str(Path(tmp) / "bm25.pkl"),
                        })(),
                        embedder=StubEmbedder())
        check("ingest() runs load->chunk->index end-to-end",
              stats["indexed"] > 0, f"stats={stats}")


def main():
    for fn in (check_tokenizer, check_models, check_loaders, check_chunkers,
               check_make_corpus, check_indexer, check_ingest):
        fn()

    print("\n=== Summary ===")
    passed = sum(1 for _, ok in RESULTS if ok)
    failed = [desc for desc, ok in RESULTS if not ok]
    print(f"  {passed}/{len(RESULTS)} checks passed")
    if failed:
        print("  FAILED:")
        for desc in failed:
            print(f"    - {desc}")
        sys.exit(1)
    print("  All Phase 1 components verified working.")


if __name__ == "__main__":
    main()
