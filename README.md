# project6-rag-scratch

A local, $0 Retrieval-Augmented Generation (RAG) system built from scratch: ingest documents,
index them with both dense vector search and sparse keyword search, retrieve with hybrid
fusion + reranking, generate grounded answers with verified inline citations, and evaluate it
all against a hand-written golden Q&A set.

This is a learn-by-writing project — implementations are written by hand, one component at a
time, rather than scaffolded in bulk.

## Stack (local, no paid APIs)

- Python 3.11+
- Embeddings: `BAAI/bge-small-en-v1.5` (sentence-transformers)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: Ollama (`llama3.1`) for generation and LLM-as-judge
- Vector store: ChromaDB
- Sparse search: `rank_bm25`
- API: FastAPI; Dashboard: Streamlit

## Progress

**Phase 1 — Ingestion & indexing** (complete)
- [x] `src/tokenizer.py` — offline token counter/splitter
- [x] `src/models.py` — `Document` / `Chunk` dataclasses
- [x] `src/loaders.py` — multi-format document loader (md/txt/html/pdf)
- [x] `src/chunkers.py` — fixed-size, recursive, and semantic chunking strategies
- [x] `scripts/make_corpus.py` — synthetic corpus generator
- [x] `src/indexer.py` — dense (Chroma) + sparse (BM25) indexing
- [x] `ingest.py` — end-to-end ingestion entry point

**Phase 2 — Hybrid retrieval** (dense + sparse + RRF + cross-encoder rerank) — not started
**Phase 3 — Generation & citations** — not started
**Phase 4 — Evaluation** — not started
**Phase 5 — API & dashboard** — not started
**Phase 6 — Polish** — not started

## Tests

```
pytest tests/
```

## Layout

```
src/        Core library (tokenizer, models, loaders, chunkers, ...)
tests/      Unit tests, mirroring src/
```
