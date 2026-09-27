# Grounded RAG Pipeline

A Retrieval-Augmented Generation system built from scratch. It answers questions
over internal documents with inline citations, and then checks each citation.
Documents are indexed with both vector and keyword search. Results are fused
and reranked. The answer cites its sources as [1], [2], and a second LLM pass
checks every claim against the passage it cites. If confidence is too low, it
answers "I don't know" instead of guessing.

Runs fully local and free (Ollama + open-source models). A hosted demo on free
infrastructure uses Groq for the LLM. See [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md).

## How a question is answered

1. **Hybrid search.** Dense (`bge-small-en-v1.5` + ChromaDB) and sparse (BM25)
   retrieval, top 10 each. Dense matches meaning. BM25 matches exact identifiers
   like `ERR_2043` that embeddings blur.
2. **Reciprocal Rank Fusion.** `score = Σ 1/(60 + rank)` merges the two
   rankings into the top 20 candidates.
3. **Cross-encoder rerank.** `ms-marco-MiniLM-L-6-v2` scores each
   (question, passage) pair and keeps the top 5.
4. **Grounded generation.** Llama 3.1 answers only from those 5 passages and
   cites every claim.
5. **Citation verification.** Each claim is paired with the passage it cites
   and judged YES/NO by the LLM. Unsupported citations are flagged.
6. **Confidence gate.** A geometric mean of retrieval confidence, citation
   coverage and completeness. Below 0.5 the answer becomes "I don't know
   based on the provided context."

## Evaluation

`golden_qa.json` holds 48 hand-written questions: 21 lookup, 7 multi-hop
(facts from two documents), 11 no-answer and 9 ambiguous. The no-answer and
ambiguous questions check that the system declines when it should.
`scripts/run_eval.py` scores answer correctness, faithfulness, retrieval
relevance, citation accuracy and fallback rate, and runs the same suite
against all three chunking strategies (fixed-size, recursive, semantic).

## Engineering

- **Consistent re-indexing.** Each ingest builds a new index version and
  switches over atomically. Questions in flight keep a consistent snapshot of
  both indexes.
- **Deduplication.** A chunk is skipped when its cosine similarity to an
  already-kept chunk is above 0.95.
- **Secured API** (FastAPI). An API key protects the write endpoints. Upload
  names are stripped of paths, and type and size limits are enforced. The
  ingest folder comes from config, never from the request.
- **Resilience.** `/healthz` and `/readyz` endpoints. LLM calls have timeouts
  and backoff retries. An LLM outage returns 503, and a rate limit returns 429.
- **Observability.** One JSON log line per request, with a request ID and
  timings for each stage.
- **Config-driven.** Every setting can be overridden with a `RAG_*`
  environment variable (see `.env.example`).
- **Tested offline.** 200+ pytest tests. The embedder and LLM are passed in as
  dependencies, so tests need no models or network.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1

python scripts/seed.py                        # generate the sample corpus + build the index
uvicorn api.main:app --port 8000              # API
streamlit run dashboard/app.py                # dashboard at http://localhost:8501
```

Or with Docker: `docker compose up`.

Tests: `pytest tests/`

## Layout

```
src/         Core pipeline: tokenizer, loaders, chunkers, indexer, retriever, LLM, confidence, eval
api/         FastAPI service (routes, schemas, security, logging)
dashboard/   Streamlit UI; backend.py switches between the API and the in-process demo
scripts/     Corpus generator, seeding, evaluation, demo-answer recording
tests/       Unit and API tests
```

## Stack

Python · sentence-transformers · ChromaDB · rank_bm25 · Ollama (Llama 3.1) ·
Groq (hosted demo) · FastAPI · Streamlit · Docker Compose
