# Resume points — Project 6 (RAG from scratch)

_Last updated: 2026-07-14_

## Where things stand

**Phase 1 (Ingestion & indexing) — done.** tokenizer, models, loaders, all
three chunkers, synthetic corpus generator, indexer (dense + sparse, dedup),
`ingest.py`.

**Phase 2 (Hybrid retrieval) — done.** `src/retriever.py`: dense + sparse
query, RRF fusion, cross-encoder rerank, full `retrieve()` funnel. Notes
captured in `PHASE2_DEEP_DIVE.md`.

**Phase 3 (Generation & citation) — done.** `src/llm.py` (grounded generation
+ citation verification via `OllamaClient`, `extract_claims`,
`verify_citations`, `NO_ANSWER_SENTINEL`), `src/confidence.py`
(`retrieval_confidence`, `citation_coverage`, `completeness`,
`compute_confidence` as a weighted **geometric** mean, `answer_with_confidence`
wiring the "I don't know" fallback on low confidence). Notes captured in
`PHASE3_DEEP_DIVE.md`.

**Phase 4 (Evaluation) — done.** `golden_qa.json` (50+ hand-written
lookup/multi-hop/no-answer/ambiguous questions), `src/eval.py` (metrics:
correctness, faithfulness, retrieval relevance, citation accuracy, all
LLM-as-judge via `OllamaClient`), `scripts/run_eval.py` (runs the golden set
across all three chunking strategies, writes `data/eval/comparison.json` for
the dashboard's Eval tab).

**Phase 5 (API & dashboard) — done.** `api/` (FastAPI: `/v1/ask`,
`/v1/ingest`, `/v1/documents`, `RAGService` wrapping the pipeline),
`dashboard/` (Streamlit: Ask / Documents / Eval comparison tabs, HTML render
helpers), `Dockerfile` + `docker-compose.yml` (api + dashboard services,
Ollama reached via `host.docker.internal`), `scripts/seed.py`.

## Next up

**Phase 6 (Polish) — not started.**
1. <4-minute demo video: ingest -> ask -> citation verification catching a
   hallucination -> hybrid beating dense.
2. Case study write-up led with the eval numbers (correctness, faithfulness,
   retrieval relevance, citation accuracy per chunking strategy).
3. Stretch goal (after Phase 6 or alongside it): fine-tune the cross-encoder
   reranker on the golden Q&A set.

## Also noted

- Branch is ahead of `origin/main` — not pushed yet.
- `PHASE1_DEEP_DIVE.md`, `PHASE2_DEEP_DIVE.md`, `PHASE3_DEEP_DIVE.md` are
  personal notes docs; `PHASE1_DEEP_DIVE.md` is committed, the other two are
  still untracked — decide if they should be gitignored or committed too.
