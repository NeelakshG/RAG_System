# Project 6 — RAG Pipeline with Hybrid Search (build-from-scratch, learn-by-writing)

This file primes you (Claude Code) to coach me through building this project myself.
Read it fully before we start.

## The mission
Build a production-grade, LOCAL, $0 Retrieval-Augmented Generation system over internal
docs: ingest documents, index them with BOTH dense vector search and sparse keyword search,
retrieve with hybrid fusion + reranking, generate grounded answers with verified inline
citations, and evaluate it all on a hand-written golden Q&A set.

## Working agreement (IMPORTANT — how we collaborate)
- **I write as much of the code as I can myself.** This is a learn-by-writing exercise; the
  goal is that I understand every line.
- **Your default is to COACH, not to write:** give me the spec, the function signature, the
  *why*, and a test I can check myself against — but let ME write the body.
- **When I explicitly ask you to write/finish/wrap something up, say yes and just do it,
  fully.** No pushback. This is "I do as much as I see fit, then hand off."
- After I write a piece, review it: point out bugs, edge cases, and explain fixes clearly.
- Keep me oriented: after each component, tell me what's next.

## The learn-by-writing loop (per component)
1. You give me: purpose, the signature (inputs/outputs), key decisions, and a test.
2. I write the implementation.
3. I paste it back; you review + explain fixes.
4. We run the test; move to the next component.

## The local $0 stack (no paid APIs, runs on my machine)
- Language: Python 3.11+ (venv)
- Embeddings: sentence-transformers `BAAI/bge-small-en-v1.5` (local, injected via interface)
- Reranker: local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM (generation + LLM-as-judge): Ollama (`ollama pull llama3.1`), host http://localhost:11434
- Vector store: ChromaDB (file-based, `data/chroma`)
- Sparse search: `rank_bm25`, persisted to `data/bm25.pkl`
- API: FastAPI + uvicorn; Dashboard: Streamlit; Containerize: Docker Compose
- Everything config-driven from a central `config.py`.

## Build order (smallest-first so each piece is graspable)
### Phase 1 — Ingestion & indexing (we are here)
1. `src/tokenizer.py` — offline token counter/splitter (regex `\w+|[^\w\s]`); approximates LLM
   tokens; used as the size unit for chunking.
2. `src/models.py` — `Document` and `Chunk` dataclasses. Chunk metadata (source, chunk_index,
   section_heading, chunking_strategy, char_count, token_count) is what powers citations + eval.
3. `src/loaders.py` — multi-format loader (md/txt/html/pdf) -> normalized `clean_text` + metadata.
   Keep `raw_text` too. PRESERVE `#`/`##` headings (chunkers need them). HTML: strip script/style,
   convert h1-h6 to markdown headings.
4. `src/chunkers.py` — three switchable strategies behind one `Chunker.chunk(doc)->list[Chunk]`
   interface, via `get_chunker(strategy, config, embedder=None)`:
   - fixed-size + overlap (slide token window, step = size - overlap; slice original string)
   - recursive/structure-aware (separator hierarchy: headings -> paragraphs -> lines -> sentences
     -> words; recurse when > max_tokens; then merge small adjacent pieces)
   - semantic (embed sentences; cut where cosine distance between neighbors exceeds a percentile;
     embedder is INJECTED so this module never imports sentence-transformers; use a StubEmbedder
     for offline tests)
5. `scripts/make_corpus.py` — generate a synthetic "internal docs" corpus with PLANTED exact
   identifiers (error codes like ERR_2043, function signatures, config keys) so hybrid search can
   beat dense-only, and ONE verbatim duplicate paragraph in two docs for the dedup step to catch.
6. `src/indexer.py` — embed each chunk (bge-small) -> ChromaDB dense index; build BM25 sparse index
   over the SAME chunks; keep chunk IDs identical across both (never let them drift).
7. Dedup — before inserting a chunk, skip if cosine > 0.95 vs existing (catches the planted dup).
8. `ingest.py` — end-to-end entry point: load -> chunk -> embed -> index.

### Phase 2 — Hybrid retrieval engine
dense retrieval (Chroma) + sparse retrieval (BM25) -> Reciprocal Rank Fusion (score = sum 1/(k+rank),
k=60) -> cross-encoder reranker over top ~20, keep top 5. Funnel: cheap-broad first, expensive-precise last.

### Phase 3 — Generation & citation layer (HARDEST — budget patience)
grounded-generation prompt (answer ONLY from context; cite [1][2]; say "I don't know" when context is
insufficient) -> citation verification (LLM-as-judge per claim/chunk pair, flag unsupported) ->
composite confidence score (retrieval confidence + citation coverage + completeness) -> graceful
"I don't know" path. Note: local models need coaxing for reliable citations/structured output.

### Phase 4 — Evaluation (2nd hardest — effort + trustworthiness)
50+ hand-written golden Q&A (lookups, multi-hop across 2 docs, no-answer, ambiguous) -> automated
metrics: answer correctness, faithfulness, retrieval relevance, citation accuracy (LLM-as-judge) ->
run the SAME suite across all three chunking strategies to produce the comparison table (headline artifact).

### Phase 5 — API & dashboard
FastAPI (POST /v1/ask, POST /v1/ingest, GET /v1/documents) + OpenAPI; Streamlit dashboard (answer with
clickable citations, retrieved chunks, confidence breakdown, hybrid-vs-dense toggle); docker-compose + seed script.

### Phase 6 — Polish
<4-min demo (ingest -> ask -> citation verification catches a hallucination -> hybrid beats dense);
case study led with eval numbers ("X% faithfulness, Y% citation accuracy on a 50-question suite").

## Key design principles to hold throughout
- Config-driven: all sizes/k-values/thresholds/model names live in `config.py`.
- Dependency injection for the embedder (keeps chunker/tests offline).
- Uniform Chunk objects from all 3 strategies so Phase 4 can compare them fairly.
- Two indexes over the EXACT SAME chunks; identical IDs; rebuild both together.
- Dedup threshold 0.95 (high on purpose — lower deletes similar-but-distinct chunks).

## Known gotchas (learned the hard way)
- The naive regex tokenizer only approximates real LLM tokens — fine for a local build.
- Semantic chunking is brittle: it splits on `.!?`, so it mangles "e.g." and can't split a
  markdown table (no sentence punctuation) — a table becomes one oversized chunk. Expect this.
- Keep dense + sparse indexes in sync or RRF fuses rankings over different corpora (silent bug).

## Reference material
- Study guide: `../project6-rag/Project6_Study_Guide.docx` (concepts: RRF, cross-encoders,
  citation verification, eval metrics).
- A prior reference implementation exists at `../project6-rag/` — treat it as an ANSWER KEY:
  I should try to write each file myself first, and only peek if stuck.

## First step
When I'm ready, I'll say "give me the spec for the tokenizer." Give me the signature, purpose,
the token-approximation caveat, and a tiny test — then let me write it.
