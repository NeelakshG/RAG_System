# Phase 2 Deep Dive — Hybrid Retrieval Engine

Phase 2 builds the funnel that turns a question into the final ranked
chunks used for generation: dense retrieval + sparse retrieval -> RRF
fusion -> cross-encoder rerank. This document walks every piece in build
order, the mechanics of each, and the real end-to-end trace that proved it
works.

## The funnel

```
query
  |  embed (bi-encoder) + tokenize
  v
dense_index.query(query_embedding, k=10)     sparse_index.query(query_tokens, k=10)
  |  ordered list[str] of chunk_ids            |  ordered list[str] of chunk_ids
  v                                             v
        reciprocal_rank_fusion(dense_results, sparse_results, k=60)
                   |  one merged ranked list[str], best first
                   v
          top ~20 candidates (fusion_candidates)
                   |  dense_index.get_texts(...) -- fetch actual text by ID
                   v
          rerank(query, candidates, cross_encoder, top_n=5)
                   |  cross-encoder re-scores query+chunk TOGETHER
                   v
              final top 5 chunk_ids
```

Cheap-and-broad first (dense/sparse scan the whole corpus, fast), then
progressively narrower and more expensive (fusion is just arithmetic,
reranking is a full neural forward pass per candidate — only run on the
already-small shortlist).

---

## 1. `DenseIndex.query` (2.1)

```python
def query(self, query_embedding: list[float], k: int = 10) -> list[str]:
    if not query_embedding:
        return []
    result = self._collection.query(query_embeddings=[query_embedding], n_results=k)
    return result["ids"][0]
```

**Purpose:** ask Chroma for the k chunks whose stored vectors are closest
to a query vector.

**Mechanics:** `self._collection.query(...)` is Chroma's own built-in
nearest-neighbor search — no manual loop, no `_cosine_similarity` call;
Chroma does the comparison internally using the index it built when
chunks were added in Phase 1. Chroma's API supports *batched* queries
(multiple query vectors in one call), so its response is nested one level
deeper than needed: `result["ids"]` is a `list[list[str]]`, one inner
list per query sent. Since exactly one query embedding is ever sent
(wrapped as `[query_embedding]`), the method unwraps that batch-of-1 down
to a flat list with `result["ids"][0]`.

**Edge case:** an empty/falsy `query_embedding` returns `[]`, not `None`
— keeps the declared `list[str]` return type honest, so callers (like RRF
fusion later) never have to special-case `None`.

**Verified:** `test_dense_index_query_returns_closest_first` — 3 chunks
added with distinct embeddings, `query([1.0, 0.0], k=2)` correctly
returns the 2 closest, nearest-first.

---

## 2. `SparseIndex.query` (2.2)

```python
def query(self, query_tokens: list[str], k: int = 10) -> list[str]:
    if self.bm25 is None:
        return []
    score = self.bm25.get_scores(query_tokens)
    paired = zip(self.chunk_ids, score)
    ranked = sorted(paired, key=lambda pair: pair[1], reverse=True)
    top_k = ranked[:k]
    result = []
    for chunk_id, score in top_k:
        result.append(chunk_id)
    return result
```

**Key difference from `DenseIndex.query`:** Chroma sorts and top-k's
internally; `BM25Okapi` does not. `self.bm25.get_scores(query_tokens)`
scores **every** chunk in the corpus in one call and returns a flat,
**unsorted** array, index-aligned with `self.chunk_ids` (`scores[i]`
belongs to `chunk_ids[i]`, since that's the order they were built in).
Ranking is the caller's job: `zip` pairs IDs with scores, `sorted(...,
reverse=True)` ranks them, `[:k]` takes the top k, and the final loop
extracts just the chunk_ids, discarding the scores.

**Edge case:** `SparseIndex.build([])` leaves `self.bm25 = None`; calling
`.get_scores()` on `None` crashes with `AttributeError`. Guarded with an
early `if self.bm25 is None: return []`.

**Verified real behavior** (from Phase 1's BM25 demo, same mechanism):
querying `"What is ERR_2043?"` against the real corpus scores
`troubleshooting.md::1` at 1.5947 — more than double the next-highest
result (0.6617) — because it's the only chunk containing the literal
token `ERR_2043`.

---

## 3. `reciprocal_rank_fusion` (2.3) — `src/retriever.py`

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60) -> list[str]:
    scores = defaultdict(float)
    for rank, chunk_id in enumerate(dense_results, start=1):
        scores[chunk_id] += 1 / (k + rank)
    for rank, chunk_id in enumerate(sparse_results, start=1):
        scores[chunk_id] += 1 / (k + rank)
    ranked_pairs = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked_pairs]
```

**The problem it solves:** dense scores (cosine similarity, roughly
0-1) and sparse scores (BM25, unbounded, corpus-dependent) are on
**completely incompatible scales** — you cannot just add `0.82` to `5.3`
and have it mean anything. RRF sidesteps this by discarding raw scores
entirely and using only **rank** (ordinal position), which is always
comparable regardless of how the ranking was produced.

**Formula:** `score(chunk) = sum of 1/(k + rank)` across every list the
chunk appears in. `k=60` is a damping constant — not derived, a
well-known standard from the original RRF paper (Cormack et al., 2009),
adopted almost universally in production hybrid search (Elasticsearch,
OpenSearch, Weaviate, Azure AI Search all ship RRF with this same
convention). Rank must be **1-indexed** (`enumerate(..., start=1)`), not
0-indexed, matching the literature's convention.

**Why it rewards agreement:** a `defaultdict(float)` accumulates
contributions per chunk_id; a chunk appearing in *both* lists gets `+=`'d
twice, summing both contributions — no special-case code needed for
"found by both methods."

**Worked trace:**
```
dense_results  = ["a::0", "b::0", "c::0"]     sparse_results = ["b::0", "d::0"]
a::0: 1/61 ≈ 0.01639                          b::0: 1/61 ≈ 0.01639  (added to existing 0.01613!)
b::0: 1/62 ≈ 0.01613                          d::0: 1/62 ≈ 0.01613
c::0: 1/63 ≈ 0.01587

totals: b::0=0.03252 (BOTH lists), a::0=0.01639, d::0=0.01613, c::0=0.01587
fused  = ["b::0", "a::0", "d::0", "c::0"]
```
`b::0` was never #1 in *either* individual list (#2 dense, #1 sparse) but
wins overall because both methods found it — this is the entire point of
hybrid fusion.

**Verified:** `test_reciprocal_rank_fusion_rewards_chunks_in_both_lists`,
plus single-list and empty-input edge cases.

---

## 4. Cross-encoder reranking (2.4) — `src/retriever.py`

### Bi-encoder vs. cross-encoder — the conceptual shift

Every embedder used so far (`bge-small`) is a **bi-encoder**: query and
chunk are embedded **completely separately**, never seeing each other,
then compared afterward via cosine similarity. Fast, precomputable ahead
of time (chunk embeddings were computed once, at ingest).

A **cross-encoder** feeds the query and **one candidate chunk's text
together, as a single combined input**, into the model at once. Internally
(it's a transformer), self-attention lets every word in the query
directly compare against every word in the chunk *during* the same
computation — this cross-referencing is why it's called a cross-encoder.
More accurate, but nothing can be precomputed: every `(query, chunk)`
pair needs a fresh full model pass, live, at query time. This is exactly
why it only ever touches the ~20 fused candidates, never the whole corpus.

```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder   # lazy import
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return self._model.predict(pairs).tolist()
```
`pairs` builds `[(query, text1), (query, text2), ...]` — same query
repeated against each candidate. `.predict()` runs the model on each pair
and returns a raw relevance score (not a bounded probability, just
"higher = more relevant" — sufficient since only sort order is used).
Trained on the MS MARCO dataset (real queries + human-judged relevant/
irrelevant passages), so its judgment is learned, not hand-coded rules.

```python
def rerank(query, candidates: list[tuple[str, str]], reranker, top_n=5) -> list[str]:
    chunk_ids = [chunk_id for chunk_id, text in candidates]
    texts = [text for chunk_id, text in candidates]
    scores = reranker.score(query, texts)
    ranked_pairs = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    top = ranked_pairs[:top_n]
    return [chunk_id for chunk_id, score in top]
```
Same `zip -> sorted -> slice -> extract` idiom as `SparseIndex.query` —
this pattern now appears three times across the funnel (sparse query, RRF
fusion, rerank). `candidates` deliberately takes `(chunk_id, chunk_text)`
pairs rather than chunk_ids alone, since nothing upstream (dense/sparse
query, RRF) ever carries chunk text — only IDs, to keep the cheap early
stages lightweight.

**Simplest-terms summary:** dense/sparse retrieval quickly finds
"probably relevant" (like an automated resume keyword scan narrowing
1,000 down to 20); reranking carefully decides "actually the best" (like
a human reading those 20 side-by-side with the job description) — and it
only works on a small shortlist because it's too slow to run on everything.

**Verified:** `test_rerank_picks_highest_scoring_candidate_first` with a
deterministic `StubReranker` (word-overlap counting, no real model needed
for the test).

---

## 5. `DenseIndex.get_texts` + `retrieve()` (2.5)

### Closing the ID-to-text gap

Every stage up to this point deliberately worked with chunk **IDs only**
(no text) to stay lightweight. But `rerank()` needs actual text. Since
`DenseIndex.add()` already stored each chunk's text as Chroma's
`"documents"` field back in Phase 1, fetching it back is a direct lookup,
not a search:

```python
def get_texts(self, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    result = self._collection.get(ids=chunk_ids)
    return dict(zip(result["ids"], result["documents"]))
```
**Contrast with `.query()`:** Chroma's `.get(ids=[...])` is a direct
"fetch these exact records" lookup, not a similarity search — its
response is **flat**, not wrapped in `.query()`'s batch-of-1 nested-list
structure.

### The orchestrator

```python
def retrieve(query, dense_index, sparse_index, embedder, reranker,
             config=None, use_hybrid=True) -> list[str]:
    dense_k = getattr(config, "dense_k", 10)
    sparse_k = getattr(config, "sparse_k", 10)
    rrf_k = getattr(config, "rrf_k", 60)
    fusion_candidates = getattr(config, "fusion_candidates", 20)
    final_k = getattr(config, "final_k", 5)

    query_embedding = embedder.embed([query])[0]   # embed() always takes/returns a list
    query_tokens = tokenize(query)

    dense_results = dense_index.query(query_embedding, k=dense_k)

    if use_hybrid:
        sparse_results = sparse_index.query(query_tokens, k=sparse_k)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)
    else:
        fused = dense_results

    top_fused = fused[:fusion_candidates]
    texts = dense_index.get_texts(top_fused)
    candidates = [(chunk_id, texts[chunk_id]) for chunk_id in top_fused]

    return rerank(query, candidates, reranker, top_n=final_k)
```

**The `use_hybrid` switch** is the "hybrid-vs-dense-only" comparison mode
`BUILD_STEPS.md` calls for. It only changes what `fused` contains —
reranking still runs identically in both branches. This matters for
Phase 4: comparing hybrid vs. dense-only eval results is only a fair test
if reranking isn't a confounding variable between the two conditions.

**Verified end-to-end against the real corpus and real models:**
```python
retrieve("What is ERR_2043?", dense, sparse, embedder, reranker, use_hybrid=True)
# -> ['troubleshooting.md::1', 'troubleshooting.md::3', 'troubleshooting.md::2',
#     'api_reference.html::1', 'api_reference.html::0']
```
`troubleshooting.md::1` (the chunk that actually defines `ERR_2043`)
correctly lands first, in both hybrid and dense-only mode — for this
particular query, both converged on the same answer, since `ERR_2043` is
unambiguous enough that dense search alone already found it. Where hybrid
is expected to pull ahead is a harder query where dense's semantic
fuzziness leads it astray and only BM25's exact-match strength saves
it — the kind of systematic comparison Phase 4's eval suite exists to
surface.

---

## Industry-standard context (not a toy simplification)

- **RRF** is not invented for this project — it's from a real IR research
  paper and is built into production systems: Elasticsearch (native
  support since 8.9, specifically for combining vector + BM25 search),
  OpenSearch, Weaviate, Azure AI Search. `k=60` is the field's de facto
  standard constant.
- **Cross-encoder reranking** as a funnel's final stage (cheap-broad
  retrieval -> expensive-precise rerank) is the standard architecture for
  production hybrid search, not specific to this project.

---

## Test coverage

105/105 tests passing across the full project by the end of Phase 2,
including: `DenseIndex.query`/`get_texts`, `SparseIndex.query`,
`reciprocal_rank_fusion` (both-lists / single-list / empty), `rerank`
(highest-first / top_n), and `retrieve()` (hybrid / dense-only).

---

## What's next — Phase 3 (Generation & Citations — flagged HARDEST)

Grounded-generation prompt (answer ONLY from retrieved context; cite
`[1][2]`; say "I don't know" when context is insufficient) -> citation
verification (LLM-as-judge per claim/chunk pair) -> composite confidence
score (retrieval confidence + citation coverage + completeness) ->
graceful "I don't know" path.

**Why this phase is a different kind of hard than Phases 1-2:** everything
built so far is deterministic — same input always produces the same,
testable output. Phase 3's core step is an LLM generating free-form text,
which has no such guarantee. Local models (via Ollama) need real prompt
iteration to reliably follow citation/formatting instructions; citation
verification uses *another* LLM call to judge the first one's output,
which is itself fuzzy; confidence scoring is a calibration/design
judgment call, not a formula to derive. Expect "try it on 10 examples and
see how it holds up" rather than clean pass/fail assertions.
