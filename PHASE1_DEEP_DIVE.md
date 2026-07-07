# Phase 1 Deep Dive — Ingestion & Indexing

RAG pipeline, Phase 1: turning raw files on disk into two searchable indexes
(dense + sparse). This document walks every file in the order it was built,
explains what each function does and *why*, and includes real traces run
against the actual generated corpus — including two real bugs discovered
along the way.

## Pipeline overview

```
raw files (.md/.txt/.html)
   |  loaders.py
   v
Document (raw_text, clean_text, metadata)
   |  chunkers.py
   v
list[Chunk]        <- text slices, NOT yet searchable
   |  indexer.py  (called by ingest.py)
   v
1. embed every chunk's text            (embedder, injected)
2. dedup near-identical embeddings     (deduplicate())
3. build Chroma collection             (DenseIndex)  -> data/chroma/
4. build BM25 index                    (SparseIndex) -> data/bm25.pkl
   |
   v
two indexes on disk, same chunk IDs, ready to be queried
   |  (Phase 2 - not built yet)
   v
retrieval: query both indexes -> fuse rankings (RRF) -> rerank -> top 5 chunks
```

`ingest.py` is the entry point you actually run. `indexer.py` is a building
block `ingest.py` calls. Everything else (`tokenizer.py`, `models.py`,
`loaders.py`, `chunkers.py`) exists to produce the `list[Chunk]` that
`indexer.py` needs.

---

## 1. `src/tokenizer.py`

```python
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text)

def count_tokens(text: str) -> int:
    return len(tokenize(text))

def find_token_spans(text: str) -> list:
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
```

**Purpose:** a cheap, local, model-agnostic approximation of "how many
tokens is this text." Every chunker needs a consistent unit for deciding
"is this piece too big" — using a real LLM's tokenizer would tie chunk
sizing to one specific model's vocabulary. This regex is not simulating any
real LLM's tokenizer; it's a **proxy measurement** used to keep chunks a
sane, bounded size.

**The regex** `\w+|[^\w\s]`:
- `\w+` greedily grabs a run of word characters (letters/digits/underscore)
  as one token — `"ERR_2043"` stays one token because `_` counts as `\w`.
- `[^\w\s]` — any single non-word, non-whitespace character (punctuation)
  becomes its own one-character token.
- Whitespace matches neither branch and is silently skipped.

**Three functions, one engine:**
- `tokenize` — the actual splitter, returns token strings.
- `count_tokens` — `len(tokenize(text))`, used wherever only a count matters.
- `find_token_spans` — same regex via `finditer`, keeps `(start_char,
  end_char)` per token instead of the string. This is what chunkers use to
  slice the *original* string at exact character boundaries.

**Known gotcha:** this regex has no concept of sentence boundaries or
abbreviations. `"e.g."` becomes four tokens (`e`, `.`, `g`, `.`), not one.
This doesn't affect `tokenizer.py` itself, but it's the root cause of a
sentence-splitting fragility that shows up later in `chunkers.py`.

---

## 2. `src/models.py`

Two `@dataclass` definitions that give every other file a shared, typed
vocabulary — the contract the whole pipeline is built on.

### `Document` — one loaded source file
```python
doc_id, source_path, source_name, fmt, raw_text, clean_text, metadata
```
- `doc_id` / `source_name` — same value in practice: the file's path
  relative to the corpus root.
- `source_path` — the absolute filesystem path (debugging only, not identity).
- `fmt` — `"md" | "txt" | "html"`.
- `raw_text` vs `clean_text` — `raw_text` is the file's contents completely
  untouched (kept for provenance/audit). `clean_text` is normalized
  (line-ending fixes, HTML stripped to markdown-style headings).
  **Chunkers only ever read `clean_text`.**
- `metadata` — free-form dict, currently always `{}`, an extension point.

### `Chunk` — one retrievable unit
```python
chunk_id, doc_id, source_name, chunk_index, text, section_heading,
chunking_strategy, char_count, token_count, start_char, end_char
```
This object flows through the entire rest of the pipeline. Every field has
a downstream job:
- `chunk_id` — `f"{doc_id}::{chunk_index}"`, globally unique. Keeps the
  dense and sparse indexes pointing at the exact same chunk.
- `section_heading` — nearest markdown heading above this chunk (or
  `None`). Powers citations like "this came from the 'Retry Utilities'
  section," not just "this came from api_reference.html."
- `chunking_strategy` — `"fixed" | "recursive" | "semantic"`, stamped on
  every chunk. Enables Phase 4's "run the same eval across all 3
  strategies" comparison.
- `char_count` / `token_count` — precomputed size stats.
- `start_char` / `end_char` — this chunk's position in the original
  `clean_text`.

**`to_metadata()`** — returns every field except `text` as a dict, for
handing to Chroma as metadata (Chroma already stores the chunk text
separately as its "document" field, so duplicating it into metadata would
be redundant).

---

## 3. `src/loaders.py`

The bridge between "a file on disk" and "a `Document` object." Different
formats need different handling to reach the same normalized shape — this
file contains that mess so nothing downstream has to know which format a
chunk originally came from.

- **`_detect_format(path)`** — `path.suffix[1:]` (`.md` -> `md`).
- **`_read_normalized(path)`** — reads a text file, replaces `\r\n` with
  `\n`. Windows-authored files often have `\r\n`; without normalizing,
  every downstream char-offset calculation could get thrown off by an
  invisible `\r`.
- **`_html_to_clean_text(html)`** — via BeautifulSoup: strips
  `<script>`/`<style>`, converts `<h1>`-`<h6>` into markdown-style `#`
  headings in place, then `get_text(separator=" ", strip=True)` extracts
  plain text.
- **`load_file(path, base_dir)`** — the public entry point.
  `source_name = path.relative_to(base_dir).as_posix()` — the path
  relative to the corpus root, forward-slashed even on Windows (stable
  `chunk_id`s regardless of OS). `md`/`txt` are treated identically
  (`_read_normalized`); `html` goes through the BeautifulSoup conversion.
  Anything else raises `ValueError`.

**Known gap:** CLAUDE.md/the project mentions `pdf` as a supported format,
but there's no `pdf` branch — it would hit the `else: raise ValueError`.
Unfinished, not a bug.

---

## 4. `src/chunkers.py`

Three interchangeable strategies behind one interface.

### `Chunker` (base class)
```python
class Chunker:
    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError
```
Just an interface. Lets `get_chunker(strategy)` return *some* `Chunker`
subclass without the caller needing to know which.

### `FixedChunker` — sliding token window with overlap

```python
def _fixed_windows(n_tokens, size, overlap) -> list[tuple[int, int]]:
    step = size - overlap
    windows = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += step
    return windows
```

Concrete trace, `size=10, overlap=3` (`step=7`), `n_tokens=25`:
```
start=0:  end=min(10,25)=10  -> window (0,10)   ; not at end, start=7
start=7:  end=min(17,25)=17  -> window (7,17)    ; not at end, start=14
start=14: end=min(24,25)=24  -> window (14,24)   ; not at end, start=21
start=21: end=min(31,25)=25  -> window (21,25)   ; end==n_tokens -> BREAK
```
Windows 1-3 are exactly 10 tokens; only the **last** window is smaller
(clipped by whatever text remains). The overlap (3 tokens between
consecutive windows) prevents a sentence that falls on a boundary from
being split with no shared context — the cost is redundancy (same text
embedded/indexed twice), an accepted tradeoff.

`_window_to_char_span(spans, start, end)` converts a token-index window
into `(char_start, char_end)` using `find_token_spans`'s per-token offsets,
so `chunk()` can slice `text[char_start:char_end]` **directly from the
original string** — preserving exact whitespace, rather than rejoining
token strings (which would mangle spacing).

**Real downside:** `FixedChunker` has zero awareness of document
structure — it can slice a chunk boundary through the middle of a
sentence. Every chunk gets `section_heading=None`.

### `RecursiveChunker` — structure-aware, separator hierarchy

Splits on headings first; within each section, recursively falls back
through a hierarchy of separators only where needed; then merges
small adjacent pieces back up toward the budget.

**Step 1 — `_split_by_headings(text)`:** finds every markdown heading line
(`_HEADING_RE = r"^#{1,6}\s+.*$"`, MULTILINE) and cuts there, returning
`(heading_text_or_None, section_text)` pairs. Text before the first
heading becomes its own `(None, ...)` preamble section.

**The separator hierarchy** (gentlest to most destructive):
```python
_RECURSIVE_LEVELS = [_split_paragraphs, _split_lines, _split_sentences, _split_words]
```
Three of these share `_split_keep_delimiter(text, pattern)` — splits on a
regex but reattaches the delimiter to the *preceding* piece, guaranteeing
`"".join(pieces) == text` exactly (this is what lets `chunk()` track
offsets with a simple running counter instead of re-searching text).

**Step 2 — `_recursive_split(text, max_tokens, levels)`:**
```python
def _recursive_split(text, max_tokens, levels=None):
    if levels is None:
        levels = _RECURSIVE_LEVELS
    if count_tokens(text) <= max_tokens:
        return [text]                 # already small enough
    if not levels:
        return [text]                 # out of separators, give up
    pieces = levels[0](text)
    if len(pieces) <= 1:
        return _recursive_split(text, max_tokens, levels[1:])   # no-op, try next
    result = []
    for piece in pieces:
        result.extend(_recursive_split(piece, max_tokens, levels[1:]))
    return result
```

Full worked trace — text = `"Sentence one is short. Sentence two is short
too. Sentence three is also short."` (17 tokens total: 5+6+6), `max_tokens=12`:

```
_recursive_split(text, 12, [paragraphs, lines, sentences, words])
  17 > 12 -> not done
  pieces = _split_paragraphs(text) -> [text]   (no blank lines, 1 piece = no-op)
  recurse, SAME text, levels=[lines, sentences, words]
    17 > 12 -> not done
    pieces = _split_lines(text) -> [text]   (no \n, 1 piece = no-op)
    recurse, SAME text, levels=[sentences, words]
      17 > 12 -> not done
      pieces = _split_sentences(text) -> [S1, S2, S3]   <- 3 pieces! worked.
      recurse into EACH piece, levels=[words]:
        _recursive_split(S1, 12, [words]): count_tokens(S1)=5 <=12 -> base case -> [S1]
        _recursive_split(S2, 12, [words]): count_tokens(S2)=6 <=12 -> base case -> [S2]
        _recursive_split(S3, 12, [words]): count_tokens(S3)=6 <=12 -> base case -> [S3]
      result = [S1, S2, S3]
```

**Two mechanics to hold onto:**
1. `levels[1:]` — the level list shrinks by one every recursive call.
   Once a level is tried, it's gone for everything below that point;
   descent is strictly one-directional (never retries a coarser separator).
2. `if len(pieces) <= 1:` — "no-op" detection. If a separator didn't
   actually split anything (returned the text unchanged as 1 piece), that
   attempt is discarded and the **next** separator is tried on the
   **original, unmodified** text.

**Important: 1 piece means opposite things in two different checks.**
`count_tokens(text) <= max_tokens -> return [text]` (before any split is
attempted) is the *good* outcome — already small enough. `len(pieces) <=
1` (after calling a separator) is the *bad*/no-op outcome — that
separator failed to do anything, try the next one.

**How `_split_sentences` actually produces multiple pieces from one
string** (this is `_split_keep_delimiter` under the hood): `re.split`
with a *capturing group* around the pattern keeps the matched delimiters
in the output, interleaved with the surrounding text:
```python
parts = re.split(f"({pattern})", text)
# n matches in the string -> n+1 segments (like n cuts making n+1 rope pieces)
```
For the sentence example (2 sentence-boundary matches found):
```python
parts = [
    "Sentence one is short.",   # text before match 1
    " ",                         # matched delimiter 1
    "Sentence two is short too.", # text before match 2
    " ",                         # matched delimiter 2
    "Sentence three is also short." # remainder
]
```
Then a pairwise loop glues each delimiter onto the piece before it:
```python
for i in range(0, len(parts), 2):
    piece = parts[i]
    if i + 1 < len(parts):
        piece += parts[i + 1]
    pieces.append(piece)
```
producing 3 final pieces, each still carrying its trailing whitespace.

**Step 3 — `_merge_small_pieces(pieces, max_tokens)`:** undoes
over-fragmentation by greedily merging adjacent small pieces:
```python
current = pieces[0]
for piece in pieces[1:]:
    merged = current + piece
    if count_tokens(merged) <= max_tokens:
        current = merged
    else:
        result.append(current)
        current = piece
result.append(current)
```

Real traced example (`max_tokens=10`):
```
pieces (with real token counts):
  A='Short intro. '                              -> 3 tok
  B='Another short one. '                         -> 4 tok
  C='Tiny. '                                       -> 2 tok
  D='This piece is a good bit longer than the rest.' -> 11 tok (already OVER budget alone!)
  E='Last bit.'                                    -> 3 tok

current=A(3)
+B: 3+4=7<=10  -> fits, current=A+B(7)
+C: 7+2=9<=10  -> fits, current=A+B+C(9)
+D: 9+11=20>10 -> doesn't fit -> flush "A+B+C"(9), current=D(11)
+E: 11+3=14>10 -> doesn't fit -> flush "D"(11, oversized), current=E(3)
end -> flush E(3)

result = ["A+B+C"(9 tok), "D"(11 tok, still oversized), "E"(3 tok)]
```

**Two properties:** (1) greedy, no lookahead — once it decides to flush,
that decision is final, it never reconsiders. (2) can only **grow** small
pieces by merging, never **shrink** an already-oversized piece — `D`
stays 11 tokens because merging has no power to split anything further,
only `_recursive_split` (upstream) controls maximum piece size.

**`chunk()`** ties it together: for each `(heading, section_text)`,
recursively split -> merge -> build one `Chunk` per surviving piece, using
a running `offset` (not re-searched, thanks to the join-guarantee) and a
running `index` **across the whole document**, not reset per section.

**Bug found and FIXED (verified against `data/corpus/api_reference.html`):**
HTML-derived `clean_text` was losing newlines around headings.
`_html_to_clean_text` inserts `"\n# Heading\n"` around each heading, but
was then calling `soup.get_text(separator=" ", strip=True)` —
`strip=True` strips whitespace from **each individual text fragment
before joining**, which stripped away the inserted `\n` characters before
they ever reached the final string. Result: the entire document ended up
on what is functionally one continuous line, e.g.:
```
'# API Reference ## Retry Utilities The retry utilities module provides
helpers for handling transient\nfailures when calling downstream
services. ### calculate_retry_backoff ...'
```
`_HEADING_RE`'s `^`/`$` (MULTILINE) only match at actual `\n` boundaries —
with none present, the regex's greedy `.*` swallowed from the first `#`
all the way to the end of the document. Verified directly:
```
sections = _split_by_headings(doc.clean_text)
-> number of sections found: 1   (should be 5)
-> heading: 'API Reference ## Retry Utilities The retry utilities module
             provides helpers for handling transient' (garbage, not "API Reference")
```
**Consequence (before the fix):** HTML documents got zero real
heading-based section splitting — the whole file became one section.
Confirmed in the real ingest run: `api_reference.html` produced exactly
**1 chunk** for its entire 1266-character document, vs. 4 chunks each for
the real `.md`/`.txt` docs.

**The fix** — `src/loaders.py`, `_html_to_clean_text`: stop stripping each
text fragment individually; strip only the final combined string once:
```python
# before:
return soup.get_text(separator=" ", strip=True)
# after:
return soup.get_text(separator=" ").strip()
```
**Verified after the fix:** `_split_by_headings` on `api_reference.html`
now finds **5 correct sections** (`API Reference`, `Retry Utilities`,
`calculate_retry_backoff`, `Document Intake Endpoint`, `Health Check
Endpoint`), each with a clean, accurate heading string. Real
`chunk_corpus` run: `api_reference.html` now produces **5 chunks**
(up from 1), for **17 chunks total** across the corpus (up from 13). All
94 tests still pass. Re-running dedup on the corrected chunks: the two
duplicate-paragraph chunks (`api_reference.html::2`,
`config_guide.txt::1`) now measure **0.8687** cosine similarity (up
slightly from 0.8507, since the paragraph's chunk is less diluted by
unrelated content) — still under the 0.95 threshold, so dedup still
correctly does not fire; the paragraph is still embedded alongside other
section content, not as a fully isolated duplicate.

### `SemanticChunker` — embedding-distance-based cuts

Cuts where consecutive sentences' embeddings differ more than usual for
this document, rather than relying on structure or fixed size.

```python
def chunk(self, doc):
    sentences = _split_sentences(doc.clean_text)
    embeddings = self.embedder.embed(sentences)   # DI - the only embed() call in this class
    distances = _adjacent_distances(embeddings)
    cut_points = _find_cut_points(distances, self.percentile)
    groups = _group_by_cuts(sentences, cut_points)
    # -> one Chunk per group, section_heading=None always
```

- **`_cosine_distance(a, b)`** = `1 - cosine_similarity(a, b)`.
- **`_adjacent_distances`** — compares only *consecutive* pairs (sentence
  0 vs 1, 1 vs 2, ...). `n` sentences -> `n-1` distances.
- **`_percentile(values, pct)`** — the threshold is computed **fresh from
  this document's own distances every time** (default 95th percentile) —
  adaptive, not a fixed absolute cosine cutoff.
- **`_find_cut_points`** — indices where distance exceeds that threshold.
- **`_group_by_cuts`** — walks sentences in order, closing a group the
  instant it hits a cut-point index.

Verified trace with synthetic topic-shift embeddings (3 sentences near
`[1,0,0]`, 3 near `[0,1,0]`):
```
distances[0] (sent 0,1): 0.0012   <- tiny, same topic
distances[1] (sent 1,2): 0.0026   <- tiny
distances[2] (sent 2,3): 0.9948   <- HUGE, topic boundary
distances[3] (sent 3,4): 0.0062   <- tiny
distances[4] (sent 4,5): 0.0026   <- tiny

95th percentile threshold: 0.7971
cut points: [2]  -> cut immediately after sentence index 2

result: 2 groups, cleanly split at the topic boundary
```

**Nothing gets dropped — only "same chunk or new chunk" is decided.**
Every sentence lands in exactly one group.

**Real limitation (linear, not clustering):** the algorithm only ever
compares a sentence to the one *immediately before it*. If sentence 4 is
actually similar to sentence 1, but sentences 2-3 were unrelated (an
"A-B-A" pattern), sentence 4 only ever gets compared to sentence 3 — its
similarity to sentence 1 is never computed anywhere. This is structural,
not a missed detection: `Chunk.start_char`/`end_char` require each chunk
to be one **contiguous** span of the source document. A clustering
approach that pulled non-adjacent similar sentences together would break
that invariant (a chunk's text would no longer be one unbroken substring
of `doc.clean_text`), so the algorithm deliberately never looks back.
**Cost:** more chunks than the document's true topic structure might
ideally produce, purely due to linear placement.

**Cost of this strategy generally:** the only chunker that requires
embedding *every sentence* just to decide where to cut — meaningfully
more compute than `FixedChunker` or `RecursiveChunker` (both zero
embedding calls).

**Shared gotcha:** reuses `_split_sentences`, so `"e.g."` mangling applies
here too — and worse than in `RecursiveChunker`, since this is the *only*
splitting unit used here, not a last resort.

### `get_chunker(strategy, config=None, embedder=None)` — the factory

```python
def get_chunker(strategy, config=None, embedder=None) -> Chunker:
    if strategy == "fixed":
        return FixedChunker(size=getattr(config, "chunk_size", 256),
                             overlap=getattr(config, "chunk_overlap", 32))
    if strategy == "recursive":
        return RecursiveChunker(max_tokens=getattr(config, "max_tokens", 256))
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return SemanticChunker(embedder=embedder,
                                percentile=getattr(config, "semantic_percentile", 95))
    raise NotImplementedError(f"chunking strategy not yet implemented: {strategy}")
```
Makes all three strategies interchangeable behind a string name — this is
what lets Phase 4's "run the same eval across all 3 strategies" be a
simple loop rather than three hardcoded paths. `getattr(config, "x",
default)` is why `config=None` always works everywhere (`getattr(None,
"x", default)` just returns `default`). The `semantic` branch **fails
fast** with a clear `ValueError` if no embedder was provided, rather than
crashing later with a confusing `AttributeError`.

---

## 5. `scripts/make_corpus.py`

Generates a synthetic "internal docs" corpus — not random content, every
piece is **deliberately planted** to exercise a specific part of the
pipeline.

**The 4 documents:**
| File | Format | What it tests |
|---|---|---|
| `troubleshooting.md` | native Markdown | 3 `##` error codes (`ERR_2043` is the real planted constant); real `\n` around headings -> heading-splitting works correctly here |
| `api_reference.html` | HTML | `FUNCTION_SIG`, 1st copy of `DUPLICATE_PARAGRAPH`; this is the file that hits the heading-swallowing bug |
| `config_guide.txt` | plain text, markdown-style headers | 2nd copy of `DUPLICATE_PARAGRAPH`; `.txt` is loaded identically to `.md` |
| `onboarding.md` | native Markdown | no plants — a "control" document |

**The two plants and why:**
- **Exact identifiers** (`ERROR_CODE="ERR_2043"`, `CONFIG_KEY="MAX_RETRY_COUNT"`,
  `FUNCTION_SIG`) — planted so BM25 can reliably find them by exact match,
  demonstrating why hybrid search beats dense-only. Verified with a real
  query against the persisted BM25 index (`"What is ERR_2043?"`):
  ```
  troubleshooting.md::1: 1.5947   <- the chunk that actually defines it
  config_guide.txt::1:   0.6617
  ...
  config_guide.txt::0:   0.0000   <- several chunks score exactly zero
  ```
  The correct chunk wins by more than 2x margin because it's the only one
  containing the literal token `ERR_2043`.
- **`DUPLICATE_PARAGRAPH`** — the exact same paragraph planted verbatim in
  two different files, meant to test `indexer.py`'s `deduplicate()`. Real
  result: it does **not** get caught with `RecursiveChunker` — real
  cosine similarity between the two chunks containing it is **0.8507**,
  below the 0.95 threshold, because each copy is embedded together with
  different surrounding section content (API docs vs. config docs),
  diluting the similarity.

**Structure** — same shape you built for `ingest.py`:
```python
def _write_file(path, content) -> Path:   # low-level disk write
def make_corpus(out_dir) -> list[Path]:    # reusable, testable core, no CLI
def main() -> None:                        # thin argparse wrapper
```

---

## 6. `src/indexer.py`

Takes a `list[Chunk]` and turns it into two searchable indexes. Doesn't
care where the chunks came from.

- **`_cosine_similarity(a, b)`** — used only inside `deduplicate()` here.
  Real value computed between the two duplicate-paragraph chunks:
  **0.8507**.
- **`_chroma_safe_metadata(metadata)`** — strips `None`-valued keys before
  handing metadata to Chroma (Chroma rejects `None`; a missing key is
  fine). Needed because `Chunk.section_heading` is `str | None` — any
  chunk without a heading above it would otherwise crash `DenseIndex.add()`.
- **`deduplicate(chunks, embeddings, threshold=0.95)`** — O(n^2): for each
  chunk, checks cosine similarity against every chunk *already kept* so
  far; drops it if any exceeds `threshold`. First occurrence wins. Real
  run: 17 chunks in, `{'indexed': 17, 'deduped': 0}` — nothing in the real
  corpus clears 0.95 (the closest pair, the duplicate paragraph, reaches
  0.8687).
- **`DenseIndex`** — wraps `chromadb.PersistentClient`. `.reset()` deletes
  and recreates the collection (full-rebuild semantics). `.add()` no-ops
  on empty input, otherwise pushes `chunk_id`s as Chroma IDs, embeddings,
  chunk text as the "document," and sanitized metadata. Chroma builds its
  own internal nearest-neighbor index (HNSW) over the vectors — this file
  never touches that structure directly.
- **`SparseIndex`** — wraps `BM25Okapi`. `.build()` tokenizes every
  chunk's text (via `src.tokenizer.tokenize`, for consistency with the
  rest of the pipeline) and hands the whole corpus to `BM25Okapi`, which
  precomputes term-frequency / inverse-document-frequency / average
  document length statistics up front — that's what makes `.get_scores()`
  fast later, at actual query time. `.save()`/`.load()` pickle
  `{chunk_ids, bm25}` together; `chunk_ids[i]` maps back to which chunk
  produced `corpus[i]`.
- **`build_indexes(chunks, embedder, config=None)`** — the orchestrator:
  embed once -> dedup -> full-rebuild `DenseIndex` -> full-rebuild
  `SparseIndex`, both over the identical surviving `kept_chunks`. This
  single function is what guarantees dense and sparse never drift out of
  sync (no window where one has chunks the other doesn't). Returns
  `{"indexed": n, "deduped": n}`.

---

## 7. `ingest.py`

The end-to-end entry point — the thing you actually run. Produces the
`list[Chunk]` from real files, then calls `indexer.py` to finish the job.

- **`discover_files(corpus_dir)`** — recursive walk (`rglob("*")`),
  filtered to the extensions `load_file` supports (`md`, `txt`, `html`),
  sorted for determinism. Real result: found all 4 corpus files.
- **`load_corpus(corpus_dir)`** — maps `discover_files` -> `load_file` for
  each path. Real result: 4 `Document`s, `clean_text` lengths 1266 / 835 /
  734 / 1701 chars.
- **`chunk_corpus(docs, strategy, config, embedder)`** — one
  `get_chunker(...)` instance, flattens `.chunk(doc)` results across all
  docs into a single list. Real result (`strategy="recursive"`, after the
  loaders.py fix): 17 chunks total — `api_reference.html: 5`,
  `config_guide.txt: 4`, `onboarding.md: 4`, `troubleshooting.md: 4`.
- **`SentenceTransformerEmbedder`** — the one place in the codebase
  allowed to import `sentence_transformers` directly (lazy import inside
  `__init__`, so tests using `StubEmbedder` never pay that cost or need
  the package installed). `.embed(texts)` = `self._model.encode(texts).tolist()`
  — converts the model's numpy output into plain `list[list[float]]`,
  matching every other embedder's interface.
- **`ingest(corpus_dir, strategy="recursive", config=None, embedder=None)`**
  — if no embedder given, constructs a real `SentenceTransformerEmbedder`;
  wires `load_corpus -> chunk_corpus -> build_indexes` together, reusing
  the same embedder instance for both chunking (if `strategy="semantic"`)
  and indexing (always) — one model load, not two. Real run:
  `{'indexed': 17, 'deduped': 0}`.
- **`main()`** — thin `argparse` CLI wrapper: `--corpus-dir` (default
  `data/corpus`), `--strategy` (default `recursive`), calls `ingest()`,
  prints the stats dict.

---

## Real bugs discovered while building/testing this phase

1. **HTML heading-swallowing** (`loaders.py` x `chunkers.py`) — **FIXED.**
   `_html_to_clean_text`'s `get_text(strip=True)` was stripping the
   newlines `loaders.py` inserted around headings before joining
   fragments, so `_HEADING_RE` (which relies on real `\n` boundaries)
   couldn't find per-line heading matches in HTML-derived text. Result:
   HTML documents got zero real section splitting — confirmed via
   `api_reference.html` producing exactly 1 chunk for its entire content,
   and `_split_by_headings` returning `1` section with a garbage,
   entire-document "heading" string.
   **Fix:** stop stripping per-fragment; strip only the final joined
   string once:
   ```python
   # before: soup.get_text(separator=" ", strip=True)
   # after:  soup.get_text(separator=" ").strip()
   ```
   **Verified after fix:** `api_reference.html` now correctly splits into
   5 sections / 5 chunks (up from 1); corpus total went from 13 to 17
   chunks. All 94 tests still pass.

2. **Dedup dilution by chunk-level context** (`indexer.py` x `chunkers.py`)
   — not a bug, a real, verified limitation: a verbatim duplicate
   paragraph planted in two documents does *not* trigger the 0.95 dedup
   threshold, because `RecursiveChunker` embeds each copy together with
   different surrounding section content. Measured real cosine
   similarity: 0.8687 (post heading-fix; was 0.8507 before, when the
   paragraph's chunk was diluted by even more unrelated swallowed
   content). Dedup only catches duplicates when the **whole chunk** is
   near-identical, not when a duplicate passage is diluted inside a
   larger, differently-surrounded chunk. Not something to "fix" — this is
   inherent to chunk-level (vs. passage-level) dedup, and is a useful,
   expected finding rather than a defect.

---

## What's next — Phase 2 (Hybrid Retrieval)

Dense retrieval (Chroma, top-k by cosine) + sparse retrieval (BM25,
top-k) -> Reciprocal Rank Fusion (`score = sum 1/(k + rank)`, k=60) ->
cross-encoder reranker over the fused top ~20 -> keep top 5. Funnel
pattern: cheap-broad first, expensive-precise last. This is the first
component that will actually *query* the two indexes built in Phase 1.
