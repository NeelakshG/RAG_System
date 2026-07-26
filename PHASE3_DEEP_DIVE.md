# Phase 3 Deep Dive — Generation & Citation Layer

Phase 3 turns the top-5 ranked chunks from Phase 2 into a trustworthy
answer: grounded generation -> citation verification -> composite
confidence score -> graceful "I don't know" fallback. This document walks
every piece in build order, the mechanics of each, and the real
end-to-end traces (including one real bug found and fixed) that proved it
works.

## The pipeline

```
query + chunks (from Phase 2's retrieve())
  |  build_prompt(): assemble numbered context + instructions
  v
OllamaClient.generate(): POST http://localhost:11434/api/generate
  |  raw answer text, e.g. "ERR_2043 means ... [1]."
  v
extract_claims(): split into sentences, pull citation numbers per sentence
  |  [(sentence, [citation_numbers]), ...]
  v
verify_citations(): per (claim, citation_number), ask the LLM AGAIN --
  |  "does this specific chunk support this specific claim?"
  v
compute_confidence(): geometric mean of retrieval / citation / completeness
  |  one float, 0.0-1.0
  v
answer_with_confidence(): if confidence < threshold, swap in
                           NO_ANSWER_SENTINEL instead
```

Same "cheap-broad first, expensive-precise last" shape as Phase 2's
funnel, but the expensive step here is an LLM call, not a neural rerank
pass -- and unlike Phase 1-2, nothing in this phase is fully
deterministic: the same prompt can get a differently-worded (though
hopefully equally correct) answer on different runs.

---

## 1. `build_prompt` (3.1) -- `src/llm.py`

```python
def build_prompt(query: str, chunks: list[Chunk]) -> str:
    prompt = INSTRUCTIONS
    for i, chunk in enumerate(chunks, start=1):
        prompt += f'[{i}], Chunk Source name: {chunk.source_name}, Chunk Text: {chunk.text} \n'
    prompt += f'Question: {query}'
    return prompt
```

**Purpose:** turn retrieved chunks into the single text blob an LLM
actually consumes. Pure string assembly -- no network, no model, fully
offline and deterministic.

**The numbering convention:** `enumerate(chunks, start=1)` is the source
of truth for what `[1]`, `[2]` mean everywhere downstream. The number is
never semantically meaningful on its own -- it's a positional pointer
back into this exact list, which is why `verify_citations` later does
`index = citation_number - 1` to reverse the lookup.

**`INSTRUCTIONS` does three jobs at once:** forces answers to come only
from the provided context (grounding), forces inline citations (`[1]`
style), and forces an exact sentinel string
(`"I don't know based on the provided context."`) when the context is
insufficient -- a fixed string, not free-form refusal text, specifically
so downstream code can check equality instead of parsing intent out of
prose.

---

## 2. `OllamaClient` (3.2) -- `src/llm.py`

```python
class OllamaClient:
    def __init__(self, model="llama3.1", host="http://localhost:11434", timeout=60.0):
        self.model = model
        self.host = host
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"]

    def answer(self, query: str, chunks: list[Chunk]) -> str:
        prompt = build_prompt(query, chunks)
        return self.generate(prompt)
```

**`__init__` vs. `generate` -- the key distinction that took a moment to
click:** `__init__` runs once, when the client object is created, and
only stores config (model name, host, timeout). It never touches the
network. `generate()` is what actually dials out -- every call is a fresh
HTTP request. Mental model: `__init__` = "save the phone number,"
`generate()` = "actually place the call."

**`stream: False`:** Ollama defaults to streaming the response back as
multiple newline-separated JSON chunks (for a live "typing" effect).
Setting it `False` collects the whole generation server-side first and
returns one single JSON object -- simpler to parse, at the cost of
blocking until the entire answer is done generating (no partial output
while waiting).

**`answer()` is pure orchestration** -- it does no string work and no
HTTP work itself, just calls the function that does the first
(`build_prompt`) and the method that does the second (`generate`). Same
shape Phase 2's `retrieve()` uses to stitch together dense query + sparse
query + fusion + rerank without doing any of those things itself.

**Verified end-to-end against real Ollama + real llama3.1:**
```python
client.answer("What does ERR_2043 mean?", [chunk])
# -> "ERR_2043 means the retry budget was exhausted after 3 attempts [1]."
```
(Note: this specific example used a hand-crafted test chunk, not the real
corpus -- the *actual* `ERR_2043` in `troubleshooting.md` is about
intake-service uploads, not retry budgets. Good reminder that hand-typed
examples during dev aren't the same as ground truth from the real
corpus -- exactly the gap Phase 4's golden Q&A suite exists to close.)

---

## 3. `extract_claims` (3.3) -- `src/llm.py`

```python
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

def extract_claims(answer: str) -> list[tuple[str, list[int]]]:
    sentences = SENTENCE_SPLIT_PATTERN.split(answer.strip())
    claims = []
    for sentence in sentences:
        if not sentence:
            continue
        citation_numbers = [int(n) for n in CITATION_PATTERN.findall(sentence)]
        if citation_numbers:
            claims.append((sentence, citation_numbers))
    return claims
```

**Purpose:** the offline half of citation verification -- pull
(sentence, cited_numbers) pairs out of generated text with regex, no LLM
call needed.

**The lookbehind split, `(?<=[.!?])\s+`:** splits on whitespace, but only
whitespace immediately *after* a `.`/`!`/`?`. Using a lookbehind (rather
than literally splitting on `". "`) keeps the punctuation attached to the
sentence it ends, instead of the split consuming it.

**Same brittleness caveat as Phase 1's semantic chunker:** this naive
sentence splitter mangles abbreviations like "e.g." the same way. Known,
accepted tradeoff for a local $0 build -- flagged in the project's
CLAUDE.md from the start.

**Sentences with zero citations are silently dropped** -- nothing to
verify if nothing was cited (e.g. connective sentences, or the
`NO_ANSWER_SENTINEL` itself, which has no `[n]` markers at all).

**Verified:**
```python
extract_claims("ERR_2043 means the retry budget was exhausted [1]. The default retry count is 3 [2].")
# -> [("ERR_2043 means the retry budget was exhausted [1].", [1]),
#     ("The default retry count is 3 [2].", [2])]

extract_claims("Both apply here [1][2].")
# -> [("Both apply here [1][2].", [1, 2])]   -- multi-citation sentences produce one entry, multiple numbers

extract_claims("I don't know based on the provided context.")
# -> []   -- no citations, nothing to check
```

---

## 4. `verify_citations` (3.4) -- `src/llm.py`

```python
@dataclass
class CitationCheck:
    claim: str
    citation_number: int
    chunk: Chunk | None
    supported: bool

def build_verification_prompt(claim: str, chunk: Chunk) -> str:
    return (
        f"{VERIFICATION_INSTRUCTIONS}"
        f"Context: {chunk.text}\nClaim: {claim}\n"
        "Does the context support the claim?"
    )

def verify_citations(answer, chunks, client) -> list[CitationCheck]:
    checks = []
    for claim, citation_numbers in extract_claims(answer):
        for number in citation_numbers:
            index = number - 1
            if index < 0 or index >= len(chunks):
                checks.append(CitationCheck(claim, number, None, False))
                continue
            chunk = chunks[index]
            response = client.generate(build_verification_prompt(claim, chunk))
            supported = response.strip().upper().startswith("YES")
            checks.append(CitationCheck(claim, number, chunk, supported))
    return checks
```

**The core idea -- what `[1]` is actually compared against:** the
citation number itself is never "checked" in any abstract sense. It's
just an address: `index = number - 1` converts it back into a list
position (`[1]` -> `chunks[0]`), mirroring `build_prompt`'s
`enumerate(chunks, start=1)` exactly. The real work is comparing the
**text** at that address against the claim's **text** -- a second,
independent LLM call, completely separate from the one that generated the
answer. That second call never sees the original question or the other
chunks; it only sees one claim and one chunk, in isolation, specifically
so it can't rationalize "the answer sounds plausible overall" -- it has
to judge whether *this specific pairing* holds up.

**Why a second LLM call at all, instead of trusting the first one's own
citations:** the model that generated the answer has every tendency (not
malice, just how next-token prediction works) to sound confident and
attach *some* citation, whether or not it verified the number correctly.
Verification is deliberately independent skepticism, not a rubber stamp.

**Out-of-range citations are treated as unsupported without spending an
LLM call** -- if the model hallucinates `[5]` but only 3 chunks existed,
that's already evidence of a broken citation; no need to ask the judge
model to confirm what bounds-checking already proved.

**Verified end-to-end** -- real chunk 1 supports the claim, real chunk 2
(about `max_upload_size`) doesn't:
```python
verify_citations(
    "ERR_2043 means the retry budget was exhausted [1]. The default retry count is 3 [2].",
    [chunk1, chunk2], client,
)
# -> [CitationCheck(..., citation_number=1, supported=True),   # chunk1 actually says this
#     CitationCheck(..., citation_number=2, supported=False)]  # chunk2 is about something else entirely
```

**Important scope boundary:** a failed check does **not** trigger
re-retrieval or an automatic retry loop. That would require an agentic
architecture (retry limits, loop termination conditions) this project
deliberately doesn't build. A failed citation is *flagged*, then fed into
confidence scoring -- the system is honest about uncertainty, not
self-healing.

---

## 5. `compute_confidence` (3.5) -- `src/confidence.py`

```python
def retrieval_confidence(retrieval_results: list[tuple[str, float]]) -> float:
    if not retrieval_results:
        return 0.0
    return 1 / (1 + math.exp(-retrieval_results[0][1]))   # sigmoid of the top rerank score

def citation_coverage(citation_checks: list[CitationCheck]) -> float:
    if not citation_checks:
        return 0.0
    return sum(1 for c in citation_checks if c.supported) / len(citation_checks)

def completeness(answer: str) -> float:
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if s]
    if not sentences:
        return 0.0
    return len(extract_claims(answer)) / len(sentences)

def compute_confidence(answer, retrieval_results, citation_checks,
                        retrieval_weight=1/3, citation_weight=1/3, completeness_weight=1/3) -> float:
    r = retrieval_confidence(retrieval_results)
    c = citation_coverage(citation_checks)
    p = completeness(answer)
    return (r ** retrieval_weight) * (c ** citation_weight) * (p ** completeness_weight)
```

**Why `retrieve()`/`rerank()` had to change first:** they originally
returned bare `chunk_id` strings, discarding the cross-encoder's actual
scores. `retrieval_confidence` needs those scores, so `rerank()` was
changed to return `list[tuple[chunk_id, score]]` instead -- a real (if
small) retrofit of already-"finished" Phase 2 code, not something
designed in from the start. Blast radius turned out to be just 4 test
assertions, since nothing downstream had been wired to the old shape yet.

**Sigmoid on the raw cross-encoder score:** the score itself is an
unbounded logit (could be `-5`, could be `+8`), so sigmoid squashes it
into a comparable `(0, 1)` range regardless of scale -- same function
used in ML generally to turn a logit into something interpretable as a
probability-like confidence.

**`completeness` is a free, offline signal** -- it reuses `extract_claims`
(no extra LLM call) to measure what fraction of the answer's sentences
carry a citation at all, as a proxy for "how much of this answer is
grounded vs. just asserted."

### The real bug: arithmetic mean let a bad citation hide

The first version of `compute_confidence` used a weighted **arithmetic
mean** (`r*w1 + c*w2 + p*w3`). Testing it against a case with one
completely unsupported citation (`citation_coverage = 0.0`) but a strong
retrieval score (`sigmoid(2.0) ≈ 0.881`) and full completeness (`1.0`)
produced:

```
(0.881 + 0.0 + 1.0) / 3 ≈ 0.627   -- ABOVE the 0.5 threshold
```

A hallucinated citation was getting **masked** by the other two strong
components -- exactly the failure mode citation verification exists to
catch, slipping through the score meant to act on it.

**The fix: switch to a weighted geometric mean.** Because geometric mean
is multiplicative, any near-zero factor collapses the *entire* product
toward zero, no matter how strong the other factors are:

```
(0.881 ** 1/3) * (0.0 ** 1/3) * (1.0 ** 1/3) = 0.0   -- correctly below threshold
```

Rerunning the *exact same inputs* that previously slipped through at
`0.627` now correctly produces `0.0`. This is the "weakest link" property
the score needed: all three signals must be strong for confidence to be
high, and any single one failing makes the whole answer untrustworthy --
matching how citation verification is actually meant to be used.

**Verified (regression-checked both directions):**
```python
# healthy case -- real chunk, real citation, strong retrieval score
answer_with_confidence(...) # -> confidence=0.999, fallback_triggered=False

# same bad-citation case that leaked through the old arithmetic mean
answer_with_confidence(...) # -> confidence=0.0, fallback_triggered=True (now correctly caught)
```

---

## 6. `answer_with_confidence` (3.6) -- `src/confidence.py`

```python
@dataclass
class AnswerResult:
    text: str
    confidence: float
    fallback_triggered: bool

def answer_with_confidence(query, chunks, retrieval_results, client, threshold=0.5) -> AnswerResult:
    if not chunks:
        return AnswerResult(NO_ANSWER_SENTINEL, 0.0, False)

    text = client.answer(query, chunks)
    if text == NO_ANSWER_SENTINEL:
        return AnswerResult(text, 0.0, False)

    checks = verify_citations(text, chunks, client)
    confidence = compute_confidence(text, retrieval_results, checks)

    if confidence < threshold:
        return AnswerResult(NO_ANSWER_SENTINEL, confidence, True)
    return AnswerResult(text, confidence, False)
```

**Two short-circuits, both deliberately *not* counted as "fallback
triggered":**
- **Empty `chunks`** (retrieval found nothing) -- skips generation
  entirely, since there's no context to even attempt an answer from. No
  wasted LLM call.
- **Model already says "I don't know"** -- skips verification and
  scoring entirely, since there are no citations to check. This isn't
  overriding the model; the model's own answer already was the honest
  one.

**`fallback_triggered=True` means something specific:** the model
*attempted* a real answer, but the composite score judged it untrustworthy
enough to withhold. This is the only path where the system actively
overrides what the LLM produced.

**The real score is preserved even on fallback** (`confidence` is the
actual computed value, not reset to `0.0`) so a caller -- API response,
dashboard -- can distinguish "confidently wrong, score 0.05" from
"a near miss, score 0.48," even though both currently produce the same
user-facing text.

**Explicitly out of scope (by design, not oversight):** no retry loop, no
automatic re-retrieval on low confidence. Per the project's Phase 3 spec,
this is a single forward pass -- retrieve once, generate once, verify
once, decide once. An agentic self-correction loop is a legitimate
future direction, but a different, heavier architecture than what this
phase builds.

---

## Test coverage

All of Phase 3 was verified against **real Ollama + real llama3.1**, not
just stubs -- a deliberate departure from Phases 1-2, where deterministic
stub embedders/rerankers were enough. Confidence scoring's arithmetic
mean -> geometric mean bug was caught specifically because a real
end-to-end run against a stub client surfaced a case the unit-level pieces
individually looked fine on.

---

## What's next -- Phase 4 (Evaluation)

50+ hand-written golden Q&A pairs (lookup / multi-hop / no-answer /
ambiguous) against the real corpus -> automated metrics (answer
correctness, faithfulness, retrieval relevance, citation accuracy, all
LLM-as-judge) -> run the same suite across all three chunking strategies
to produce the headline comparison table.

**Why this phase exists:** everything tested so far has been "does this
one hand-picked example look right." Phase 4 replaces spot-checking with
a fixed, repeatable exam graded against ground truth *you* define by
hand -- the only way to know a change (new chunking strategy, reweighted
confidence score, prompt tweak) actually helped instead of just feeling
different. `GoldenQuestion` (`src/eval.py`) and its JSON-backed loader are
the first piece: `id`, `question`, `category`, `expected_answer`,
`expected_source_docs` per entry, stored in `eval/golden_qa.json` (hand
authored, tracked in git -- unlike `data/`, which is regenerable and
gitignored). The four metrics functions are the next component to spec.
