# Deployment plan

_Last updated: 2026-09-26_

## Public demo: Streamlit Community Cloud + Groq (free, always on)

This is the link for your resume. The dashboard runs the whole pipeline
in-process on Streamlit's free hosting (`RAG_DASHBOARD_BACKEND=embedded`), and
LLM calls go to Groq's free tier instead of local Ollama. Your PC can be off.

**Cost: $0, and it can't start charging.** Neither service has a card on
file. If usage runs out, Groq refuses requests until its limit resets, and
the demo shows a "try again in a minute" message. **Never add a payment
method to the Groq account.**

Built-in safeguards (all in `dashboard/backend.py`):
- A shared answer cache. Repeat questions cost nothing.
- Pre-recorded answers for the example buttons
  (`dashboard/demo_answers.json`). These work even when the quota is used up.
- A global limit of 4 new questions per minute across all visitors, and 5
  per visitor session. Both are set in config.
- Uploads are off. The Groq key lives only in Streamlit's secrets
  store and never reaches the browser.

### One-time setup
1. **Groq key:** sign up at https://console.groq.com (no card), then go to
   API Keys and create a key. Check the model list for the current Llama 3.1
   8B id. The default is `llama-3.1-8b-instant`. If it has been renamed, set
   `RAG_GROQ_MODEL` in the secrets.
2. **Push the code to GitHub** (`main`).
3. **Streamlit:** go to https://share.streamlit.io, sign in with GitHub,
   choose **Create app → Deploy a public app from GitHub**, and set:
   - Repository `NeelakshG/RAG_System`, branch `main`, main file `dashboard/app.py`
   - **Advanced settings → Python 3.12**
   - **Secrets:**
     ```toml
     RAG_DASHBOARD_BACKEND = "embedded"
     RAG_LLM_PROVIDER = "groq"
     RAG_GROQ_API_KEY = "gsk_..."
     ```
   - Pick a custom subdomain, e.g. `neelaksh-rag`, for a clean
     `https://neelaksh-rag.streamlit.app` link.
4. The first boot installs the dependencies and downloads the two models,
   which takes a few minutes. After that, a sleeping app wakes in about 30 seconds.

### Updating it
Push to `main` and Streamlit redeploys automatically. To refresh the
recorded example answers, run `python scripts/record_demo_answers.py`
locally and commit `dashboard/demo_answers.json`. If you've run
`scripts/run_eval.py`, also commit `dashboard/eval_comparison.json`, which
fills the Eval tab.

## Where we are (Stage 0 — DONE 2026-07-26)

The dashboard is public at **https://unchagrined-ungotten-kaden.ngrok-free.dev**
through an ngrok free static domain. Everything runs as three hand-started
processes on this PC:

- API: `uvicorn api.main:app --port 8000`
- Dashboard: `streamlit run dashboard/app.py --server.port 8501`
- Tunnel: `ngrok http --domain=unchagrined-ungotten-kaden.ngrok-free.dev 8501`

The ngrok authtoken is already saved in `C:\Users\neela\AppData\Local\ngrok\ngrok.yml`.

**Why this isn't production yet:**
1. **Open to abuse.** Anyone with the URL can call `/v1/ingest` with any
   server path (`api/service.py` passes `corpus_dir` straight to the loader).
   The dashboard's "Advanced: re-index from a server-side folder" box hands
   that ability to every visitor.
2. **Upload is broken in Docker Compose.** The dashboard writes uploads to
   its *own* `data/corpus`, then asks the API to ingest that path. In
   Compose, the dashboard container has no `./data` volume, so the API
   never sees those files.
3. **Ingest can break queries that run at the same time.** `ingest()`
   rebuilds the index files in place, then swaps the dense and sparse
   indexes one after the other. A question that arrives mid-rebuild can
   fuse results from two different corpora.
4. **Fragile.** Three processes are started by hand. There are no
   health checks, no restart policy, and no timeouts on Ollama calls, and
   the site dies when this PC sleeps.
5. **Unobservable.** There are no logs, request IDs or per-stage timings.

## Target

A one-command deploy (`docker compose up -d`) that is locked down, restarts
itself, reports its health, gets tested in CI with an eval quality gate, and
runs on a host that stays up. Budget stays at **$0**.

---

## Stage 1 — Harden the app (host-agnostic) — DONE 2026-09-26

Implemented and tested (185 tests pass). Where the implementation differs from the plan below:
- **Env names changed.** The environment variable is now `RAG_OLLAMA_HOST`,
  not `OLLAMA_HOST`. `docker-compose.yml` is updated to match. See
  `.env.example` for every setting.
- **Question limits are fixed in code**, not in config:
  `MAX_QUESTION_CHARS = 1000` and `MAX_SOURCE_NAMES = 100` in `api/schemas.py`.
- **The API key protects write endpoints only** (`POST /v1/ingest` and
  `POST /v1/documents`). `/v1/ask` and `GET /v1/documents` stay open. The
  plan is for the API to be reachable only through the dashboard (Stage 2.3
  binds it to localhost).
- **The upload size cap is checked late.** The API enforces it only after
  the multipart body has been received. The real first line of defence is
  Streamlit's `maxUploadSize = 10` in `.streamlit/config.toml`. Stage 2.3
  must keep the API off the public internet.
- **Index layout.** Versioned indexes live in
  `data/indexes/<strategy>/<UTC timestamp>_<id>/`, and the live one is named
  in `data/CURRENT_<strategy>`. Until the first versioned build, the old
  `data/chroma_<strategy>` / `data/bm25_<strategy>.pkl` paths are still read.
  So an existing deployment keeps working, and switches over on its next
  ingest. At least 2 versions are always kept. On Windows, deleting an old
  version can fail while a Chroma handle is open; the leftover directory is
  retried on the next prune.
- **All ingest paths are versioned.** `python ingest.py`, `scripts/seed.py`
  and `scripts/run_eval.py` now all build a new version and publish it. A
  running API only picks up a CLI-built index after a restart or a
  `POST /v1/ingest` call.
- **Not verified yet:** booting the real server with real models and Ollama.
  The dev venv used for these tests has no torch.

Each step is small and ends with a check you can run.

### 1.1 Settings from env
- Convert `Config` in `config.py` to a `pydantic_settings.BaseSettings`
  class with `env_prefix="RAG_"`. Keep the same field names and defaults.
- Add new fields: `api_key: str | None`, `data_root: str = "data/corpus"`,
  `max_question_chars: int = 1000`, `max_upload_mb: int = 10`,
  `ollama_timeout_s: float = 60`, `ollama_retries: int = 2`.
- Delete the hand-rolled `os.environ` reads in `api/main.py`.
- Add `.env.example` to the repo. Keep `.env` out of git via `.gitignore`.
- ✅ Check: `RAG_FINAL_K=3 python -c "from config import Config; print(Config().final_k)"` prints `3`.

### 1.2 Security
- **API key.** Write a FastAPI dependency that compares the `X-API-Key`
  header to `config.api_key` using `secrets.compare_digest`. Require it on
  every write endpoint. If `api_key` is unset, skip the check (local dev only).
- **Path containment.** `/v1/ingest` no longer accepts a path. It always
  re-indexes `config.data_root`.
- **Real upload endpoint.** Add `POST /v1/documents` (multipart;
  `python-multipart` is already installed). It should:
  - keep only the filename (`Path(name).name`),
  - allow only `.md/.txt/.html/.pdf`,
  - enforce `max_upload_mb`,
  - write into `data_root`, then re-index.

  The dashboard sends the files to this endpoint instead of writing to disk.
  That fixes problem #2.
- **Dashboard.** Delete the "re-index from a server-side folder" free-text
  box. The dashboard reads `RAG_API_KEY` from its own env and sends the
  header.
- **Input limits.** In `AskRequest`, set a max length on `question` and on
  `source_names`.
- ✅ Tests in `tests/test_api.py`:
  - missing or wrong key → 401
  - upload named `../../evil.md` → saved as `evil.md` inside `data_root`
  - `.csv` → 415
  - oversized file → 413
  - question over the limit → 422

### 1.3 Atomic re-index
- `build_indexes` writes to a new versioned directory,
  `data/index_<strategy>_<timestamp>/`, containing both Chroma and BM25.
  When the build finishes, write a `data/CURRENT_<strategy>` pointer file.
- `RAGService` holds `(dense, sparse)` as **one tuple** and swaps it in a
  single assignment. `ask()` reads the tuple once at the start. Put a
  `threading.Lock` around `ingest()` so two ingests can't overlap.
- Delete old index directories, keeping the last 2 for rollback.
- ✅ Test: start an ask, trigger ingest from a second thread, and assert
  every returned chunk ID exists in the snapshot the ask started with.

### 1.4 Reliability
- Add `GET /healthz` for liveness. It always returns 200 once the process is up.
- Add `GET /readyz` for readiness. It returns 200 only if Ollama answers
  `GET /api/tags` with the configured model listed **and** the index is
  non-empty. Otherwise it returns 503 with the reason.
- Wrap Ollama calls in `OllamaClient` with the configured timeout, plus
  `tenacity` retries on connection errors and 5xx responses. If Ollama is
  still down after the retries, `/v1/ask` returns 503 `{"detail": "LLM unavailable"}`,
  not a 500 stack trace.
- ✅ Check: stop Ollama. `/readyz` → 503 and `/v1/ask` → 503. Start Ollama
  again: both recover without restarting the API.

### 1.5 Observability
- Log one JSON line per request with:
  - `request_id` (from the `X-Request-ID` header or a generated uuid, echoed back in the response),
  - route, status, total ms,
  - **per-stage ms**: dense, sparse, fuse, rerank, generate, verify,
  - composite confidence and `fallback_triggered`.
- Use the stdlib `logging` module with a JSON formatter. Don't add a new dependency.
- ✅ Check: one `/v1/ask` produces one parseable JSON log line containing all
  of the fields above.

## Stage 2 — Package it

### 2.1 Split requirements
- `requirements.txt` keeps only runtime dependencies: fastapi, uvicorn,
  pydantic-settings, chromadb, rank-bm25, sentence-transformers, streamlit,
  requests, beautifulsoup4, pypdf, tenacity, python-multipart.
- `requirements-dev.txt` = `-r requirements.txt` + pytest (+ fpdf2 if only
  the corpus script needs it).
- Drop `kubernetes`, `GitPython` and the other leftovers from the old
  `pip freeze`.

### 2.2 Dockerfile
- Use a multi-stage build on `python:3.11-slim`.
- Install torch from the **CPU-only** wheel index
  (`--extra-index-url https://download.pytorch.org/whl/cpu`). This cuts
  gigabytes from the image.
- **Bake the models into the image.** A build step runs a short script
  that loads both `BAAI/bge-small-en-v1.5` and the cross-encoder, so
  containers don't download them from Hugging Face at startup. Set
  `HF_HOME=/models`.
- Run as a non-root `app` user.
- Add a `HEALTHCHECK` that curls `/healthz`.
- ✅ Check: the image is well under 2 GB (`docker images`), and
  `docker run --network none <image> python -c "<load both models>"`
  succeeds offline.

### 2.3 docker-compose.yml
Services:
- **`ollama`**: `ollama/ollama` with a named volume for models.
- **`ollama-init`**: a one-shot `ollama pull llama3.1` that exits when done.
- **`api`**:
  - `depends_on: ollama-init: service_completed_successfully`,
  - healthcheck on `/readyz`,
  - `restart: unless-stopped`,
  - `./data` volume,
  - `env_file: .env`.
- **`dashboard`**:
  - `depends_on: api: condition: service_healthy`,
  - `restart: unless-stopped`,
  - `RAG_API_KEY` from `.env`.
- **`ngrok`**: `ngrok/ngrok` with `NGROK_AUTHTOKEN` from `.env`, running
  `http --domain=... dashboard:8501`, `restart: unless-stopped`. The tunnel
  is now part of the stack, not a hand-run process.
- Only the dashboard is exposed publicly. The API port binds to
  `127.0.0.1:8000` so it's reachable only from this machine.
- ✅ Check: on a clean checkout, `cp .env.example .env` (fill in values),
  `docker compose up -d`, then `python scripts/seed.py`. Within a few
  minutes the public URL answers a golden question with verified citations.

## Stage 3 — CI/CD (GitHub Actions, free for public repos)

- **`ci.yml` on every push/PR:**
  - install `requirements-dev.txt` with the CPU torch wheel and pip cache,
  - run `pytest` (any test that needs Ollama or a model download gets a
    marker and is skipped in CI),
  - check that `docker build` succeeds.
- **`eval.yml` — manual trigger + nightly.** GitHub's runners can't run
  llama3.1, so this runs on a **self-hosted runner on the deploy box**:
  - it runs `scripts/run_eval.py` on the golden set and uploads the results
    table as an artifact,
  - it **fails if faithfulness or citation accuracy falls more than 5
    points below** the baseline in `eval_baseline.json` (committed).
- **Deploy:** on push to `main` after CI passes, the self-hosted runner runs
  `git pull && docker compose up -d --build`.
- ✅ Check: a deliberately broken prompt in a PR turns `eval.yml` red.

## Stage 4 — Host it (DECISION NEEDED)

Stages 1–3 work on any host. Pick where the compose stack lives:

| Option | Cost | Always on? | LLM | Notes |
|---|---|---|---|---|
| **A. This PC** (Docker Desktop + ngrok service) | $0 | Only while the PC is awake | llama3.1 local, fast-ish | Fewest moving parts. Set Docker Desktop to start at login and turn off sleep. |
| **B. Oracle Cloud Always Free** Ampere A1 VM (4 ARM cores / 24 GB RAM) | $0 (needs a card for signup, not charged) | Yes | llama3.1 8B quantized on CPU, **slow** (answer + per-claim verification may take tens of seconds) | Free-tier capacity is often "out of stock" in popular regions. ARM images needed; the python/ollama base images are multi-arch. |
| **C. Cheap VM + hosted LLM** | Not $0 | Yes | API model via an injected `LLMClient` interface | Breaks the $0/local rule. Only worth it if speed matters more than the rule. |

**Recommendation:** ship on **A** first, because it's the same machine the
Stage 1–3 work already runs on. Then try **B** as the always-on upgrade.
Moving from A to B is just `git clone` + `.env` + `docker compose up -d` on
the VM, plus pointing ngrok (or the VM's public IP with Caddy for HTTPS)
at it.

For B, run the golden eval on the VM once and record latency; if p95 is
unacceptable, drop to a smaller Ollama model (e.g. `llama3.2:3b`) and
re-run the eval to confirm quality holds.

## Stage 5 — Operate

- **Backups:** a nightly job tars `data/corpus` + the `CURRENT_*` index
  directory. Keep 7 days. The indexes can always be rebuilt from the
  corpus, so the corpus is what actually matters.
- **Rollback:** point `CURRENT_<strategy>` back at the previous index
  directory (1.3 keeps 2). For code, run `git checkout <prev tag> &&
  docker compose up -d --build`.
- **Monitoring:** use a free uptime pinger (e.g. UptimeRobot) on the public
  dashboard URL. The API isn't public, so it can't ping `/readyz` directly.
  Have the dashboard show an "API unavailable" banner when `/readyz` fails. Check the JSON logs with `docker compose logs api | grep '"status": 5'`.
- **Runbook** (add to README): start/stop, rotate the API key, re-index,
  roll back, and what to do when `/readyz` says the model is missing.

## Order of work

1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 2.1 → 2.2 → 2.3 → deploy on A →
3 (CI, then eval gate, then auto-deploy) → decide B → 5.

Security (1.2) is the only step that should be done **before** the public
URL is next brought up. Until then, run the dashboard locally only, or stop
the ngrok tunnel.
