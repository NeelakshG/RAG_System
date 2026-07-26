# Deployment plan — public URL for the dashboard

_Last updated: 2026-07-26_

## Goal

Get a public, shareable URL for the Streamlit dashboard (recruiter/portfolio use),
at $0 cost.

## Decisions made (and why)

- **Ruled out:** buying a domain + Cloudflare named tunnel. Real domains cost
  money (~$3-12/yr); the ask is fully free.
- **Ruled out:** Hugging Face Spaces / paid cloud VM. Would mean swapping the
  local `llama3.1` Ollama setup for a smaller model or ongoing hosting cost —
  more moving parts than needed right now.
- **Chosen: ngrok free static domain.** Free account (no payment info), gives
  one permanent URL (`<name>.ngrok-free.app`) that stays the same across
  restarts — unlike Cloudflare's free "quick tunnel," which hands out a new
  random `*.trycloudflare.com` address every time it starts.
- Everything (API, dashboard, Ollama) keeps running locally on this machine
  exactly as today. The tunnel just exposes the dashboard's port (8501)
  publicly. This means the public URL only works while this PC is on and the
  tunnel + servers are running — that's the accepted tradeoff for staying at
  $0.

## Status

- `ngrok` CLI is **already installed** on this machine
  (`C:\Users\neela\AppData\Local\Microsoft\WindowsApps\ngrok.exe`, v3.39.8).
- Nothing else has been set up yet — waiting on account creation, which only
  the user can do (Claude doesn't create accounts or handle payment info).

## Status: DONE (2026-07-26)

Live at **https://unchagrined-ungotten-kaden.ngrok-free.dev** — verified
end-to-end in browser (Ask tab generates a cited, verified answer;
Documents tab lists indexed docs and renders the upload widget).

Only works while this PC is on with all three processes running:
- API: `uvicorn api.main:app --port 8000` (the `api/` package — NOT the
  root `api.py`, see note below)
- Dashboard: `streamlit run dashboard/app.py --server.port 8501`
- Tunnel: `ngrok http --domain=unchagrined-ungotten-kaden.ngrok-free.dev 8501`

Authtoken is already saved to ngrok's config
(`C:\Users\neela\AppData\Local\ngrok\ngrok.yml`), so `ngrok config
add-authtoken` never needs to be re-run on this machine.

### Note on `api.py` vs `api/`
Found during this deployment: there are two competing API implementations
in this repo — the `api/` package (`main.py`/`service.py`/`schemas.py`,
built 2026-07-10, matches the dashboard's actual request/response contract)
and a root-level `api.py` + `tests/test_api.py` (built 2026-07-21, different
field names, incompatible with the dashboard, missing `/v1/ingest` and
`/v1/documents`). They also collide on import (`import api` non-
deterministically resolves to whichever one Python's package-vs-module
resolution prefers), which is why `pytest tests/test_api.py` currently fails
standalone. Deployed using the `api/` package since it's the one the
dashboard is actually built against. `api.py`/`test_api.py` are unfinished
work-in-progress — untouched, still there, not wired up to anything. Decide
later whether to finish porting it (and update the dashboard + retire
`api/`) or drop it.

## How to resume

Restart the three processes above (API, dashboard, tunnel) any time this
machine reboots or the processes get killed — nothing needs to be
reconfigured.

## Related context (not part of deployment, but recent work this session)

- `dashboard/app.py` — Documents tab was reworked: added a working file
  upload widget ("Add documents"), and rewrote the "Advanced: re-index from a
  server-side folder" section with an in-depth, plain-language explanation
  (what the path means, local-vs-Docker gotcha, what re-indexing actually
  does, when to use it vs. the upload widget). These changes are **not yet
  committed** — sitting as local edits.
