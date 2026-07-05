import argparse
from pathlib import Path

DUPLICATE_PARAGRAPH = (
    "All internal services must implement exponential backoff when retrying "
    "failed requests to downstream dependencies. Retries should never exceed "
    "the configured maximum attempt count, and clients must respect the "
    "Retry-After header when a downstream service provides one. Failure to "
    "implement backoff correctly is a common root cause of cascading "
    "failures during partial outages."
)
ERROR_CODE = "ERR_2043"
CONFIG_KEY = "MAX_RETRY_COUNT"
FUNCTION_SIG = "def calculate_retry_backoff(attempt: int, base: float = 1.5) -> float:"


def _build_troubleshooting_doc() -> str:
    return f"""# Troubleshooting Guide

## {ERROR_CODE}: Document not received by intake service

**Symptom:** A document submitted through the internal upload portal never
appears in the processing queue, and the submitter receives no confirmation
email.

**Cause:** The intake service silently drops uploads larger than the
configured buffer size before writing to the queue, and does not currently
emit a failure event when this happens.

**Fix:** Resubmit the document through the CLI uploader (`docctl submit`)
instead of the web portal, which retries automatically on drop, or contact
platform support to confirm queue depth.

## ERR_1004: Unsupported document format rejected

**Symptom:** The intake service returns an immediate rejection when a
document is submitted in a format outside the supported list (md, txt, html).

**Cause:** The format-detection step only recognizes a fixed allow-list of
file extensions and fails closed on anything else, including common types
like .docx or .pdf that have not yet been added to the intake pipeline.

**Fix:** Convert the document to a supported format (Markdown is preferred)
before resubmitting, or file a request to add support for the new format.

## ERR_3390: Document exceeds maximum size limit

**Symptom:** Large documents (typically over 5MB) fail intake with a
size-limit error, even though smaller documents from the same source submit
successfully.

**Cause:** The intake service enforces a hard per-document size ceiling to
bound embedding and indexing latency, and does not currently support chunked
or streaming uploads.

**Fix:** Split the document into smaller files before submitting, or request
a size-limit exception for bulk historical imports.
"""


def _build_api_reference_doc() -> str:
    return f"""<html>
<body>
<h1>API Reference</h1>

<h2>Retry Utilities</h2>
<p>The retry utilities module provides helpers for handling transient
failures when calling downstream services.</p>

<h3>calculate_retry_backoff</h3>
<pre>{FUNCTION_SIG}</pre>
<p>Computes the delay in seconds before the next retry attempt, using
exponential backoff with a configurable base multiplier. The first attempt
uses no delay; each subsequent attempt multiplies the previous delay by
<code>base</code>.</p>

<p>{DUPLICATE_PARAGRAPH}</p>

<h2>Document Intake Endpoint</h2>
<p>POST /v1/documents accepts a single document for ingestion. Supported
content types are text/markdown, text/plain, and text/html. Requests
exceeding the configured size limit are rejected with a 413 status code.</p>

<h2>Health Check Endpoint</h2>
<p>GET /v1/health returns the current status of the intake service and its
downstream dependencies, including queue depth and last successful
ingestion timestamp.</p>
</body>
</html>
"""


def _build_config_guide_doc() -> str:
    return f"""# Configuration Reference

## {CONFIG_KEY}
Maximum number of retry attempts for a failed request to a downstream
service before the call is abandoned and an error is surfaced to the
caller. Default: 3.

{DUPLICATE_PARAGRAPH}

## CACHE_TTL_SECONDS
How long a cached embedding or retrieval result is considered valid before
it must be recomputed. Default: 900 (15 minutes).

## LOG_LEVEL
Controls the verbosity of service logs. One of DEBUG, INFO, WARNING, ERROR.
Default: INFO.
"""


def _build_onboarding_doc() -> str:
    return """# New Engineer Onboarding

## Welcome

Welcome to the platform team. This guide covers what to set up in your
first week and where to find things.

## First Week Checklist

- Request access to the internal wiki and the team's shared drive.
- Set up your local development environment following the README in the
  platform-tools repository.
- Join the #platform-eng and #on-call channels.
- Schedule a 1:1 with your onboarding buddy.

## Where to Find Things

- Architecture diagrams live on the internal wiki under Platform > Design.
- On-call rotations are managed in the scheduling tool; the current
  rotation is posted weekly in #on-call.
- Questions about local setup issues should go to #platform-eng before
  filing a ticket.
"""


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def make_corpus(out_dir: Path) -> list[Path]:
    """Write the synthetic corpus to out_dir; return the list of file paths written."""
    docs = {
        "troubleshooting.md": _build_troubleshooting_doc(),
        "api_reference.html": _build_api_reference_doc(),
        "config_guide.txt": _build_config_guide_doc(),
        "onboarding.md": _build_onboarding_doc(),
    }

    paths = []
    for filename, content in docs.items():
        paths.append(_write_file(out_dir / filename, content))
    return paths


def main() -> None:
    """CLI entry point: argparse --out-dir (default data/corpus), calls make_corpus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus"))
    args = parser.parse_args()

    paths = make_corpus(args.out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
