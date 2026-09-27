import secrets
from pathlib import PureWindowsPath

from fastapi import Header, HTTPException, Request


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency guarding write endpoints. Compares the X-API-Key
    header to config.api_key in constant time. When no key is configured
    the check is skipped -- that's for local dev only; any deployment
    reachable by others must set RAG_API_KEY.
    """
    configured = request.app.state.service.config.api_key
    expected = configured.get_secret_value() if configured else ""
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )


def safe_filename(raw: str | None) -> str | None:
    """Reduce a client-supplied upload name to a bare filename, or None if
    nothing safe is left. PureWindowsPath splits on BOTH / and \\ (and drops
    drive letters), so "../../x.md", "..\\\\x.md" and "C:x.md" all become
    "x.md" on any OS. Hidden files and control characters are rejected.
    """
    if not raw:
        return None
    name = PureWindowsPath(raw).name
    if not name or name.startswith(".") or any(ord(ch) < 32 for ch in name):
        return None
    return name
