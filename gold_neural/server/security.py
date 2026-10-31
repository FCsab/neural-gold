"""Bearer-token authentication and the optional source-IP allowlist.

The token is never written to config or SQLite: it comes from an environment
variable, or from a 0600 file generated on first boot. Every ``/api`` route
requires it; a failed check logs nothing but the source IP (a token has no
business in a log file).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Request, status

log = logging.getLogger(__name__)

TOKEN_BYTES = 32


def resolve_token(config) -> tuple[str, bool]:
    """Return ``(token, generated_now)``.

    Order: environment variable, then token file, then generate and persist.
    """
    env_name = config.server.token_env
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env, False

    path = Path(config.server.token_file)
    if path.is_file():
        stored = path.read_text().strip()
        if stored:
            return stored, False

    token = secrets.token_urlsafe(TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - non-POSIX
        pass
    return token, True


def ip_allowed(config, host: str | None) -> bool:
    allowed = config.server.allowed_ips
    if not allowed:
        return True
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    for entry in allowed:
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def token_matches(supplied: str | None, expected: str) -> bool:
    if not supplied:
        return False
    return secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def make_auth_dependency(config, token: str):
    """Build a FastAPI dependency that enforces IP allowlist + bearer token."""

    def dependency(request: Request) -> None:
        client = request.client.host if request.client else None
        if not ip_allowed(config, client):
            log.warning("rejected request from non-allowlisted IP %s", client)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="source IP not allowed"
            )

        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not token_matches(supplied.strip(), token):
            log.warning("unauthorized API request from %s", client)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency
