"""The optional source-IP allowlist.

There is no token: the server is meant for the owner's own machine or home
network. The allowlist is the one gate left, and it is empty (any source)
unless ``server.allowed_ips`` is set.
"""

from __future__ import annotations

import ipaddress
import logging

from fastapi import HTTPException, Request, status

log = logging.getLogger(__name__)


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


def make_ip_dependency(config):
    """Build a FastAPI dependency that enforces the IP allowlist."""

    def dependency(request: Request) -> None:
        client = request.client.host if request.client else None
        if not ip_allowed(config, client):
            log.warning("rejected request from non-allowlisted IP %s", client)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="source IP not allowed"
            )

    return dependency
