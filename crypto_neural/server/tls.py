"""Self-signed certificate bootstrap.

Generating the certificate here keeps ``python run.py`` a single command on a
fresh checkout. The certificate is for LAN use only: it is not trusted by any
CA, so browsers will warn once per device. Replace it with a real certificate
by pointing ``server.cert_path`` / ``server.key_path`` at one.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

CERT_DAYS = 825


def ensure_certificate(cert_path: str, key_path: str, host: str) -> tuple[Path, Path]:
    """Return existing cert/key paths, generating a self-signed pair if absent."""
    cert = Path(cert_path)
    key = Path(key_path)

    if cert.is_file() and key.is_file():
        return cert, key

    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl not found; cannot generate a self-signed certificate")

    subject_alt_names = ["DNS:localhost", "IP:127.0.0.1"]
    try:
        address = ipaddress.ip_address(host)
        if not address.is_loopback:
            subject_alt_names.append(f"IP:{address}")
    except ValueError:
        if host not in {"0.0.0.0", "::", ""}:
            subject_alt_names.append(f"DNS:{host}")

    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)

    command = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        str(CERT_DAYS),
        "-keyout",
        str(key),
        "-out",
        str(cert),
        "-subj",
        "/CN=crypto-neural",
        "-addext",
        f"subjectAltName={','.join(subject_alt_names)}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"openssl failed: {result.stderr.strip()[:400]}")

    try:
        os.chmod(key, 0o600)
    except OSError:  # pragma: no cover - non-POSIX
        pass

    log.info("generated self-signed certificate at %s", cert)
    return cert, key
