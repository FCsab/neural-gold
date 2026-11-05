#!/usr/bin/env python3
"""Single entrypoint: start the crypto-neural web UI and API.

    python run.py

Everything the workbench can do is reachable from the page it serves:
backfill, verification, label preview, dataset builds, the leakage suite,
and the exchange filters. No CLI step is required after this file starts.

Options exist for bootstrap cases (a different port, HTTP on a loopback
interface) but the normal path is config.yaml + this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import socket
import sys
from dataclasses import replace

import uvicorn

from crypto_neural.config import AppConfig, load_config
from crypto_neural.server.app import create_app
from crypto_neural.server.tls import ensure_certificate

log = logging.getLogger("crypto_neural.run")


def _lan_address(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            probe.close()
            return address
        except OSError:  # pragma: no cover - offline machine
            return "127.0.0.1"
    return host


def _banner(config: AppConfig, url: str, tls: bool) -> None:
    print()
    print(f"  crypto-neural {__import__('crypto_neural').__version__}")
    print(f"  UI          {url}")
    print(f"  symbol      {config.exchange.symbol} {config.exchange.interval}")
    print(f"  data        {config.data_root.resolve()}")
    print(f"  transport   {'HTTPS (self-signed)' if tls else 'HTTP (no TLS)'}")
    torch_spec = importlib.util.find_spec("torch")
    print(
        "  training    "
        + ("pytorch present (device probed per run)" if torch_spec else "unavailable: pip install torch")
    )
    print("  auth        none (open to anyone who can reach the port)")
    if config.server.allowed_ips:
        print(f"  allowlist   {', '.join(config.server.allowed_ips)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="crypto-neural web UI (single entrypoint)"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--host", default=None, help="bind address override")
    parser.add_argument("--port", type=int, default=None, help="port override")
    parser.add_argument("--no-tls", action="store_true", help="serve plain HTTP")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.host or args.port:
        config = replace(
            config,
            server=replace(
                config.server,
                host=args.host or config.server.host,
                port=args.port or config.server.port,
            ),
        )

    tls = config.server.tls_enabled and not args.no_tls
    ssl_kwargs: dict[str, str] = {}
    if tls:
        try:
            cert, key = ensure_certificate(
                config.server.cert_path, config.server.key_path, config.server.host
            )
            ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        except Exception as exc:  # noqa: BLE001 - fall back rather than refuse to boot
            print(f"[warn] TLS unavailable ({exc}); serving plain HTTP", file=sys.stderr)
            tls = False

    scheme = "https" if tls else "http"
    display_host = _lan_address(config.server.host)
    url = f"{scheme}://{display_host}:{config.server.port}/"
    _banner(config, url, tls)

    app = create_app(config, tls=tls)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=args.log_level,
        **ssl_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
