#!/usr/bin/env python3
"""Smoke-test a running crypto-neural server.

    python scripts/smoke.py --url https://host:8443 --insecure

Checks the page, the read endpoints, and (with ``--leakage``)
runs the leakage suite through the job API. Exits non-zero on the first
failure, so it works as a post-deploy gate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        raise SystemExit(1)


def wait_for_job(client: httpx.Client, headers: dict, job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if payload["status"] in {"ok", "failed"}:
            return payload
        time.sleep(0.4)
    raise SystemExit(f"job {job_id} did not finish within {timeout}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("CRYPTONEURAL_URL", "https://127.0.0.1:8443"),
    )
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification")
    parser.add_argument("--leakage", action="store_true", help="run the leakage suite too")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)

    headers: dict[str, str] = {}
    print(f"crypto-neural smoke test -> {args.url}")

    with httpx.Client(
        base_url=args.url, verify=not args.insecure, timeout=30.0
    ) as client:
        page = client.get("/")
        check("UI page served", page.status_code == 200 and "crypto-neural" in page.text)

        assets = client.get("/assets/app.js")
        check("UI javascript served", assets.status_code == 200 and "crypto-neural" in assets.text)

        status = client.get("/api/status", headers=headers).json()
        check("status reports the symbol", status["symbol"] == "BTCUSDC", str(status)[:200])
        check("TLS is on", status["tls"] is True)

        info = client.get("/api/data/info", headers=headers).json()
        rows = info["coverage"]["rows"]
        check("bars are stored", rows > 0, f"{rows} rows")

        bars = client.get("/api/data/bars?limit=20", headers=headers).json()["bars"]
        check("chart data returned", len(bars) > 0, f"{len(bars)} candles")
        if bars:
            check("candles are ordered", bars == sorted(bars, key=lambda b: b["time"]))

        paper = client.get("/api/paper", headers=headers)
        body = paper.json() if paper.status_code == 200 else {}
        # A scored record carries the live-order gate; an empty one says why not.
        check(
            "paper record served",
            bool(body.get("verdict")) and ("gate" in body or not body.get("days")),
            str(body)[:200],
        )

        job_id = client.post("/api/data/verify", headers=headers, json={}).json()["job_id"]
        job = wait_for_job(client, headers, job_id)
        check("verify job completed", job["status"] == "ok", str(job.get("error")))
        check("stored bars verify clean", job["result"]["ok"], str(job["result"]["problems"])[:300])

        if args.leakage:
            job_id = client.post(
                "/api/leakage/run", headers=headers, json={"samples": args.samples}
            ).json()["job_id"]
            job = wait_for_job(client, headers, job_id)
            check("leakage job completed", job["status"] == "ok", str(job.get("error")))
            check("leakage suite passed", job["result"]["passed"] is True)

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
