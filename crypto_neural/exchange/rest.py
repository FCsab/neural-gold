"""Minimal async client for Binance public REST endpoints.

Only the endpoints the data layer needs are implemented. No API key is
required (and none is read here): spot klines, exchangeInfo and the futures
funding history are public.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pandas as pd

log = logging.getLogger(__name__)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]
_INT_COLUMNS = ["open_time", "close_time", "trades"]

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


class ExchangeError(RuntimeError):
    """Raised when Binance returns an error we cannot recover from."""


def interval_to_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}; known: {sorted(INTERVAL_MS)}"
        ) from None


def _klines_to_frame(payload: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(payload, columns=KLINE_COLUMNS)
    if frame.empty:
        return frame

    frame = frame.drop(columns=["ignore"])
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    for column in _INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame.sort_values("open_time", kind="stable").reset_index(drop=True)


async def _request_with_retry(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any],
    *,
    max_retries: int,
) -> Any:
    """GET with retry on rate limits, server errors, and transient network faults."""
    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.get(path, params=params)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_error = exc
            log.warning("network error on %s (attempt %d/%d): %s", path, attempt, max_retries, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue

        if response.status_code == 200:
            used_weight = response.headers.get("x-mbx-used-weight-1m")
            if used_weight is not None and int(used_weight) > 5000:
                # 6000/min budget; back off before Binance starts rejecting.
                log.warning("request weight %s/6000, sleeping 30s", used_weight)
                await asyncio.sleep(30)
            return response.json()

        if response.status_code in {418, 429}:
            retry_after = float(response.headers.get("retry-after", delay))
            log.warning(
                "rate limited (%d) on %s, sleeping %.1fs",
                response.status_code,
                path,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            delay = min(delay * 2, 60.0)
            continue

        if 500 <= response.status_code < 600:
            last_error = ExchangeError(f"{response.status_code}: {response.text[:200]}")
            log.warning("server error %d on %s, retrying", response.status_code, path)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue

        raise ExchangeError(
            f"GET {path} failed with {response.status_code}: {response.text[:500]}"
        )

    raise ExchangeError(f"GET {path} failed after {max_retries} attempts: {last_error}")


async def fetch_klines(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    interval: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 1000,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch one page of klines (max 1000). Returned bars may be unclosed."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be within 1..1000")

    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    payload = await _request_with_retry(
        client, "/api/v3/klines", params, max_retries=max_retries
    )
    return _klines_to_frame(payload)


async def fetch_exchange_info(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Fetch the symbol's trading filters.

    Precision is never hardcoded: callers must round prices to
    PRICE_FILTER.tickSize and quantities to LOT_SIZE.stepSize, and re-fetch
    periodically because Binance changes filters without notice.
    """
    payload = await _request_with_retry(
        client,
        "/api/v3/exchangeInfo",
        {"symbol": symbol},
        max_retries=max_retries,
    )
    symbols = payload.get("symbols") or []
    if not symbols:
        raise ExchangeError(f"symbol {symbol} not present in exchangeInfo")

    entry = symbols[0]
    filters = {f["filterType"]: f for f in entry.get("filters", [])}
    return {
        "symbol": entry["symbol"],
        "status": entry.get("status"),
        "base_asset": entry.get("baseAsset"),
        "quote_asset": entry.get("quoteAsset"),
        "base_asset_precision": entry.get("baseAssetPrecision"),
        "quote_asset_precision": entry.get("quoteAssetPrecision"),
        "filters": filters,
        "server_time": payload.get("serverTime"),
    }


async def fetch_funding_rates(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int | None = None,
    limit: int = 1000,
    max_retries: int = 5,
) -> pd.DataFrame:
    """One page of USDⓈ-M perpetual funding settlements (public, no key).

    ``client`` must point at the futures host (``FUTURES_BASE_URL``). Returns
    ``funding_time`` (ms, the settlement moment) and ``funding_rate``.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be within 1..1000")
    params: dict[str, Any] = {"symbol": symbol, "startTime": int(start_ms), "limit": limit}
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    payload = await _request_with_retry(
        client, "/fapi/v1/fundingRate", params, max_retries=max_retries
    )
    frame = pd.DataFrame(payload, columns=["fundingTime", "fundingRate"])
    return pd.DataFrame(
        {
            "funding_time": pd.to_numeric(frame["fundingTime"], errors="raise").astype("int64"),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="raise").astype("float64"),
        }
    ).sort_values("funding_time", kind="stable").reset_index(drop=True)


FUTURES_BASE_URL = "https://fapi.binance.com"


def make_client(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s),
        headers={"User-Agent": "crypto-neural/0.1 (+research)"},
    )
