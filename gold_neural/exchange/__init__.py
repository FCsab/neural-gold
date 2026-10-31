"""Binance spot REST access (public endpoints only)."""

from .rest import (
    INTERVAL_MS,
    KLINE_COLUMNS,
    ExchangeError,
    fetch_exchange_info,
    fetch_klines,
    interval_to_ms,
)

__all__ = [
    "INTERVAL_MS",
    "KLINE_COLUMNS",
    "ExchangeError",
    "fetch_exchange_info",
    "fetch_klines",
    "interval_to_ms",
]
