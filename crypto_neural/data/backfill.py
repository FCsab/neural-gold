"""Paginated historical backfill from Binance spot klines."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..config import AppConfig
from ..exchange.rest import fetch_klines, interval_to_ms, make_client
from .store import BAR_COLUMNS, Coverage, store_from_config

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


@dataclass
class BackfillStats:
    pages: int = 0
    bars_fetched: int = 0
    bars_written: int = 0
    bars_skipped_unclosed: int = 0
    start_ms: int | None = None
    end_ms: int | None = None
    elapsed_s: float = 0.0

    def format(self) -> str:
        span = ""
        if self.start_ms is not None and self.end_ms is not None:
            days = (self.end_ms - self.start_ms) / DAY_MS
            span = f" span={days:.0f}d"
        return (
            f"pages={self.pages} fetched={self.bars_fetched} "
            f"written={self.bars_written} unclosed_skipped={self.bars_skipped_unclosed}"
            f"{span} elapsed={self.elapsed_s:.1f}s"
        )


def plan_missing_windows(
    coverage: Coverage,
    interval_ms: int,
    *,
    years: float,
    now_ms: int,
) -> list[tuple[int, int]]:
    """Return the windows needed so the store holds ``years`` of history.

    ``years`` means "at least this much history ending now", not "add this
    much at the end". A store with two years that is asked for six gets one
    window extending backwards and one topping up the tail — the middle is
    never re-downloaded, so the call stays cheap and idempotent.

    Returns an empty list when the store already covers the request.
    """
    target_start = now_ms - int(years * 365.25 * DAY_MS)
    target_start -= target_start % interval_ms

    if coverage.empty:
        return [(target_start, now_ms)]

    windows: list[tuple[int, int]] = []

    if target_start < int(coverage.start_ms):
        windows.append((target_start, int(coverage.start_ms) - interval_ms))

    tail_start = int(coverage.end_ms) + interval_ms
    if tail_start < now_ms:
        windows.append((tail_start, now_ms))

    return windows


def plan_missing_range(
    coverage: Coverage,
    interval_ms: int,
    *,
    years: float,
    now_ms: int,
) -> tuple[int, int] | None:
    """The first window from :func:`plan_missing_windows`, for single-shot callers."""
    windows = plan_missing_windows(coverage, interval_ms, years=years, now_ms=now_ms)
    return windows[0] if windows else None


async def backfill(
    config: AppConfig,
    *,
    years: float = 2.0,
    start_ms: int | None = None,
    end_ms: int | None = None,
    page_limit: int = 1000,
    flush_bars: int = 10_000,
    progress: Callable[[BackfillStats], None] | None = None,
) -> BackfillStats:
    """Download bars until the store holds at least ``years`` of history."""
    store = store_from_config(config)
    interval_ms = interval_to_ms(config.exchange.interval)
    now_ms = int(time.time() * 1000)
    stop_at = now_ms if end_ms is None else min(int(end_ms), now_ms)

    if start_ms is None:
        windows = plan_missing_windows(
            store.coverage(), interval_ms, years=years, now_ms=stop_at
        )
        if not windows:
            coverage = store.coverage()
            return BackfillStats(
                start_ms=coverage.start_ms, end_ms=coverage.end_ms, elapsed_s=0.0
            )
    else:
        windows = [(int(start_ms), stop_at)]

    stats = BackfillStats(start_ms=min(window[0] for window in windows))
    began = time.monotonic()
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0

    def flush() -> None:
        nonlocal buffer, buffered_rows
        if not buffer:
            return
        page = pd.concat(buffer, ignore_index=True)
        # write_bars reports the row count of every touched partition (it
        # rewrites whole months), which is not "new bars". Count what was
        # actually handed over instead.
        store.write_bars(page)
        stats.bars_written += len(page)
        buffer = []
        buffered_rows = 0

    async with make_client(
        config.exchange.base_url, config.exchange.request_timeout_s
    ) as client:
        for window_start, window_stop in windows:
            log.info("backfill window %d -> %d (ms)", window_start, window_stop)
            cursor = window_start
            while cursor <= window_stop:
                page = await fetch_klines(
                    client,
                    symbol=config.exchange.symbol,
                    interval=config.exchange.interval,
                    start_ms=cursor,
                    end_ms=window_stop,
                    limit=page_limit,
                    max_retries=config.exchange.max_retries,
                )
                stats.pages += 1

                if page.empty:
                    break

                closed = page[page["close_time"] < now_ms]
                stats.bars_skipped_unclosed += len(page) - len(closed)
                stats.bars_fetched += len(closed)

                if not closed.empty:
                    buffer.append(closed[BAR_COLUMNS])
                    buffered_rows += len(closed)
                    stats.end_ms = int(closed["open_time"].iloc[-1])

                last_open = int(page["open_time"].iloc[-1])
                if last_open < cursor:
                    log.warning("exchange returned bars before the cursor; stopping")
                    break
                cursor = last_open + interval_ms

                if buffered_rows >= flush_bars:
                    flush()

                if progress is not None and stats.pages % 10 == 0:
                    progress(stats)

                await asyncio.sleep(config.exchange.min_request_spacing_s)

    flush()
    stats.elapsed_s = time.monotonic() - began
    return stats


async def top_up(
    config: AppConfig, *, max_age_days: float = 60.0
) -> BackfillStats | None:
    """Fetch only the bars after the newest stored one.

    A reading meant for the next bar is only as current as the store, and a
    job that silently scores a week-old bar is worse than one that says so.
    Returns ``None`` (and fetches nothing) when the store is empty or ends
    more than ``max_age_days`` ago: that is a full backfill's job, with its
    own button, not a side effect of asking for a reading.
    """
    store = store_from_config(config)
    coverage = store.coverage()
    if coverage.empty:
        return None
    now_ms = int(time.time() * 1000)
    if now_ms - int(coverage.end_ms) > max_age_days * DAY_MS:
        return None
    interval_ms = interval_to_ms(config.exchange.interval)
    return await backfill(config, start_ms=int(coverage.end_ms) + interval_ms)


def verify_bars(bars: pd.DataFrame, max_report: int = 20) -> list[str]:
    """Return human-readable problems found in a bar frame (empty = clean)."""
    problems: list[str] = []

    required = set(BAR_COLUMNS)
    missing = required - set(bars.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]

    if bars.empty:
        return ["no bars"]

    if bars["open_time"].duplicated().any():
        dupes = int(bars["open_time"].duplicated().sum())
        problems.append(f"{dupes} duplicate open_time values")

    times = bars["open_time"].to_numpy()
    if len(times) > 1 and not pd.Series(times).is_monotonic_increasing:
        problems.append("open_time is not sorted ascending")

    bad_prices = (
        (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | bars[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    if bad_prices.any():
        problems.append(f"{int(bad_prices.sum())} rows with non-positive/NaN prices")

    bad_range = (bars["high"] < bars["low"]) | (bars["high"] < bars["open"]) | (
        bars["high"] < bars["close"]
    ) | (bars["low"] > bars["open"]) | (bars["low"] > bars["close"])
    if bad_range.any():
        problems.append(f"{int(bad_range.sum())} rows violating OHLC ordering")

    bad_time = bars["close_time"] <= bars["open_time"]
    if bad_time.any():
        problems.append(f"{int(bad_time.sum())} rows with close_time <= open_time")

    if (bars["volume"] < 0).any():
        problems.append(f"{int((bars['volume'] < 0).sum())} rows with negative volume")

    if len(problems) > max_report:
        problems = problems[:max_report]
    return problems


def print_progress(stats: BackfillStats) -> None:
    print(f"  {stats.format()}", file=sys.stderr, flush=True)
