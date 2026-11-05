"""Bar storage: Parquet partitions + a SQLite index of what is on disk.

Layout::

    data/
      klines/BTCUSDC/5m/year=2024/month=10/part.parquet
      klines/BTCUSDC/5m/year=2024/month=11/part.parquet
      index-BTCUSDC-5m.sqlite

One file per (year, month) keeps reads columnar and cheap, and keeps
writes small enough that merging a backfill page is not a rewrite of the
whole history.

All timestamps are Unix epoch milliseconds, UTC. ``open_time`` is the
primary key of a bar.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BAR_COLUMNS = [
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
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS partitions (
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    year        INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    path        TEXT    NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    rows        INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (symbol, interval, year, month)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def month_key(open_time_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
    return dt.year, dt.month


@dataclass(frozen=True)
class Coverage:
    """Inclusive millisecond range of bars on disk."""

    start_ms: int | None
    end_ms: int | None
    rows: int

    @property
    def empty(self) -> bool:
        return self.rows == 0 or self.start_ms is None or self.end_ms is None


class BarStore:
    """Parquet-backed store for one (symbol, interval) pair."""

    def __init__(self, root: str | Path, symbol: str, interval: str, compression: str = "zstd"):
        self.root = Path(root)
        self.symbol = symbol
        self.interval = interval
        self.compression = compression
        self.partitions_dir = self.root / "klines" / symbol / interval
        self.index_path = self.root / f"index-{symbol}-{interval}.sqlite"

        self.partitions_dir.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- sqlite plumbing -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.index_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- writes ----------------------------------------------------------
    def write_bars(self, bars: pd.DataFrame) -> int:
        """Merge ``bars`` into the month partitions and refresh the index.

        Returns the number of rows written across all touched partitions.
        Duplicate ``open_time`` values are resolved in favour of the *new*
        row: a later download of a bar is assumed to be the corrected one.
        """
        if bars.empty:
            return 0

        missing = set(BAR_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")

        frame = bars[BAR_COLUMNS].copy()
        frame["open_time"] = frame["open_time"].astype("int64")
        frame = frame.drop_duplicates(subset="open_time", keep="last")
        frame = frame.sort_values("open_time", kind="stable")

        written = 0
        keys = frame["open_time"].map(month_key)
        for (year, month), chunk in frame.groupby(keys, sort=True):
            written += self._write_partition(int(year), int(month), chunk)
        return written

    def _partition_path(self, year: int, month: int) -> Path:
        return self.partitions_dir / f"year={year}" / f"month={month:02d}" / "part.parquet"

    def _write_partition(self, year: int, month: int, chunk: pd.DataFrame) -> int:
        path = self._partition_path(year, month)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, chunk], ignore_index=True)
            merged = merged.drop_duplicates(subset="open_time", keep="last")
            merged = merged.sort_values("open_time", kind="stable").reset_index(drop=True)
        else:
            merged = chunk.reset_index(drop=True)

        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, compression=self.compression, index=False)
        os.replace(tmp, path)  # atomic: readers never see a partial file

        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO partitions
                    (symbol, interval, year, month, path, start_ms, end_ms, rows, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, year, month) DO UPDATE SET
                    path=excluded.path,
                    start_ms=excluded.start_ms,
                    end_ms=excluded.end_ms,
                    rows=excluded.rows,
                    updated_at=excluded.updated_at
                """,
                (
                    self.symbol,
                    self.interval,
                    year,
                    month,
                    str(path),
                    int(merged["open_time"].iloc[0]),
                    int(merged["open_time"].iloc[-1]),
                    int(len(merged)),
                    now_ms,
                ),
            )
        return int(len(merged))

    # -- reads -----------------------------------------------------------
    def coverage(self) -> Coverage:
        row = self._connect().execute(
            """
            SELECT MIN(start_ms), MAX(end_ms), SUM(rows)
            FROM partitions WHERE symbol=? AND interval=?
            """,
            (self.symbol, self.interval),
        ).fetchone()
        start_ms, end_ms, rows = row
        return Coverage(start_ms, end_ms, int(rows or 0))

    def partitions(self) -> list[dict[str, object]]:
        rows = self._connect().execute(
            """
            SELECT year, month, path, start_ms, end_ms, rows
            FROM partitions WHERE symbol=? AND interval=?
            ORDER BY year, month
            """,
            (self.symbol, self.interval),
        ).fetchall()
        return [
            {
                "year": r[0],
                "month": r[1],
                "path": r[2],
                "start_ms": r[3],
                "end_ms": r[4],
                "rows": r[5],
            }
            for r in rows
        ]

    def gaps(self, expected_interval_ms: int, tolerance: float = 1.5) -> list[tuple[int, int]]:
        """Unfilled stretches between stored bars.

        A gap is any spacing larger than ``tolerance`` x the bar interval.
        Missing bars are normal for a thin book with no trades, so treat
        this as a diagnostic, not an error.
        """
        bars = self.load_bars()
        if len(bars) < 2:
            return []
        times = bars["open_time"].to_numpy()
        deltas = pd.Series(times).diff().to_numpy()
        limit = expected_interval_ms * tolerance
        gaps: list[tuple[int, int]] = []
        for i in range(1, len(times)):
            if deltas[i] > limit:
                gaps.append((int(times[i - 1]), int(times[i])))
        return gaps

    def load_bars(self, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        """Load bars in ``[start_ms, end_ms]`` (inclusive), sorted by open_time."""
        entries = self.partitions()
        # Use the index instead of reading every month: the chart asks for the
        # last few hundred bars, not two years of them.
        if start_ms is not None:
            entries = [e for e in entries if int(e["end_ms"]) >= int(start_ms)]
        if end_ms is not None:
            entries = [e for e in entries if int(e["start_ms"]) <= int(end_ms)]

        paths = [e["path"] for e in entries]
        if not paths:
            return pd.DataFrame(columns=BAR_COLUMNS)

        frames = [pd.read_parquet(p) for p in paths]
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.sort_values("open_time", kind="stable")

        if start_ms is not None:
            frame = frame[frame["open_time"] >= int(start_ms)]
        if end_ms is not None:
            frame = frame[frame["open_time"] <= int(end_ms)]

        return frame.reset_index(drop=True)


def store_from_config(config) -> BarStore:  # noqa: ANN001 - avoids circular import
    return BarStore(
        root=config.data_root,
        symbol=config.exchange.symbol,
        interval=config.exchange.interval,
        compression=config.data.compression,
    )
