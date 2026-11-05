"""Perpetual-futures funding rates: positioning information from outside the
spot price.

Funding is what leveraged longs pay shorts (or the reverse) every eight
hours. It measures how crowded one side is, which is the mechanism behind
liquidation cascades, the kind of 10%-in-5-days fall the risk model watches
for. It is the cheapest information about BTC that is not BTC's own price.

Look-ahead rule: a bar's funding features use only settlements *strictly
before the bar opens*. A settlement at 08:00 is not available to the 08:00
bar, even though it is known a moment later; the strict version costs
nothing and removes the argument.

:func:`funding_builder` hands the settlement table to the walk-forward
explicitly (``risk.ExternalFeatures``), so its look-ahead check can hide
every settlement after a bar opened and require that bar's row to stay the
same. The as-of rule is checked on every run, not taken on trust.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .exchange.rest import FUTURES_BASE_URL, fetch_funding_rates, make_client
from .risk import ExternalFeatures

FUNDING_FEATURES = ["fund_last", "fund_mean_9", "fund_mean_90", "fund_z"]
# Windows in settlements (8h each): 9 = three days, 90 = thirty days.
SHORT, LONG = 9, 90
FIRST_FUNDING_MS = 1_568_000_000_000  # 2019-09-09; BTCUSDT perp listed 2019-09-10


def funding_path(data_root: str | Path, symbol: str) -> Path:
    return Path(data_root) / "funding" / f"{symbol}.parquet"


def load_funding(data_root: str | Path, symbol: str) -> pd.DataFrame:
    path = funding_path(data_root, symbol)
    if not path.exists():
        return pd.DataFrame({"funding_time": pd.Series(dtype="int64"),
                             "funding_rate": pd.Series(dtype="float64")})
    return pd.read_parquet(path)


async def _download(symbol: str, start_ms: int, timeout_s: float, spacing_s: float) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    cursor = int(start_ms)
    now_ms = int(time.time() * 1000)
    async with make_client(FUTURES_BASE_URL, timeout_s) as client:
        while cursor < now_ms:
            page = await fetch_funding_rates(client, symbol=symbol, start_ms=cursor)
            if page.empty:
                break
            pages.append(page)
            last = int(page["funding_time"].iloc[-1])
            if last < cursor or len(page) < 1000:
                break
            cursor = last + 1
            await asyncio.sleep(spacing_s)
    if not pages:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    return pd.concat(pages, ignore_index=True)


def update_funding(
    data_root: str | Path,
    symbol: str,
    *,
    timeout_s: float = 20.0,
    spacing_s: float = 0.25,
) -> pd.DataFrame:
    """Download settlements after the newest stored one; return the full table."""
    stored = load_funding(data_root, symbol)
    start = int(stored["funding_time"].max()) + 1 if len(stored) else FIRST_FUNDING_MS
    fresh = asyncio.run(_download(symbol, start, timeout_s, spacing_s))
    table = (
        pd.concat([stored, fresh], ignore_index=True)
        .drop_duplicates("funding_time", keep="first")
        .sort_values("funding_time", kind="stable")
        .reset_index(drop=True)
    )
    table["funding_time"] = table["funding_time"].astype("int64")
    table["funding_rate"] = table["funding_rate"].astype("float64")
    path = funding_path(data_root, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table


def settlement_features(funding: pd.DataFrame) -> pd.DataFrame:
    """Per-settlement statistics, each using that settlement and earlier ones."""
    rate = funding["funding_rate"].astype("float64").reset_index(drop=True)
    mean_short = rate.rolling(SHORT, min_periods=SHORT).mean()
    mean_long = rate.rolling(LONG, min_periods=LONG).mean()
    std_long = rate.rolling(LONG, min_periods=LONG).std()
    # Funding sat at exactly 0.01% for weeks in 2019-20. With no spread the
    # two means are equal too, so the deviation is zero, not undefined; NaN
    # would silently drop those rows from one arm of a comparison.
    z = ((mean_short - mean_long) / std_long.where(std_long > 1e-12)).where(
        std_long > 1e-12, 0.0
    ).where(std_long.notna())
    return pd.DataFrame(
        {
            "funding_time": funding["funding_time"].astype("int64").to_numpy(),
            "fund_last": rate.to_numpy(),
            "fund_mean_9": mean_short.to_numpy(),
            "fund_mean_90": mean_long.to_numpy(),
            "fund_z": z.to_numpy(),
        }
    )


def funding_features(open_time: np.ndarray, funding: pd.DataFrame) -> pd.DataFrame:
    """Funding features for bars opening at ``open_time``, as of strictly
    before each open. Rows with no settlement yet are NaN (warm-up)."""
    stats = settlement_features(funding.sort_values("funding_time", kind="stable"))
    bars = pd.DataFrame({"key": np.asarray(open_time, dtype=np.int64) - 1})
    order = np.argsort(bars["key"].to_numpy(), kind="stable")
    merged = pd.merge_asof(
        bars.iloc[order],
        stats.rename(columns={"funding_time": "key"}),
        on="key",
        direction="backward",  # settlement time <= open - 1 ms, i.e. strictly before
    )
    out = pd.DataFrame(index=np.arange(len(bars)), columns=FUNDING_FEATURES, dtype="float64")
    out.iloc[order] = merged[FUNDING_FEATURES].to_numpy(dtype=np.float64)
    return out


def funding_builder(funding: pd.DataFrame) -> ExternalFeatures:
    """The funding columns as an ``ExternalFeatures`` for the walk-forward,
    whose look-ahead check hides later settlements and later bars and
    requires each checked row to stay the same."""

    def build(bars: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        frame = funding_features(bars["open_time"].to_numpy(dtype=np.int64), table)
        frame.index = bars.index
        return frame

    return ExternalFeatures(
        name="funding", table=funding.reset_index(drop=True), time_column="funding_time",
        build=build,
    )


# --------------------------------------------------------------------------
# the pre-registered experiment (docs/risk.md, 2026-11-05)
# --------------------------------------------------------------------------
MIN_AUC_GAIN = 0.010
MIN_BLOCK_SHARE = 0.60


def experiment_bars(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Bars from the first one with ``LONG`` settlements strictly before it.

    Both arms run on exactly these bars. Trimming only the funding arm would
    let the baseline train on 2017-19 and compare two different models on
    two different histories.
    """
    times = np.sort(funding["funding_time"].to_numpy(dtype=np.int64))
    if len(times) < LONG:
        raise ValueError(f"need at least {LONG} funding settlements, have {len(times)}")
    first = bars["open_time"].to_numpy(dtype=np.int64) > times[LONG - 1]
    return bars.loc[first].reset_index(drop=True)


def compare_arms(base: dict, fund: dict) -> dict:
    """Apply the pre-registered decision rule to two risk-report dicts.

    Refuses to compare unless the out-of-sample rows and the blocks are the
    same in both arms: a pooled AUC over different rows is not a comparison.
    """
    if base["prediction"]["rows"] != fund["prediction"]["rows"]:
        raise ValueError("the two arms scored different rows")
    starts = [(f["start_ms"], f["end_ms"]) for f in base["folds"]]
    if starts != [(f["start_ms"], f["end_ms"]) for f in fund["folds"]]:
        raise ValueError("the two arms used different blocks")

    pairs = [
        (b["auc"], f["auc"])
        for b, f in zip(base["folds"], fund["folds"])
        if b["auc"] is not None and f["auc"] is not None
    ]
    wins = sum(f > b for b, f in pairs)
    share = wins / len(pairs) if pairs else 0.0
    gain = fund["prediction"]["auc"] - base["prediction"]["auc"]
    adds = gain >= MIN_AUC_GAIN and share >= MIN_BLOCK_SHARE

    def economics(report: dict) -> dict:
        keys = ("cagr", "max_drawdown", "sharpe", "exposure", "placebo_p")
        return {k: report["strategy"].get(k) for k in keys}

    text = (
        f"Pooled AUC {base['prediction']['auc']:.3f} without funding, "
        f"{fund['prediction']['auc']:.3f} with ({gain:+.3f}; the rule needs "
        f"{MIN_AUC_GAIN:+.3f}). Funding had the higher AUC in {wins} of "
        f"{len(pairs)} blocks ({share:.0%}; the rule needs {MIN_BLOCK_SHARE:.0%})."
    )
    if adds:
        text = ("Funding adds information by the pre-registered rule. Defaults "
                "stay unchanged; the predictions log is the forward test. " + text)
    else:
        text = ("Funding does not add information by the pre-registered rule. "
                "No variants are to be tried on this period. " + text)
    return {
        "adds_information": adds,
        "auc_base": base["prediction"]["auc"],
        "auc_funding": fund["prediction"]["auc"],
        "auc_gain": round(gain, 4),
        "rows": base["prediction"]["rows"],
        "blocks_compared": len(pairs),
        "blocks_won": wins,
        "block_share": round(share, 3),
        "rule": {"min_auc_gain": MIN_AUC_GAIN, "min_block_share": MIN_BLOCK_SHARE},
        "economics": {
            "base": economics(base),
            "funding": economics(fund),
            "hold": {k: base["hold"].get(k) for k in ("cagr", "max_drawdown", "sharpe")},
        },
        "period": base["period"],
        "block_auc": [
            {"start": _iso_day(b["start_ms"]), "base": b["auc"], "funding": f["auc"],
             "falls": b["falls"]}
            for b, f in zip(base["folds"], fund["folds"])
        ],
        "text": text,
    }


def _iso_day(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")
