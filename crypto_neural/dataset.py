"""Assemble the supervised dataset: features at t, label at t.

The merge is done on ``open_time`` with ``validate="one_to_one"``. Joining
features and labels positionally is the single most common source of
silent lookahead in this kind of pipeline — a one-row misalignment turns a
useless model into a backtest hero.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FeatureConfig, LabelConfig
from .features import build_features, feature_names
from .labels import label_triple_barrier

META_COLUMNS = [
    "open_time",
    "close",
    "label",
    "barrier_up",
    "barrier_down",
    "exit_index",
    "exit_time",
    "exit_price",
    "realized_return",
    "holding_bars",
]


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    feature_columns: list[str]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame["label"].astype("int8")


def build_dataset(
    bars: pd.DataFrame,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
) -> Dataset:
    """Build a leakage-safe feature/label frame, dropping incomplete rows."""
    bars = bars.reset_index(drop=True)
    if bars.empty:
        raise ValueError("cannot build a dataset from zero bars")

    feats = build_features(bars, features_cfg)
    columns = feature_names(features_cfg)
    feats = feats[columns].copy()
    feats.insert(0, "open_time", bars["open_time"].to_numpy())
    feats["close"] = bars["close"].to_numpy()

    labels = label_triple_barrier(bars, labels_cfg)
    if labels.empty:
        raise ValueError(
            f"no bar has a complete {labels_cfg.max_holding_bars}-bar forward window"
        )

    merged = feats.merge(labels, on="open_time", how="inner", validate="one_to_one")
    merged = merged.sort_values("open_time", kind="stable").reset_index(drop=True)

    before = len(merged)
    merged = merged[merged["valid"]].dropna(subset=columns + ["label"])
    merged["label"] = merged["label"].astype("int8")
    merged = merged.reset_index(drop=True)
    dropped = before - len(merged)

    if merged.empty:
        raise ValueError(
            "dataset is empty after dropping warm-up and unresolved rows; "
            "download more history"
        )
    if dropped:
        # Kept as an attribute rather than a log line so callers can report it.
        merged.attrs["dropped_rows"] = dropped

    return Dataset(frame=merged, feature_columns=columns)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def walk_forward_split(
    dataset: Dataset,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
) -> Split:
    """Chronological split: oldest train, middle validation, newest test."""
    if not 0 < train_frac < 1 or not 0 <= validation_frac < 1:
        raise ValueError("invalid split fractions")
    if train_frac + validation_frac >= 1:
        raise ValueError("train_frac + validation_frac must leave room for test")

    frame = dataset.frame
    n = len(frame)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + validation_frac))

    return Split(
        train=frame.iloc[:train_end].reset_index(drop=True),
        validation=frame.iloc[train_end:val_end].reset_index(drop=True),
        test=frame.iloc[val_end:].reset_index(drop=True),
    )
