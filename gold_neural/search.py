"""Automated search over label settings.

The question this answers is not "which k_up is correct" — there is no correct
value. It is "which label settings produce trades big enough to survive costs,
and do they hold up on data the model never saw".

Two rules keep the results honest:

1. **Selection happens on the validation slice.** Every candidate is scored on
   validation. The winner's *test* result is then reported as out-of-sample.
   Selecting on test would guarantee a pretty number and teach nothing.
2. **Hopeless candidates are pruned before training.** If the profit target is
   not at least ``min_target_ratio`` times the round-trip cost, no model can
   win, so spending four minutes training it is waste. The same goes for a
   target that a random walk of the holding period could not plausibly reach.

Costs come from ``fee_bps`` and ``slippage_bps``; the ATR is measured on the
stored bars, so the pruning adapts to the instrument and the timeframe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Wider than the defaults on purpose: the defaults were too tight to trade.
K_UP_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
K_DOWN_GRID = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
HOLDING_GRID = (12, 24, 48, 96, 288)


@dataclass(frozen=True)
class Candidate:
    k_up: float
    k_down: float
    holding: int
    window: int = 64
    phase: int = 1
    target_bps: float = 0.0
    ratio_to_cost: float = 0.0

    def key(self) -> str:
        return f"u{self.k_up:g}d{self.k_down:g}h{self.holding}w{self.window}"

    def label(self) -> str:
        return f"k_up={self.k_up:g} · k_down={self.k_down:g} · hold={self.holding} · w={self.window}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_up": self.k_up,
            "k_down": self.k_down,
            "holding": self.holding,
            "window": self.window,
            "phase": self.phase,
            "target_bps": round(self.target_bps, 1),
            "ratio_to_cost": round(self.ratio_to_cost, 2),
            "key": self.key(),
        }


@dataclass(frozen=True)
class SearchConfig:
    total_minutes: float = 180.0
    per_candidate_minutes: float = 4.0
    max_candidates: int = 40
    epochs: int = 400
    batch_size: int = 512
    lr: float = 3e-4
    patience: int = 8
    device: str = "auto"
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    min_target_ratio: float = 2.5
    min_trades: int = 20
    refine_top: int = 2
    refine_windows: tuple[int, ...] = (32, 128)
    seed: int = 7

    def __post_init__(self) -> None:
        if self.total_minutes <= 0:
            raise ValueError("total_minutes must be positive")
        if self.per_candidate_minutes <= 0:
            raise ValueError("per_candidate_minutes must be positive")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")

    @property
    def round_trip_cost_bps(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps)


def generate_candidates(
    atr_bps: float,
    config: SearchConfig,
    *,
    phase: int = 1,
    windows: tuple[int, ...] = (64,),
    label_pairs: list[tuple[float, float, int]] | None = None,
) -> list[Candidate]:
    """Build the label grid, prune the untradeable corners, order by headroom."""
    cost = config.round_trip_cost_bps
    if atr_bps <= 0:
        raise ValueError("atr_bps must be positive")

    pairs = label_pairs
    if pairs is None:
        pairs = [
            (k_up, k_down, holding)
            for k_up in K_UP_GRID
            for k_down in K_DOWN_GRID
            for holding in HOLDING_GRID
        ]

    candidates: list[Candidate] = []
    for k_up, k_down, holding in pairs:
        target = k_up * atr_bps
        if target < config.min_target_ratio * cost:
            continue  # cannot pay for the round trip
        if k_down > k_up:
            continue  # a stop wider than the target is a different strategy
        # A driftless walk covers roughly ATR * sqrt(holding) over the window.
        if k_up > 1.5 * math.sqrt(holding):
            continue
        for window in windows:
            candidates.append(
                Candidate(
                    k_up=k_up,
                    k_down=k_down,
                    holding=holding,
                    window=window,
                    phase=phase,
                    target_bps=target,
                    ratio_to_cost=(target / cost) if cost > 0 else float("inf"),
                )
            )

    # Smallest viable target first: those trade most often and give the most
    # statistical evidence per unit of training time.
    candidates.sort(key=lambda c: (c.ratio_to_cost, c.holding, c.k_down, c.window))
    return candidates[: config.max_candidates]


def refine_candidates(
    best: list[dict[str, Any]],
    config: SearchConfig,
) -> list[Candidate]:
    """Second pass: same labels, different window, for the top few candidates."""
    if config.refine_top <= 0 or not config.refine_windows:
        return []

    out: list[Candidate] = []
    for entry in best[: config.refine_top]:
        candidate = entry["candidate"]
        for window in config.refine_windows:
            if int(window) == int(candidate["window"]):
                continue
            out.append(
                Candidate(
                    k_up=float(candidate["k_up"]),
                    k_down=float(candidate["k_down"]),
                    holding=int(candidate["holding"]),
                    window=int(window),
                    phase=2,
                    target_bps=float(candidate.get("target_bps") or 0.0),
                    ratio_to_cost=float(candidate.get("ratio_to_cost") or 0.0),
                )
            )
    return out


def score(result: dict[str, Any], min_trades: int) -> float:
    """Ranking score: validation return, gated on having traded enough.

    Return alone would reward one lucky trade; the trade floor makes a
    candidate prove it acts often enough to mean anything.
    """
    metrics = (result.get("validation") or {}).get("metrics") or {}
    if metrics.get("trades", 0) < min_trades:
        return float("-inf")
    total = metrics.get("total_return")
    if total is None:
        return float("-inf")
    drawdown = metrics.get("max_drawdown") or 0.0
    # Mild penalty: two candidates with the same return are not equal if one
    # got there through a 90% drawdown.
    return float(total) - 0.25 * float(drawdown)


def summarise(results: list[dict[str, Any]], config: SearchConfig) -> list[dict[str, Any]]:
    ranked = [r for r in results if r.get("status") == "ok"]
    ranked.sort(key=lambda r: r["score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def best_of(results: list[dict[str, Any]], config: SearchConfig) -> dict[str, Any] | None:
    ranked = summarise(results, config)
    return ranked[0] if ranked else None
