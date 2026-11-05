"""The mandatory leakage test.

Two different questions get confused in most "shuffle the labels" checks,
so this module is explicit about what each check proves:

1. **Prefix invariance (the real leak detector).** Recompute the feature
   frame on ``bars[:i+1]`` and compare row ``i`` against the full-history
   row. If a feature at ``i`` used anything after ``i``, the two runs
   cannot agree. This is necessary and sufficient for temporal leakage in
   the feature builder, and it is the check that must never be relaxed.

2. **Future dependence of the label.** Perturb a bar inside the forward
   window and assert the label changes; truncate past the forward window
   and assert it does not. This proves the labeller is actually looking
   forward (a labeller that quietly returns zeros would pass everything
   else).

3. **Shuffled-label correlation.** Shuffling labels destroys the X-y
   alignment, so a *correct* pipeline shows ~0 correlation for every
   feature. A non-zero result means the harness itself is broken (for
   example, both sides indexed by position while one was reset). It does
   **not** detect leakage on its own, because leakage also disappears when
   the link is shuffled. The plan asks for this check; keeping its real
   meaning in the docstring prevents someone from trusting it for the
   wrong reason.

4. **Dataset invariants.** Alignment, monotonic time, exit strictly in the
   future, label/return sign agreement.

Finally, feature-label mutual information is reported as a diagnostic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import FeatureConfig, LabelConfig
from .dataset import build_dataset
from .features import build_features
from .labels import label_triple_barrier


class LeakageError(AssertionError):
    """Raised when a leakage check fails."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Report:
    checks: tuple[Check, ...]
    bars: int
    rows: int
    labels: dict[str, int]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "bars": self.bars,
            "rows": self.rows,
            "labels": self.labels,
            "checks": [asdict(c) for c in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format(self) -> str:
        head = "LEAKAGE SUITE: " + ("PASS" if self.passed else "FAIL")
        lines = [head, f"  bars={self.bars} rows={self.rows} labels={self.labels}", ""]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name}")
            lines.append(f"         {check.detail}")
            for key, value in check.metrics.items():
                if isinstance(value, float):
                    lines.append(f"         {key} = {value:.6g}")
                else:
                    lines.append(f"         {key} = {value}")
        return "\n".join(lines)


def _evenly_spaced(indices: np.ndarray, count: int) -> list[int]:
    if len(indices) <= count:
        return [int(i) for i in indices]
    picks = np.linspace(0, len(indices) - 1, count)
    return [int(indices[int(round(p))]) for p in picks]


def _close_enough(a: pd.Series, b: pd.Series) -> bool:
    nan_a = a.isna().to_numpy()
    nan_b = b.isna().to_numpy()
    if not np.array_equal(nan_a, nan_b):
        return False
    valid = ~nan_a
    if not valid.any():
        return True
    return bool(
        np.allclose(
            a.to_numpy()[valid].astype(float),
            b.to_numpy()[valid].astype(float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        )
    )


def check_prefix_invariance(
    bars: pd.DataFrame,
    cfg: FeatureConfig,
    samples: int = 20,
) -> Check:
    """Every feature at t must be reproducible from bars[:t+1] alone."""
    full = build_features(bars, cfg)
    complete = np.flatnonzero(full.notna().all(axis=1).to_numpy())
    if len(complete) == 0:
        return Check(
            name="features_prefix_invariant",
            passed=False,
            detail="no complete feature row; history too short for warm-up",
        )

    picks = _evenly_spaced(complete, samples)
    checked = 0
    for i in picks:
        prefix = bars.iloc[: i + 1].reset_index(drop=True)
        recomputed = build_features(prefix, cfg)
        left = full.iloc[i]
        right = recomputed.iloc[-1]
        if not _close_enough(left, right):
            bad = [
                column
                for column in full.columns
                if not _close_enough(
                    pd.Series([left[column]]), pd.Series([right[column]])
                )
            ]
            return Check(
                name="features_prefix_invariant",
                passed=False,
                detail=(
                    f"feature row at bar {i} changed when the history was truncated "
                    f"(columns: {bad[:5]}); the builder reads the future"
                ),
                metrics={"first_bad_index": float(i)},
            )
        checked += 1

    return Check(
        name="features_prefix_invariant",
        passed=True,
        detail=(
            f"{checked} feature rows recomputed from truncated history match the "
            "full-history values exactly"
        ),
        metrics={"checked_rows": float(checked)},
    )


def check_label_future_dependence(
    bars: pd.DataFrame,
    cfg: LabelConfig,
    samples: int = 10,
) -> Check:
    """Forcing a barrier touch inside the window must change the label."""
    base = label_triple_barrier(bars, cfg)
    horizon = cfg.max_holding_bars
    n = len(bars)

    eligible = np.flatnonzero(
        (base["valid"] & (base.index + horizon + 2 < n)).to_numpy()
    )
    if len(eligible) < samples:
        picks = eligible[-1:] if len(eligible) else []
    else:
        picks = _evenly_spaced(eligible, samples)

    if len(picks) == 0:
        return Check(
            name="labels_depend_on_future",
            passed=False,
            detail="no labellable bar with a complete forward window",
        )

    changed = 0
    for t in picks:
        row = base.iloc[t]
        up = float(row["barrier_up"])
        down = float(row["barrier_down"])

        perturbed = bars.copy()
        # Guarantee an upper touch on the next bar and prevent a lower touch,
        # so the expected label is unambiguous (+1).
        perturbed.loc[perturbed.index[t + 1], "high"] = up * 1.001
        perturbed.loc[perturbed.index[t + 1], "low"] = max(
            float(perturbed.loc[perturbed.index[t + 1], "low"]), down * 1.01
        )
        perturbed.loc[perturbed.index[t + 1], "close"] = max(
            float(perturbed.loc[perturbed.index[t + 1], "close"]), down * 1.01
        )

        relabelled = label_triple_barrier(perturbed, cfg)
        new_label = relabelled.iloc[t]["label"]
        if pd.notna(new_label) and int(new_label) == 1:
            changed += 1
        else:
            return Check(
                name="labels_depend_on_future",
                passed=False,
                detail=(
                    f"perturbing bar {t + 1} did not flip the label at {t} to +1 "
                    f"(got {new_label}); the labeller may not be looking forward"
                ),
                metrics={"first_bad_index": float(t)},
            )

    return Check(
        name="labels_depend_on_future",
        passed=True,
        detail=f"{changed} sampled labels flipped to +1 when a future bar was forced through the upper barrier",
        metrics={"checked_rows": float(changed)},
    )


def check_label_far_future_independence(
    bars: pd.DataFrame,
    cfg: LabelConfig,
    samples: int = 10,
) -> Check:
    """Data after the timeout bar must not change a label."""
    base = label_triple_barrier(bars, cfg)
    horizon = cfg.max_holding_bars
    n = len(bars)

    eligible = np.flatnonzero(
        (base["valid"] & (base.index + horizon + 5 < n)).to_numpy()
    )
    picks = _evenly_spaced(eligible, samples) if len(eligible) else []
    if not picks:
        return Check(
            name="labels_ignore_far_future",
            passed=False,
            detail="not enough bars to test truncation past the forward window",
        )

    for t in picks:
        truncated = bars.iloc[: t + horizon + 1].reset_index(drop=True)
        relabelled = label_triple_barrier(truncated, cfg)
        expected = base.iloc[t]["label"]
        got = relabelled.iloc[t]["label"]
        if pd.isna(expected) or pd.isna(got) or int(expected) != int(got):
            return Check(
                name="labels_ignore_far_future",
                passed=False,
                detail=(
                    f"label at bar {t} changed from {expected} to {got} when data "
                    f"after its {horizon}-bar window was removed"
                ),
                metrics={"first_bad_index": float(t)},
            )

    return Check(
        name="labels_ignore_far_future",
        passed=True,
        detail=f"{len(picks)} labels survived truncation past their forward window",
        metrics={"checked_rows": float(len(picks))},
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv, yv = x[mask], y[mask]
    if xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def mutual_information_bits(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Histogram mutual information between a continuous feature and labels."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < bins * 20:
        return float("nan")

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    xb = np.digitize(x, edges[1:-1])
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")

    table = np.zeros((int(xb.max()) + 1, len(classes)))
    for j, cls in enumerate(classes):
        table[:, j] = np.bincount(xb[y == cls], minlength=table.shape[0])

    total = table.sum()
    p_xy = table / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_xy > 0, p_xy / (p_x * p_y), 1.0)
    return float(np.nansum(np.where(p_xy > 0, p_xy * np.log2(ratio), 0.0)))


def check_shuffled_labels(
    dataset,
    seed: int = 1234,
    sigma: float = 4.0,
) -> Check:
    """Shuffled labels must show ~zero correlation with every feature."""
    frame = dataset.frame
    labels = frame["label"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)

    n = len(frame)
    threshold = sigma / math.sqrt(n)
    worst_feature = ""
    worst = 0.0
    for column in dataset.feature_columns:
        corr = _pearson(frame[column].to_numpy(dtype=float), shuffled)
        if np.isfinite(corr) and abs(corr) > abs(worst):
            worst = corr
            worst_feature = column

    passed = abs(worst) <= threshold
    return Check(
        name="shuffled_label_correlation",
        passed=passed,
        detail=(
            f"max |corr(feature, shuffled label)| = {abs(worst):.5f} "
            f"({worst_feature or 'n/a'}), threshold {threshold:.5f} = "
            f"{sigma:g}/sqrt(n={n}). Non-zero here means a harness bug, not leakage."
        ),
        metrics={
            "max_abs_correlation": abs(worst),
            "threshold": threshold,
            "rows": float(n),
        },
    )


def check_dataset_invariants(dataset, max_holding_bars: int) -> Check:
    """Structural checks that catch alignment and sign mistakes."""
    frame = dataset.frame
    problems: list[str] = []

    times = frame["open_time"].to_numpy()
    if len(times) and not np.all(np.diff(times) > 0):
        problems.append("open_time is not strictly increasing")
    if frame["open_time"].duplicated().any():
        problems.append("duplicate open_time rows")

    if not frame["label"].isin([-1, 0, 1]).all():
        problems.append("label outside {-1, 0, +1}")

    if not (frame["exit_time"] > frame["open_time"]).all():
        problems.append("exit_time is not strictly after open_time")

    if not (
        (frame["holding_bars"] >= 1) & (frame["holding_bars"] <= max_holding_bars)
    ).all():
        problems.append("holding_bars outside [1, max_holding_bars]")

    long_rows = frame["label"] == 1
    if long_rows.any() and not (frame.loc[long_rows, "realized_return"] > 0).all():
        problems.append("label +1 rows with non-positive realized_return")
    short_rows = frame["label"] == -1
    if short_rows.any() and not (frame.loc[short_rows, "realized_return"] < 0).all():
        problems.append("label -1 rows with non-negative realized_return")

    return Check(
        name="dataset_invariants",
        passed=not problems,
        detail="; ".join(problems) if problems else "all structural invariants hold",
        metrics={"rows": float(len(frame))},
    )


def check_feature_label_information(dataset, top: int = 5) -> Check:
    """Diagnostic only: MI between each feature and the real labels."""
    frame = dataset.frame
    y = frame["label"].to_numpy(dtype=float)
    scores = {
        column: mutual_information_bits(frame[column].to_numpy(dtype=float), y)
        for column in dataset.feature_columns
    }
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    ranked = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)[:top]
    top_text = ", ".join(f"{name}={value:.4f}" for name, value in ranked)
    return Check(
        name="feature_label_mutual_information",
        passed=True,
        detail=(
            "informational, never a pass/fail gate (real signal raises MI): "
            + (top_text or "n/a")
        ),
        metrics={f"mi_{name}": value for name, value in ranked},
    )


def run_leakage_suite(
    bars: pd.DataFrame,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
    *,
    samples: int = 10,
    seed: int = 1234,
) -> Report:
    dataset = build_dataset(bars, features_cfg, labels_cfg)

    checks = [
        check_prefix_invariance(bars, features_cfg, samples=samples * 2),
        check_label_future_dependence(bars, labels_cfg, samples=samples),
        check_label_far_future_independence(bars, labels_cfg, samples=samples),
        check_dataset_invariants(dataset, labels_cfg.max_holding_bars),
        check_shuffled_labels(dataset, seed=seed),
        check_feature_label_information(dataset),
    ]

    label_counts = dataset.frame["label"].value_counts().to_dict()
    return Report(
        checks=tuple(checks),
        bars=len(bars),
        rows=len(dataset),
        labels={
            "long": int(label_counts.get(1, 0)),
            "flat": int(label_counts.get(0, 0)),
            "stop": int(label_counts.get(-1, 0)),
        },
    )


def assert_no_leakage(
    bars: pd.DataFrame,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
    **kwargs,
) -> Report:
    report = run_leakage_suite(bars, features_cfg, labels_cfg, **kwargs)
    if not report.passed:
        failed = ", ".join(c.name for c in report.failures)
        raise LeakageError(f"leakage suite failed: {failed}\n{report.format()}")
    return report
