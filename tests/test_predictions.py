from __future__ import annotations

import pandas as pd
import pytest

from crypto_neural.predictions import (
    GRACE_MS,
    PredictionLog,
    config_fingerprint,
    reading_from_report,
    score_paper,
)

from .conftest import BAR_MS, flat_bars, synthetic_bars

BARS_PER_YEAR = 288 * 365


def reading(bar_ms: int, action: str = "long", *, recorded: int | None = None,
            risk: float = 0.1, trend: str = "long", config_hash: str = "abc") -> dict:
    return {
        "recorded_at_ms": bar_ms if recorded is None else recorded,
        "for_bar_ms": bar_ms, "symbol": "BTCUSDT", "interval": "5m", "source": "test",
        "config_hash": config_hash, "model": "lgbm", "drawdown": 0.01, "horizon_bars": 6,
        "risk": risk, "rule": "q0.9", "exit_line": 0.3, "entry_line": 0.27,
        "action": action, "trend_action": trend, "trend_distance": 0.01,
        "last_close": 100.0, "status": "draft",
    }


def test_the_first_reading_for_a_bar_is_kept(tmp_path):
    log = PredictionLog(tmp_path / "p.sqlite")
    assert log.record(reading(1_000, "long", risk=0.1)) is True
    # A rerun for the same bar must not replace it: no re-rolling the answer.
    assert log.record(reading(1_000, "flat", risk=0.9, recorded=2_000)) is False
    rows = log.readings()
    assert len(rows) == 1
    assert rows["action"].iloc[0] == "long" and rows["risk"].iloc[0] == 0.1
    # Another configuration is another series.
    assert log.record(reading(1_000, config_hash="other")) is True
    assert len(log.readings("abc")) == 1 and log.latest_hash() == "other"


def test_record_rejects_an_incomplete_row(tmp_path):
    log = PredictionLog(tmp_path / "p.sqlite")
    row = reading(1_000)
    del row["action"]
    with pytest.raises(ValueError):
        log.record(row)


def test_fingerprint_changes_with_any_setting():
    assert config_fingerprint({"drawdown": 0.1}) == config_fingerprint({"drawdown": 0.1})
    assert config_fingerprint({"drawdown": 0.1}) != config_fingerprint({"drawdown": 0.15})


def test_reading_from_report_needs_a_next_bar_reading():
    report = {"params": {"model": "lgbm", "drawdown": 0.1, "horizon_bars": 30},
              "status": "draft",
              "today": {"action": "long", "risk": 0.1, "rule": "hold", "for_bar_ms": 5,
                        "trend": {"action": "flat", "distance": -0.02, "close": 1.0}}}
    row = reading_from_report(report, symbol="BTCUSDT", interval="4h", source="t",
                              recorded_at_ms=7)
    assert row["for_bar_ms"] == 5 and row["trend_action"] == "flat"
    assert row["config_hash"] == config_fingerprint(report["params"])
    del report["today"]["for_bar_ms"]
    with pytest.raises(ValueError):
        reading_from_report(report, symbol="BTCUSDT", interval="4h", source="t")


def test_a_flat_reading_takes_the_paper_position_out_and_pays_the_cost():
    bars = flat_bars([100.0] * 10 + [100.0 * 0.99**k for k in range(1, 21)])
    t = bars["open_time"].to_numpy()
    # Flat from bar 9: the open-to-open moves after that are all falls.
    readings = pd.DataFrame([reading(int(t[0]), "long"), reading(int(t[9]), "flat")])

    paper = score_paper(readings, bars, fee_bps=10, slippage_bps=2, bars_per_year=BARS_PER_YEAR)

    assert paper.readings == 2 and paper.on_time == 2
    # Flat for the falling stretch: it lost only the entry and exit costs.
    assert paper.risk_rule["total_return"] == pytest.approx(-0.0024, abs=1e-4)
    assert paper.hold["total_return"] < -0.15
    assert paper.risk_rule["exposure"] < paper.hold["exposure"] == 1.0


def test_a_late_reading_is_filled_at_the_first_open_after_it_was_recorded():
    bars = flat_bars([100.0 * 0.99**k for k in range(30)])
    t = bars["open_time"].to_numpy()
    on_time = pd.DataFrame([reading(int(t[0]), "flat")])
    # Recorded just after bar 5 opened: it cannot trade bar 0's open.
    assert int(t[5]) - int(t[0]) > GRACE_MS
    late = pd.DataFrame([reading(int(t[0]), "flat", recorded=int(t[5]) + 1)])

    a = score_paper(on_time, bars, fee_bps=0, slippage_bps=0, bars_per_year=BARS_PER_YEAR)
    b = score_paper(late, bars, fee_bps=0, slippage_bps=0, bars_per_year=BARS_PER_YEAR)

    assert a.late == 0 and b.late == 1
    assert b.rows[0]["filled_at"] == pd.Timestamp(int(t[6]), unit="ms").strftime("%Y-%m-%d %H:%M UTC")
    assert b.first_bar != a.first_bar


def test_keep_holds_the_previous_position_until_the_next_reading():
    bars = flat_bars([100.0] * 20)
    t = bars["open_time"].to_numpy()
    readings = pd.DataFrame(
        [reading(int(t[0]), "flat"), reading(int(t[5]), "keep"), reading(int(t[10]), "long")]
    )
    paper = score_paper(readings, bars, fee_bps=0, slippage_bps=0, bars_per_year=BARS_PER_YEAR)
    # Flat for bars 0..9, long for 10..19.
    assert paper.risk_rule["exposure"] == pytest.approx(0.5)


def test_outcomes_use_the_walkforward_label():
    bars = synthetic_bars(200)
    t = bars["open_time"].to_numpy()
    readings = pd.DataFrame([reading(int(t[i]), risk=0.5) for i in range(0, 190, 10)])
    paper = score_paper(readings, bars, fee_bps=10, slippage_bps=2, bars_per_year=BARS_PER_YEAR)

    from crypto_neural.labels import label_drawdown

    labels = label_drawdown(bars, 0.01, 6).set_index("open_time")["label"]
    expected = labels.loc[t[0:190:10]].dropna()
    assert paper.decided == len(expected)
    assert paper.falls == int(expected.sum())
    assert paper.brier is not None


def test_readings_from_two_configurations_are_not_scored_together():
    bars = synthetic_bars(50)
    t = bars["open_time"].to_numpy()
    mixed = pd.DataFrame([reading(int(t[0])), reading(int(t[1]), config_hash="zzz")])
    with pytest.raises(ValueError):
        score_paper(mixed, bars, fee_bps=10, slippage_bps=2, bars_per_year=BARS_PER_YEAR)


def test_no_readings_and_no_later_bar_are_reported_in_words():
    bars = synthetic_bars(50)
    empty = score_paper(pd.DataFrame(columns=list(reading(0))), bars, fee_bps=10,
                        slippage_bps=2, bars_per_year=BARS_PER_YEAR)
    assert empty.readings == 0 and "No readings" in empty.verdict

    future = int(bars["open_time"].iloc[-1]) + BAR_MS
    pending = score_paper(pd.DataFrame([reading(future, recorded=future + GRACE_MS + 5)]), bars,
                          fee_bps=10, slippage_bps=2, bars_per_year=BARS_PER_YEAR)
    assert pending.readings == 1 and "no bar has opened" in pending.verdict
