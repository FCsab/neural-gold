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


def test_gate_passes_only_with_enough_days_coverage_and_agreement():
    from crypto_neural.predictions import paper_gate

    ok = paper_gate(0.11, 0.10, days=120, coverage=0.95)
    assert ok["passed"] and ok["gap"] == pytest.approx(0.1) and ok["text"].startswith("PASS")

    short = paper_gate(0.10, 0.10, days=30, coverage=1.0)
    assert not short["passed"] and "30 of 90 days" in short["text"]

    patchy = paper_gate(0.10, 0.10, days=120, coverage=0.5)
    assert not patchy["passed"] and "50% of bars" in patchy["text"]

    apart = paper_gate(0.05, 0.10, days=120, coverage=1.0)
    assert not apart["passed"] and apart["gap"] == pytest.approx(0.5)


def test_gate_measures_a_near_zero_pnl_against_one_percent_of_capital():
    from crypto_neural.predictions import paper_gate

    # 0.1% vs 0.2% is a 50% ratio, but only 0.1% of capital apart.
    near_zero = paper_gate(0.001, 0.002, days=120, coverage=1.0)
    assert near_zero["passed"] and near_zero["gap"] == pytest.approx(0.1)


def test_on_time_readings_match_the_simulation_and_late_ones_do_not():
    bars = flat_bars([100.0] * 10 + [100.0 * 0.99**k for k in range(1, 31)])
    t = bars["open_time"].to_numpy()
    on_time = pd.DataFrame([reading(int(t[i]), "long" if i < 10 else "flat")
                            for i in range(len(t))])
    paper = score_paper(on_time, bars, fee_bps=10, slippage_bps=2, bars_per_year=BARS_PER_YEAR)
    gate = paper.gate
    assert gate["paper_return"] == gate["simulated_return"] and gate["gap"] == 0.0
    assert gate["coverage"] == pytest.approx(1.0)
    # Agreement alone is not enough: the record is minutes long.
    assert not gate["passed"] and "of 90 days" in gate["text"]
    assert "gate:" in paper.format() and "trend_rule" in gate

    # Bar 10's flat call recorded an hour late: the paper record is long for
    # one more falling bar (bar 11's on-time call takes it out), and the late
    # bar does not count as covered.
    late = on_time.copy()
    late.loc[10, "recorded_at_ms"] = int(t[10]) + 12 * BAR_MS
    late_gate = score_paper(late, bars, fee_bps=10, slippage_bps=2,
                            bars_per_year=BARS_PER_YEAR).gate
    assert late_gate["coverage"] < 1.0
    assert late_gate["simulated_return"] == gate["simulated_return"]
    assert late_gate["paper_return"] == pytest.approx(
        (1 + gate["paper_return"]) * 0.99 - 1, abs=2e-4
    )
