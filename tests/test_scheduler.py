from __future__ import annotations

from crypto_neural.server.scheduler import ReadingScheduler, next_run_s

FOUR_HOURS_MS = 4 * 3600 * 1000


def test_next_run_is_just_after_the_next_bar_opens():
    bar = 4 * 3600
    # 01:00 -> 04:00:45
    assert next_run_s(3600, FOUR_HOURS_MS, 45) == bar + 45
    # 00:00:10 is before this bar's run time: run at 00:00:45, not four hours later.
    assert next_run_s(10, FOUR_HOURS_MS, 45) == 45
    # Exactly at the run time -> the next bar's.
    assert next_run_s(45, FOUR_HOURS_MS, 45) == bar + 45


def test_scheduler_stops_without_submitting_early():
    calls = []
    scheduler = ReadingScheduler(FOUR_HOURS_MS, lambda: calls.append(1), clock=lambda: 3600.0)
    scheduler.start()
    scheduler.stop()
    assert calls == []
    assert scheduler.next_at == 4 * 3600 + 45
