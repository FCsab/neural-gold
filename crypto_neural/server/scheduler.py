"""Run the risk job just after every bar opens, so readings are on time.

A reading counts as on time only if it is recorded within a few minutes of
its bar opening (``predictions.GRACE_MS``). Nobody presses a button at 04:00
UTC, so without this the paper record would be all late readings, booked a
bar later than the walk-forward books them, and the comparison the plan
requires would measure the operator's sleep instead of the model.

Off by default (``server.record_readings``): each run trains the full
walk-forward, a few CPU-minutes every four hours. The scheduler only submits
ordinary jobs to the one queue, so it never runs beside another job; if a
training run holds the queue, the reading waits and is booked late, which is
the honest outcome.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


def next_run_s(now_s: float, interval_ms: int, delay_s: float) -> float:
    """The first bar boundary + ``delay_s`` strictly after ``now_s``."""
    interval_s = interval_ms / 1000.0
    boundary = (now_s // interval_s) * interval_s + delay_s
    while boundary <= now_s:
        boundary += interval_s
    return boundary


class ReadingScheduler:
    """A daemon thread that calls ``submit`` once per bar."""

    def __init__(
        self,
        interval_ms: int,
        submit: Callable[[], None],
        *,
        delay_s: float = 45.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.interval_ms = int(interval_ms)
        self.submit = submit
        self.delay_s = float(delay_s)
        self.clock = clock
        self.next_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "ReadingScheduler":
        self._thread = threading.Thread(target=self._loop, name="reading-scheduler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            self.next_at = next_run_s(self.clock(), self.interval_ms, self.delay_s)
            if self._stop.wait(max(0.0, self.next_at - self.clock())):
                return
            try:
                self.submit()
            except Exception:  # noqa: BLE001 - thread boundary: log and keep the schedule
                log.exception("scheduled risk reading could not be submitted")
