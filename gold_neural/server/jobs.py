"""Background job runner for long actions triggered from the web UI.

One worker thread, deliberately. Backfill, dataset builds, and leakage runs
all touch the same Parquet store and SQLite index; running two at once would
contend for the single SQLite writer and produce confusing partial results.
The API stays responsive while a job runs, and the UI streams progress over
the WebSocket.

Jobs live in memory only. Their *results* are persisted separately by
:mod:`gold_neural.server.reports`, so a restart loses the log but not the
evidence.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_LOG_LINES = 400
MAX_JOBS = 50

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    seq: int
    status: str = "queued"
    message: str = "queued"
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    log: list[str] = field(default_factory=list)
    stop_requested: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    stop_hooks: list[Callable[[], None]] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.status in {"queued", "running"}

    def to_dict(self, include_log: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_lines": len(self.log),
        }
        if include_log:
            payload["log"] = list(self.log)
        return payload


class JobContext:
    """Handle passed to job functions so they can report progress."""

    def __init__(self, manager: "JobManager", job: Job):
        self._manager = manager
        self._job = job

    @property
    def id(self) -> str:
        return self._job.id

    def log(self, message: str) -> None:
        self._manager._append_log(self._job, message)

    def message(self, message: str) -> None:
        self._manager._set_message(self._job, message)

    def update(self, message: str | None = None, progress: float | None = None) -> None:
        self._manager._update(self._job, message=message, progress=progress)

    def progress(self, value: float, message: str | None = None) -> None:
        self._manager._update(self._job, message=message, progress=value)

    def should_stop(self) -> bool:
        """True once someone pressed Stop for this job."""
        return self._job.stop_event.is_set()

    def on_stop(self, hook: Callable[[], None]) -> None:
        """Register a callback fired when Stop is pressed.

        Training uses this to set a ``multiprocessing.Event`` the child polls;
        the job worker only knows about its own threading.Event.
        """
        self._manager._add_stop_hook(self._job, hook)


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="goldnerual-job"
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._seq = itertools.count(1)

    # -- mutation --------------------------------------------------------
    def submit(
        self,
        kind: str,
        params: dict[str, Any],
        fn: Callable[[JobContext], dict[str, Any]],
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            params=params,
            seq=next(self._seq),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()
        self._executor.submit(self._run, job, fn)
        return job

    def _evict_locked(self) -> None:
        finished = [jid for jid in self._order if not self._jobs[jid].running]
        while len(self._jobs) > MAX_JOBS and finished:
            victim = finished.pop(0)
            self._jobs.pop(victim, None)
            self._order.remove(victim)

    def _append_log(self, job: Job, message: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        with self._lock:
            job.log.append(f"{stamp} {message}")
            if len(job.log) > MAX_LOG_LINES:
                del job.log[: len(job.log) - MAX_LOG_LINES]

    def _add_stop_hook(self, job: Job, hook: Callable[[], None]) -> None:
        with self._lock:
            job.stop_hooks.append(hook)

    def request_stop(self, job_id: str) -> bool:
        """Ask a running job to stop. Returns False if it is not running."""
        job = self.get(job_id)
        if job is None or not job.running:
            return False
        with self._lock:
            job.stop_requested = True
            job.stop_event.set()
            hooks = list(job.stop_hooks)
            job.message = "stopping…"
        for hook in hooks:
            try:
                hook()
            except Exception:  # noqa: BLE001 - a bad hook must not block the stop
                log.exception("stop hook for job %s failed", job_id)
        self._append_log(job, "stop requested")
        return True

    def _set_message(self, job: Job, message: str) -> None:
        with self._lock:
            job.message = message

    def _update(
        self, job: Job, message: str | None = None, progress: float | None = None
    ) -> None:
        with self._lock:
            if message is not None:
                job.message = message
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))

    # -- execution -------------------------------------------------------
    def _run(self, job: Job, fn: Callable[[JobContext], dict[str, Any]]) -> None:
        ctx = JobContext(self, job)
        with self._lock:
            job.status = "running"
            job.started_at = time.time()
            job.message = "running"
        ctx.log(f"started {job.kind}")
        try:
            result = fn(ctx) or {}
            stopped = job.stop_requested
            with self._lock:
                job.result = result
                job.status = "stopped" if stopped else "ok"
                if not stopped:
                    job.progress = 1.0
                job.message = "stopped" if stopped else "done"
            ctx.log("stopped" if stopped else "finished")
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "failed"
            ctx.log(f"FAILED: {type(exc).__name__}: {exc}")
            for line in traceback.format_exc().splitlines()[-8:]:
                ctx.log(line)
        finally:
            with self._lock:
                job.finished_at = time.time()

    # -- reads -----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, include_log: bool = False) -> list[dict[str, Any]]:
        """Job summaries, newest first.

        Logs are excluded by default: the socket sends this every 500 ms and a
        full log per job would make that payload grow without bound. The UI
        fetches one job's log from ``GET /api/jobs/{id}`` when it is opened.
        """
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.seq, reverse=True)
            return [job.to_dict(include_log=include_log) for job in jobs]

    def counts(self) -> dict[str, int]:
        with self._lock:
            running = sum(1 for job in self._jobs.values() if job.running)
            return {"running": running, "total": len(self._jobs)}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
