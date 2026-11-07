"""FastAPI application: JSON API + the bundled single-page UI.

Design notes that matter:

* There is no login: the server runs on the owner's machine. Every ``/api``
  route still passes the optional IP allowlist (``server.allowed_ips``),
  which is empty, i.e. open to any source, by default.
* Long actions return a job id immediately. Progress and logs arrive over
  ``/api/ws``; the console pane fetches a job's log on demand so the socket
  payload stays small.
* OpenAPI/docs are disabled: the UI is the interface, and an unauthenticated
  schema dump is not worth the extra surface.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from ..config import AppConfig, ConfigError, FeatureConfig, LabelConfig
from ..data.store import store_from_config
from ..exchange.rest import interval_to_ms
from ..registry import compare_to_baselines, registry_from_config
from . import reports, tasks
from .jobs import JobManager
from .scheduler import ReadingScheduler
from .security import ip_allowed, make_ip_dependency

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------
class BackfillParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    years: float = Field(default=2.0, gt=0, le=10)
    start: str | None = None
    end: str | None = None


class DatasetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_up: float | None = Field(default=None, gt=0, le=20)
    k_down: float | None = Field(default=None, gt=0, le=20)
    max_holding_bars: int | None = Field(default=None, ge=1, le=288)
    atr_period: int | None = Field(default=None, ge=2, le=200)
    ambiguity: Literal["stop_priority", "up_priority"] | None = None
    feature_lag: int | None = Field(default=None, ge=0, le=10)


class LeakageParams(DatasetParams):
    samples: int = Field(default=10, ge=1, le=50)


class RiskParams(BaseModel):
    """The drawdown-risk walk-forward. Unset fields use the config defaults."""

    model_config = ConfigDict(extra="forbid")

    drawdown: float | None = Field(default=None, gt=0.01, le=0.5)
    horizon_days: int | None = Field(default=None, ge=1, le=180)
    # Fetch bars newer than the store's last one first, so the next-bar
    # reading is current. Off only for offline use.
    refresh: bool = True


class BacktestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    initial_equity: float = Field(default=10_000.0, gt=0, le=10_000_000)
    fee_bps: float = Field(default=10.0, ge=0, le=100)
    slippage_bps: float = Field(default=2.0, ge=0, le=100)
    min_confidence: float = Field(default=0.0, ge=0, lt=1)
    entry_rule: Literal["expected_value", "argmax"] = "expected_value"


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None = run until early stopping or Stop is pressed. That is the intended
    # default: a wall-clock cap that cuts a run before it converges is worse
    # than no cap, and patience already bounds a healthy run.
    minutes: float | None = Field(default=None, ge=0, le=1440)
    # 0 = no epoch limit: run until Stop is pressed.
    epochs: int = Field(default=0, ge=0, le=100000)
    window: int = Field(default=64, ge=8, le=512)
    batch_size: int = Field(default=256, ge=16, le=4096)
    lr: float = Field(default=3e-4, gt=0, le=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.8)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    # 0 = no early stopping: a human ends the run.
    patience: int = Field(default=0, ge=0, le=1000)
    # Start over with a new seed when an attempt stalls, keep the best by
    # validation loss. Patience 0 then means the default restart patience.
    restarts: bool = False
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    device: Literal["auto", "cpu"] = "auto"


def training_availability() -> dict[str, Any]:
    """Report whether PyTorch is installed without importing it.

    Importing torch in the API process would add ~300 MB of RSS and a couple of
    seconds of startup for a banner. The training child imports it anyway and
    probes the device there, which is the only place the answer matters.
    """
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):  # pragma: no cover - broken install
        spec = None

    if spec is None:
        return {
            "available": False,
            "reason": (
                "PyTorch is not installed in this environment. Install it with: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu "
                "(or /rocm6.4 for an AMD GPU)."
            ),
        }

    try:
        from importlib.metadata import version

        torch_version = version("torch")
    except Exception:  # pragma: no cover - metadata can be missing in odd installs
        torch_version = "unknown"

    return {"available": True, "torch": torch_version, "gpu_reported": None}


def _parse_iso_ms(text: str | None) -> int | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"cannot parse time {text!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _resolve_labels(config: AppConfig, params: DatasetParams) -> LabelConfig:
    overrides = {
        name: getattr(params, name)
        for name in ("k_up", "k_down", "max_holding_bars", "atr_period", "ambiguity")
        if getattr(params, name, None) is not None
    }
    try:
        return replace(config.labels, **overrides)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _resolve_features(config: AppConfig, params: DatasetParams) -> FeatureConfig:
    if params.feature_lag is None:
        return config.features
    try:
        return replace(config.features, feature_lag=params.feature_lag)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


# --------------------------------------------------------------------------
# app factory
# --------------------------------------------------------------------------
def create_app(
    config: AppConfig,
    manager: JobManager | None = None,
    *,
    tls: bool = False,
) -> FastAPI:
    manager = manager or JobManager()
    auth = make_ip_dependency(config)

    def submit_risk(params: dict[str, Any]):
        return manager.submit("risk", params, lambda ctx: tasks.run_risk(config, params, ctx))

    def scheduled_reading() -> None:
        # One queue: if a risk job is already waiting or running, it will
        # produce this bar's reading; a second would only be ignored.
        if any(job["kind"] == "risk" and job["status"] in {"queued", "running"}
               for job in manager.snapshot()):
            return
        submit_risk({"drawdown": None, "horizon_days": None, "refresh": True})

    scheduler = (
        ReadingScheduler(interval_to_ms(config.risk.interval), scheduled_reading)
        if config.server.record_readings
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()

    app = FastAPI(
        title="crypto-neural",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.scheduler = scheduler
    app.state.config = config
    app.state.jobs = manager
    app.state.started_at = time.time()
    app.state.tls = tls

    # -- status ----------------------------------------------------------
    @app.get("/api/status")
    def status(_: None = Depends(auth)) -> dict[str, Any]:
        store = store_from_config(config)
        coverage = store.coverage()
        return {
            "app": "crypto-neural",
            "version": __version__,
            "symbol": config.exchange.symbol,
            "interval": config.exchange.interval,
            "data_root": str(config.data_root.resolve()),
            "bar_seconds": interval_to_ms(config.exchange.interval) // 1000,
            "bars": coverage.rows,
            "current": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "uptime_s": round(time.time() - app.state.started_at, 1),
            "jobs": manager.counts(),
            "tls": app.state.tls,
            "training": training_availability(),
            "readings": {
                "scheduled": scheduler is not None,
                "next_at": (
                    datetime.fromtimestamp(scheduler.next_at, tz=timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    if scheduler is not None and scheduler.next_at
                    else None
                ),
            },
        }

    # -- data ------------------------------------------------------------
    @app.get("/api/data/info")
    def data_info(_: None = Depends(auth)) -> dict[str, Any]:
        store = store_from_config(config)
        coverage = store.coverage()
        return {
            "coverage": {
                "rows": coverage.rows,
                "start_ms": coverage.start_ms,
                "end_ms": coverage.end_ms,
            },
            "partitions": store.partitions(),
            "bars_dir": str(config.bars_dir.resolve()),
            "dataset": reports.read_report(config, "dataset"),
            "leakage": reports.read_report(config, "leakage"),
            "backtest": reports.read_report(config, "backtest"),
            "risk": reports.read_report(config, "risk"),
            "verify": reports.read_report(config, "verify"),
            "exchange": reports.read_report(config, "exchange"),
        }

    @app.get("/api/data/bars")
    def data_bars(
        limit: int = Query(default=600, ge=10, le=5000),
        end_ms: int | None = None,
        _: None = Depends(auth),
    ) -> dict[str, Any]:
        store = store_from_config(config)
        interval_ms = interval_to_ms(config.exchange.interval)
        start_ms = None if end_ms is None else int(end_ms) - limit * interval_ms * 2

        frame = store.load_bars(start_ms=start_ms, end_ms=end_ms)
        if frame.empty:
            return {"bars": [], "interval_s": interval_ms // 1000}

        frame = frame.iloc[-limit:]
        times = (frame["open_time"] // 1000).astype("int64").tolist()
        return {
            "interval_s": interval_ms // 1000,
            "bars": [
                {
                    "time": int(t),
                    "open": float(o),
                    "high": float(h),
                    "low": float(lo),
                    "close": float(c),
                }
                for t, o, h, lo, c in zip(
                    times, frame["open"], frame["high"], frame["low"], frame["close"]
                )
            ],
        }

    @app.post("/api/data/backfill")
    def post_backfill(
        params: BackfillParams, _: None = Depends(auth)
    ) -> dict[str, Any]:
        start_ms = _parse_iso_ms(params.start)
        end_ms = _parse_iso_ms(params.end)
        job = manager.submit(
            "backfill",
            params.model_dump(),
            lambda ctx: tasks.run_backfill(config, params.years, start_ms, end_ms, ctx),
        )
        return {"job_id": job.id}

    @app.post("/api/data/verify")
    def post_verify(_: None = Depends(auth)) -> dict[str, Any]:
        job = manager.submit("verify", {}, lambda ctx: tasks.run_verify(config, ctx))
        return {"job_id": job.id}

    @app.post("/api/data/exchange-info")
    def post_exchange_info(_: None = Depends(auth)) -> dict[str, Any]:
        job = manager.submit(
            "exchange-info", {}, lambda ctx: tasks.refresh_exchange_info(config, ctx)
        )
        return {"job_id": job.id}

    # -- dataset ---------------------------------------------------------
    @app.post("/api/dataset/preview")
    def post_dataset_preview(
        params: DatasetParams, _: None = Depends(auth)
    ) -> dict[str, Any]:
        labels_cfg = _resolve_labels(config, params)
        job = manager.submit(
            "label-preview",
            params.model_dump(),
            lambda ctx: tasks.preview_labels(config, labels_cfg, ctx),
        )
        return {"job_id": job.id}

    @app.post("/api/dataset/build")
    def post_dataset_build(
        params: DatasetParams, _: None = Depends(auth)
    ) -> dict[str, Any]:
        labels_cfg = _resolve_labels(config, params)
        features_cfg = _resolve_features(config, params)
        out_path = str(config.data_root / "dataset.parquet")
        job = manager.submit(
            "dataset-build",
            params.model_dump(),
            lambda ctx: tasks.run_dataset_build(
                config, features_cfg, labels_cfg, out_path, ctx
            ),
        )
        return {"job_id": job.id}

    # -- leakage ---------------------------------------------------------
    @app.post("/api/leakage/run")
    def post_leakage(params: LeakageParams, _: None = Depends(auth)) -> dict[str, Any]:
        labels_cfg = _resolve_labels(config, params)
        features_cfg = _resolve_features(config, params)
        job = manager.submit(
            "leakage",
            params.model_dump(),
            lambda ctx: tasks.run_leakage(
                config, features_cfg, labels_cfg, params.samples, ctx
            ),
        )
        return {"job_id": job.id}

    # -- training --------------------------------------------------------
    @app.post("/api/train/start")
    def post_train(params: TrainParams, _: None = Depends(auth)) -> dict[str, Any]:
        availability = training_availability()
        if not availability["available"]:
            raise HTTPException(status_code=409, detail=availability["reason"])
        if not (config.data_root / "dataset.parquet").is_file():
            raise HTTPException(
                status_code=409,
                detail="no dataset on disk; press “Train and test” in the web UI, or run `cli dataset`",
            )
        payload = params.model_dump()
        job = manager.submit(
            "training", payload, lambda ctx: tasks.run_training(config, payload, ctx)
        )
        return {"job_id": job.id}

    @app.get("/api/train/status")
    def train_status(_: None = Depends(auth)) -> dict[str, Any]:
        return {
            "availability": training_availability(),
            "dataset_ready": (config.data_root / "dataset.parquet").is_file(),
        }

    @app.get("/api/models")
    def list_models(_: None = Depends(auth)) -> dict[str, Any]:
        registry = registry_from_config(config)
        models = registry.list_models(limit=50)
        for model in models:
            model["comparison"] = compare_to_baselines(model.get("metrics"))
        return {"models": models}

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str, _: None = Depends(auth)) -> dict[str, Any]:
        registry = registry_from_config(config)
        model = registry.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="unknown model")
        return {"model": model, "runs": registry.list_runs(model_id, limit=20)}

    # -- backtest --------------------------------------------------------
    @app.post("/api/backtest/run")
    def post_backtest(params: BacktestParams, _: None = Depends(auth)) -> dict[str, Any]:
        if not (config.data_root / "dataset.parquet").is_file():
            raise HTTPException(
                status_code=409,
                detail="no dataset on disk; build one from the Labels tab first",
            )
        payload = params.model_dump()
        job = manager.submit(
            "backtest", payload, lambda ctx: tasks.run_backtest(config, payload, ctx)
        )
        return {"job_id": job.id}

    @app.get("/api/backtest/last")
    def backtest_last(_: None = Depends(auth)) -> dict[str, Any]:
        report = reports.read_report(config, "backtest")
        if report is None:
            raise HTTPException(status_code=404, detail="no backtest has been run")
        return report

    # -- risk model ------------------------------------------------------
    @app.post("/api/risk/run")
    def post_risk(params: RiskParams, _: None = Depends(auth)) -> dict[str, Any]:
        return {"job_id": submit_risk(params.model_dump()).id}

    @app.get("/api/paper")
    def get_paper(_: None = Depends(auth)) -> dict[str, Any]:
        """The predictions log scored against the bars that followed."""
        return tasks.paper_payload(config)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str, _: None = Depends(auth)) -> dict[str, Any]:
        if not manager.request_stop(job_id):
            raise HTTPException(status_code=409, detail="job is not running")
        return {"stopped": True}

    @app.get("/api/jobs")
    def list_jobs(_: None = Depends(auth)) -> dict[str, Any]:
        return {"jobs": manager.snapshot(include_log=False)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(auth)) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.to_dict(include_log=True)

    # -- websocket -------------------------------------------------------
    @app.websocket("/api/ws")
    async def websocket_jobs(websocket: WebSocket) -> None:
        client = websocket.client.host if websocket.client else None
        if not ip_allowed(config, client):
            await websocket.close(code=4403)
            return

        await websocket.accept()
        last_payload = ""
        try:
            while True:
                payload = json.dumps(
                    manager.snapshot(include_log=False), default=str, sort_keys=True
                )
                if payload != last_payload:
                    await websocket.send_text(json.dumps({"type": "jobs", "jobs": json.loads(payload)}))
                    last_payload = payload
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    # -- static UI -------------------------------------------------------
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    if WEB_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:  # pragma: no cover - packaging error
        @app.get("/", include_in_schema=False)
        def index_missing() -> JSONResponse:
            return JSONResponse(
                {"error": f"web directory not found at {WEB_DIR}"}, status_code=500
            )

    return app
