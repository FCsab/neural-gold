from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from crypto_neural.config import AppConfig, DataConfig, ServerConfig
from crypto_neural.data.store import BarStore
from crypto_neural.server.app import create_app

from .conftest import small_risk_config, synthetic_bars


@pytest.fixture()
def config(tmp_path) -> AppConfig:
    # The model learns from the traded pair here: the fixtures store one
    # series, and a different training symbol would reach for the network.
    return AppConfig(
        data=DataConfig(root=str(tmp_path / "data"), training_symbol="BTCUSDC"),
        server=ServerConfig(tls_enabled=False),
    )


@pytest.fixture()
def seeded(config: AppConfig) -> AppConfig:
    store = BarStore(config.data_root, config.exchange.symbol, config.exchange.interval)
    store.write_bars(synthetic_bars(n=3000))
    return config


@pytest.fixture()
def client(config: AppConfig) -> TestClient:
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def seeded_client(seeded: AppConfig) -> TestClient:
    app = create_app(seeded)
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"ok", "failed", "stopped"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# -- static + access ---------------------------------------------------------
def test_index_is_served_without_auth(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "crypto-neural" in response.text
    assert "/assets/app.js" in response.text


def test_static_assets_are_served(client: TestClient):
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/vendor/lightweight-charts.standalone.production.js").status_code == 200


def test_api_needs_no_token(client: TestClient):
    assert client.get("/api/status").status_code == 200


def test_ip_allowlist_blocks_unknown_source(tmp_path):
    config = AppConfig(
        data=DataConfig(root=str(tmp_path / "data")),
        server=ServerConfig(tls_enabled=False, allowed_ips=("10.0.0.0/8",)),
    )
    with TestClient(create_app(config)) as test_client:
        assert test_client.get("/api/status").status_code == 403


# -- data endpoints --------------------------------------------------------
def test_status_payload(client: TestClient):
    payload = client.get("/api/status").json()
    assert payload["symbol"] == "BTCUSDC"
    assert payload["interval"] == "5m"
    assert payload["bar_seconds"] == 300
    training = payload["training"]
    assert isinstance(training["available"], bool)
    if training["available"]:
        assert "torch" in training
    else:
        assert "reason" in training


def test_data_info_on_empty_store(client: TestClient):
    payload = client.get("/api/data/info").json()
    assert payload["coverage"]["rows"] == 0
    assert payload["partitions"] == []
    assert payload["dataset"] is None
    assert payload["leakage"] is None


def test_bars_endpoint_returns_candles(seeded_client: TestClient):
    payload = seeded_client.get("/api/data/bars?limit=50").json()
    assert payload["interval_s"] == 300
    bars = payload["bars"]
    assert len(bars) == 50
    assert set(bars[0]) == {"time", "open", "high", "low", "close"}
    times = [bar["time"] for bar in bars]
    assert times == sorted(times)
    assert bars[0]["time"] < bars[-1]["time"]


def test_bars_endpoint_empty_store(client: TestClient):
    payload = client.get("/api/data/bars?limit=10").json()
    assert payload["bars"] == []


# -- jobs ------------------------------------------------------------------
def test_verify_job_runs(seeded_client: TestClient):
    job_id = seeded_client.post("/api/data/verify", json={}).json()["job_id"]
    job = wait_for_job(seeded_client, job_id)
    assert job["status"] == "ok", job
    assert job["result"]["rows"] == 3000
    assert job["result"]["ok"] is True
    assert job["log"], "job log should not be empty"

    info = seeded_client.get("/api/data/info").json()
    assert info["verify"]["rows"] == 3000


def test_dataset_preview_job(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/dataset/preview",
        json={"k_up": 2.0, "k_down": 1.5, "max_holding_bars": 12},
    )
    job = wait_for_job(seeded_client, response.json()["job_id"])
    assert job["status"] == "ok", job
    result = job["result"]
    assert result["labelled"] > 0
    assert result["distribution"]["total"] == result["labelled"]


def test_dataset_build_job_writes_report(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/dataset/build",
        json={"k_up": 2.0, "k_down": 1.5, "max_holding_bars": 12, "feature_lag": 1},
    )
    job = wait_for_job(seeded_client, response.json()["job_id"])
    assert job["status"] == "ok", job
    result = job["result"]
    assert result["rows"] > 0
    from crypto_neural.config import FeatureConfig
    from crypto_neural.features import feature_names

    assert result["features"] == len(feature_names(FeatureConfig()))
    assert result["split"]["train"] > result["split"]["test"] > 0
    assert "1" in result["by_class"] or "-1" in result["by_class"]

    info = seeded_client.get("/api/data/info").json()
    assert info["dataset"]["rows"] == result["rows"]
    assert info["dataset"]["params"]["k_up"] == 2.0


def test_leakage_job_reports_pass(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/leakage/run", json={"samples": 2}
    )
    job = wait_for_job(seeded_client, response.json()["job_id"], timeout=120)
    assert job["status"] == "ok", job
    assert job["result"]["passed"] is True

    info = seeded_client.get("/api/data/info").json()
    assert info["leakage"]["passed"] is True
    assert len(info["leakage"]["checks"]) >= 5


def test_invalid_params_are_rejected(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/dataset/build", json={"k_up": -1}
    )
    assert response.status_code == 422


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/api/jobs/nope").status_code == 404


def test_job_list_is_available(seeded_client: TestClient):
    seeded_client.post("/api/data/verify", json={})
    payload = seeded_client.get("/api/jobs").json()
    assert payload["jobs"]
    assert "log" not in payload["jobs"][0]


# -- websocket -------------------------------------------------------------
def test_websocket_respects_the_ip_allowlist(tmp_path):
    config = AppConfig(
        data=DataConfig(root=str(tmp_path / "data")),
        server=ServerConfig(tls_enabled=False, allowed_ips=("10.0.0.0/8",)),
    )
    with TestClient(create_app(config)) as test_client:
        with pytest.raises(Exception):
            with test_client.websocket_connect("/api/ws") as socket:
                socket.receive_json()


def test_websocket_streams_jobs(client: TestClient):
    with client.websocket_connect("/api/ws") as socket:
        message = socket.receive_json()
        assert message["type"] == "jobs"
        assert message["jobs"] == []


def test_websocket_ends_quietly_when_uvicorn_already_closed_it(client, monkeypatch):
    """A client that stops answering pings (a sleeping laptop) is closed by
    uvicorn itself; over TLS the next send then raises RuntimeError rather
    than WebSocketDisconnect. That is a gone client, not a server error."""
    from starlette.websockets import WebSocket

    real_send = WebSocket.send_text
    sends = []

    async def send_text(self, data):
        sends.append(data)
        if len(sends) > 1:
            raise RuntimeError(
                "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
            )
        await real_send(self, data)

    monkeypatch.setattr(WebSocket, "send_text", send_text)
    # Leaving the block re-raises anything the handler raised.
    with client.websocket_connect("/api/ws") as socket:
        assert socket.receive_json()["jobs"] == []
        client.post("/api/data/verify", json={})  # a job change forces a send
        deadline = time.monotonic() + 10
        while len(sends) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
    assert len(sends) == 2


# -- training --------------------------------------------------------------
def _build_dataset(client: TestClient) -> None:
    job_id = client.post("/api/dataset/build", json={}).json()["job_id"]
    job = wait_for_job(client, job_id, timeout=120)
    assert job["status"] == "ok", job


def test_train_requires_dataset(client: TestClient):
    response = client.post("/api/train/start", json={"minutes": 1})
    assert response.status_code == 409
    assert "dataset" in response.json()["detail"].lower()


def test_train_status_reports_torch(client: TestClient):
    payload = client.get("/api/train/status").json()
    assert payload["dataset_ready"] is False
    assert payload["availability"]["available"] is True
    assert "torch" in payload["availability"]


def test_training_job_trains_and_registers_a_model(seeded_client: TestClient):
    _build_dataset(seeded_client)

    response = seeded_client.post(
        "/api/train/start",
        json={
            "minutes": 3,
            "epochs": 1,
            "window": 16,
            "batch_size": 128,
            "patience": 5,
            "device": "cpu",
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(seeded_client, response.json()["job_id"], timeout=600)

    assert job["status"] == "ok", job
    result = job["result"]
    assert result["model_id"].startswith("model_")
    assert result["epochs_done"] == 1
    assert result["checkpoint_path"]
    # One live-chart point per epoch, carried from the child to the job.
    assert [point["epoch"] for point in job["series"]] == [1.0]
    assert set(job["series"][0]) >= {"train_loss", "val_loss", "val_balanced_accuracy"}
    assert result["metrics"]["test_rows"] > 0

    models = seeded_client.get("/api/models").json()["models"]
    assert any(model["id"] == result["model_id"] for model in models)
    stored = next(model for model in models if model["id"] == result["model_id"])
    assert stored["status"] in {"candidate", "draft"}
    assert stored["window"] == 16
    # Each model comes with its comparison against both guessing baselines.
    assert stored["comparison"]["verdict"]
    assert stored["comparison"]["majority_accuracy"] == stored["metrics"]["baseline_accuracy"]


def test_stop_requests_are_honoured(seeded_client: TestClient):
    _build_dataset(seeded_client)

    job_id = seeded_client.post(
        "/api/train/start",
        json={
            "minutes": 10,
            "epochs": 100000,
            "window": 16,
            "batch_size": 128,
            "patience": 1000,
            "device": "cpu",
        },
    ).json()["job_id"]

    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        payload = seeded_client.get(f"/api/jobs/{job_id}").json()
        assert payload["status"] not in {"ok", "failed"}, payload
        if payload["status"] == "running" and payload["message"].startswith("epoch"):
            ready = True
            break
        time.sleep(0.2)
    assert ready, "training never started producing epochs"

    assert seeded_client.post(f"/api/jobs/{job_id}/stop").status_code == 200
    job = wait_for_job(seeded_client, job_id, timeout=300)
    assert job["status"] == "stopped", job
    assert job["result"]["stop_reason"] == "stopped"


def test_stop_unknown_job_is_409(client: TestClient):
    assert client.post("/api/jobs/nope/stop").status_code == 409


# -- backtest --------------------------------------------------------------
def test_backtest_requires_dataset(client: TestClient):
    response = client.post("/api/backtest/run", json={})
    assert response.status_code == 409


def test_backtest_without_a_model_fails_clearly(seeded_client: TestClient):
    _build_dataset(seeded_client)
    job_id = seeded_client.post("/api/backtest/run", json={}).json()["job_id"]
    job = wait_for_job(seeded_client, job_id, timeout=120)
    assert job["status"] == "failed"
    assert "model" in (job["error"] or "").lower()


def test_backtest_after_training_produces_an_equity_curve(seeded_client: TestClient):
    _build_dataset(seeded_client)

    training = seeded_client.post(
        "/api/train/start",
        json={
            "minutes": 3,
            "epochs": 1,
            "window": 16,
            "batch_size": 128,
            "patience": 5,
            "device": "cpu",
        },
    )
    assert training.status_code == 200, training.text
    assert wait_for_job(seeded_client, training.json()["job_id"], timeout=600)["status"] == "ok"

    response = seeded_client.post(
        "/api/backtest/run",
        json={"initial_equity": 1000, "fee_bps": 10, "slippage_bps": 2},
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(seeded_client, response.json()["job_id"], timeout=600)
    assert job["status"] == "ok", job

    result = job["result"]
    metrics = result["metrics"]
    assert metrics["initial_equity"] == 1000
    assert "buy_hold_return" in metrics
    assert "max_drawdown" in metrics
    assert result["verdict"]["text"]
    assert isinstance(result["equity_curve"], list)

    info = seeded_client.get("/api/data/info").json()
    assert info["backtest"]["metrics"]["initial_equity"] == 1000

    last = seeded_client.get("/api/backtest/last").json()
    assert last["model"]["id"].startswith("model_")


# -- risk model -------------------------------------------------------------
def test_risk_route_is_bounded(client: TestClient):
    assert client.post("/api/risk/run", json={"drawdown": 0.9}).status_code == 422
    assert client.post("/api/risk/run", json={"nope": 1}).status_code == 422


def test_risk_job_runs_and_records(seeded: AppConfig):
    from dataclasses import replace

    # Small windows so the 3000 synthetic 5m bars hold several test blocks.
    config = replace(seeded, risk=small_risk_config())
    with TestClient(create_app(config)) as client:
        response = client.post("/api/risk/run", json={})
        assert response.status_code == 200, response.text
        job = wait_for_job(client, response.json()["job_id"], timeout=300)

        assert job["status"] == "ok", job
        assert job["result"]["status"] in {"candidate", "draft"}
        info = client.get("/api/data/info").json()
        report = info["risk"]
        assert report["period"]["folds"] >= 3
        assert report["verdict"]["text"]
        assert report["params"]["fee_bps"] == 10.0
        assert job["result"]["recorded"] is True

        # The reading is in the predictions log; a rerun for the same bar
        # does not replace it.
        again = wait_for_job(
            client, client.post("/api/risk/run", json={}).json()["job_id"],
            timeout=300,
        )
        assert again["status"] == "ok" and again["result"]["recorded"] is False

        # A run with other settings is an experiment, not part of the series.
        other = wait_for_job(
            client,
            client.post("/api/risk/run", json={"drawdown": 0.2}).json()["job_id"],
            timeout=300,
        )
        assert other["status"] == "ok" and other["result"]["recorded"] is False

        paper = client.get("/api/paper").json()
        assert paper["readings"] == 1
        assert paper["verdict"]


def test_risk_job_with_the_neural_model_runs_in_a_child_process(seeded: AppConfig):
    pytest.importorskip("torch")
    from dataclasses import replace

    config = replace(
        seeded, risk=small_risk_config(model="ensemble", mlp_epochs=2, mlp_hidden=(8,))
    )
    with TestClient(create_app(config)) as client:
        response = client.post("/api/risk/run", json={})
        job = wait_for_job(client, response.json()["job_id"], timeout=300)

        assert job["status"] == "ok", job
        report = client.get("/api/data/info").json()["risk"]
        assert report["params"]["model"] == "ensemble"


def test_status_reports_whether_readings_are_scheduled(config: AppConfig):
    from dataclasses import replace

    with TestClient(create_app(config)) as client:
        readings = client.get("/api/status").json()["readings"]
        assert readings == {"scheduled": False, "next_at": None}

    scheduled = replace(config, server=replace(config.server, record_readings=True))
    with TestClient(create_app(scheduled)) as client:
        for _ in range(50):
            readings = client.get("/api/status").json()["readings"]
            if readings["next_at"]:
                break
        assert readings["scheduled"] is True and readings["next_at"]


def test_job_series_is_returned_with_the_job_and_capped():
    from crypto_neural.server import jobs as jobs_module
    from crypto_neural.server.jobs import JobManager

    manager = JobManager()

    def body(ctx):
        for epoch in range(1, jobs_module.MAX_SERIES_POINTS + 11):
            ctx.point({"epoch": epoch, "val_loss": 1.0 / epoch})
        return {}

    job_id = manager.submit("training", {}, body).id
    deadline = time.time() + 30
    while manager.get(job_id).running and time.time() < deadline:
        time.sleep(0.02)
    job = manager.get(job_id)
    full = job.to_dict(include_log=True)
    assert len(full["series"]) == jobs_module.MAX_SERIES_POINTS
    assert full["series"][-1]["epoch"] == jobs_module.MAX_SERIES_POINTS + 10
    # The websocket snapshot stays small: no series, like no log.
    assert "series" not in job.to_dict(include_log=False)


# -- training restarts ------------------------------------------------------
class _FakeCtx:
    def __init__(self, stop_after: int | None = None):
        self.lines: list[str] = []
        self.calls = 0
        self.stop_after = stop_after

    def log(self, line: str) -> None:
        self.lines.append(line)

    def should_stop(self) -> bool:
        return self.stop_after is not None and self.calls >= self.stop_after


def _fake_child(outcomes, ctx, seen):
    """Stand-in for the training child: attempt n ends as ``outcomes[n-1]``."""

    def child(config, dataset_path, feature_names, checkpoint_dir, ctx_, *, attempt, epoch_offset):
        ctx.calls += 1
        seen.append({"attempt": attempt, "seed": config.seed, "offset": epoch_offset})
        val_loss, reason = outcomes[attempt - 1]
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "v1.pt").write_text("weights")
        return {
            "status": "ok" if reason == "early_stopping" else "stopped",
            "stop_reason": reason,
            "epochs_done": 5,
            "checkpoint_path": str(checkpoint_dir / "v1.pt"),
            "metrics": {"best_val_loss": val_loss, "balanced_accuracy": 0.4},
        }

    return child


def _restart_config():
    from crypto_neural.training import TrainConfig

    return TrainConfig(window=16, patience=10, seed=7)


def test_restarts_keep_the_attempt_with_the_lowest_validation_loss(tmp_path, monkeypatch):
    from crypto_neural.server import tasks

    ctx, seen = _FakeCtx(), []
    outcomes = [(1.10, "early_stopping"), (1.02, "early_stopping"), (1.05, "time_budget")]
    monkeypatch.setattr(tasks, "train_child", _fake_child(outcomes, ctx, seen))
    result = tasks._train_with_restarts(_restart_config(), tmp_path / "d.parquet", [], tmp_path, ctx)

    assert [s["seed"] for s in seen] == [7, 8, 9]  # a new seed each attempt
    assert [s["offset"] for s in seen] == [0, 5, 10]  # live epochs keep rising
    assert result["metrics"]["chosen_attempt"] == 2
    assert result["metrics"]["attempts"] == 3
    assert result["checkpoint_path"].endswith("attempt-2/v1.pt")
    assert result["epochs_done"] == 15
    assert result["stop_reason"] == "time_budget"  # the run ended on time
    # Only the kept attempt's checkpoints stay on disk.
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["attempt-2"]


def test_restarts_end_when_stop_is_pressed(tmp_path, monkeypatch):
    from crypto_neural.server import tasks

    ctx, seen = _FakeCtx(stop_after=2), []
    outcomes = [(1.1, "early_stopping")] * 5
    monkeypatch.setattr(tasks, "train_child", _fake_child(outcomes, ctx, seen))
    result = tasks._train_with_restarts(_restart_config(), tmp_path / "d.parquet", [], tmp_path, ctx)
    assert len(seen) == 2 and result["metrics"]["chosen_attempt"] == 1


def test_restarts_need_patience_to_mean_anything():
    """Patience 0 means an attempt never stalls, so it would never restart."""
    from crypto_neural.server.tasks import RESTART_PATIENCE, training_patience

    assert training_patience({"patience": 0, "restarts": True}) == RESTART_PATIENCE > 0
    assert training_patience({"patience": 4, "restarts": True}) == 4
    assert training_patience({"patience": 0, "restarts": False}) == 0
