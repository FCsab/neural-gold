from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from gold_neural.config import AppConfig, DataConfig, ServerConfig
from gold_neural.data.store import BarStore
from gold_neural.server.app import create_app

from .conftest import synthetic_bars

TOKEN = "test-token-do-not-use"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture()
def config(tmp_path) -> AppConfig:
    return AppConfig(
        data=DataConfig(root=str(tmp_path / "data")),
        server=ServerConfig(tls_enabled=False),
    )


@pytest.fixture()
def seeded(config: AppConfig) -> AppConfig:
    store = BarStore(config.data_root, config.exchange.symbol, config.exchange.interval)
    store.write_bars(synthetic_bars(n=3000))
    return config


@pytest.fixture()
def client(config: AppConfig) -> TestClient:
    app = create_app(config, TOKEN)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def seeded_client(seeded: AppConfig) -> TestClient:
    app = create_app(seeded, TOKEN)
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if payload["status"] in {"ok", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# -- static + auth ---------------------------------------------------------
def test_index_is_served_without_auth(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "gold-neural" in response.text
    assert "/assets/app.js" in response.text


def test_static_assets_are_served(client: TestClient):
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/vendor/lightweight-charts.standalone.production.js").status_code == 200


def test_api_requires_token(client: TestClient):
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/status", headers=AUTH).status_code == 200


def test_ip_allowlist_blocks_unknown_source(tmp_path):
    config = AppConfig(
        data=DataConfig(root=str(tmp_path / "data")),
        server=ServerConfig(tls_enabled=False, allowed_ips=("10.0.0.0/8",)),
    )
    with TestClient(create_app(config, TOKEN)) as test_client:
        assert test_client.get("/api/status", headers=AUTH).status_code == 403


# -- data endpoints --------------------------------------------------------
def test_status_payload(client: TestClient):
    payload = client.get("/api/status", headers=AUTH).json()
    assert payload["symbol"] == "PAXGUSDT"
    assert payload["interval"] == "5m"
    assert payload["bar_seconds"] == 300
    assert payload["training"]["available"] is False
    assert "reason" in payload["training"]


def test_data_info_on_empty_store(client: TestClient):
    payload = client.get("/api/data/info", headers=AUTH).json()
    assert payload["coverage"]["rows"] == 0
    assert payload["partitions"] == []
    assert payload["dataset"] is None
    assert payload["leakage"] is None


def test_bars_endpoint_returns_candles(seeded_client: TestClient):
    payload = seeded_client.get("/api/data/bars?limit=50", headers=AUTH).json()
    assert payload["interval_s"] == 300
    bars = payload["bars"]
    assert len(bars) == 50
    assert set(bars[0]) == {"time", "open", "high", "low", "close"}
    times = [bar["time"] for bar in bars]
    assert times == sorted(times)
    assert bars[0]["time"] < bars[-1]["time"]


def test_bars_endpoint_empty_store(client: TestClient):
    payload = client.get("/api/data/bars?limit=10", headers=AUTH).json()
    assert payload["bars"] == []


# -- jobs ------------------------------------------------------------------
def test_verify_job_runs(seeded_client: TestClient):
    job_id = seeded_client.post("/api/data/verify", headers=AUTH, json={}).json()["job_id"]
    job = wait_for_job(seeded_client, job_id)
    assert job["status"] == "ok", job
    assert job["result"]["rows"] == 3000
    assert job["result"]["ok"] is True
    assert job["log"], "job log should not be empty"

    info = seeded_client.get("/api/data/info", headers=AUTH).json()
    assert info["verify"]["rows"] == 3000


def test_dataset_preview_job(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/dataset/preview",
        headers=AUTH,
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
        headers=AUTH,
        json={"k_up": 2.0, "k_down": 1.5, "max_holding_bars": 12, "feature_lag": 1},
    )
    job = wait_for_job(seeded_client, response.json()["job_id"])
    assert job["status"] == "ok", job
    result = job["result"]
    assert result["rows"] > 0
    assert result["features"] == 21
    assert result["split"]["train"] > result["split"]["test"] > 0
    assert "1" in result["by_class"] or "-1" in result["by_class"]

    info = seeded_client.get("/api/data/info", headers=AUTH).json()
    assert info["dataset"]["rows"] == result["rows"]
    assert info["dataset"]["params"]["k_up"] == 2.0


def test_leakage_job_reports_pass(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/leakage/run", headers=AUTH, json={"samples": 2}
    )
    job = wait_for_job(seeded_client, response.json()["job_id"], timeout=120)
    assert job["status"] == "ok", job
    assert job["result"]["passed"] is True

    info = seeded_client.get("/api/data/info", headers=AUTH).json()
    assert info["leakage"]["passed"] is True
    assert len(info["leakage"]["checks"]) >= 5


def test_invalid_params_are_rejected(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/dataset/build", headers=AUTH, json={"k_up": -1}
    )
    assert response.status_code == 422


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/api/jobs/nope", headers=AUTH).status_code == 404


def test_job_list_is_available(seeded_client: TestClient):
    seeded_client.post("/api/data/verify", headers=AUTH, json={})
    payload = seeded_client.get("/api/jobs", headers=AUTH).json()
    assert payload["jobs"]
    assert "log" not in payload["jobs"][0]


# -- websocket -------------------------------------------------------------
def test_websocket_requires_token(client: TestClient):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/ws?token=wrong") as socket:
            socket.receive_json()


def test_websocket_streams_jobs(client: TestClient):
    with client.websocket_connect(f"/api/ws?token={TOKEN}") as socket:
        message = socket.receive_json()
        assert message["type"] == "jobs"
        assert message["jobs"] == []
