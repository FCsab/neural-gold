"""The detached research campaign: file contract and API surface."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from gold_neural.config import AppConfig, DataConfig, ServerConfig
from gold_neural.server import research
from gold_neural.server.app import create_app

TOKEN = "test-token-do-not-use"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

STATUS = {
    "status": "running",
    "policy": "hold",
    "ranked_by": "median_excess",
    "interval": "4h",
    "elapsed_minutes": 1.5,
    "folds_total": 2,
    "specs_done": 1,
    "folds_per_minute": 12.0,
    "current": "u2.5d3.0h288w128s7|hold fold 3/19",
    "leaderboard": [
        {
            "spec": "u2.5d3.0h288w128s7",
            "folds": 2,
            "median_return": 0.04,
            "median_excess": 0.01,
            "worst_return": -0.05,
            "share_over_5pct": 0.5,
            "share_beating_hold": 0.5,
            "compounded": 0.08,
        }
    ],
}


def _config(tmp_path) -> AppConfig:
    return AppConfig(data=DataConfig(root=str(tmp_path)))


def _seed(tmp_path, status=None) -> AppConfig:
    config = _config(tmp_path)
    out = research.campaign_dir(config)
    out.mkdir(parents=True)
    (out / "status.json").write_text(json.dumps(status or STATUS))
    rows = [
        {
            "status": "ok",
            "policy": "hold",
            "key": "u2.5d3.0h288w128s7|hold|f0",
            "fold": 0,
            "test_start_ms": 1000,
            "test_end_ms": 2000,
            "return": 0.05,
            "buy_hold": 0.03,
            "excess": 0.02,
            "trades": 2,
            "stop_threshold": 1.0,
            "validation_excess": 0.01,
        },
        {
            "status": "ok",
            "policy": "hold",
            "key": "u2.5d3.0h288w128s7|hold|f1",
            "fold": 1,
            "test_start_ms": 3000,
            "test_end_ms": 4000,
            "return": -0.01,
            "buy_hold": 0.02,
            "excess": -0.03,
            "trades": 4,
            "stop_threshold": 0.55,
            "validation_excess": 0.02,
        },
        {
            "status": "ok",
            "policy": "hold",
            "key": "u4.0d2.0h192w64s8|hold|f0",
            "fold": 0,
            "test_start_ms": 1000,
            "test_end_ms": 2000,
            "return": 0.5,
            "buy_hold": 0.0,
            "excess": 0.5,
        },
        {
            "status": "ok",
            "policy": "signal",
            "key": "u2.5d3.0h288w128s7|f0",
            "fold": 0,
            "test_start_ms": 1000,
            "test_end_ms": 2000,
            "return": 0.9,
            "buy_hold": 0.0,
            "excess": 0.9,
        },
    ]
    with (out / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (out / "log.txt").write_text("one\ntwo\nthree\n")
    return config


def test_status_reads_the_best_spec_and_its_folds(tmp_path):
    config = _seed(tmp_path)

    payload = research.campaign_status(config)

    assert payload["best_spec"] == "u2.5d3.0h288w128s7"
    assert [fold["fold"] for fold in payload["folds"]] == [0, 1]
    assert payload["folds"][0]["return"] == pytest.approx(0.05)
    assert payload["status"]["folds_total"] == 2
    assert payload["log_tail"] == ["one", "two", "three"]
    assert payload["running"] is False
    assert payload["stop_requested"] is False


def test_status_prefers_best_folds_cached_in_status(tmp_path):
    status = dict(STATUS)
    status["best_folds"] = [
        {
            "fold": 0,
            "return": 0.123,
            "buy_hold": 0.01,
            "excess": 0.113,
            "trades": 1,
        }
    ]
    config = _seed(tmp_path, status=status)

    payload = research.campaign_status(config)
    assert len(payload["folds"]) == 1
    assert payload["folds"][0]["return"] == pytest.approx(0.123)

    other = research.campaign_status(config, spec="u4.0d2.0h192w64s8")
    assert [fold["return"] for fold in other["folds"]] == [0.5]


def test_status_treats_legacy_rows_without_policy_as_signal(tmp_path):
    legacy = {key: value for key, value in STATUS.items() if key != "policy"}
    config = _seed(tmp_path, status=legacy)

    payload = research.campaign_status(config)

    assert payload["status"]["policy"] == "signal"
    assert len(payload["folds"]) == 1
    assert payload["folds"][0]["return"] == pytest.approx(0.9)


def test_status_normalises_a_legacy_policy_suffixed_leaderboard(tmp_path):
    legacy = dict(STATUS)
    legacy["leaderboard"] = [dict(STATUS["leaderboard"][0], spec="u2.5d3.0h288w128s7|hold")]
    config = _seed(tmp_path, status=legacy)

    payload = research.campaign_status(config)

    assert payload["best_spec"] == "u2.5d3.0h288w128s7"
    assert payload["status"]["leaderboard"][0]["spec"] == "u2.5d3.0h288w128s7"
    assert len(payload["folds"]) == 2


def test_status_can_skip_folds_and_pick_another_spec(tmp_path):
    config = _seed(tmp_path)

    assert research.campaign_status(config, include_folds=False)["folds"] == []

    other = research.campaign_status(config, spec="u4.0d2.0h192w64s8")
    assert [fold["return"] for fold in other["folds"]] == [0.5]


def test_stop_writes_the_stop_file(tmp_path):
    config = _config(tmp_path)

    payload = research.stop_campaign(config)

    assert payload["stopped"] is True
    assert (research.campaign_dir(config) / "STOP").exists()
    assert research.campaign_status(config)["stop_requested"] is True


def test_start_builds_an_argument_list_and_records_the_pid(tmp_path):
    config = _config(tmp_path)
    captured = {}

    class FakeProcess:
        pid = 10_000_001

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    params = {
        "interval": "4h",
        "policy": "hold",
        "k_up": [2.5, 3.0],
        "k_down": [2.0],
        "holding": [288],
        "window": [64],
        "seeds": [7],
        "epochs": 3,
        "minutes_per_fold": 1.0,
        "test_days": 90,
        "val_days": 90,
        "min_train_days": 365,
        "fee_bps": 10.0,
        "slippage_bps": 2.0,
        "device": "cpu",
        "stop_thresholds": [0.5, 1.0],
        "reentry_margin": 0.05,
        "min_flat_bars": 1,
    }

    payload = research.start_campaign(config, params, popen=fake_popen)

    command = captured["command"]
    assert command[0] == "bash"
    assert "--policy" in command and "hold" in command
    assert "--k-up" in command and "2.5,3.0" in command
    assert "--stop-thresholds" in command and "0.5,1.0" in command
    assert str(research.campaign_dir(config)) in command
    assert "--shell" not in command
    assert captured["kwargs"]["start_new_session"] is True
    assert payload["pid"] == 10_000_001
    assert (research.campaign_dir(config) / "campaign.pid").read_text() == str(
        FakeProcess.pid
    )


def test_start_refuses_a_second_campaign(tmp_path):
    config = _config(tmp_path)
    out = research.campaign_dir(config)
    out.mkdir(parents=True)
    (out / "campaign.pid").write_text(str(os.getpid()))

    with pytest.raises(RuntimeError, match="already running"):
        research.start_campaign(config, {})


def test_an_exited_child_campaign_is_not_reported_running(tmp_path):
    """A supervisor we launched lingers as a zombie unless reaped."""
    import subprocess

    config = _config(tmp_path)
    out = research.campaign_dir(config)
    out.mkdir(parents=True)
    child = subprocess.Popen(["true"])
    (out / "campaign.pid").write_text(str(child.pid))
    # Wait for it to exit without reaping it, leaving a zombie behind.
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)

    assert research.running_pid(config) is None
    child.returncode = 0  # already reaped; stop Popen.__del__ from warning


def test_research_routes_are_authenticated_and_bounded(tmp_path):
    config = _seed(tmp_path)
    app = create_app(
        AppConfig(
            data=config.data,
            server=ServerConfig(tls_enabled=False),
        ),
        TOKEN,
    )

    with TestClient(app) as client:
        assert client.get("/api/research/status").status_code == 401

        payload = client.get("/api/research/status", headers=AUTH).json()
        assert payload["best_spec"] == "u2.5d3.0h288w128s7"
        assert len(payload["folds"]) == 2

        compact = client.get("/api/research/status?folds=false", headers=AUTH).json()
        assert compact["folds"] == []

        assert client.post("/api/research/start", headers=AUTH, json={"interval": "nope"}).status_code == 422
        assert client.post("/api/research/start", headers=AUTH, json={"k_up": [-1]}).status_code == 422

        stopped = client.post("/api/research/stop", headers=AUTH).json()
        assert stopped["stopped"] is True
        assert (research.campaign_dir(config) / "STOP").exists()


def test_research_start_route_passes_defaults(tmp_path, monkeypatch):
    config = _config(tmp_path)
    app = create_app(
        AppConfig(data=config.data, server=ServerConfig(tls_enabled=False)),
        TOKEN,
    )
    seen = {}

    def fake_start(_config, params):
        seen.update(params)
        return {"started": True, "pid": 123, "out": "somewhere"}

    monkeypatch.setattr(research, "start_campaign", fake_start)

    with TestClient(app) as client:
        response = client.post("/api/research/start", headers=AUTH, json={})

    assert response.status_code == 200, response.text
    assert response.json()["pid"] == 123
    assert seen["policy"] == "hold"
    assert seen["seeds"] == [7]
    assert seen["stop_thresholds"]
