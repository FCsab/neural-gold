from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gold_neural.config import FeatureConfig, LabelConfig
from gold_neural.dataset import build_dataset
from gold_neural.training import TrainConfig, train_model

from .conftest import synthetic_bars


@pytest.fixture(scope="module")
def prepared():
    bars = synthetic_bars(n=2600)
    dataset = build_dataset(bars, FeatureConfig(feature_lag=1), LabelConfig())
    return dataset.frame, dataset.feature_columns


def _config(**overrides) -> TrainConfig:
    base = dict(
        window=16,
        epochs=2,
        batch_size=128,
        d_model=32,
        nhead=4,
        layers=1,
        dim_feedforward=64,
        patience=5,
        max_minutes=None,
        num_threads=1,
        device="cpu",
        seed=3,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_training_completes_and_writes_checkpoint(prepared, tmp_path):
    frame, features = prepared
    result = train_model(frame, features, _config(), tmp_path)

    assert result.status == "ok"
    assert result.stop_reason == "completed"
    assert result.epochs_done == 2
    assert len(result.history) == 2
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_file()

    metrics = result.metrics
    assert len(metrics["confusion_matrix"]) == 3
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == metrics["test_rows"]
    assert set(metrics["per_class"]) == {"-1", "0", "1"}
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["baseline_accuracy"] <= 1.0

    payload = torch.load(result.checkpoint_path, weights_only=False)
    assert payload["feature_names"] == features
    assert payload["window"] == 16
    assert len(payload["scaler"]["mean"]) == len(features)


def test_stop_flag_exits_cleanly(prepared, tmp_path):
    frame, features = prepared
    seen = {"epochs": 0}

    def progress(_payload):
        seen["epochs"] += 1

    result = train_model(
        frame,
        features,
        _config(epochs=50),
        tmp_path,
        progress=progress,
        should_stop=lambda: seen["epochs"] >= 1,
    )

    assert result.stop_reason == "stopped"
    assert result.status == "stopped"
    assert result.epochs_done == 1
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_file()


def test_time_budget_before_first_epoch(prepared, tmp_path):
    frame, features = prepared
    result = train_model(frame, features, _config(max_minutes=0.0001), tmp_path)

    assert result.stop_reason == "time_budget"
    assert result.status == "stopped"
    assert result.epochs_done == 0
    assert result.checkpoint_path is None
    assert "stopped before the first epoch" in result.metrics["note"]


def test_history_records_every_epoch(prepared, tmp_path):
    frame, features = prepared
    seen: list[dict] = []
    train_model(frame, features, _config(epochs=3), tmp_path, progress=seen.append)

    assert len(seen) == 3
    assert [int(item["epoch"]) for item in seen] == [1, 2, 3]
    assert all("val_loss" in item and "train_loss" in item for item in seen)
    assert seen[-1]["elapsed_s"] >= seen[0]["elapsed_s"]


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        TrainConfig(window=1)


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        TrainConfig(train_frac=0.95, val_frac=0.1)
