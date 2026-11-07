from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crypto_neural.config import FeatureConfig, LabelConfig
from crypto_neural.dataset import build_dataset
from crypto_neural.training import TrainConfig, train_model

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
    assert 0.0 <= metrics["baseline_balanced_accuracy"] <= 1.0

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
    # The live chart plots these against each other every epoch.
    assert all(0.0 <= item["val_balanced_accuracy"] <= 1.0 for item in seen)
    assert all(0.0 < item["baseline_balanced_accuracy"] <= 1.0 for item in seen)


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        TrainConfig(window=1)


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        TrainConfig(train_frac=0.95, val_frac=0.1)


def test_balanced_baseline_matches_a_constant_guess():
    """A constant predictor scores 1/3 on balanced accuracy, not 50%."""
    import numpy as np

    from crypto_neural.training import _metrics

    y_true = np.array([-1] * 50 + [0] * 30 + [1] * 20)
    y_pred = np.full(100, -1)
    metrics = _metrics(y_true, y_pred, np.zeros(100))

    assert metrics["baseline_accuracy"] == 0.5
    assert metrics["baseline_balanced_accuracy"] == 0.3333
    assert metrics["balanced_accuracy"] == 0.3333
    assert metrics["beats_baseline"] is False


def test_open_ended_run_stops_only_when_asked(prepared, tmp_path):
    """epochs=0 with patience=0 means: run until Stop, not until a counter."""
    frame, features = prepared
    seen = {"epochs": 0}

    def progress(_payload):
        seen["epochs"] += 1

    result = train_model(
        frame,
        features,
        _config(epochs=0, patience=0),
        tmp_path,
        progress=progress,
        should_stop=lambda: seen["epochs"] >= 3,
    )

    assert result.stop_reason == "stopped"
    assert result.status == "stopped"
    assert result.epochs_done == 3
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_file()


def test_checkpoint_is_written_at_the_best_epoch(prepared, tmp_path):
    frame, features = prepared
    result = train_model(frame, features, _config(epochs=3, patience=0), tmp_path)

    assert result.metrics["best_epoch"] >= 1
    files = sorted(path.name for path in tmp_path.glob("v*.pt"))
    assert files, "training should leave at least one checkpoint"

    payload = torch.load(result.checkpoint_path, weights_only=False)
    assert payload["epoch"] == result.metrics["best_epoch"]
    assert result.checkpoint_version == result.metrics["best_epoch"]


def test_learning_rate_warms_up_then_decays(prepared, tmp_path):
    frame, features = prepared
    seen: list[dict] = []
    train_model(frame, features, _config(epochs=8, patience=0), tmp_path, progress=seen.append)

    rates = [entry["lr"] for entry in seen]
    assert len(rates) == 8
    assert rates[0] < rates[3], "warm-up should raise the rate over the first epochs"
    assert rates[-1] < rates[3], "cosine decay should end below the peak"


def test_purge_drops_rows_whose_label_resolves_in_the_next_block():
    import numpy as np

    from crypto_neural.training import purge_overlap

    bars = synthetic_bars(n=1500)
    frame = build_dataset(
        bars, FeatureConfig(), LabelConfig(max_holding_bars=48)
    ).frame
    train = np.arange(0, 800)
    val = np.arange(800, 1100)
    test = np.arange(1100, 1400)

    kept_train, kept_val, purged = purge_overlap(frame, train, val, test)

    exit_time = frame["exit_time"].to_numpy()
    open_time = frame["open_time"].to_numpy()
    # Every surviving row is decided before the next block opens...
    assert (exit_time[kept_train] < open_time[val[0]]).all()
    assert (exit_time[kept_val] < open_time[test[0]]).all()
    # ...something was actually purged, and only from the tail.
    assert purged["train"] > 0 and purged["validation"] > 0
    assert kept_train[0] == train[0] and kept_val[0] == val[0]
    assert purged["train"] <= 48 and purged["validation"] <= 48


def test_live_balanced_accuracy_agrees_with_the_final_metrics():
    import numpy as np

    from crypto_neural.training import _metrics, balanced_accuracy

    rng = np.random.default_rng(3)
    y_true = rng.choice([-1, 0, 1], size=300, p=[0.5, 0.3, 0.2])
    y_pred = rng.choice([-1, 0, 1], size=300)
    live, guess = balanced_accuracy(y_true, y_pred)
    final = _metrics(y_true, y_pred, np.zeros(300))
    assert round(live, 4) == final["balanced_accuracy"]
    assert round(guess, 4) == final["baseline_balanced_accuracy"]
    # Only the classes present count towards the guess.
    assert balanced_accuracy(np.array([1, 1, -1]), np.array([1, 1, 1]))[1] == 0.5


def test_probabilities_undo_the_class_weights(prepared, tmp_path):
    """The weighted loss inflates rare classes by their weight; the corrected
    probabilities divide it back out, row by row, and still sum to one."""
    import numpy as np

    from crypto_neural.training import load_checkpoint, predict_proba

    frame, features = prepared
    result = train_model(frame, features, _config(epochs=1), tmp_path)
    checkpoint = load_checkpoint(result.checkpoint_path)
    weights = np.asarray(checkpoint["class_weights"])
    assert weights.shape == (3,) and not np.allclose(weights, weights[0])

    rows = np.arange(len(frame) - 50, len(frame))
    raw = predict_proba(checkpoint, frame, rows, undo_class_weights=False)
    fixed = predict_proba(checkpoint, frame, rows)
    assert np.allclose(fixed.sum(axis=1), 1.0, atol=1e-5)
    ratio = (raw / fixed) / (raw / fixed)[:, :1]
    assert np.allclose(ratio, weights / weights[0], rtol=1e-4)
