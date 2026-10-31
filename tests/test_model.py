from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gold_neural.model import WindowTransformer, probe_device


def test_forward_shape_and_logits():
    model = WindowTransformer(n_features=7, window=16, d_model=32, nhead=4, layers=1, dim_feedforward=64)
    batch = torch.randn(4, 16, 7)
    logits = model(batch)
    assert logits.shape == (4, 3)


def test_parameter_budget_stays_small():
    model = WindowTransformer(n_features=21, window=64)
    assert model.parameter_count() < 5_000_000
    assert model.parameter_count() > 10_000


def test_short_window_supported():
    model = WindowTransformer(n_features=3, window=8, d_model=16, nhead=2, layers=1, dim_feedforward=32)
    assert model(torch.randn(2, 8, 3)).shape == (2, 3)


def test_bad_head_divisibility_rejected():
    with pytest.raises(ValueError, match="divisible"):
        WindowTransformer(n_features=3, window=8, d_model=30, nhead=4)


def test_probe_respects_cpu_request():
    result = probe_device("cpu")
    assert result.device == "cpu"
    assert result.is_gpu is False
    assert "cpu" in result.note.lower()


def test_probe_prints_through_forward_and_backward():
    """Whatever the machine, the probe must return a usable device."""
    result = probe_device("auto")
    assert result.device in {"cpu", "cuda"}
    assert result.note
