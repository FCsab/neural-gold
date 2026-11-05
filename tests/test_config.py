from __future__ import annotations

import pytest

from crypto_neural.config import ConfigError, load_config


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.exchange.symbol == "BTCUSDC"
    assert cfg.exchange.interval == "5m"
    assert cfg.features.feature_lag == 1
    assert cfg.labels.max_holding_bars == 12
    assert cfg.bars_dir.parts[-3:] == ("klines", "BTCUSDC", "5m")


def test_yaml_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
exchange:
  symbol: BTCUSDC
  interval: 15m
labels:
  k_up: 3.0
  max_holding_bars: 8
features:
  return_lookbacks: [1, 5, 20]
"""
    )
    cfg = load_config()
    assert cfg.exchange.interval == "15m"
    assert cfg.labels.k_up == 3.0
    assert cfg.labels.k_down == 1.5  # untouched default
    assert cfg.features.return_lookbacks == (1, 5, 20)


def test_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("labels:\n  k_up: 3.0\n")
    monkeypatch.setenv("CRYPTONEURAL_LABELS_K_UP", "2.5")
    monkeypatch.setenv("CRYPTONEURAL_FEATURES_FEATURE_LAG", "0")
    cfg = load_config()
    assert cfg.labels.k_up == 2.5
    assert cfg.features.feature_lag == 0


def test_unknown_key_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("labels:\n  k_up: 2.0\n  kup: 5.0\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config()


def test_unknown_section_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("execution:\n  max_position: 0.25\n")
    with pytest.raises(ConfigError, match="unknown config sections"):
        load_config()


def test_invalid_ambiguity_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("labels:\n  ambiguity: optimistic\n")
    with pytest.raises(ConfigError, match="ambiguity"):
        load_config()


def test_example_config_loads():
    cfg = load_config("config.example.yaml")
    assert cfg.exchange.symbol == "BTCUSDC"
    assert cfg.features.feature_lag == 1
