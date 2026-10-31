"""Configuration loading: YAML defaults + environment overrides.

Precedence (lowest to highest):
    dataclass defaults  <  config.yaml  <  GOLDNEURAL_* environment variables

Environment variables are namespaced per section, e.g.
``GOLDNEURAL_EXCHANGE_SYMBOL`` or ``GOLDNEURAL_LABELS_K_UP``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

ENV_PREFIX = "GOLDNEURAL_"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config file or environment override is malformed."""


def _coerce_scalar(value: Any, default: Any) -> Any:
    """Coerce ``value`` to the type of ``default``."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"cannot parse {value!r} as bool")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, (list, tuple)):
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
        else:
            parts = list(value)
        item_default = default[0] if default else ""
        return [_coerce_scalar(p, item_default) for p in parts]
    return _coerce_scalar(value, default)


def _build(cls: type[T], mapping: Mapping[str, Any], env_section: str) -> T:
    """Build a dataclass from a mapping, rejecting unknown keys.

    Unknown keys are almost always typos, and a silently ignored
    ``k_up: 20`` (instead of 2.0) would quietly change what the model is
    trained to predict. Fail loudly instead.
    """
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(mapping) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        default = f.default
        value = mapping.get(name, default)
        if isinstance(default, (list, tuple)) and isinstance(value, (list, tuple)):
            value = _coerce(list(value), default)
        elif isinstance(value, str) or isinstance(default, str):
            # YAML usually gives us correct types already; coerce strings.
            if isinstance(value, str) and not isinstance(default, str):
                value = _coerce(value, default)
        kwargs[name] = value

    env_var = f"{ENV_PREFIX}{env_section.upper()}_"
    for name in known:
        raw = os.environ.get(env_var + name.upper())
        if raw is not None:
            kwargs[name] = _coerce(raw, kwargs[name])

    # Tuple-typed fields (lookbacks, spans, ...) must stay tuples: a list from
    # YAML would make the config hashable-inconsistent and break equality.
    for name, f in known.items():
        if isinstance(f.default, tuple):
            kwargs[name] = tuple(kwargs[name])
    return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExchangeConfig:
    base_url: str = "https://api.binance.com"
    symbol: str = "PAXGUSDT"
    interval: str = "5m"
    request_timeout_s: float = 20.0
    max_retries: int = 5
    min_request_spacing_s: float = 0.15


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    compression: str = "zstd"


@dataclass(frozen=True)
class FeatureConfig:
    # 1 => features at bar t use data through bar t-1 only.
    feature_lag: int = 1
    return_lookbacks: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    ewma_vol_spans: tuple[int, ...] = (12, 48)
    realized_vol_windows: tuple[int, ...] = (12, 48)
    rsi_period: int = 14
    macd: tuple[int, ...] = (12, 26, 9)
    atr_period: int = 14
    volume_z_window: int = 48
    range_window: int = 48
    ema_distances: tuple[int, ...] = (20, 50)


@dataclass(frozen=True)
class LabelConfig:
    k_up: float = 2.0
    k_down: float = 1.5
    max_holding_bars: int = 12
    atr_period: int = 14
    ambiguity: str = "stop_priority"

    def __post_init__(self) -> None:
        if self.k_up <= 0 or self.k_down <= 0:
            raise ConfigError("k_up and k_down must be positive")
        if self.max_holding_bars < 1:
            raise ConfigError("max_holding_bars must be >= 1")
        if self.ambiguity not in {"stop_priority", "up_priority"}:
            raise ConfigError(
                "ambiguity must be 'stop_priority' or 'up_priority', "
                f"got {self.ambiguity!r}"
            )


@dataclass(frozen=True)
class AppConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)

    @property
    def data_root(self) -> Path:
        return Path(self.data.root)

    @property
    def bars_dir(self) -> Path:
        return self.data_root / "klines" / self.exchange.symbol / self.exchange.interval

    @property
    def index_path(self) -> Path:
        return self.data_root / f"index-{self.exchange.symbol}-{self.exchange.interval}.sqlite"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG_PATHS = ("config.yaml", "config.yml")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from ``path`` (or the default file if present)."""
    raw: dict[str, Any] = {}

    if path is None:
        for candidate in DEFAULT_CONFIG_PATHS:
            if Path(candidate).is_file():
                path = candidate
                break

    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p} must contain a YAML mapping at the top level")
        raw = loaded

    unknown_sections = set(raw) - {"exchange", "data", "features", "labels"}
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {sorted(unknown_sections)}")

    return AppConfig(
        exchange=_build(ExchangeConfig, raw.get("exchange") or {}, "exchange"),
        data=_build(DataConfig, raw.get("data") or {}, "data"),
        features=_build(FeatureConfig, raw.get("features") or {}, "features"),
        labels=_build(LabelConfig, raw.get("labels") or {}, "labels"),
    )
