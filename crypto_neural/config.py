"""Configuration loading: YAML defaults + environment overrides.

Precedence (lowest to highest):
    dataclass defaults  <  config.yaml  <  CRYPTONEURAL_* environment variables

Environment variables are namespaced per section, e.g.
``CRYPTONEURAL_EXCHANGE_SYMBOL`` or ``CRYPTONEURAL_LABELS_K_UP``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

ENV_PREFIX = "CRYPTONEURAL_"

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
    symbol: str = "BTCUSDC"
    interval: str = "5m"
    request_timeout_s: float = 20.0
    max_retries: int = 5
    min_request_spacing_s: float = 0.15


@dataclass(frozen=True)
class DataConfig:
    """Where bars live, and which series the 5-minute model learns from.

    ``training_symbol`` is the price series the dataset, the leakage suite and
    the backtest use. BTCUSDC on Binance starts in 2024 here and was delisted
    from 2022-09 to 2023-03; BTCUSDT is continuous from 2017-08 and trades
    within basis points of it, so it gives the model about four times the
    history (the risk model's ``data_symbol`` exists for the same reason).
    Set it to the traded symbol to learn from the traded pair only.
    """

    root: str = "data"
    compression: str = "zstd"
    training_symbol: str = "BTCUSDT"


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
    # Long-horizon regime features, one window list per family. Empty by
    # default: the 5m pipeline does not use them; the risk model sets them
    # from its own section.
    momentum_windows: tuple[int, ...] = ()
    momentum_z_windows: tuple[int, ...] = ()
    sma_windows: tuple[int, ...] = ()
    drawdown_windows: tuple[int, ...] = ()
    # Two or more: each gets a volatility feature, plus first/last as a ratio.
    rvol_windows: tuple[int, ...] = ()


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


REGIME_FIELDS = (
    "momentum_windows",
    "momentum_z_windows",
    "sma_windows",
    "drawdown_windows",
    "rvol_windows",
)


@dataclass(frozen=True)
class RiskConfig:
    """The crash-risk model: what it predicts, how it is tested, how it acts.

    Bar counts are in bars of ``interval`` (4h by default: 6 bars a day).
    The label is "does price fall at least ``drawdown`` within
    ``horizon_bars``", and the policy is long by default, stepping aside when
    that risk is in the top tail of what the model has seen.

    Why these defaults (docs/risk.md has the numbers): on PAXG and on nine
    years of BTC, direction was not predictable at any horizon tried, and the
    one target that carried signal on both assets was short-horizon crash risk
    from the standard short-horizon features (PAXG 3% in 5 days: AUC 0.66;
    BTC 10% in 5 days: AUC 0.65). The long-horizon regime features made BTC
    predictions worse, so they are off. 10% is PAXG's 3% scaled by the
    BTC/PAXG volatility ratio (3.41), fixed before the BTC test; the 5-day
    horizon was chosen after seeing the BTC target table, on the strength of
    it replicating PAXG. On that target an MLP matched LightGBM (AUC 0.650 vs
    0.652), the windowed transformer did worse (0.600), and the average of
    the MLP and LightGBM did best (0.664), hence ``model = "ensemble"``.
    Changing any of this after seeing the economic result is selecting on
    the test period.

    ``data_symbol`` is the price series the model learns from and is tested
    on. BTCUSDC was delisted from 2022-09 to 2023-03, which leaves too little
    contiguous history to test anything; BTCUSDT is continuous from 2017 and
    trades within basis points of it. Set it to "BTCUSDC" to use only the
    traded pair (history after the gap).
    """

    data_symbol: str = "BTCUSDT"
    interval: str = "4h"
    drawdown: float = 0.10
    horizon_bars: int = 30  # 5 days
    # "lgbm", "mlp" (a small neural network), or "ensemble" (their average).
    model: str = "ensemble"
    mlp_hidden: tuple[int, ...] = (64, 32)
    mlp_dropout: float = 0.3
    mlp_epochs: int = 40
    mlp_patience: int = 6
    # Optional long-horizon regime features, in bars. Off by default: on BTC
    # they made out-of-sample predictions worse (patterns from earlier
    # regimes reversed).
    momentum_windows: tuple[int, ...] = ()
    momentum_z_windows: tuple[int, ...] = ()
    sma_windows: tuple[int, ...] = ()
    drawdown_windows: tuple[int, ...] = ()
    rvol_windows: tuple[int, ...] = ()
    test_bars: int = 540  # 90-day blocks
    min_train_bars: int = 2190  # one year before the first test block
    # Exit when risk exceeds this quantile of the training predictions. Which
    # one is used for a block is chosen from earlier blocks only.
    exit_quantiles: tuple[float, ...] = (0.8, 0.85, 0.9, 0.925, 0.95)
    reentry_ratio: float = 0.9
    min_flat_bars: int = 30
    # Blocks traded as plain buy-and-hold before any rule has a track record.
    warmup_folds: int = 2
    seeds: int = 5
    trend_sma_bars: int = 1200  # 200-day average, for the trend baseline
    fee_bps: float = 10.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.drawdown < 1.0:
            raise ConfigError("risk.drawdown must be within (0, 1)")
        if self.horizon_bars < 1 or self.test_bars < 1 or self.min_train_bars < 1:
            raise ConfigError("risk bar counts must be >= 1")
        for name in REGIME_FIELDS:
            if any(int(window) < 2 for window in getattr(self, name)):
                raise ConfigError(f"risk.{name} needs windows of >= 2 bars")
        if not self.exit_quantiles or not all(
            0.0 < q < 1.0 for q in self.exit_quantiles
        ):
            raise ConfigError("risk.exit_quantiles must be within (0, 1)")
        if not 0.0 < self.reentry_ratio <= 1.0:
            raise ConfigError("risk.reentry_ratio must be within (0, 1]")
        if self.min_flat_bars < 0 or self.warmup_folds < 0 or self.seeds < 1:
            raise ConfigError("risk.min_flat_bars, warmup_folds >= 0; seeds >= 1")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ConfigError("risk costs must be >= 0")
        if self.model not in {"lgbm", "mlp", "ensemble"}:
            raise ConfigError("risk.model must be 'lgbm', 'mlp' or 'ensemble'")
        if not self.mlp_hidden or min(self.mlp_hidden) < 1:
            raise ConfigError("risk.mlp_hidden needs positive layer sizes")
        if not 0.0 <= self.mlp_dropout < 1.0 or self.mlp_epochs < 1 or self.mlp_patience < 1:
            raise ConfigError("risk.mlp_dropout in [0, 1); mlp_epochs, mlp_patience >= 1")


@dataclass(frozen=True)
class ServerConfig:
    """Web UI / API server settings.

    There is no API token. ``allowed_ips`` is a list of addresses or CIDR
    blocks; empty means "any source".
    """

    host: str = "0.0.0.0"
    port: int = 8443
    tls_enabled: bool = True
    cert_path: str = "certs/server.crt"
    key_path: str = "certs/server.key"
    allowed_ips: tuple[str, ...] = ()
    # Run the risk job just after every bar opens and log its reading
    # (server/scheduler.py). Off by default: a few CPU-minutes per bar.
    record_readings: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError("server.port must be within 1..65535")
        for entry in self.allowed_ips:
            import ipaddress

            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError:
                raise ConfigError(f"server.allowed_ips entry {entry!r} is not an IP or CIDR") from None


@dataclass(frozen=True)
class AppConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def data_root(self) -> Path:
        return Path(self.data.root)

    @property
    def bars_dir(self) -> Path:
        return self.data_root / "klines" / self.exchange.symbol / self.exchange.interval

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    @property
    def index_path(self) -> Path:
        return self.data_root / f"index-{self.exchange.symbol}-{self.exchange.interval}.sqlite"

    def training_market(self) -> "AppConfig":
        """This config pointed at the series the 5-minute model learns from
        (``data.training_symbol``): its bar store, index and top-ups."""
        symbol = self.data.training_symbol or self.exchange.symbol
        return replace(self, exchange=replace(self.exchange, symbol=symbol))

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

    unknown_sections = set(raw) - {"exchange", "data", "features", "labels", "risk", "server"}
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {sorted(unknown_sections)}")

    return AppConfig(
        exchange=_build(ExchangeConfig, raw.get("exchange") or {}, "exchange"),
        data=_build(DataConfig, raw.get("data") or {}, "data"),
        features=_build(FeatureConfig, raw.get("features") or {}, "features"),
        labels=_build(LabelConfig, raw.get("labels") or {}, "labels"),
        risk=_build(RiskConfig, raw.get("risk") or {}, "risk"),
        server=_build(ServerConfig, raw.get("server") or {}, "server"),
    )
