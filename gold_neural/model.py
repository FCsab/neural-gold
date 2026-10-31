"""Model definition and the mandatory GPU probe.

The probe is deliberately not ``torch.cuda.is_available()``: a ROCm build can
report a device and then page-fault on the first real kernel. A tiny
forward/backward/synchronize is the only reliable signal, and it costs
milliseconds at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

log = logging.getLogger(__name__)

CLASSES = (-1, 0, 1)
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


class WindowTransformer(nn.Module):
    """Compact encoder over a window of feature rows.

    ~200k parameters at the default sizes: small enough to train on a CPU in
    minutes, large enough to model cross-feature interactions across the
    window. Everything is in one file so swapping in a CNN hybrid stays local.
    """

    def __init__(
        self,
        n_features: int,
        window: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.window = window
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.positional, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, window, n_features) -> logits (batch, 3)."""
        hidden = self.input_proj(features) + self.positional[:, : features.size(1)]
        hidden = self.encoder(hidden)
        return self.head(self.norm(hidden[:, -1]))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class ProbeResult:
    device: str
    note: str

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"


def probe_device(preference: str = "auto") -> ProbeResult:
    """Pick a device by actually running a forward/backward pass on it."""
    preference = (preference or "auto").lower()

    if preference == "cpu":
        return ProbeResult("cpu", "cpu requested by configuration")
    if not torch.cuda.is_available():
        return ProbeResult(
            "cpu",
            "torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, build may be CPU-only)",
        )

    try:
        probe = nn.Linear(64, 64).to("cuda")
        inputs = torch.randn(64, 64, device="cuda")
        probe(inputs).sum().backward()
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda")
        name = torch.cuda.get_device_name(0)
        torch.cuda.empty_cache()
        return ProbeResult("cuda", f"{name} passed a forward/backward probe")
    except Exception as exc:  # noqa: BLE001 - any failure means "do not trust it"
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass
        return ProbeResult("cpu", f"GPU probe failed: {type(exc).__name__}: {exc}")
