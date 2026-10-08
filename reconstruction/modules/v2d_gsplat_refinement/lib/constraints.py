"""Conservative bounds around independent pose initialization, without GT."""
from __future__ import annotations

import math

import torch


class PoseTrustRegion:
    """Project translation/axis-angle blocks into fixed initialization balls.

    Axis-angle parameter distance bounds the physical rotation change. Each
    finger joint is a separate3-vector block. No observations are created.
    These priors preserve input reliability; they do not establish accuracy.
    """

    def __init__(self):
        self.entries = []

    def add(self, name: str, parameter: torch.Tensor, radius: float) -> None:
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError(f"Invalid pose trust radius for {name}: {radius}")
        if radius == 0:
            return
        if parameter.numel() % 3:
            raise ValueError("Pose trust parameters must contain3-vector blocks")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"Nonfinite initialization for {name}")
        self.entries.append((name, parameter, parameter.detach().clone(), radius))

    @torch.no_grad()
    def project(self) -> dict[str, int]:
        clipped = {}
        for name, parameter, initial, radius in self.entries:
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"Nonfinite parameter before pose projection: {name}")
            delta = (parameter - initial).reshape(-1, 3)
            length = delta.norm(dim=-1, keepdim=True)
            scale = (radius / length.clamp_min(1e-12)).clamp_max(1.0)
            parameter.copy_(initial + (delta * scale).reshape_as(parameter))
            clipped[name] = int((length > radius).sum())
        return clipped
