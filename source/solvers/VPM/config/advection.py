"""Advection configuration for the VPM solver."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AdvectionConfig:
    """Configure integration of the particle-advection equation ``dx/dt = u``.

    ``RK3`` is the default production scheme. ``NONE`` freezes particle
    positions and is useful for stationary diffusion benchmarks.
    """

    scheme: Literal["NONE", "EULER", "RK2", "RK3", "RK4"] = "RK3"
    """Time-integration scheme for particle position."""
