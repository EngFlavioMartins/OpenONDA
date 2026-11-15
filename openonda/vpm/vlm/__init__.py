"""Vortex-lattice-method (VLM) package: surface geometry and the VLM solver."""

from .config import ForceConfig, VLMConfig, VLMMeshConfig, VLMSurface
from .geometry import Aircraft, Wing, WingSegment
from .lattice import VLMLattice
from .solver import VLMSolver

__all__ = [
    "Aircraft",
    "ForceConfig",
    "VLMConfig",
    "VLMLattice",
    "VLMMeshConfig",
    "VLMSolver",
    "VLMSurface",
    "Wing",
    "WingSegment",
]
