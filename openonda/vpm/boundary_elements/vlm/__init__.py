"""Vortex-lattice-method (VLM) subpackage: surface geometry and the VLM solver."""

from .config import ForceConfig, VLMConfig, VLMMeshConfig, VLMSurface
from .geometry.aircraft import Aircraft, Wing, WingSegment
from .solver.lattice import VLMLattice
from .solver.vlm_solver import VLMSolver

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
