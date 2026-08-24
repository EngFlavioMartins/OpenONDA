"""Viscous-diffusion schemes for the VPM solver.

One module per scheme:

- ``core_spreading.py``  Core Spreading Method (CSM), analytic Gaussian core growth.
- ``random_walk.py``     Random Walk Method (RWM).
- ``grid.py``            The stateful grid mixin shared by DVH and GBD (grid
                         allocation, body masking, M4' scattering, Laplacian
                         stepping, particle regeneration, transfers).
- ``schemes.py``         DiffusionPhysics, the handler that composes them.

DVH and GBD are driven from ``grid.py`` (``grid_based_diffusion`` /
``gbd_diffusion``) because they share one stateful grid; their dispatch from
the time-step algorithm lives in ``core/evolution.py``.
"""

from .core_spreading import apply_core_spreading
from .grid import (
    _DVH_BETA,
    _GRID_TRANSFER_CHUNK,
    _M4_SCATTER_BATCH_SIZE,
    _REGEN_RADIUS_RATIO,
    _GridDiffusionMixin,
)
from .random_walk import apply_random_walk
from .schemes import DiffusionPhysics

__all__ = [
    "DiffusionPhysics",
    "_DVH_BETA",
    "_GRID_TRANSFER_CHUNK",
    "_GridDiffusionMixin",
    "_M4_SCATTER_BATCH_SIZE",
    "_REGEN_RADIUS_RATIO",
    "apply_core_spreading",
    "apply_random_walk",
]
