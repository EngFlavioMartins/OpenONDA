"""Viscous-diffusion schemes for the VPM solver.

One module per scheme or responsibility:

- ``core_spreading.py``     Core Spreading Method (CSM), analytic Gaussian core growth.
- ``random_walk.py``        Random Walk Method (RWM).
- ``lattice.py``            The stateful grid shared by DVH and GBD: allocation, the fixed
                            lattice origin, active box, Laplacian stepping and host transfers.
- ``wall.py``               The solid body on that lattice: body mask, wall links, slip-slab
                            folding and the M4' corrections of wall-adjacent particles.
- ``regeneration.py``       Threshold selection and construction of the regenerated particles.
- ``moment_correction.py``  Moment-preserving removal of weak grid nodes.
- ``dvh.py``, ``gbd.py``    The DVH and GBD steps (``dvh_diffusion``, ``gbd_diffusion``).

:class:`~openonda.vpm.physics.engine.PhysicsEngine` composes the first two and owns the
lattice; the dispatch of DVH and GBD from the time-step algorithm lives in
``core/evolution.py``.
"""

from .core_spreading import apply_core_spreading
from .random_walk import apply_random_walk

__all__ = [
    "apply_core_spreading",
    "apply_random_walk",
]
