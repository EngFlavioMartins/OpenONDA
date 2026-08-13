"""Advection configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AdvectionConfig:
    """
    Configuration for the advection substep: dx/dt = u.

    Advection advances each particle's position by solving the material-derivative
    equation dx/dt = u( x(t), t ) over the configured time-integration scheme.
    The substep is nested inside the solver's macro time-step (the DVH-pinned
    dt for DVH runs).

    Available schemes
    -----------------
    ``NONE``
        Freeze particle positions.  Useful for stationary-flow viscous tests
        where particle motion is undesirable.
    ``EULER``
        Forward / explicit Euler — 1st order.  Lowest accuracy per step;
        generally not recommended for production runs.
    ``RK2``
        Heun's method — 2nd-order Runge–Kutta.  Good balance of accuracy
        and cost for moderate advection-dominated applications.
    ``RK3``
        Strong-Stability-Preserving Runge–Kutta — 3rd order (Gottlieb, Shu &
        Tadmor 2001).  Default for production VPM runs.
    ``RK4``
        Classical 4th-order Runge–Kutta.  Highest per-step accuracy but
        4 evaluations per step versus 2 for RK2.

    Examples
    --------
    .. code-block:: python

        # Default (RK3)
        adv = AdvectionConfig()

        # 4th-order advection
        adv = AdvectionConfig(scheme="RK4")

        # Freeze particles (stationary-flow test)
        adv = AdvectionConfig(scheme="NONE")

    References
    ----------
    - Gottlieb, Shu & Tadmor (2001) SIAM J. Numer. Anal. 39(5), 1984–2012.
    """

    scheme: Literal["NONE", "EULER", "RK2", "RK3", "RK4"] = "RK3"
    """Time integration scheme for the advection substep (dx/dt = u).

    Options:
      - 'NONE':  freeze particle positions (useful for stationary-flow viscous tests)
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
      - 'RK4':   classical 4th-order Runge–Kutta

    Advection advances the configured scheme over the macro time-step set by the
    solver (the DVH-pinned dt for DVH runs)."""


# =========================================================
# VISCOUS CONFIGURATION
# =========================================================
