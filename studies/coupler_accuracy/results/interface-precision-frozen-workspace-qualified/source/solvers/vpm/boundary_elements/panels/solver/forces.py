"""
Panel force evaluation module — stateless helpers for force computation.

Mirrors ``vlm/solver/forces.py`` conventions.  Extracts force kernel logic
from ``panel_solver.py`` and ``influence.py`` into a dedicated module for
clarity and testability.

Two methods:
  - Bernoulli:     F = 0.5·ρ·(V∞² − V²)·A·n
  - Kutta-Joukowski: F = ρ·Γ·(V × l_bound)

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np


class PanelForceEvaluator:
    """Stateless NumPy helpers for panel pressure and circulation forces.

    All methods operate on active panels only.  Inputs are copied or read but
    never modified; returned forces have shape ``(N, 3)`` and SI units N.
    """

    @staticmethod
    def compute_bernoulli(
        surface_velocity: np.ndarray,
        freestream_speed: float,
        area: np.ndarray,
        normal: np.ndarray,
        density: float,
    ) -> np.ndarray:
        """Evaluate pressure traction from the steady Bernoulli relation.

        Parameters
        ----------
        surface_velocity : ndarray, shape (N, 3)
            Local fluid velocity at panel centres, in m/s.
        freestream_speed : float
            Reference speed in m/s.  It is used to form dynamic pressure and
            the pressure coefficient; it must be non-zero in physical use.
        area : ndarray, shape (N,)
            Panel areas in m².
        normal : ndarray, shape (N, 3)
            Outward unit normals.
        density : float
            Fluid density in kg/m³.

        Returns
        -------
        ndarray, shape (N, 3)
            Pressure force on each panel in N, with the sign set by
            ``normal``.

        Notes
        -----
        The implementation uses ``C_p = 1 - |V|²/U∞²`` and
        ``F_i = q∞ C_p A_i n_i``.  It does not include unsteady potential,
        viscous shear, or wake-history terms.
        """
        dynamic_pressure = 0.5 * density * freestream_speed * freestream_speed
        surface_speed_squared = np.sum(surface_velocity**2, axis=1)
        pressure_coefficient = 1.0 - surface_speed_squared / (
            freestream_speed * freestream_speed + 1e-30
        )
        forces = dynamic_pressure * pressure_coefficient[:, None] * area[:, None] * normal
        return forces

    @staticmethod
    def compute_kutta_joukowski(
        doublet_strength: np.ndarray,
        surface_velocity: np.ndarray,
        vertex_position: np.ndarray,
        density: float,
    ) -> np.ndarray:
        """Evaluate a simplified per-panel Kutta--Joukowski force.

        Parameters
        ----------
        doublet_strength : ndarray, shape (N,)
            Per-panel doublet/circulation strength in m²/s.
        surface_velocity : ndarray, shape (N, 3)
            Local relative velocity in m/s.
        vertex_position : ndarray, shape (N, 3, 3)
            Triangle vertices in metres.  The first edge is used as the bound
            vortex leg in this simplified evaluator.
        density : float
            Fluid density in kg/m³.

        Returns
        -------
        ndarray, shape (N, 3)
            Force vectors in N.

        Notes
        -----
        Each force is ``rho * Gamma * (V x l_bound)``.  This is intended for
        thin lifting surfaces; it is not an unsteady added-mass or thick-body
        pressure model.
        """
        n = len(doublet_strength)
        forces = np.zeros((n, 3), dtype=float)
        for i in range(n):
            v0, v1, _ = vertex_position[i, 0], vertex_position[i, 1], vertex_position[i, 2]
            # Bound vortex vector along the first edge (simplified KJ)
            bound_vortex_leg = v1 - v0
            relative_velocity = surface_velocity[i]
            vortex_strength = doublet_strength[i]
            forces[i] = density * vortex_strength * np.cross(relative_velocity, bound_vortex_leg)
        return forces
