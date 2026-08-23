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
    """Stateless helper for computing panel forces via multiple methods."""

    @staticmethod
    def compute_bernoulli(
        surface_velocity: np.ndarray,
        freestream_speed: float,
        area: np.ndarray,
        normal: np.ndarray,
        density: float,
    ) -> np.ndarray:
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
