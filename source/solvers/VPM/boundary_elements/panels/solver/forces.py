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
        V_surface: np.ndarray,
        freestream_velocity_mag: float,
        areas: np.ndarray,
        normals: np.ndarray,
        density: float,
    ) -> np.ndarray:
        q_inf = 0.5 * density * freestream_velocity_mag * freestream_velocity_mag
        V_sq = np.sum(V_surface**2, axis=1)
        cp = 1.0 - V_sq / (freestream_velocity_mag * freestream_velocity_mag + 1e-30)
        forces = q_inf * cp[:, None] * areas[:, None] * normals
        return forces

    @staticmethod
    def compute_kutta_joukowski(
        strengths: np.ndarray,
        V_surface: np.ndarray,
        vertices: np.ndarray,
        density: float,
    ) -> np.ndarray:
        n = len(strengths)
        forces = np.zeros((n, 3), dtype=float)
        for i in range(n):
            v0, v1, _ = vertices[i, 0], vertices[i, 1], vertices[i, 2]
            # Bound vortex vector along the first edge (simplified KJ)
            l_bound = v1 - v0
            V_rel = V_surface[i]
            gamma = strengths[i]
            forces[i] = density * gamma * np.cross(V_rel, l_bound)
        return forces
