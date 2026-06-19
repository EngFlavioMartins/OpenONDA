"""
Panel force evaluation module — stateless helpers for force computation.

Mirrors ``vlm/solver/forces.py`` conventions.  Extracts force kernel logic
from ``panel_solver.py`` and ``influence.py`` into a dedicated module for
clarity and testability.

Three methods:
  - Bernoulli:     F = 0.5·ρ·(V∞² − V²)·A·n
  - Kutta-Joukowski: F = ρ·Γ·(V × l_bound)
  - Impulse:       F = −ρ·d/dt(Σ μ·A·n)

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np


class PanelForceEvaluator:
    """Stateless helper for computing panel forces via multiple methods."""

    @staticmethod
    def make_impulse_state() -> dict:
        return {
            "impulse_prev": np.zeros(3, dtype=float),
            "impulse_prev2": np.zeros(3, dtype=float),
        }

    @staticmethod
    def compute_bernoulli(
        V_surface: np.ndarray,
        V_inf_mag: float,
        areas: np.ndarray,
        normals: np.ndarray,
        density: float,
    ) -> np.ndarray:
        q_inf = 0.5 * density * V_inf_mag * V_inf_mag
        V_sq = np.sum(V_surface**2, axis=1)
        cp = 1.0 - V_sq / (V_inf_mag * V_inf_mag + 1e-30)
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

    @staticmethod
    def compute_impulse(
        impulse_state: dict,
        strengths: np.ndarray,
        strengths_old: np.ndarray,
        areas: np.ndarray,
        normals: np.ndarray,
        density: float,
        dt: float,
        step: int,
    ) -> np.ndarray:
        n = len(strengths)
        impulse_curr = np.sum(strengths[:, None] * areas[:, None] * normals, axis=0)

        if step == 0:
            impulse_state["impulse_prev"] = impulse_curr.copy()
            impulse_state["impulse_prev2"] = impulse_curr.copy()
            return np.zeros((n, 3), dtype=float)

        impulse_prev = impulse_state["impulse_prev"]
        impulse_prev2 = impulse_state["impulse_prev2"]

        if step == 1:
            dI_dt = (impulse_curr - impulse_prev) / max(dt, 1e-16)
        else:
            dI_dt = (1.5 * impulse_curr - 2.0 * impulse_prev + 0.5 * impulse_prev2) / max(dt, 1e-16)

        impulse_state["impulse_prev2"] = impulse_prev.copy()
        impulse_state["impulse_prev"] = impulse_curr.copy()

        total_force = -dI_dt * density
        # Distribute proportionally to panel area
        total_area = np.sum(areas)
        if total_area > 0:
            forces = total_force[None, :] * (areas[:, None] / total_area)
        else:
            forces = np.zeros((n, 3), dtype=float)
        return forces
