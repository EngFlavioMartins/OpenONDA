"""
VLM force evaluation module.

Owns all logic for:
  - Kutta-Joukowski force computation from bound vortex panels.

Nothing in this module imports from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
orchestrator.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .vlm_solver import VLMSolver


class VLMForceEvaluator:
    """Static helpers for aerodynamic force evaluation in VLM-VPM coupling.

    All methods are static and receive their required data explicitly from
    the solver (no solver reference is held here).
    """

    # KUTTA-JOUKOWSKI METHOD

    @staticmethod
    def compute_kutta_joukowski(
        vlm_solver: VLMSolver | None,
        freestream_velocity: np.ndarray,
        density: float,
        reference_speed: float | None,
    ) -> dict[str, np.ndarray | float | str]:
        """Return bound-leg KJ loads plus the configured unsteady pressure term."""
        if vlm_solver is None or not vlm_solver._solved:
            raise RuntimeError("Force evaluation requires a solved VLM lattice")
        reference = vlm_solver._resolve_reference_velocity(None, vlm_solver.lattice.n_panels)
        if reference_speed is not None:
            reference = reference / np.linalg.norm(reference) * reference_speed
        forces = vlm_solver.compute_forces(density, reference_velocity=reference)
        return {
            "method": "KUTTA_JOUKOWSKI",
            "unsteady_pressure": vlm_solver.force.unsteady,
            "force": np.array([forces["force_x"], forces["force_y"], forces["force_z"]]),
            **forces,
        }
