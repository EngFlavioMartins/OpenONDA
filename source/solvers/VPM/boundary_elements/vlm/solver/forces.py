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
        V_ref_mag: float | None,
    ) -> dict[str, np.ndarray | float | str]:
        """Compute forces using the conventional Kutta-Joukowski theorem.

        Integrates pressure forces over bound vortex panels: F = ρ Γ × V_local.
        """
        if vlm_solver is None or not vlm_solver._solved:
            return {
                "method": "KUTTA_JOUKOWSKI",
                "force": np.zeros(3),
                "Fx": 0.0,
                "Fy": 0.0,
                "Fz": 0.0,
                "error": "No VLM solver or not solved",
            }
        if V_ref_mag is None:
            V_ref_mag = float(np.linalg.norm(freestream_velocity))
        try:
            forces_dict = vlm_solver.compute_forces(density, V_ref_mag)
            force_vector = np.array([forces_dict["Fx"], forces_dict["Fy"], forces_dict["Fz"]])
            return {
                "method": "KUTTA_JOUKOWSKI",
                "force": force_vector,
                "Fx": forces_dict["Fx"],
                "Fy": forces_dict["Fy"],
                "Fz": forces_dict["Fz"],
                "CL": forces_dict.get("CL", 0.0),
                "CD": forces_dict.get("CD", 0.0),
                **forces_dict,
            }
        except Exception as exc:
            return {
                "method": "KUTTA_JOUKOWSKI",
                "force": np.zeros(3),
                "Fx": 0.0,
                "Fy": 0.0,
                "Fz": 0.0,
                "error": str(exc),
            }
