"""
VLM force evaluation module.

Owns all logic for:
  - Kutta-Joukowski force computation from bound vortex panels.
  - Impulse-based force computation (F = -dI/dt).
  - Impulse history differentiation (BDF1, BDF2, central differences).
  - Impulse history shift after particle removal to maintain dI/dt continuity.
  - Comparison between both methods for validation.

Nothing in this module imports from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
orchestrator.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .vlm_solver import ForceConfig, VLMSolver


class VLMForceEvaluator:
    """Static helpers for aerodynamic force evaluation in VLM-VPM coupling.

    All methods are static and receive their required data explicitly from
    the solver (no solver reference is held here).
    """

    # -------------------------------------------------------------------------
    # STATE FACTORY
    # -------------------------------------------------------------------------

    @staticmethod
    def make_impulse_state() -> dict[str, Any]:
        """Return a fresh impulse state dict.  Call once during solver init."""
        return {
            "history": [],
            "time_history": [],
            "force_history": [],
            "removed_accumulated": np.zeros(3),
        }

    # -------------------------------------------------------------------------
    # KUTTA-JOUKOWSKI METHOD
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_kutta_joukowski(
        vlm_solver: VLMSolver | None,
        background_velocity: np.ndarray,
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
            V_ref_mag = float(np.linalg.norm(background_velocity))
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

    # -------------------------------------------------------------------------
    # IMPULSE METHOD — helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def apply_history_shift(impulse_state: dict[str, Any], density: float) -> None:
        """Shift historical impulse values to maintain dI/dt continuity after particle removal.

        Must be called before appending the new impulse value each time step.
        """
        shift_raw: np.ndarray = impulse_state.get("removed_accumulated", np.zeros(3))
        impulse_shift = shift_raw * density
        if float(np.linalg.norm(impulse_shift)) > 1e-15:
            for i in range(len(impulse_state["history"])):
                impulse_state["history"][i] = impulse_state["history"][i] - impulse_shift
            impulse_state["removed_accumulated"] = np.zeros(3)

    @staticmethod
    def differentiate(
        impulse_state: dict[str, Any],
        diff_mode: str,
        effective_order: int,
    ) -> np.ndarray:
        """Compute dI/dt from impulse history using the specified differentiation scheme.

        Schemes:
          CENTRAL  — central differences (requires ≥ 3 points).
          BACKWARD — BDF1 (order=1) or BDF2 (order=2, requires ≥ 3 points).
        """
        n_hist = len(impulse_state["history"])
        if diff_mode == "CENTRAL" and n_hist >= 3:
            dt_total = impulse_state["time_history"][-1] - impulse_state["time_history"][-3]
            return (impulse_state["history"][-1] - impulse_state["history"][-3]) / dt_total
        if effective_order == 1 or n_hist < 3:
            dt = impulse_state["time_history"][-1] - impulse_state["time_history"][-2]
            if dt > 0:
                return (impulse_state["history"][-1] - impulse_state["history"][-2]) / dt
            return np.zeros(3)
        # Second-order backward difference (BDF2) with non-uniform time steps
        dt1 = impulse_state["time_history"][-1] - impulse_state["time_history"][-2]
        dt2 = impulse_state["time_history"][-2] - impulse_state["time_history"][-3]
        a0 = (2 * dt1 + dt2) / (dt1 * (dt1 + dt2))
        a1 = -(dt1 + dt2) / (dt1 * dt2)
        a2 = dt1 / (dt2 * (dt1 + dt2))
        return (
            a0 * impulse_state["history"][-1]
            + a1 * impulse_state["history"][-2]
            + a2 * impulse_state["history"][-3]
        )

    # -------------------------------------------------------------------------
    # IMPULSE METHOD — main
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_impulse(
        impulse_state: dict[str, Any],
        vlm_solver: VLMSolver | None,
        linear_impulse: np.ndarray,
        flow_time: float,
        density: float,
        force_config: ForceConfig,
    ) -> dict[str, np.ndarray | float | str | None]:
        """Compute forces using the impulse-based method F = -dI/dt.

        Uses I = (ρ/2) ∫ x × ω dV; invariant to wake truncation.
        Impulse state (history lists, removed_accumulated) is mutated in-place.
        """
        order: int = force_config.impulse_order
        diff_mode: str = getattr(force_config, "impulse_differentiation", "BACKWARD")

        impulse_wake = linear_impulse * density
        impulse_bound = np.zeros(3)
        if vlm_solver is not None and vlm_solver._solved:
            impulse_bound = vlm_solver.compute_bound_impulse(density)
        impulse_total = impulse_wake + impulse_bound

        VLMForceEvaluator.apply_history_shift(impulse_state, density)

        time_hist = impulse_state["time_history"]
        if len(time_hist) == 0 or flow_time > time_hist[-1]:
            impulse_state["history"].append(impulse_total.copy())
            time_hist.append(flow_time)
        else:
            impulse_state["history"][-1] = impulse_total.copy()

        n_hist = len(impulse_state["history"])
        if n_hist < 2:
            dt = time_hist[0]
            if dt > 1e-12:
                dI_dt = impulse_state["history"][0] / dt
            else:
                return {
                    "method": "IMPULSE",
                    "force": np.zeros(3),
                    "Fx": 0.0,
                    "Fy": 0.0,
                    "Fz": 0.0,
                    "impulse_total": impulse_total,
                    "status": "t=0 (initial state)",
                }
        else:
            dI_dt = VLMForceEvaluator.differentiate(impulse_state, diff_mode, order)

        # F = -dI/dt: upward force on body corresponds to downward fluid impulse change
        force = -dI_dt
        impulse_state["force_history"].append(force.copy())

        # Bound history to the last 10 steps (sufficient for BDF2)
        if len(impulse_state["history"]) > 10:
            impulse_state["history"].pop(0)
            time_hist.pop(0)
        if len(impulse_state["force_history"]) > 20:
            impulse_state["force_history"].pop(0)

        return {
            "method": "IMPULSE",
            "force": force,
            "Fx": float(force[0]),
            "Fy": float(force[1]),
            "Fz": float(force[2]),
            "impulse_total": impulse_total,
            "impulse_wake": impulse_wake,
            "impulse_bound": impulse_bound,
            "order": order,
            "smoothing": None,  # Deactivated: raw physics requirement
            "status": "OK",
        }

    # -------------------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------------------

    @staticmethod
    def compare_methods(
        impulse_state: dict[str, Any],
        vlm_solver: VLMSolver | None,
        linear_impulse: np.ndarray,
        background_velocity: np.ndarray,
        flow_time: float,
        density: float,
        force_config: ForceConfig,
        V_ref_mag: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Compare impulse-based vs Kutta-Joukowski force evaluation."""
        result_impulse = VLMForceEvaluator.compute_impulse(
            impulse_state, vlm_solver, linear_impulse, flow_time, density, force_config
        )
        result_kj = VLMForceEvaluator.compute_kutta_joukowski(
            vlm_solver, background_velocity, density, V_ref_mag
        )
        force_impulse: np.ndarray = result_impulse["force"]  # type: ignore[assignment]
        force_kj: np.ndarray = result_kj["force"]  # type: ignore[assignment]
        difference = float(np.linalg.norm(force_impulse - force_kj))
        imp_norm = float(np.linalg.norm(force_impulse))
        relative_error = 100.0 * difference / imp_norm if imp_norm > 1e-10 else 0.0
        return {
            "force_impulse": force_impulse,
            "force_kj": force_kj,
            "difference": difference,
            "relative_error": relative_error,
        }
