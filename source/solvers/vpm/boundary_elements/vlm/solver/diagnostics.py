"""
VLM diagnostics module — recording and CSV export of force/vector-strength history.

Owns all logic for:
  - Appending per-step VLM scalars to the solver diagnostics history dict.
  - Writing vlm_forces.csv.
  - Appending time / observed_time_step_size history entries.

Nothing in this module should import from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
orchestrator.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np

from ....io.sampling import resolve_samples_dir


class VLMDiagnostics:
    """Static helpers for recording VLM diagnostics and writing CSV output."""

    @staticmethod
    def record_time(diagnostics_history: dict, time: float, observed_time_step_size: float) -> None:
        """Append *time* and *observed_time_step_size* to diagnostics history.

        Parameters
        ----------
        diagnostics_history:
            The solver's ``_diagnostics_history`` dict (mutated in-place).
        time:
            Current simulation time [s].
        observed_time_step_size:
            Wall-clock or physical output interval observed for this step [s].
        """
        if "time" not in diagnostics_history:
            return
        ft_hist = diagnostics_history["time"]
        if len(ft_hist) > 0 and ft_hist[-1] == time:
            return
        ft_hist.append(time)
        if len(diagnostics_history["observed_time_step_size"]) < len(ft_hist):
            diagnostics_history["observed_time_step_size"].append(float(observed_time_step_size))

    @staticmethod
    def record_vlm_diagnostics(
        vlm_solver,
        particles,
        particle_vortex_strength: np.ndarray,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Record VLM force and vector-strength scalars and, when due, flush to CSV.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        particles:
            Current ``Particles`` container (for particle count).
        particle_vortex_strength:
            Vortex-strength array (N, 3) for the current particles [m³/s].
        diagnostics_history:
            Solver's ``_diagnostics_history`` dict (mutated in-place).
        step:
            Current integer step counter.
        time:
            Current simulation time [s].
        case_dir:
            Solver backup/output root directory (CSV is written under
            ``<case_dir>/samples/vlm_forces.csv``).
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_last_forces"):
            return
        try:
            forces = vlm_solver._last_forces
            n_panels = vlm_solver.lattice.n_panels
            is_tenp = vlm_solver.lattice.is_trailing_edge.to_numpy()[:n_panels]
            circulation_cum = vlm_solver.lattice.cumulative_circulation.to_numpy()[:n_panels]

            # VPM particles store vector vortex strength
            # alpha = integral(omega dV), with units m^3/s.  A plain sum with
            # scalar VLM circulation is both dimensionally incompatible and
            # cancels between mirrored halves.  Convert the TE bound vortices
            # to the same vector-strength measure using their oriented legs.
            vortex_point_position = vlm_solver.lattice.vortex_point_position.to_numpy()[:n_panels]
            te_mask = is_tenp == 1
            bound_legs = vortex_point_position[te_mask, 2] - vortex_point_position[te_mask, 1]
            bound_vortex_strength = np.sum(circulation_cum[te_mask, None] * bound_legs, axis=0)
            bound_vortex_strength_y = float(bound_vortex_strength[1])

            n_p = particles.n_particles_total
            wake_vortex_strength_y = float(particle_vortex_strength[:, 1].sum()) if n_p > 0 else 0.0

            lespnp = vlm_solver.lattice.leading_edge_suction_parameter.to_numpy()[:n_panels]
            max_leading_edge_suction_parameter = float(np.max(lespnp)) if n_panels > 0 else 0.0

            diagnostics_history["vlm_lift_coefficient"].append(float(forces["lift_coefficient"]))
            diagnostics_history["vlm_drag_coefficient"].append(float(forces["drag_coefficient"]))
            diagnostics_history["vlm_bound_vortex_strength_y"].append(bound_vortex_strength_y)
            diagnostics_history["vlm_wake_vortex_strength_y"].append(wake_vortex_strength_y)
            diagnostics_history["vlm_max_leading_edge_suction_parameter"].append(
                max_leading_edge_suction_parameter
            )
            diagnostics_history["vlm_n_particles_total"].append(float(n_p))

            freq = max(1, int(getattr(vlm_solver, "logging_interval_steps", 1)))
            if step % freq == 0:
                VLMDiagnostics.export_forces_csv(
                    vlm_solver,
                    forces,
                    bound_vortex_strength_y,
                    wake_vortex_strength_y,
                    max_leading_edge_suction_parameter,
                    n_p,
                    time,
                    step,
                    case_dir,
                    sample_directory,
                )
        except Exception as exc:
            print(f"(Warning) Failed to record VLM diagnostics: {exc}")

    # CSV export

    @staticmethod
    def export_forces_csv(
        vlm_solver,
        forces: dict,
        bound_vortex_strength: float,
        wake_vortex_strength: float,
        max_leading_edge_suction_parameter: float,
        n_p: int,
        time: float,
        step: int,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Append one row to ``<case_dir>/samples/vlm_forces.csv``.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        forces:
            Force dict returned by ``vlm_solver.compute_forces()``.
        bound_vortex_strength, wake_vortex_strength:
            Pre-computed bound and wake y-components of vector strength [m³/s].
        max_leading_edge_suction_parameter:
            Maximum Leading Edge Suction Parameter for this step.
        n_p:
            Number of VPM particles at this step.
        time:
            Current simulation time [s].
        step:
            Integer step counter.
        case_dir:
            Output root; CSV is written under ``<case_dir>/samples/``.
        """
        import pandas as pd

        samples_dir = resolve_samples_dir(case_dir, sample_directory)
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_forces.csv"

        row = {
            "time": time,
            "step": step,
            "lift_coefficient": forces.get("lift_coefficient", 0.0),
            "drag_coefficient": forces.get("drag_coefficient", 0.0),
            "side_force_coefficient": forces.get("side_force_coefficient", 0.0),
            "force_x": forces.get("force_x", 0.0),
            "force_y": forces.get("force_y", 0.0),
            "force_z": forces.get("force_z", 0.0),
            "moment_x": forces.get("moment_x", 0.0),
            "moment_y": forces.get("moment_y", 0.0),
            "moment_z": forces.get("moment_z", 0.0),
            "lift": forces.get("lift", 0.0),
            "drag": forces.get("drag", 0.0),
            "dynamic_pressure": forces.get("dynamic_pressure", 0.0),
            "reference_area": forces.get("reference_area", 0.0),
            "rolling_moment_coefficient": forces.get("rolling_moment_coefficient", 0.0),
            "pitching_moment_coefficient": forces.get("pitching_moment_coefficient", 0.0),
            "yawing_moment_coefficient": forces.get("yawing_moment_coefficient", 0.0),
            "rolling_moment_coefficient_quarter_chord": forces.get(
                "rolling_moment_coefficient_quarter_chord", 0.0
            ),
            "pitching_moment_coefficient_quarter_chord": forces.get(
                "pitching_moment_coefficient_quarter_chord", 0.0
            ),
            "yawing_moment_coefficient_quarter_chord": forces.get(
                "yawing_moment_coefficient_quarter_chord", 0.0
            ),
            "bound_vortex_strength_y": bound_vortex_strength,
            "wake_vortex_strength_y": wake_vortex_strength,
            "max_leading_edge_suction_parameter": max_leading_edge_suction_parameter,
            "n_particles_total": n_p,
        }
        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)
