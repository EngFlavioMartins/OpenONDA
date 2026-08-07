"""
VLM diagnostics module — recording and CSV export of VLM force/circulation history.

Owns all logic for:
  - Appending per-step VLM scalars to the solver diagnostics history dict.
  - Writing vlm_forces.csv.
  - Appending flow_time / observed_dt history entries.

Nothing in this module should import from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
orchestrator.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np

from ....utils.field_samplers import resolve_samples_dir


class VLMDiagnostics:
    """Static helpers for recording VLM diagnostics and writing CSV output."""

    @staticmethod
    def record_flow_time(diagnostics_history: dict, flow_time: float, observed_dt: float) -> None:
        """Append *flow_time* and *observed_dt* to diagnostics history (no duplicates).

        Parameters
        ----------
        diagnostics_history:
            The solver's ``_diagnostics_history`` dict (mutated in-place).
        flow_time:
            Current simulation time [s].
        observed_dt:
            Wall-clock or physical dt observed for this step [s].
        """
        if "flow_time" not in diagnostics_history:
            return
        ft_hist = diagnostics_history["flow_time"]
        if len(ft_hist) > 0 and ft_hist[-1] == flow_time:
            return
        ft_hist.append(flow_time)
        if len(diagnostics_history["observed_dt"]) < len(ft_hist):
            diagnostics_history["observed_dt"].append(float(observed_dt))

    @staticmethod
    def record_vlm_diagnostics(
        vlm_solver,
        particles,
        particles_strengths: np.ndarray,
        diagnostics_history: dict,
        time_step: int,
        flow_time: float,
        backup_directory: str,
    ) -> None:
        """Record VLM force and circulation scalars and (if due) flush to CSV.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        particles:
            Current ``Particles`` container (for particle count).
        particles_strengths:
            Circulation array (N, 3) for the current particle set.
        diagnostics_history:
            Solver's ``_diagnostics_history`` dict (mutated in-place).
        time_step:
            Current integer step counter.
        flow_time:
            Current simulation time [s].
        backup_directory:
            Solver backup/output root directory (CSV is written under
            ``<backup_directory>/samples/vlm_forces.csv``).
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_last_forces"):
            return
        try:
            forces = vlm_solver._last_forces
            n_panels = vlm_solver.lattice.num_panels
            is_tenp = vlm_solver.lattice.is_TE_panel.to_numpy()[:n_panels]
            gamma_cum = vlm_solver.lattice.cumulative_circulation.to_numpy()[:n_panels]

            # VPM particle ``circulation`` stores vector vortex strength
            # alpha = integral(omega dV), with units m^3/s.  A plain sum of
            # scalar VLM circulation is both dimensionally incompatible and
            # cancels between mirrored halves.  Convert the TE bound vortices
            # to the same vector-strength measure using their oriented legs.
            vortex_points = vlm_solver.lattice.vortex_points.to_numpy()[:n_panels]
            te_mask = is_tenp == 1
            bound_legs = vortex_points[te_mask, 2] - vortex_points[te_mask, 1]
            gamma_bound_vec = np.sum(gamma_cum[te_mask, None] * bound_legs, axis=0)
            gamma_bound_y = float(gamma_bound_vec[1])

            n_p = particles.number_of_particles
            gamma_wake_y = float(particles_strengths[:, 1].sum()) if n_p > 0 else 0.0

            lespnp = vlm_solver.lattice.lesp.to_numpy()[:n_panels]
            lesp_max = float(np.max(lespnp)) if n_panels > 0 else 0.0

            diagnostics_history["vlm_CL"].append(float(forces["CL"]))
            diagnostics_history["vlm_CD"].append(float(forces["CD"]))
            diagnostics_history["vlm_gamma_bound_y"].append(gamma_bound_y)
            diagnostics_history["vlm_gamma_wake_y"].append(gamma_wake_y)
            diagnostics_history["vlm_lesp_max"].append(lesp_max)
            diagnostics_history["vlm_n_particles"].append(float(n_p))

            freq = max(1, int(getattr(vlm_solver, "logging_frequency", 1)))
            if time_step % freq == 0:
                VLMDiagnostics.export_forces_csv(
                    vlm_solver,
                    forces,
                    gamma_bound_y,
                    gamma_wake_y,
                    lesp_max,
                    n_p,
                    flow_time,
                    time_step,
                    backup_directory,
                )
        except Exception as exc:
            print(f"(Warning) Failed to record VLM diagnostics: {exc}")

    # CSV export

    @staticmethod
    def export_forces_csv(
        vlm_solver,
        forces: dict,
        gamma_bound: float,
        gamma_wake: float,
        lesp_max: float,
        n_p: int,
        flow_time: float,
        time_step: int,
        backup_directory: str,
    ) -> None:
        """Append one row to ``<backup_directory>/samples/vlm_forces.csv``.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        forces:
            Force dict returned by ``vlm_solver.compute_forces()``.
        gamma_bound, gamma_wake:
            Pre-computed bound / wake y-circulation scalars.
        lesp_max:
            Maximum Leading Edge Suction Parameter for this step.
        n_p:
            Number of VPM particles at this step.
        flow_time:
            Current simulation time [s].
        time_step:
            Integer step counter.
        backup_directory:
            Output root; CSV is written under ``<backup_directory>/samples/``.
        """
        import pandas as pd

        samples_dir = resolve_samples_dir(backup_directory)
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_forces.csv"

        row = {
            "time": flow_time,
            "step": time_step,
            "CL": forces.get("CL", 0.0),
            "CD": forces.get("CD", 0.0),
            "CC": forces.get("CC", 0.0),
            "Fx": forces.get("Fx", 0.0),
            "Fy": forces.get("Fy", 0.0),
            "Fz": forces.get("Fz", 0.0),
            "Mx": forces.get("Mx", 0.0),
            "My": forces.get("My", 0.0),
            "Mz": forces.get("Mz", 0.0),
            "L": forces.get("L", 0.0),
            "D": forces.get("D", 0.0),
            "q": forces.get("q", 0.0),
            "S_ref": forces.get("S_ref", 0.0),
            "Cl": forces.get("Cl", 0.0),
            "Cm": forces.get("Cm", 0.0),
            "Cn": forces.get("Cn", 0.0),
            "Cl_c4": forces.get("Cl_c4", 0.0),
            "Cm_c4": forces.get("Cm_c4", 0.0),
            "Cn_c4": forces.get("Cn_c4", 0.0),
            "gamma_bound_y": gamma_bound,
            "gamma_wake_y": gamma_wake,
            "lesp_max": lesp_max,
            "n_particles": n_p,
        }

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)
