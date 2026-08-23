"""
Panel loading-distribution module — per-panel force extraction for unstructured meshes.

Mirrors ``vlm/solver/loading_distribution.py`` conventions.
Since panel methods operate on unstructured triangular meshes (not structured
wing segments), the "spanwise station" concept does not apply directly.
Instead, this module exports the full per-panel dataset (position, normal,
doublet strength, pressure coefficient, panel force) for external post-processing.

Output CSVs:
  <case_dir>/samples/panel_distribution_<surface>.csv

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np

from ....io.sampling import resolve_samples_dir


class PanelLoadingDistribution:
    """Static helpers for exporting per-panel loading distributions."""

    @staticmethod
    def record_loading_distributions(
        panel_solver,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
    ) -> None:
        if panel_solver is None or panel_solver.lattice is None:
            return
        freq = max(1, int(getattr(panel_solver, "logging_interval_steps", 1)))
        if step % freq != 0:
            return

        n = panel_solver.lattice.n_panels
        if n == 0:
            return

        try:
            position = panel_solver.lattice.panel_centre.to_numpy()[:n]
            normal = panel_solver.lattice.normal.to_numpy()[:n]
            doublet_strength = panel_solver.lattice.doublet_strength.to_numpy()[:n]
            area = panel_solver.lattice.area.to_numpy()[:n]
            group_id = panel_solver.lattice.group_id.to_numpy()[:n]
            pressure_coefficient = panel_solver.lattice.pressure_coefficient.to_numpy()[:n]
            panel_force = (
                panel_solver.panel_force.to_numpy()[:n]
                if panel_solver.panel_force is not None
                else np.zeros((n, 3))
            )

            samples_dir = resolve_samples_dir(case_dir)
            samples_dir.mkdir(parents=True, exist_ok=True)
            csv_path = samples_dir / f"panel_distribution_step{step:06d}.csv"

            with open(csv_path, "w") as f:
                f.write(
                    "time,step,position_x,position_y,position_z,"
                    "normal_x,normal_y,normal_z,doublet_strength,area,"
                    "pressure_coefficient,panel_force_x,panel_force_y,panel_force_z,group_id\n"
                )
                for i in range(n):
                    panel_position = position[i]
                    panel_normal = normal[i]
                    f.write(
                        f"{time},{step},"
                        f"{panel_position[0]:.10e},{panel_position[1]:.10e},"
                        f"{panel_position[2]:.10e},"
                        f"{panel_normal[0]:.10e},{panel_normal[1]:.10e},"
                        f"{panel_normal[2]:.10e},"
                        f"{doublet_strength[i]:.10e},{area[i]:.10e},"
                        f"{pressure_coefficient[i]:.10e},"
                        f"{panel_force[i][0]:.10e},{panel_force[i][1]:.10e},"
                        f"{panel_force[i][2]:.10e},{int(group_id[i])}\n"
                    )
        except Exception as exc:
            print(f"(Warning) Failed to record panel loading distribution: {exc}")
