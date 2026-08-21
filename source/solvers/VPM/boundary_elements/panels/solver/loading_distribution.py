"""
Panel loading-distribution module — per-panel force extraction for unstructured meshes.

Mirrors ``vlm/solver/loading_distribution.py`` conventions.
Since panel methods operate on unstructured triangular meshes (not structured
wing segments), the "spanwise station" concept does not apply directly.
Instead, this module exports the full per-panel dataset (position, normal,
doublet strength, Cp, panel force) for external post-processing.

Output CSVs:
  <checkpoint_dir>/samples/panel_distribution_<surface>.csv

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
        checkpoint_directory: str,
    ) -> None:
        if panel_solver is None or panel_solver.lattice is None:
            return
        freq = max(1, int(getattr(panel_solver, "logging_interval_steps", 1)))
        if step % freq != 0:
            return

        n = panel_solver.lattice.num_panels
        if n == 0:
            return

        try:
            centers = panel_solver.lattice.centers.to_numpy()[:n]
            normals = panel_solver.lattice.normals.to_numpy()[:n]
            strengths = panel_solver.lattice.strengths.to_numpy()[:n]
            areas = panel_solver.lattice.areas.to_numpy()[:n]
            group_ids = panel_solver.lattice.panel_group_id.to_numpy()[:n]
            cp = (
                panel_solver.lattice.Cp.to_numpy()[:n]
                if hasattr(panel_solver.lattice, "Cp")
                else np.zeros(n)
            )
            forces = (
                panel_solver.panel_forces.to_numpy()[:n]
                if panel_solver.panel_forces is not None
                else np.zeros((n, 3))
            )

            samples_dir = resolve_samples_dir(checkpoint_directory)
            samples_dir.mkdir(parents=True, exist_ok=True)
            csv_path = samples_dir / f"panel_distribution_step{step:06d}.csv"

            with open(csv_path, "w") as f:
                f.write("time,step,cx,cy,cz,nx,ny,nz,strength,area,Cp,Fx,Fy,Fz,group_id\n")
                for i in range(n):
                    c = centers[i]
                    nv = normals[i]
                    f.write(
                        f"{time},{step},"
                        f"{c[0]:.10e},{c[1]:.10e},{c[2]:.10e},"
                        f"{nv[0]:.10e},{nv[1]:.10e},{nv[2]:.10e},"
                        f"{strengths[i]:.10e},{areas[i]:.10e},{cp[i]:.10e},"
                        f"{forces[i][0]:.10e},{forces[i][1]:.10e},{forces[i][2]:.10e},"
                        f"{int(group_ids[i])}\n"
                    )
        except Exception as exc:
            print(f"(Warning) Failed to record panel loading distribution: {exc}")
