"""
VLM diagnostics module — per-step VLM loads and vector strengths in vlm_forces.csv.

Nothing in this module should import from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
step controller.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....particles.container import Particles
    from .vlm_solver import VLMSolver


class VLMDiagnostics:
    """Static helpers for recording VLM diagnostics and writing CSV output."""

    @staticmethod
    def record_vlm_diagnostics(
        vlm_solver: VLMSolver,
        particles: Particles,
        particle_vortex_strength: np.ndarray,
        time: float,
        samples_dir: Path,
    ) -> None:
        """Record VLM force and vector-strength scalars and flush one solver sample.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        particles:
            Current ``Particles`` container (for particle count).
        particle_vortex_strength:
            Vortex-strength array (N, 3) for the current particles [m³/s].
        time:
            Current simulation time [s].
        samples_dir:
            Sample directory of the case; the table is ``vlm_forces.csv``.
        """
        if vlm_solver is None or vlm_solver._last_forces is None:
            return
        forces = vlm_solver._last_forces
        n_panels = vlm_solver.lattice.n_panels
        # Use the same integrated three-leg field as the solver budget. A
        # quarter-chord-only reconstruction is wrong for tapered/twisted wings.
        bound_vortex_strength = vlm_solver.compute_total_bound_vortex_strength()
        bound_vortex_strength_y = float(bound_vortex_strength[1])

        n_p = particles.n_particles_total
        wake_vortex_strength_y = float(particle_vortex_strength[:, 1].sum(dtype=np.float64))

        lespnp = vlm_solver.lattice.leading_edge_suction_parameter.to_numpy()[:n_panels]
        max_leading_edge_suction_parameter = float(np.max(lespnp)) if n_panels > 0 else 0.0

        # Coupled VLM is an owned VPM component.  The caller reaches this
        # method only after an accepted VPM step, so every row belongs to the
        # solver's accepted clock; no VLM-specific cadence may skip it.
        VLMDiagnostics.export_forces_csv(
            vlm_solver,
            forces,
            bound_vortex_strength_y,
            wake_vortex_strength_y,
            max_leading_edge_suction_parameter,
            n_p,
            time,
            samples_dir,
        )

    # CSV export

    @staticmethod
    def export_forces_csv(
        vlm_solver: VLMSolver,
        forces: dict,
        bound_vortex_strength: float,
        wake_vortex_strength: float,
        max_leading_edge_suction_parameter: float,
        n_p: int,
        time: float,
        samples_dir: Path,
    ) -> None:
        """Append one row to ``<samples_dir>/vlm_forces.csv``.

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
        samples_dir:
            Sample directory of the case.
        """
        import pandas as pd

        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_forces.csv"

        force_density = float(vlm_solver._force_density)
        reference_velocity = vlm_solver._last_reference_velocity
        if reference_velocity is None:
            reference_velocity = vlm_solver.freestream_velocity
        reference_speed = (
            float(np.linalg.norm(reference_velocity)) if reference_velocity is not None else 0.0
        )

        row = {
            "time": time,
            "lift_coefficient": forces.get("lift_coefficient", 0.0),
            "drag_coefficient": forces.get("drag_coefficient", 0.0),
            "side_force_coefficient": forces.get("side_force_coefficient", 0.0),
            "force_x": forces.get("force_x", 0.0),
            "force_y": forces.get("force_y", 0.0),
            "force_z": forces.get("force_z", 0.0),
            "unsteady_force_x": forces.get("unsteady_force_x", 0.0),
            "unsteady_force_y": forces.get("unsteady_force_y", 0.0),
            "unsteady_force_z": forces.get("unsteady_force_z", 0.0),
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
            "force_density": force_density,
            "force_units": "N",
            "moment_units": "N*m",
            "reference_speed": reference_speed,
            "reference_speed_units": "m/s",
        }
        reference_velocity = vlm_solver._last_reference_velocity
        if reference_velocity is None:
            reference_velocity = vlm_solver.freestream_velocity
        surface_forces = vlm_solver.compute_per_surface_forces(
            vlm_solver.density, reference_velocity
        )
        for key in ("power", "rotational_power", "translational_power"):
            row[key] = sum(surface[key] for surface in surface_forces.values())
        surfaces_path = csv_path.with_name("vlm_surface_forces.csv")
        pd.DataFrame(
            [
                {
                    "time": time,
                    "surface": name,
                    "force_density": force_density,
                    "force_units": "N",
                    "moment_units": "N*m",
                    **values,
                }
                for name, values in surface_forces.items()
            ]
        ).to_csv(surfaces_path, mode="a", header=not surfaces_path.exists(), index=False)
        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)
