"""
Solver I/O (SolverIO): writes particle/field state and results to disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from source.solution_layout import component_directory

from .backup import _BackupIO
from .logging import Logging

if TYPE_CHECKING:
    from ..core.solver import VPMSolver


class SolverIO:
    """
    Unified IO manager for VPM Solver.

    Consolidates all IO operations into a single, clean interface.
    """

    def __init__(self, solver: "VPMSolver"):
        """Create the I/O facade for one VPM solver.

        Parameters
        ----------
        solver : VPMSolver
            Parent solver.  Its resolved backup path and accepted clock are
            read dynamically; the solver object is retained by reference.

        Notes
        -----
        Construction does not write files.  Backup, CSV, VTK, and VTU output
        occur only when their explicit methods or schedules are invoked.
        """
        self.solver = solver

        self.solution_dir = self.solver._backup_path
        self.export_dir = component_directory(self.solution_dir, "vpm")

    @property
    def vpm_prefix(self) -> str:
        """Return the stable filename prefix for particle backups."""
        return "vpm"

    @property
    def vlm_prefix(self) -> str:
        """Return the stable filename prefix for VLM outputs."""
        return "vlm"

    @property
    def step(self) -> int:
        """Return the parent solver's accepted-step index."""
        return self.solver.step

    @property
    def time(self) -> float:
        """Return the parent solver's accepted physical time in seconds."""
        return self.solver.time

    def write_backup(self, verbose: bool = True) -> None:
        """Write restart state, its VTU frame, and the ``vpm.pvd`` index.

        VLM surface companions share the sparse backup clock with VPM particles.
        Accepted-step VLM force/loading tables are emitted through the owner's
        sample path; no VLM-specific backup cadence or output root is created.
        """
        os.makedirs(self.export_dir, exist_ok=True)
        backup_path = os.path.join(self.export_dir, self.vpm_prefix)
        _BackupIO.save(self.solver, backup_path, verbose=verbose)
        _BackupIO.write_pvd(self.solution_dir)
        vlm = getattr(self.solver, "vlm_solver", None)
        if vlm is not None:
            from .vlm_backup import write_vlm_backup

            write_vlm_backup(vlm, self.solution_dir, step=self.step, time=self.time)

    def export_diagnostics_csv(self, diagnostics_history: dict, filename: str) -> None:
        """Export diagnostics history to CSV for offline analysis.

        Args:
            diagnostics_history: Solver's ``_diagnostics_history`` dict.
            filename: Destination CSV file path.
        """
        import csv

        fld = diagnostics_history
        if len(fld.get("time", [])) == 0:
            Logging.info("No diagnostic records are available to export")
            return
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "time",
                    "vpm_vortex_strength_magnitude_sum",
                    "fvm_vortex_strength_magnitude_sum",
                    "interpolated_vortex_strength_magnitude_sum",
                    "n_particles_injected",
                    "n_particle_candidates",
                    "observed_time_step_size",
                    "vortex_centroid_x",
                    "vortex_centroid_y",
                    "vortex_centroid_z",
                ]
            )
            for i in range(len(fld["time"])):
                vortex_centroid = (
                    fld["vortex_centroid"][i]
                    if i < len(fld["vortex_centroid"])
                    else (0.0, 0.0, 0.0)
                )
                writer.writerow(
                    [
                        fld["time"][i],
                        fld["vpm_vortex_strength_magnitude_sum"][i]
                        if i < len(fld["vpm_vortex_strength_magnitude_sum"])
                        else 0.0,
                        fld["fvm_vortex_strength_magnitude_sum"][i]
                        if i < len(fld["fvm_vortex_strength_magnitude_sum"])
                        else 0.0,
                        fld["interpolated_vortex_strength_magnitude_sum"][i]
                        if i < len(fld["interpolated_vortex_strength_magnitude_sum"])
                        else 0.0,
                        fld["n_particles_injected"][i]
                        if i < len(fld["n_particles_injected"])
                        else 0,
                        fld["n_particle_candidates"][i]
                        if i < len(fld["n_particle_candidates"])
                        else 0,
                        fld["observed_time_step_size"][i]
                        if i < len(fld["observed_time_step_size"])
                        else 0.0,
                        vortex_centroid[0],
                        vortex_centroid[1],
                        vortex_centroid[2],
                    ]
                )
        Logging.info(f"Diagnostics written: {filename}")

    def export_flow_integrals_csv(self, solver: "VPMSolver", csv_path) -> None:
        """Append one row of flow integrals to ``<case_dir>/samples/flow_integrals.csv``.

        Args:
            solver: Parent solver instance with evaluated flow integrals and
                diagnostics available.
        """
        import numpy as np
        import pandas as pd

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        linear_impulse = solver._flow_integrals.get("linear_impulse", np.zeros(3))
        angular_impulse = solver._flow_integrals.get("angular_impulse", np.zeros(3))
        net_vortex_strength = solver._flow_integrals.get("net_vortex_strength", np.zeros(3))
        particle_vortex_strength = solver.particle_vortex_strength
        particle_core_radius = solver.particle_core_radius
        eddy_viscosity = solver.particles.eddy_viscosity_cpu()
        effective_viscosity = solver.particles.effective_viscosity_cpu()
        row = {
            "time": solver.time,
            "step": solver.step,
            "total_kinetic_energy": solver.total_kinetic_energy,
            "energy_measurement": solver._flow_integrals.get("energy_measurement", "unknown"),
            "total_enstrophy": solver.total_enstrophy,
            "test_filtered_enstrophy": solver._flow_integrals.get("test_filtered_enstrophy", 0.0),
            "kinetic_energy_rate": solver.kinetic_energy_rate,
            "kinetic_energy_rate_source": solver._flow_integrals.get(
                "kinetic_energy_rate_source", "unknown"
            ),
            "viscous_kinetic_energy_rate": solver.viscous_kinetic_energy_rate,
            "total_helicity": solver.total_helicity,
            "vortex_strength_magnitude_sum": solver.vortex_strength_magnitude_sum,
            "net_vortex_strength_x": float(net_vortex_strength[0]),
            "net_vortex_strength_y": float(net_vortex_strength[1]),
            "net_vortex_strength_z": float(net_vortex_strength[2]),
            "linear_impulse_x": float(linear_impulse[0]),
            "linear_impulse_y": float(linear_impulse[1]),
            "linear_impulse_z": float(linear_impulse[2]),
            "angular_impulse_x": float(angular_impulse[0]),
            "angular_impulse_y": float(angular_impulse[1]),
            "angular_impulse_z": float(angular_impulse[2]),
            "n_particles_total": solver.particles.n_particles_total,
            "max_vortex_strength_magnitude": float(
                np.linalg.norm(particle_vortex_strength, axis=1).max(initial=0.0)
            ),
            "min_particle_core_radius": float(particle_core_radius.min())
            if len(particle_core_radius)
            else 0.0,
            "mean_particle_core_radius": float(particle_core_radius.mean())
            if len(particle_core_radius)
            else 0.0,
            "max_particle_core_radius": float(particle_core_radius.max(initial=0.0)),
            "mean_eddy_viscosity": float(eddy_viscosity.mean()) if len(eddy_viscosity) else 0.0,
            "max_eddy_viscosity": float(eddy_viscosity.max(initial=0.0)),
            "mean_effective_viscosity": float(effective_viscosity.mean())
            if len(effective_viscosity)
            else 0.0,
            "max_effective_viscosity": float(effective_viscosity.max(initial=0.0)),
            "invariant_projection_correction_ratio": float(
                solver.physics.rate_projection_max_correction_ratio
            ),
        }
        row.update(solver._discretization_health)
        row.update(solver.stabilization.diagnostics)
        vlm = getattr(solver, "vlm_solver", None)
        if vlm is not None:
            bound_impulse = vlm.compute_bound_linear_impulse()
            bound_strength = vlm.compute_total_bound_vortex_strength()
            for index, axis in enumerate("xyz"):
                # Like the established particle integrals, impulse is per density [m^4/s].
                row[f"bound_linear_impulse_{axis}"] = float(bound_impulse[index])
                row[f"coupled_linear_impulse_{axis}"] = float(
                    linear_impulse[index] + bound_impulse[index]
                )
                row[f"bound_vortex_strength_{axis}"] = float(bound_strength[index])
                row[f"coupled_vortex_strength_{axis}"] = float(
                    net_vortex_strength[index] + bound_strength[index]
                )
        health = getattr(solver, "_accepted_health_snapshot", None)
        if health is not None:
            row.update(
                {
                    "strain_increment_infinity": health.strain_increment_infinity,
                    "strain_increment_spectral": health.strain_increment_spectral,
                    "maximum_particle_strength": health.maximum_particle_strength,
                    "maximum_particle_vorticity": health.maximum_vorticity,
                }
            )

        df = pd.DataFrame([row])
        if csv_path.exists() and csv_path.stat().st_size:
            previous = pd.read_csv(csv_path)
            if not previous.empty and float(previous.iloc[-1]["time"]) >= float(row["time"]):
                raise ValueError(
                    "flow-integrals CSV event is duplicate or nonmonotonic: "
                    f"path={str(csv_path)!r}, "
                    f"previous step={previous.iloc[-1].get('step', 'unknown')}, "
                    f"time={float(previous.iloc[-1]['time']):.17g}; "
                    f"incoming step={row['step']}, time={float(row['time']):.17g}. "
                    "Use a separate output directory for a replay or concurrent run; "
                    "resume after the last written event."
                )
            if "energy_measurement" not in previous:
                raise ValueError(
                    f"flow-integrals CSV {str(csv_path)!r} is missing energy_measurement; "
                    "generate current samples in a separate output directory."
                )
            df = pd.concat((previous, df), ignore_index=True)
        temporary = csv_path.with_name(f".{csv_path.name}.tmp")
        df.to_csv(temporary, index=False)
        os.replace(temporary, csv_path)

    def load_particle_field(
        self, filename: str | Path, remove_current_particles: bool = False
    ) -> None:
        """Load particle field from file."""
        self.solver.particles.load_vortex_particles(str(filename), remove_current_particles)
        Logging.info(f"Particle field loaded: {filename}")

    def export_state(
        self,
        filename: str | Path,
        include_particles: bool = True,
    ) -> None:
        """Export solver state for visualization and post-processing."""
        if include_particles and self.solver.particles.n_particles_total > 0:
            self.solver.particles.save_vortex_particles(
                f"{filename}_particles.vtp",
                write_precision=self.solver.write_precision,
            )

        # Field export is not yet implemented; particles are handled above.


__all__ = ["SolverIO"]
