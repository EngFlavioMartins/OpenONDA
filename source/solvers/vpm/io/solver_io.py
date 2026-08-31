"""
Solver I/O (SolverIO): writes particle/field state and results to disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
import os
from typing import TYPE_CHECKING

from .backup import _BackupIO
from .logging import Logging
from .sampling import resolve_samples_dir

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

# =========================================================


class SolverIO:
    """
    Unified IO manager for VPM Solver.

    Consolidates all IO operations into a single, clean interface.
    """

    def __init__(self, solver: "VPMSolver"):
        """
        Initialize IO manager.

        Args:
            solver: Parent solver instance
        """
        self.solver = solver

        self.export_dir = self.solver._backup_path

        self._vlm_pvd_entries = []  # Track VLM time-series entries
        self._xdmf_series_entries = []  # Track VPM particle time-series entries

    @property
    def vpm_prefix(self) -> str:
        return "vpm"

    @property
    def vlm_prefix(self) -> str:
        return "vlm"

    @property
    def step(self) -> int:
        return self.solver.step

    @property
    def time(self) -> float:
        return self.solver.time

    def should_backup(self, step: int | None = None) -> bool:
        """
        Check if a backup should be written at the given timestep.

        Args:
            step: Step index to check (default: current solver step)

        Returns:
            True if a backup should be written
        """
        ts = step if step is not None else self.step
        interval_steps = self.solver.setup.backup.interval_steps
        return interval_steps > 0 and ts % interval_steps == 0

    def _write_scheduled_backup(self, verbose: bool = True):
        """
        Write a complete backup: HDF5 state + VTK visualization + CSV loads.

        This consolidates all backup logic from the solver into a single call.
        """
        if not self.should_backup():
            return

        # Ensure export directory exists
        os.makedirs(self.export_dir, exist_ok=True)

        # 1. HDF5 backup (for restart)
        backup_path = os.path.join(self.export_dir, self.vpm_prefix)
        _BackupIO.save(self.solver, backup_path, verbose=verbose)

        # Track VPM particle data for XDMF series
        xdmf_filename = f"{self.vpm_prefix}_{self.step:06d}.xdmf"
        # Use float64 for consistent time with HDF5
        time_val = float(self.time)
        self._xdmf_series_entries.append((time_val, xdmf_filename))

        # 2. VTK Visualization Export
        # Particles are not exported here: the HDF5 state written above already
        # carries them, and its XDMF descriptor is what ParaView opens.
        vtk_base = f"{self.export_dir}/{self.vpm_prefix}_{self.step:06d}"
        self.export_state(vtk_base, include_particles=False)

        # 3. Panel Solver Aerodynamic Loads (CSV)
        self._export_panel_loads(time_val)

        # 4. VLM Solver Export (VTK + CSV)
        self._export_vlm_results(time_val)

        # Note: XDMF temporal series (_series.xdmf) no longer written
        # Individual per-timestep .xdmf files are sufficient for ParaView

    def export_diagnostics_csv(self, diagnostics_history: dict, filename: str) -> None:
        """Export diagnostics history to CSV for offline analysis.

        Args:
            diagnostics_history: Solver's ``_diagnostics_history`` dict.
            filename: Destination CSV file path.
        """
        import csv

        fld = diagnostics_history
        if len(fld.get("time", [])) == 0:
            Logging.info("component=diagnostics_export status=skipped reason=no_records")
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
        Logging.info(f"component=diagnostics_export status=written path={filename!r}")

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

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    def save_setup(self, filename: str) -> None:
        """Save solver configuration to JSON."""
        setup_dict = self.solver.setup.to_dict()
        with open(filename, "w") as f:
            json.dump(setup_dict, f, indent=4)
        Logging.info(f"component=configuration status=written path={filename!r}")

    def load_particle_field(self, filename: str, remove_current_particles: bool = False):
        """Load particle field from file."""
        self.solver.particles.load_particle_field(filename, remove_current_particles)
        Logging.info(f"component=particle_field status=loaded path={filename!r}")

    def export_state(
        self,
        filename: str,
        include_panels: bool = True,
        include_particles: bool = True,
        format: str = "vtp",
        compression: bool = True,
    ):
        """Export solver state for visualization and post-processing."""
        # Export panels
        if (
            include_panels
            and self.solver.panel_solver is not None
            and getattr(self.solver.panel_solver, "lattice", None) is not None
        ):
            from .vtk_export import export_panels_vtk

            panel_file = f"{filename}_panels.{format}"
            export_panels_vtk(self.solver, panel_file, compression)

        if include_particles and self.solver.particles.n_particles_total > 0:
            self.solver.particles.save_vortex_particles(
                f"{filename}_particles.vtp",
                write_precision=self.solver.write_precision,
            )

        # Field export is not yet implemented; particles are handled above.

    def _export_panel_loads(self, time_val: float):
        """Export panel solver aerodynamic loads to CSV."""
        panel_solver = getattr(self.solver, "panel_solver", None)
        if panel_solver is None:
            return
        lattice = getattr(panel_solver, "lattice", None)
        if lattice is None or lattice.n_panels == 0:
            return

        # Compute forces using the cached panel_force field
        forces = panel_solver.compute_forces_coefficients(
            density=panel_solver.density,
            reference_velocity=panel_solver.freestream_velocity,
        )

        import pandas as pd

        samples_dir = resolve_samples_dir(
            self.solver.case_dir,
            self.solver.setup.samplers.directory,
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / f"{self.vpm_prefix}_forces.csv"

        row = {
            "time": time_val,
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
        }

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    def _export_vlm_results(self, time_val: float):
        """Export VLM solver results (VTK + PVD collection).

        VLM force CSV is written by solver.py's _export_vlm_forces_to_csv,
        which uses the correct cached reference velocity.
        """
        vlm_solver = getattr(self.solver, "vlm_solver", None)
        if vlm_solver is None or not getattr(vlm_solver, "_mesh_generated", False):
            return

        # Export VLM lattice visualization using consistent time
        vlm_filename = f"{self.vlm_prefix}_{self.step:06d}"
        vlm_base = f"{self.export_dir}/{vlm_filename}"
        try:
            vlm_solver.save_results(vlm_base, time=time_val)
            # Track for PVD collection
            self._vlm_pvd_entries.append((time_val, f"{vlm_filename}.vtp"))
            # Write PVD collection file for ParaView time-series
            self._write_surface_pvd_file(self.vlm_prefix)
        except Exception as exc:
            Logging.warning(
                f"component=vlm_output format=vtk status=write_failed path={vlm_base!r} "
                f"error={exc!r}"
            )

    def _write_surface_pvd_file(self, base_name: str):
        """Write ParaView Data (PVD) collection file for surface (VLM) time-series."""
        pvd_path = os.path.join(self.export_dir, f"{base_name}.pvd")

        try:
            with open(pvd_path, "w") as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
                f.write("  <Collection>\n")

                for time_value, filename in self._vlm_pvd_entries:
                    f.write(f'    <DataSet timestep="{time_value:.6g}" file="{filename}"/>')

                f.write("  </Collection>\n")
                f.write("</VTKFile>\n")
            Logging.info(
                f"component=vlm_output format=pvd status=written path={pvd_path!r} "
                f"time_levels={len(self._vlm_pvd_entries)}"
            )
        except Exception as exc:
            Logging.warning(
                f"component=vlm_output format=pvd status=write_failed path={pvd_path!r} "
                f"error={exc!r}"
            )


__all__ = ["SolverIO"]
