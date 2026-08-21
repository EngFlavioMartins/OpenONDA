"""
Solver I/O (SolverIO): writes particle/field state and results to disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
import os
from typing import TYPE_CHECKING

from .checkpoint import CheckpointManager
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

        # solver.checkpoint_directory is already resolved against case_dir
        # (see VPMSolver.__init__); solver.setup.checkpoint_directory is the
        # raw, possibly-relative user value and must not be used directly,
        # or checkpoints land under the caller's cwd instead of the case dir.
        self.export_dir = self.solver.checkpoint_directory or "solution"

        self._vlm_pvd_entries = []  # Track VLM time-series entries
        self._xdmf_series_entries = []  # Track VPM particle time-series entries

    @property
    def checkpoint_interval_steps(self) -> int:
        return self.solver.checkpoint_interval_steps

    @property
    def checkpoint_name(self) -> str:
        return (self.solver.checkpoint_name or "").strip()

    @property
    def vpm_prefix(self) -> str:
        return f"vpm_{self.checkpoint_name}" if self.checkpoint_name else "vpm"

    @property
    def vlm_prefix(self) -> str:
        return f"vlm_{self.checkpoint_name}" if self.checkpoint_name else "vlm"

    @property
    def step(self) -> int:
        return self.solver.step

    @property
    def time(self) -> float:
        return self.solver.time

    def should_checkpoint(self, step: int | None = None) -> bool:
        """
        Check if a checkpoint should be written at the given timestep.

        Args:
            step: Step index to check (default: current solver step)

        Returns:
            True if a checkpoint should be written
        """
        ts = step if step is not None else self.step
        return self.checkpoint_interval_steps > 0 and (
            ts % self.checkpoint_interval_steps == 0 or ts == 0
        )

    def write_checkpoint(self, verbose: bool = True):
        """
        Write a complete checkpoint: HDF5 state + VTK visualization + CSV loads.

        This consolidates all checkpoint logic from the solver into a single call.
        """
        if not self.should_checkpoint():
            return

        # Ensure export directory exists
        os.makedirs(self.export_dir, exist_ok=True)

        # 1. HDF5 checkpoint (for restart)
        checkpoint_path = os.path.join(self.export_dir, self.vpm_prefix)
        CheckpointManager.write_checkpoint(self.solver, checkpoint_path, verbose=verbose)

        # Track VPM particle data for XDMF series
        xdmf_filename = f"{self.vpm_prefix}_{self.step:06d}.xdmf"
        # Use float64 for consistent time with HDF5
        time_val = float(self.time)
        self._xdmf_series_entries.append((time_val, xdmf_filename))

        # 2. VTK Visualization Export
        vtk_base = f"{self.export_dir}/{self.vpm_prefix}_{self.step:06d}"
        self.export_state(vtk_base)

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
            print("[INFO] No diagnostics to export.")
            return
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "time",
                    "vpm_total_vortex_strength_magnitude",
                    "fvm_total_vortex_strength_magnitude",
                    "interpolated_total_vortex_strength_magnitude",
                    "n_injected",
                    "n_candidates",
                    "observed_time_step_size",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                ]
            )
            for i in range(len(fld["time"])):
                centroid = fld["centroid"][i] if i < len(fld["centroid"]) else (0.0, 0.0, 0.0)
                writer.writerow(
                    [
                        fld["time"][i],
                        fld["vpm_total_vortex_strength_magnitude"][i]
                        if i < len(fld["vpm_total_vortex_strength_magnitude"])
                        else 0.0,
                        fld["fvm_total_vortex_strength_magnitude"][i]
                        if i < len(fld["fvm_total_vortex_strength_magnitude"])
                        else 0.0,
                        fld["interpolated_total_vortex_strength_magnitude"][i]
                        if i < len(fld["interpolated_total_vortex_strength_magnitude"])
                        else 0.0,
                        fld["n_injected"][i] if i < len(fld["n_injected"]) else 0,
                        fld["n_candidates"][i] if i < len(fld["n_candidates"]) else 0,
                        fld["observed_time_step_size"][i]
                        if i < len(fld["observed_time_step_size"])
                        else 0.0,
                        centroid[0],
                        centroid[1],
                        centroid[2],
                    ]
                )
        print(f"[INFO] Diagnostics exported to {filename}")

    def export_flow_integrals_csv(self, solver: "VPMSolver") -> None:
        """Append one row of flow integrals to ``<case_dir>/samples/flow_integrals.csv``.

        Args:
            solver: Parent solver instance with evaluated flow integrals and
                diagnostics available.
        """
        import numpy as np
        import pandas as pd

        samples_dir = resolve_samples_dir(
            solver.case_dir,
            getattr(solver.setup, "sample_subdirectory", None),
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "flow_integrals.csv"

        impulse = solver._flow_integrals.get("linear_impulse", np.zeros(3))
        ang_impulse = solver._flow_integrals.get("angular_impulse", np.zeros(3))
        strength = solver._flow_integrals.get("vortex_strength", np.zeros(3))
        particle_vortex_strength = solver.particle_vortex_strength
        turbulent_viscosity = solver.particles.eddy_viscosity_cpu()
        effective_viscosity = solver.particles.effective_viscosity_cpu()

        row = {
            "time": solver.time,
            "step": solver.step,
            "kinetic_energy": solver.total_kinetic_energy,
            "enstrophy": solver.total_enstrophy,
            "enstrophy_test": solver._flow_integrals.get("enstrophy_test", 0.0),
            "dEdt": solver.kinetic_energy_dissipation_rate,
            "neg_nu_enstrophy": solver.vorticity_dissipation_rate,
            "helicity": solver.total_helicity,
            "vortex_strength_magnitude": solver.total_vortex_strength_magnitude,
            "vortex_strength_x": float(strength[0]),
            "vortex_strength_y": float(strength[1]),
            "vortex_strength_z": float(strength[2]),
            "impulse_x": float(impulse[0]),
            "impulse_y": float(impulse[1]),
            "impulse_z": float(impulse[2]),
            "angular_impulse_x": float(ang_impulse[0]),
            "angular_impulse_y": float(ang_impulse[1]),
            "angular_impulse_z": float(ang_impulse[2]),
            "n_particles": solver.particles.n_particles,
            "max_vortex_strength": float(
                np.linalg.norm(particle_vortex_strength, axis=1).max(initial=0.0)
            ),
            "eddy_viscosity_mean": float(turbulent_viscosity.mean())
            if len(turbulent_viscosity)
            else 0.0,
            "eddy_viscosity_max": float(turbulent_viscosity.max(initial=0.0)),
            "effective_viscosity_mean": float(effective_viscosity.mean())
            if len(effective_viscosity)
            else 0.0,
            "effective_viscosity_max": float(effective_viscosity.max(initial=0.0)),
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
        print(f"Configuration saved to {filename}")

    def load_particle_field(self, filename: str, remove_current_particles: bool = False):
        """Load particle field from file."""
        self.solver.particles.load_particle_field(filename, remove_current_particles)
        print(f"Particle field loaded from {filename}")

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

        if include_particles and self.solver.particles.n_particles > 0:
            self.solver.particles.save_vortex_particles(f"{filename}_particles.vtp")

        # Field export is not yet implemented; particles are handled above.

    def _export_panel_loads(self, time_val: float):
        """Export panel solver aerodynamic loads to CSV."""
        panel_solver = getattr(self.solver, "panel_solver", None)
        if panel_solver is None:
            return
        lattice = getattr(panel_solver, "lattice", None)
        if lattice is None or lattice.num_panels == 0:
            return

        # Compute forces using the cached panel_forces field
        forces = panel_solver.compute_forces_coefficients(
            density=panel_solver.density,
            reference_velocity=panel_solver.freestream_velocity,
        )

        import pandas as pd

        samples_dir = resolve_samples_dir(
            self.solver.case_dir,
            getattr(self.solver.setup, "sample_subdirectory", None),
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / f"{self.vpm_prefix}_forces.csv"

        row = {
            "time": time_val,
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
        except Exception as e:
            print(f"   (Warning) Could not save VLM VTK: {e}")

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
            print(
                f"   Updated surface PVD file: {pvd_path} ({len(self._vlm_pvd_entries)} timesteps)"
            )
        except Exception as e:
            print(f"   (Warning) Could not write VLM PVD file: {e}")


__all__ = ["SolverIO"]
