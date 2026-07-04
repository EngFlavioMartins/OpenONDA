"""
Solver I/O (SolverIO): writes particle/field state and results to disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import asdict
import json
import os
from typing import TYPE_CHECKING

import h5py

from .backup import BackupSystem
from .vtk_export import export_panels_vtk

if TYPE_CHECKING:
    from ..core.solver import Solver

# =========================================================

class SolverIO:
    """
    Unified IO manager for VPM Solver.

    Consolidates all IO operations into a single, clean interface.
    """

    def __init__(self, solver: "Solver"):
        """
        Initialize IO manager.

        Args:
            solver: Parent solver instance
        """
        self.solver = solver

        if hasattr(self.solver.config, "backup_directory") and self.solver.config.backup_directory:
            self.export_dir = self.solver.config.backup_directory
        elif hasattr(self.solver.config, "solution_name"):
            self.export_dir = self.solver.config.solution_name
        else:
            self.export_dir = "solution"

        self._vlm_pvd_entries = []  # Track VLM time-series entries
        self._xdmf_series_entries = []  # Track VPM particle time-series entries
        self._xdmf_series_entries = []  # Track VPM particle time-series entries

    @property
    def backup_frequency(self) -> int:
        return self.solver.backup_frequency

    @property
    def backup_file_name(self) -> str:
        return os.path.basename((self.solver.backup_file_name or "").strip())

    @property
    def vpm_prefix(self) -> str:
        return f"vpm_{self.backup_file_name}" if self.backup_file_name else "vpm"

    @property
    def vlm_prefix(self) -> str:
        return f"vlm_{self.backup_file_name}" if self.backup_file_name else "vlm"

    @property
    def time_step(self) -> int:
        return self.solver.time_step

    @property
    def flow_time(self) -> float:
        return self.solver.flow_time

    def should_backup(self, time_step: int | None = None) -> bool:
        """
        Check if backup should be performed at given timestep.

        Args:
            time_step: Timestep to check (default: current solver timestep)

        Returns:
            True if backup should be performed
        """
        ts = time_step if time_step is not None else self.time_step
        return self.backup_frequency > 0 and (ts % self.backup_frequency == 0 or ts == 0)

    def backup(self, verbose: bool = True):
        """
        Perform complete backup: HDF5 state + VTK visualization + CSV loads.

        This consolidates all backup logic from the solver into a single call.
        """
        if not self.should_backup():
            return

        # Ensure export directory exists
        os.makedirs(self.export_dir, exist_ok=True)

        # 1. HDF5 Backup (for restart)
        backup_path = os.path.join(self.export_dir, self.vpm_prefix)
        BackupSystem.backup_solver(self.solver, backup_path, verbose=verbose)

        # Track VPM particle data for XDMF series
        xdmf_filename = f"{self.vpm_prefix}_{self.time_step:06d}.xdmf"
        # Use float64 for consistent time with HDF5
        time_val = float(self.flow_time)
        self._xdmf_series_entries.append((time_val, xdmf_filename))

        # 2. VTK Visualization Export
        vtk_base = f"{self.export_dir}/{self.vpm_prefix}_{self.time_step:06d}"
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
        if len(fld.get("flow_time", [])) == 0:
            print("[INFO] No diagnostics to export.")
            return
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "flow_time",
                    "vpm_total_circ_mag",
                    "ofw_total_circ_mag",
                    "interp_total_circ_mag",
                    "n_injected",
                    "n_candidates",
                    "observed_dt",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                ]
            )
            for i in range(len(fld["flow_time"])):
                centroid = fld["centroid"][i] if i < len(fld["centroid"]) else (0.0, 0.0, 0.0)
                writer.writerow(
                    [
                        fld["flow_time"][i],
                        fld["vpm_total_circ_mag"][i] if i < len(fld["vpm_total_circ_mag"]) else 0.0,
                        fld["ofw_total_circ_mag"][i] if i < len(fld["ofw_total_circ_mag"]) else 0.0,
                        fld["interp_total_circ_mag"][i]
                        if i < len(fld["interp_total_circ_mag"])
                        else 0.0,
                        fld["n_injected"][i] if i < len(fld["n_injected"]) else 0,
                        fld["n_candidates"][i] if i < len(fld["n_candidates"]) else 0,
                        fld["observed_dt"][i] if i < len(fld["observed_dt"]) else 0.0,
                        centroid[0],
                        centroid[1],
                        centroid[2],
                    ]
                )
        print(f"[INFO] Diagnostics exported to {filename}")

    def save_config(self, filename: str) -> None:
        """Save solver configuration to JSON."""
        config_dict = asdict(self.solver.config)
        with open(filename, "w") as f:
            json.dump(config_dict, f, indent=4)
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
        include_fields: bool = False,
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
            panel_file = f"{filename}_panels.{format}"
            export_panels_vtk(self.solver, panel_file, compression)

        # Particle and field export are delegated to BackupSystem (XDMF/HDF5).
        # Standalone VTK export of particles and fields is not yet implemented.
        if include_particles or include_fields:
            pass  # Stub: handled by BackupSystem.backup_solver.

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
            U_ref=panel_solver.U_inf,
        )

        from pathlib import Path

        import pandas as pd

        samples_dir = Path(self.export_dir) / "samples"
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
        vlm_filename = f"{self.vlm_prefix}_{self.time_step:06d}"
        vlm_base = f"{self.export_dir}/{vlm_filename}"
        try:
            vlm_solver.save_results(vlm_base, flow_time=time_val)
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

    def _write_xdmf_series(self, base_name: str):
        """Write/update XDMF temporal collection file for VPM particle time-series."""
        try:
            series_path = f"{self.export_dir}/{base_name}_series.xdmf"

            # Filter entries to only include those with existing HDF5 files
            valid_entries = []
            for time_val, xdmf_filename in self._xdmf_series_entries:
                h5_filename = xdmf_filename.replace(".xdmf", ".h5")
                h5_path = os.path.join(self.export_dir, h5_filename)
                if os.path.exists(h5_path):
                    valid_entries.append((time_val, xdmf_filename))

            # Update internal list to remove stale entries
            self._xdmf_series_entries = valid_entries

            if not valid_entries:
                return  # No valid entries, skip writing

            # Load number of particles from the latest HDF5 file
            latest_h5 = os.path.join(self.export_dir, valid_entries[-1][1].replace(".xdmf", ".h5"))
            with h5py.File(latest_h5, "r") as f:
                int(f["solver"].attrs["number_of_particles"])

            # Write XDMF temporal collection
            with open(series_path, "w") as f:
                f.write('<?xml version="1.0" ?>\n')
                f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                f.write('<Xdmf Version="3.0">\n')
                f.write("  <Domain>\n")
                f.write(
                    '    <Grid Name="VortexParticles_TimeSeries" GridType="Collection" CollectionType="Temporal">\n'
                )

                for time_val, xdmf_filename in valid_entries:
                    h5_filename = xdmf_filename.replace(".xdmf", ".h5")
                    timestep_num = int(xdmf_filename.split("_")[-1].replace(".xdmf", ""))

                    # Get actual particle count from each file
                    h5_path = os.path.join(self.export_dir, h5_filename)
                    with h5py.File(h5_path, "r") as hf:
                        n_particles_this = int(hf["solver"].attrs["number_of_particles"])

                    f.write(f'      <Grid Name="TimeStep_{timestep_num:06d}" GridType="Uniform">\n')
                    f.write(
                        f'        <Topology TopologyType="Polyvertex" NumberOfElements="{n_particles_this}"/>\n'
                    )
                    f.write('        <Geometry GeometryType="XYZ">\n')
                    f.write(
                        f'          <DataItem Dimensions="{n_particles_this} 3" NumberType="Float" Precision="4" Format="HDF">\n'
                    )
                    f.write(f"            {h5_filename}:/particles/position\n")
                    f.write("          </DataItem>\n")
                    f.write("        </Geometry>\n")
                    f.write(f'        <Time Value="{time_val:.6g}"/>')
                    f.write(
                        '        <Attribute Name="Velocity" AttributeType="Vector" Center="Node">\n'
                    )
                    f.write(
                        f'          <DataItem Dimensions="{n_particles_this} 3" NumberType="Float" Precision="4" Format="HDF">\n'
                    )
                    f.write(f"            {h5_filename}:/particles/velocity\n")
                    f.write("          </DataItem>\n")
                    f.write("        </Attribute>\n")
                    f.write(
                        '        <Attribute Name="BackgroundVelocity" AttributeType="Vector" Center="Node">\n'
                    )
                    f.write(
                        f'          <DataItem Dimensions="{n_particles_this} 3" NumberType="Float" Precision="4" Format="HDF">\n'
                    )
                    f.write(f"            {h5_filename}:/particles/background_velocity\n")
                    f.write("          </DataItem>\n")
                    f.write("        </Attribute>\n")
                    f.write(
                        '        <Attribute Name="Circulation" AttributeType="Vector" Center="Node">\n'
                    )
                    f.write(
                        f'          <DataItem Dimensions="{n_particles_this} 3" NumberType="Float" Precision="4" Format="HDF">\n'
                    )
                    f.write(f"            {h5_filename}:/particles/circulation\n")
                    f.write("          </DataItem>\n")
                    f.write("        </Attribute>\n")
                    f.write(
                        '        <Attribute Name="Vorticity" AttributeType="Vector" Center="Node">\n'
                    )
                    f.write(
                        f'          <DataItem Dimensions="{n_particles_this} 3" NumberType="Float" Precision="4" Format="HDF">\n'
                    )
                    f.write(f"            {h5_filename}:/particles/vorticity\n")
                    f.write("          </DataItem>\n")
                    f.write("        </Attribute>\n")
                    f.write("      </Grid>\n")

                f.write("    </Grid>\n")
                f.write("  </Domain>\n")
                f.write("</Xdmf>\n")

        except Exception as e:
            print(f"   (Warning) Could not write XDMF series file: {e}")

__all__ = ["SolverIO"]
