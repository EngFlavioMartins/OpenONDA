"""
Backup module for VPM solver.
==================
Backup module for VPM solver. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
import os

import h5py
import numpy as np

from ..config.constants import *  # noqa: F403
from ..config.types import SolverConfig


class BackupSystem:
    """
    Unified backup and restore system using HDF5 for numerical data
    and JSON for configuration metadata.

    This eliminates precision loss while maintaining human-readable configuration.
    """

    @staticmethod
    def backup_solver(
        solver, backup_file_name: str, time_float32: float = None, verbose: bool = True
    ) -> None:
        """
        Create a complete backup of the solver state using HDF5 + JSON + XDMF.

        Args:
            solver: The Solver instance to backup
            backup_file_name: Base filename (without extension)
            time_float32: Optional float32 time value for consistency (computed if None)
            verbose: Whether to print backup completion message
        """
        try:
            # Compute float32 time if not provided
            if time_float32 is None:
                import numpy as np

                time_float32 = float(np.float32(solver.flow_time))

            # For timestamped backups: add 6-digit sequential id
            # For "latest" backups: do NOT add timestep (keep it consistent)
            if "_latest" in backup_file_name:
                # This is a "latest" checkpoint; don't add timestep suffix
                hdf5_file = f"{backup_file_name}.h5"
                xdmf_file = f"{backup_file_name}.xdmf"
            else:
                # This is a timestamped backup; add 6-digit sequential id
                step_str = str(getattr(solver, "time_step", 0)).zfill(6)
                backup_base = f"{backup_file_name}_{step_str}"
                # Generate filenames: <name>_XXXXXX.ext
                hdf5_file = f"{backup_base}.h5"
                xdmf_file = f"{backup_base}.xdmf"

            # Ensure destination directory exists (create if missing)
            # If backup_file_name contains a directory, create it. If it's a bare filename,
            # os.path.dirname() returns an empty string and no directory creation is needed.
            backup_dir = os.path.dirname(hdf5_file)
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)

            # 1. Save numerical data to HDF5 (preserves precision) - use float32 time
            BackupSystem._save_numerical_data(solver, hdf5_file, time_float32)

            # 2. Configuration JSON is NOT saved at every timestep (use save_state() explicitly)
            # This reduces IO overhead during simulation

            # 3. Create XDMF file for ParaView visualization (references the per-step HDF5) - use float32 time
            backup_base_for_xdmf = (
                backup_file_name
                if "_latest" in backup_file_name
                else f"{backup_file_name}_{str(getattr(solver, 'time_step', 0)).zfill(6)}"
            )
            BackupSystem._create_xdmf_file(solver, backup_base_for_xdmf, xdmf_file, time_float32)

            # Print message only if verbose is True
            if verbose:
                backup_dir = (
                    os.path.dirname(hdf5_file) if os.path.dirname(hdf5_file) else os.getcwd()
                )
                print(f"Solution data saved to {backup_dir}")

        except Exception as e:
            raise RuntimeError(f"Backup failed: {e}") from e

    @staticmethod
    def _save_particle_optional_fields(particles_group, solver, n_particles: int) -> None:
        """Save optional/advanced particle fields to HDF5, silently skipping unavailable ones."""
        try:
            zone_id_data = solver.particles.zone_id.to_numpy()[:n_particles]
            particles_group.create_dataset("zone_id", data=zone_id_data)
        except (AttributeError, Exception) as e:
            print(f"(Info) Warning: zone_id field not available for backup: {e}")

        try:
            vg_data = solver.particles.velocity_gradient.to_numpy()
            if vg_data.shape[0] >= n_particles:
                particles_group.create_dataset(
                    "velocity_gradient", data=vg_data[:n_particles].reshape(n_particles, 9)
                )
            else:
                print(
                    f"(Info) Warning: velocity_gradient field size mismatch "
                    f"({vg_data.shape[0]} < {n_particles}), skipping backup"
                )
        except (AttributeError, RuntimeError, ValueError) as e:
            print(f"(Info) Warning: velocity_gradient field not available for backup: {e}")

        try:
            sr_data = solver.particles.strain_rate.to_numpy()
            if sr_data.shape[0] >= n_particles:
                sr_slice = sr_data[:n_particles]
                particles_group.create_dataset("strain_rate", data=sr_slice.reshape(n_particles, 9))
                strain_rate_sym6 = np.stack(
                    [
                        sr_slice[:, 0, 0],
                        sr_slice[:, 1, 1],
                        sr_slice[:, 2, 2],
                        sr_slice[:, 0, 1],
                        sr_slice[:, 1, 2],
                        sr_slice[:, 0, 2],
                    ],
                    axis=1,
                ).astype(np.float32)
                particles_group.create_dataset("strain_rate_sym6", data=strain_rate_sym6)
            else:
                print(
                    f"(Info) Warning: strain_rate field size mismatch "
                    f"({sr_data.shape[0]} < {n_particles}), skipping backup"
                )
        except (AttributeError, RuntimeError, ValueError) as e:
            print(f"(Info) Warning: strain_rate tensor field not available for backup: {e}")

        try:
            bg_vel = solver.background_velocity
            particles_group.create_dataset(
                "background_velocity", data=np.tile(bg_vel, (n_particles, 1))
            )
        except Exception as e:
            print(f"(Info) Warning: Could not save background velocity: {e}")

        try:
            if hasattr(solver, "physics") and hasattr(solver.physics, "get_total_enstrophy"):
                enstrophy = solver.physics.get_total_enstrophy(
                    solver.particles.position.to_numpy()[:n_particles],
                    solver.particles.circulation.to_numpy()[:n_particles],
                    solver.particles.radius.to_numpy()[:n_particles],
                )
                particles_group.create_dataset("total_enstrophy", data=enstrophy)
        except (AttributeError, RuntimeError) as e:
            print(f"(Info) Warning: Could not compute enstrophy for backup: {e}")

    @staticmethod
    def _save_numerical_data(solver, hdf5_file: str, time_float32: float) -> None:
        """Save all numerical data to HDF5 with consistent float32 time."""
        with h5py.File(hdf5_file, "w") as f:
            solver_group = f.create_group("solver")
            solver_group.attrs["flow_time"] = time_float32
            solver_group.attrs["time_step"] = solver.time_step
            solver_group.attrs["time_step_size"] = solver.time_step_size
            solver_group.attrs["number_of_particles"] = solver.particles.number_of_particles

            particles_group = f.create_group("particles")
            n_particles = solver.particles.number_of_particles

            if n_particles == 0:
                return

            for name in (
                "position",
                "velocity",
                "circulation",
                "radius",
                "volume",
                "viscosity",
                "viscosity_turbulent",
                "group_id",
                "vorticity",
            ):
                particles_group.create_dataset(
                    name, data=getattr(solver.particles, name).to_numpy()[:n_particles]
                )
            BackupSystem._save_particle_optional_fields(particles_group, solver, n_particles)

    @staticmethod
    def _save_configuration(solver, config_file: str) -> None:
        """Save solver configuration and metadata to JSON."""
        config_data = {
            "solver_config": solver.config.to_dict(),
            "backup_metadata": {
                "backup_format_version": "2.1",  # Bumped for background velocity support
                "original_backend": solver.config.processing_unit,
                "openonda_version": getattr(solver, "version", "unknown"),
                "particle_count": int(
                    solver.particles.number_of_particles
                ),  # Convert to Python int
                "flow_time": float(solver.flow_time),
                "time_step": int(solver.time_step),  # Convert to Python int
            },
        }

        # Ensure directory for config file exists as well (same directory as HDF5)
        config_dir = os.path.dirname(config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _create_xdmf_file(
        solver, backup_file_name: str, xdmf_file: str, time_float32: float
    ) -> None:
        """Create XDMF file that references the HDF5 data for ParaView visualization.

        Only datasets that actually exist in the HDF5 file are referenced,
        so ParaView never tries to read a missing dataset.
        """
        n_particles = solver.particles.number_of_particles

        if n_particles == 0:
            print("(Info) Warning: No particles to export to XDMF")
            return

        h5_file = f"{backup_file_name}.h5"
        h5_basename = os.path.basename(h5_file)

        # Detect which optional datasets are present in the HDF5 file
        optional_strain_rate_attr = ""
        optional_zone_id_attr = ""
        try:
            with h5py.File(h5_file, "r") as hf:
                particle_keys = set(hf["particles"].keys()) if "particles" in hf else set()
            # Strain-rate: use flat (N,9) tensor — XDMF2 Tensor type is
            # compatible with both vtkXdmfReader and vtkXdmf3ReaderT in
            # ParaView, unlike Tensor6 which requires the XDMF3 reader.
            if "strain_rate" in particle_keys:
                optional_strain_rate_attr = f'''
      <!-- Strain-rate tensor (row-major 3x3: xx,xy,xz,yx,yy,yz,zx,zy,zz). -->
      <Attribute Name="StrainRate" AttributeType="Tensor" Center="Node">
        <DataItem Dimensions="{n_particles} 9" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/strain_rate
        </DataItem>
      </Attribute>'''
            if "zone_id" in particle_keys:
                optional_zone_id_attr = f'''
      <Attribute Name="ZoneID" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Int" Format="HDF">
          {h5_basename}:/particles/zone_id
        </DataItem>
      </Attribute>'''
        except Exception:
            pass

        # XDMF template for point-cloud particles using XDMF3 format.
        xdmf_content = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="VortexParticles" GridType="Uniform">
      <Topology TopologyType="Polyvertex" NumberOfElements="{n_particles}"/>

      <!-- Particle positions -->
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/position
        </DataItem>
      </Geometry>

      <!-- Time information -->
      <Time Value="{time_float32:.6g}"/>

      <!-- Particle attributes -->
      <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/velocity
        </DataItem>
      </Attribute>

      <Attribute Name="BackgroundVelocity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/background_velocity
        </DataItem>
      </Attribute>

      <Attribute Name="Circulation" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/circulation
        </DataItem>
      </Attribute>

      <Attribute Name="Vorticity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/vorticity
        </DataItem>
      </Attribute>

      <Attribute Name="Radius" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/radius
        </DataItem>
      </Attribute>

      <Attribute Name="Volume" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/volume
        </DataItem>
      </Attribute>

      <Attribute Name="Viscosity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/viscosity
        </DataItem>
      </Attribute>

      <Attribute Name="ViscosityTurbulent" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Float" Precision="4" Format="HDF">
          {h5_basename}:/particles/viscosity_turbulent
        </DataItem>
      </Attribute>

      <Attribute Name="GroupID" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles}" NumberType="Int" Format="HDF">
          {h5_basename}:/particles/group_id
        </DataItem>
      </Attribute>
{optional_zone_id_attr}
{optional_strain_rate_attr}

    </Grid>
  </Domain>
</Xdmf>'''

        # Ensure directory for XDMF exists too
        xdmf_dir = os.path.dirname(xdmf_file)
        if xdmf_dir:
            os.makedirs(xdmf_dir, exist_ok=True)

        # Write XDMF file
        with open(xdmf_file, "w") as f:
            f.write(xdmf_content)

    @staticmethod
    def create_temporal_xdmf(backup_pattern: str, output_file: str | None = None) -> str:
        """
        Create a temporal XDMF file that references multiple timesteps for animation.

        Args:
            backup_pattern: Pattern like "Results/*/Raw/simulation_t*" (without .h5 extension)
            output_file: Output XDMF filename (None = auto-generate)

        Returns:
            Path to the created temporal XDMF file
        """
        import glob

        # Find all matching backup files
        h5_files = glob.glob(f"{backup_pattern}.h5")
        h5_files.sort()  # Ensure chronological order

        if not h5_files:
            raise FileNotFoundError(f"No backup files found matching pattern: {backup_pattern}.h5")

        # Auto-generate output filename if not provided
        if output_file is None:
            base_pattern = backup_pattern.replace("*", "series")
            output_file = f"{base_pattern}_temporal.xdmf"

        # Ensure directory for output XDMF exists
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Create temporal XDMF content
        xdmf_content = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="VortexParticles_TimeSeries" GridType="Collection" CollectionType="Temporal">
"""

        # Add each timestep
        for h5_file in h5_files:
            # Load time information
            with h5py.File(h5_file, "r") as f:
                flow_time = float(f["solver"].attrs["flow_time"])
                time_step = int(f["solver"].attrs["time_step"])
                n_particles = int(f["solver"].attrs["number_of_particles"])
                particle_keys = set(f["particles"].keys()) if "particles" in f else set()

            h5_basename = os.path.basename(h5_file)

            # Optional attributes depend on which datasets are present in this snapshot
            _optional = ""
            if "strain_rate" in particle_keys:
                _optional += f'''
        <Attribute Name="StrainRate" AttributeType="Tensor" Center="Node">
          <DataItem Dimensions="{n_particles} 9" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/strain_rate
          </DataItem>
        </Attribute>'''
            if "zone_id" in particle_keys:
                _optional += f'''
        <Attribute Name="ZoneID" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{n_particles}" NumberType="Int" Format="HDF">
            {h5_basename}:/particles/zone_id
          </DataItem>
        </Attribute>'''

            xdmf_content += f'''
      <Grid Name="TimeStep_{time_step:06d}" GridType="Uniform">
        <Topology TopologyType="Polyvertex" NumberOfElements="{n_particles}"/>

        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/position
          </DataItem>
        </Geometry>

        <Time Value="{flow_time}"/>

        <!-- Key attributes for animation -->
        <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/velocity
          </DataItem>
        </Attribute>

        <Attribute Name="BackgroundVelocity" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/background_velocity
          </DataItem>
        </Attribute>

        <Attribute Name="Circulation" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/circulation
          </DataItem>
        </Attribute>

        <Attribute Name="Vorticity" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="{n_particles} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_basename}:/particles/vorticity
          </DataItem>
        </Attribute>
{_optional}
      </Grid>'''

        xdmf_content += """
    </Grid>
  </Domain>
</Xdmf>"""

        # Write temporal XDMF file
        with open(output_file, "w") as f:
            f.write(xdmf_content)

        print(f"Created temporal XDMF file: {output_file}")
        print(f"Contains {len(h5_files)} timesteps")

        return output_file

    @staticmethod
    def restore_solver(backup_file_name: str):
        """
        Restore a complete solver instance from HDF5 + JSON backup.

        Args:
            backup_file_name: Base filename (without extension)

        Returns:
            Fully restored Solver instance
        """
        try:
            # Generate filenames
            hdf5_file = f"{backup_file_name}.h5"
            config_file = f"{backup_file_name}_config.json"

            # Verify files exist
            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Numerical data file not found: {hdf5_file}")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            # 1. Load configuration
            config = BackupSystem._load_configuration(config_file)

            # 2. Create new solver with exact configuration
            from ..core.solver import Solver  # Import here to avoid circular dependency

            solver = Solver(config=config)

            # 3. Load numerical data with full precision
            BackupSystem._load_numerical_data(solver, hdf5_file)

            print(f"Solver restored from: {hdf5_file}")
            print(f"Configuration loaded from: {config_file}")

            return solver

        except Exception as e:
            raise RuntimeError(f"Restore failed: {e}") from e

    @staticmethod
    def _load_configuration(config_file: str) -> SolverConfig:
        """Load and validate solver configuration from JSON."""
        with open(config_file) as f:
            data = json.load(f)

        # Extract SolverConfig data
        config_dict = data["solver_config"]

        # Validate backup format
        metadata = data.get("backup_metadata", {})
        format_version = metadata.get("backup_format_version", "1.0")

        if format_version < "2.0":
            print(f"(Info) Warning: Loading older backup format {format_version}")

        # Create SolverConfig instance with validation
        config = SolverConfig(**config_dict)

        print(f"Configuration validated (format v{format_version})")
        return config

    @staticmethod
    def _load_optional_particle_fields(particles_group) -> dict:
        """Load optional advanced particle fields; returns dict of available arrays."""
        result = {"zone_id": None, "velocity_gradient": None, "strain_rate": None}
        if "zone_id" in particles_group:
            result["zone_id"] = particles_group["zone_id"][:]
        if "velocity_gradient" in particles_group:
            vg = particles_group["velocity_gradient"][:]
            result["velocity_gradient"] = (
                vg.reshape(-1, 3, 3) if vg.ndim == 2 and vg.shape[1] == 9 else vg
            )
        if "strain_rate" in particles_group:
            sr = particles_group["strain_rate"][:]
            result["strain_rate"] = (
                sr.reshape(-1, 3, 3) if sr.ndim == 2 and sr.shape[1] == 9 else sr
            )
        if "total_enstrophy" in particles_group:
            print(f"(Info) Restored enstrophy value: {float(particles_group['total_enstrophy'])}")
        return result

    @staticmethod
    def _load_numerical_data(solver, hdf5_file: str) -> None:
        """Load all numerical data from HDF5 with full precision."""
        with h5py.File(hdf5_file, "r") as f:
            solver_group = f["solver"]
            solver.flow_time = float(solver_group.attrs["flow_time"])
            solver.time_step = int(solver_group.attrs["time_step"])
            solver.time_step_size = float(solver_group.attrs["time_step_size"])

            particles_group = f["particles"]
            n_particles = int(solver_group.attrs["number_of_particles"])

            if n_particles == 0:
                return

            position = particles_group["position"][:]
            velocity = particles_group["velocity"][:]
            circulation = particles_group["circulation"][:]
            radius = particles_group["radius"][:]
            volume = particles_group["volume"][:]
            viscosity = particles_group["viscosity"][:]
            viscosity_turbulent = particles_group["viscosity_turbulent"][:]
            group_id = particles_group["group_id"][:]
            vorticity = particles_group["vorticity"][:]

            optional = BackupSystem._load_optional_particle_fields(particles_group)

            solver.add_vortex_particles(
                position=position,
                velocity=velocity,
                circulation=circulation,
                radius=radius,
                volume=volume,
                viscosity=viscosity,
                viscosity_turbulent=viscosity_turbulent,
                group_id=group_id,
                velocity_gradient=optional["velocity_gradient"],
                zone_id=optional["zone_id"],
            )

            if vorticity is not None:
                solver.particles.vorticity.from_numpy(vorticity)
            if optional["strain_rate"] is not None:
                solver.particles.strain_rate.from_numpy(optional["strain_rate"])

            print(f"Restored {solver.particles.number_of_particles} particles with full precision")

    @staticmethod
    def _validate_hdf5_structure(hdf5_file: str) -> bool:
        """Check that the HDF5 file has valid structure and required attributes."""
        required_attrs = ["flow_time", "time_step", "number_of_particles"]
        with h5py.File(hdf5_file, "r") as f:
            if "solver" not in f or "particles" not in f:
                return False
            return all(attr in f["solver"].attrs for attr in required_attrs)

    @staticmethod
    def validate_backup(backup_file_name: str) -> bool:
        """
        Validate the integrity of backup files.

        Args:
            backup_file_name: Base filename (without extension)

        Returns:
            True if backup is valid, False otherwise
        """
        try:
            hdf5_file = f"{backup_file_name}.h5"
            config_file = f"{backup_file_name}_config.json"
            xdmf_file = f"{backup_file_name}.xdmf"

            if not (os.path.exists(hdf5_file) and os.path.exists(config_file)):
                return False

            has_xdmf = os.path.exists(xdmf_file)

            if not BackupSystem._validate_hdf5_structure(hdf5_file):
                return False

            with open(config_file) as f:
                data = json.load(f)
                if "solver_config" not in data:
                    return False

            print(f"(Info) Backup validation passed: {backup_file_name}")
            if has_xdmf:
                print(f"(Info) ParaView XDMF file available: {xdmf_file}")
            else:
                print("(Info) ParaView XDMF file missing (can be regenerated)")

            return True

        except Exception as e:
            print(f"(Info) Backup validation failed: {e}")
            return False
