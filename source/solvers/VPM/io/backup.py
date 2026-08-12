"""
Checkpoint/restart backup system for VPM simulations (BackupSystem).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from ..config.constants import *  # noqa: F403
from ..config.types import VPMSetup


def _atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _stabilization(solver):
    """The stabilization master, or ``None`` for a solver stand-in without one."""
    return getattr(solver, "stabilization", None)


class BackupSystem:
    """Atomic HDF5 snapshots and JSON-backed restart checkpoints."""

    @staticmethod
    def load_numerical_state(solver, hdf5_file: str | Path) -> None:
        path = str(hdf5_file)
        if not BackupSystem._validate_hdf5_structure(path):
            raise ValueError(f"invalid VPM checkpoint: {path}")
        stabilization = _stabilization(solver)
        reference_strengths = getattr(stabilization, "reference_strengths", None)
        reference_lengths = getattr(stabilization, "reference_lengths", None)
        if solver.particles.number_of_particles:
            solver.remove_particles(remove_all=True)
        if reference_strengths is not None and reference_lengths is not None:
            stabilization.reference_strengths = reference_strengths
            stabilization.reference_lengths = reference_lengths
        BackupSystem._load_numerical_data(solver, path)

    @staticmethod
    def backup_solver(
        solver,
        backup_file_name: str,
        flow_time: float | None = None,
        *,
        append_step: bool = True,
        verbose: bool = True,
    ) -> None:
        """Write an HDF5 particle snapshot and its XDMF descriptor.

        Args:
            solver: The Solver instance to backup
            backup_file_name: Base filename (without extension)
            flow_time: Optional authoritative simulation time.
            append_step: Append the zero-padded solver step to the filename.
            verbose: Whether to print backup completion message
        """
        try:
            time_value = float(solver.flow_time if flow_time is None else flow_time)
            backup_base = backup_file_name
            if append_step:
                backup_base = f"{backup_base}_{int(getattr(solver, 'time_step', 0)):06d}"
            hdf5_file = f"{backup_base}.h5"
            xdmf_file = f"{backup_base}.xdmf"

            backup_dir = os.path.dirname(hdf5_file)
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)

            hdf5_tmp = f"{hdf5_file}.tmp"
            try:
                if os.path.exists(hdf5_tmp):
                    os.remove(hdf5_tmp)
                BackupSystem._save_numerical_data(solver, hdf5_tmp, time_value)
                os.replace(hdf5_tmp, hdf5_file)
            finally:
                if os.path.exists(hdf5_tmp):
                    os.remove(hdf5_tmp)

            BackupSystem._create_xdmf_file(solver, backup_base, xdmf_file, time_value)

            if verbose:
                print(f"Snapshot saved: {hdf5_file}")

        except Exception as e:
            raise RuntimeError(f"Backup failed: {e}") from e

    @staticmethod
    def _save_particle_optional_fields(particles_group, solver, n_particles: int) -> None:
        """Save optional/advanced particle fields to HDF5, silently skipping unavailable ones."""
        stabilization = _stabilization(solver)
        reference_strengths = getattr(stabilization, "reference_strengths", None)
        reference_lengths = getattr(stabilization, "reference_lengths", None)
        if (
            reference_strengths is not None
            and reference_lengths is not None
            and len(reference_strengths) == n_particles
            and len(reference_lengths) == n_particles
        ):
            particles_group.create_dataset(
                "filament_reference_strength",
                data=np.asarray(reference_strengths),
            )
            particles_group.create_dataset(
                "filament_reference_length",
                data=np.asarray(reference_lengths),
            )

        try:
            zone_id_data = solver.particles.zone_id_cpu()
            particles_group.create_dataset("zone_id", data=zone_id_data)
        except (AttributeError, Exception) as e:
            print(f"(Info) Warning: zone_id field not available for backup: {e}")

        try:
            vg_data = solver.particles.velocity_gradient_cpu()
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
            sr_data = solver.particles.strain_rate_cpu()
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
                    solver.particles.position_cpu(),
                    solver.particles.circulation_cpu(),
                    solver.particles.radius_cpu(),
                )
                particles_group.create_dataset("total_enstrophy", data=enstrophy)
        except (AttributeError, RuntimeError) as e:
            print(f"(Info) Warning: Could not compute enstrophy for backup: {e}")

    @staticmethod
    def _save_numerical_data(solver, hdf5_file: str, flow_time: float) -> None:
        """Save all numerical data to HDF5."""
        with h5py.File(hdf5_file, "w") as f:
            solver_group = f.create_group("solver")
            solver_group.attrs["flow_time"] = flow_time
            solver_group.attrs["time_step"] = solver.time_step
            solver_group.attrs["time_step_size"] = solver.time_step_size
            # DVH fires once every _dvh_substeps steps off this counter; without it
            # a restart resumes at phase 0 and the viscous update lands on different
            # steps than the uninterrupted run.
            solver_group.attrs["dvh_fire_counter"] = int(getattr(solver, "_dvh_fire_counter", 0))
            solver_group.attrs["number_of_particles"] = solver.particles.number_of_particles
            stabilization = _stabilization(solver)
            for name, value in getattr(stabilization, "diagnostics", {}).items():
                solver_group.attrs[name] = value
            reference_moments = getattr(stabilization, "reference_moments", None)
            if reference_moments is not None:
                reference_array = np.asarray(reference_moments, dtype=np.float64)
                if reference_array.shape != (3, 3):
                    raise ValueError(
                        "divergence-relaxation reference moments must have shape (3, 3)"
                    )
                solver_group.create_dataset(
                    "divergence_relaxation_reference_moments",
                    data=reference_array,
                )

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
                    name, data=getattr(solver.particles, f"{name}_cpu")()
                )
            BackupSystem._save_particle_optional_fields(particles_group, solver, n_particles)

    @staticmethod
    def _save_configuration(solver, config_file: str) -> None:
        """Save solver configuration and metadata to JSON."""
        config_data = {
            "solver_config": solver.config.to_dict(),
            "backup_metadata": {
                "backup_format_version": "2.2",
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

        _atomic_write_text(
            config_file,
            json.dumps(config_data, indent=2, ensure_ascii=False) + "\n",
        )

    @staticmethod
    def _create_xdmf_file(solver, backup_file_name: str, xdmf_file: str, flow_time: float) -> None:
        """Create XDMF file that references the HDF5 data for ParaView visualization.

        Only datasets that actually exist in the HDF5 file are referenced,
        so ParaView never tries to read a missing dataset.
        """
        n_particles = solver.particles.number_of_particles

        if n_particles == 0:
            if os.path.exists(xdmf_file):
                os.remove(xdmf_file)
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
      <Time Value="{flow_time:.17g}"/>

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

        _atomic_write_text(xdmf_file, xdmf_content)

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

        _atomic_write_text(output_file, xdmf_content)

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
            config_file = f"{backup_file_name}.config.json"
            legacy_config_file = f"{backup_file_name}_config.json"
            if not os.path.exists(config_file) and os.path.exists(legacy_config_file):
                config_file = legacy_config_file

            # Verify files exist
            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Numerical data file not found: {hdf5_file}")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            # 1. Load configuration
            config = BackupSystem._load_configuration(config_file)

            # 2. Create new solver with exact configuration
            from ..core.solver import Solver  # Import here to avoid circular dependency

            solver = Solver(setup=config)

            # 3. Load numerical data with full precision
            BackupSystem._load_numerical_data(solver, hdf5_file)

            print(f"Solver restored from: {hdf5_file}")
            print(f"Configuration loaded from: {config_file}")

            return solver

        except Exception as e:
            raise RuntimeError(f"Restore failed: {e}") from e

    @staticmethod
    def _load_configuration(config_file: str) -> VPMSetup:
        """Load and validate solver configuration from JSON."""
        with open(config_file) as f:
            data = json.load(f)

        # Extract VPMSetup data
        config_dict = data["solver_config"]

        # Validate backup format
        metadata = data.get("backup_metadata", {})
        format_version = metadata.get("backup_format_version", "1.0")

        if format_version < "2.0":
            print(f"(Info) Warning: Loading older backup format {format_version}")

        # Create VPMSetup instance with validation
        config = VPMSetup.from_dict(config_dict)

        print(f"Configuration validated (format v{format_version})")
        return config

    @staticmethod
    def _load_optional_particle_fields(particles_group) -> dict:
        """Load optional advanced particle fields; returns dict of available arrays."""
        result = {
            "zone_id": None,
            "velocity_gradient": None,
            "strain_rate": None,
            "filament_reference_strength": None,
            "filament_reference_length": None,
        }
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
        if "filament_reference_strength" in particles_group:
            result["filament_reference_strength"] = particles_group["filament_reference_strength"][
                :
            ]
        if "filament_reference_length" in particles_group:
            result["filament_reference_length"] = particles_group["filament_reference_length"][:]
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
            if "dvh_fire_counter" in solver_group.attrs:
                solver._dvh_fire_counter = int(solver_group.attrs["dvh_fire_counter"])
            stabilization = _stabilization(solver)
            if stabilization is not None:
                stabilization.restore_diagnostics(
                    {
                        name: value.item() if hasattr(value, "item") else value
                        for name, value in solver_group.attrs.items()
                        if name.startswith("stabilization_")
                    }
                )
            if "divergence_relaxation_reference_moments" in solver_group:
                reference_array = np.asarray(
                    solver_group["divergence_relaxation_reference_moments"][:],
                    dtype=np.float64,
                )
                if reference_array.shape != (3, 3):
                    raise ValueError(
                        "checkpoint divergence-relaxation reference moments must have shape (3, 3)"
                    )
                if stabilization is not None:
                    stabilization.reference_moments = tuple(row.copy() for row in reference_array)

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

            solver._loading_numerical_state = True
            try:
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
            finally:
                solver._loading_numerical_state = False

            if vorticity is not None:
                solver.particles.set_field("vorticity", vorticity)
            if optional["strain_rate"] is not None:
                solver.particles.set_field("strain_rate", optional["strain_rate"])
            saved_reference_strength = optional["filament_reference_strength"]
            saved_reference_length = optional["filament_reference_length"]
            stabilization = _stabilization(solver)
            if (
                saved_reference_strength is not None
                and saved_reference_length is not None
                and stabilization is not None
            ):
                stabilization.reference_strengths = np.asarray(
                    saved_reference_strength,
                    dtype=np.float64,
                )
                stabilization.reference_lengths = np.asarray(
                    saved_reference_length,
                    dtype=np.float64,
                )
            elif solver.stabilization_config.filament_refinement.enabled:
                references = getattr(stabilization, "reference_strengths", None)
                lengths = getattr(stabilization, "reference_lengths", None)
                if references is None or lengths is None or len(references) != n_particles:
                    raise ValueError(
                        "checkpoint has no filament-lineage state and its particle "
                        "count does not match the initialized cloud; restarting an "
                        "already-refined legacy checkpoint would reset its material "
                        "stretch history"
                    )

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
            config_file = f"{backup_file_name}.config.json"
            legacy_config_file = f"{backup_file_name}_config.json"
            if not os.path.exists(config_file) and os.path.exists(legacy_config_file):
                config_file = legacy_config_file
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
