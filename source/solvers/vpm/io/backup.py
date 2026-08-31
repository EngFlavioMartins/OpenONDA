"""Backup/restart I/O for VPM simulations.

Backups use the same canonical names as the live VPM state. Readers reject
every backup format other than the current canonical contract.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from source.write_precision import DEFAULT_WRITE_PRECISION, cast_for_write, storage_dtype

from .logging import Logging

_BACKUP_FORMAT_VERSION = "9.0"
_COMPRESSION = {
    "chunks": True,
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
}
_STABILIZATION_DIAGNOSTIC_NAMES = (
    "n_stabilization_events",
    "n_regularization_events",
    "last_stabilization_mechanism",
    "stabilization_vortex_strength_error",
    "stabilization_vortex_strength_growth",
    "stabilization_vorticity_growth",
    "max_stabilization_vorticity_growth",
    "lagrangian_cfl",
    "stretching_viscosity_feedback_coefficient",
)


def _atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a text file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _stabilization(solver: Any):
    """Return the stabilization manager when present."""
    return getattr(solver, "stabilization", None)


def _read_attribute(group: h5py.Group, canonical_name: str) -> Any:
    if canonical_name not in group.attrs:
        raise KeyError(f"Backup is missing solver attribute {canonical_name!r}")
    return group.attrs[canonical_name]


def _read_particle_count(group: h5py.Group) -> int:
    """Read the canonical particle count."""
    return int(_read_attribute(group, "n_particles_total"))


def _read_dataset(
    group: h5py.Group,
    canonical_name: str,
    *,
    required: bool = True,
):
    if canonical_name not in group:
        if not required:
            return None
        raise KeyError(f"Backup is missing particle field {canonical_name!r}")
    return group[canonical_name][:]


class _BackupIO:
    """Read and write VPM restart backups."""

    @staticmethod
    def load(
        solver,
        hdf5_file: str | Path,
    ) -> None:
        """Replace ``solver`` state from an HDF5 backup."""
        path = str(hdf5_file)
        if not _BackupIO._validate_hdf5_structure(path):
            raise ValueError(f"Invalid VPM backup: {path}")

        stabilization = _stabilization(solver)
        reference_vortex_strength = getattr(
            stabilization,
            "reference_vortex_strength",
            None,
        )
        reference_lengths = getattr(
            stabilization,
            "reference_lengths",
            None,
        )

        if solver.particles.n_particles_total:
            solver.remove_particles(remove_all=True)

        if (
            stabilization is not None
            and reference_vortex_strength is not None
            and reference_lengths is not None
        ):
            stabilization.reference_vortex_strength = reference_vortex_strength
            stabilization.reference_lengths = reference_lengths

        _BackupIO._load_numerical_data(solver, path)

    @staticmethod
    def save(
        solver,
        backup_path: str | Path,
        time: float | None = None,
        *,
        append_step: bool = True,
        verbose: bool = True,
    ) -> None:
        """Write HDF5 restart state and its XDMF descriptor."""
        try:
            time_value = float(solver.time if time is None else time)
            backup_base = str(backup_path)
            if append_step:
                backup_base = f"{backup_base}_{int(solver.step):06d}"

            hdf5_file = f"{backup_base}.h5"
            xdmf_file = f"{backup_base}.xdmf"
            Path(hdf5_file).parent.mkdir(parents=True, exist_ok=True)

            temporary_hdf5 = f"{hdf5_file}.tmp"
            try:
                if os.path.exists(temporary_hdf5):
                    os.remove(temporary_hdf5)
                _BackupIO._write_numerical_data(
                    solver,
                    temporary_hdf5,
                    time_value,
                )
                os.replace(temporary_hdf5, hdf5_file)
            finally:
                if os.path.exists(temporary_hdf5):
                    os.remove(temporary_hdf5)

            _BackupIO._write_xdmf(
                solver,
                backup_base,
                xdmf_file,
                time_value,
            )

            if verbose:
                Logging.record(
                    "backup written",
                    ("step", f"{solver.step:,}"),
                    ("time", f"{time_value:.6e}", "s"),
                    ("particles", f"{solver.particles.n_particles_total:,}"),
                    ("path", str(hdf5_file)),
                )
        except Exception as exc:
            raise RuntimeError(f"Backup write failed: {exc}") from exc

    @staticmethod
    def _write_optional_particle_fields(
        particles_group: h5py.Group,
        solver,
        n_particles_total: int,
        write_precision: str,
    ) -> None:
        stabilization = _stabilization(solver)
        reference_vortex_strength = getattr(
            stabilization,
            "reference_vortex_strength",
            None,
        )
        reference_lengths = getattr(
            stabilization,
            "reference_lengths",
            None,
        )
        if (
            reference_vortex_strength is not None
            and reference_lengths is not None
            and len(reference_vortex_strength) == n_particles_total
            and len(reference_lengths) == n_particles_total
        ):
            particles_group.create_dataset(
                "filament_reference_vortex_strength",
                data=cast_for_write(reference_vortex_strength, write_precision),
                **_COMPRESSION,
            )
            particles_group.create_dataset(
                "filament_reference_length",
                data=cast_for_write(reference_lengths, write_precision),
                **_COMPRESSION,
            )

        particles_group.create_dataset(
            "zone_id",
            data=solver.particles.zone_id_cpu(),
            **_COMPRESSION,
        )

        if (
            n_particles_total > 0
            and hasattr(solver, "physics")
            and hasattr(solver.physics, "get_total_enstrophy")
        ):
            total_enstrophy = solver.physics.get_total_enstrophy(
                solver.particles.position_cpu(),
                solver.particles.vortex_strength_cpu(),
                solver.particles.core_radius_cpu(),
            )
            particles_group.create_dataset(
                "total_enstrophy",
                data=cast_for_write(total_enstrophy, write_precision),
                **_COMPRESSION,
            )

    @staticmethod
    def _write_numerical_data(
        solver,
        hdf5_file: str,
        time: float,
    ) -> None:
        """Write canonical solver and particle state."""
        write_precision = getattr(solver, "write_precision", DEFAULT_WRITE_PRECISION)
        with h5py.File(hdf5_file, "w") as file:
            solver_group = file.create_group("solver")
            solver_group.attrs["backup_format_version"] = _BACKUP_FORMAT_VERSION
            solver_group.attrs["write_precision"] = write_precision
            solver_group.attrs["freestream_velocity"] = np.asarray(
                solver.freestream_velocity,
                dtype=np.float64,
            )
            solver_group.attrs["time"] = time
            solver_group.attrs["step"] = int(solver.step)
            solver_group.attrs["time_step_size"] = float(solver.time_step_size)
            solver_group.attrs["n_steps_since_dvh_diffusion"] = int(
                solver._n_steps_since_dvh_diffusion
            )
            solver_group.attrs["is_particle_regeneration_pending"] = int(
                solver._is_particle_regeneration_pending
            )
            solver_group.attrs["n_particles_total"] = int(solver.particles.n_particles_total)

            stabilization = _stabilization(solver)
            for name, value in getattr(
                stabilization,
                "diagnostics",
                {},
            ).items():
                if name not in _STABILIZATION_DIAGNOSTIC_NAMES:
                    raise ValueError(f"Unknown stabilization diagnostic {name!r}")
                solver_group.attrs[name] = value

            reference_moments = getattr(
                stabilization,
                "reference_moments",
                None,
            )
            if reference_moments is not None:
                reference_array = np.asarray(
                    reference_moments,
                    dtype=np.float64,
                )
                if reference_array.shape != (3, 3):
                    raise ValueError(
                        "divergence-relaxation reference moments must have shape (3, 3)"
                    )
                solver_group.create_dataset(
                    "divergence_relaxation_reference_moments",
                    data=reference_array,
                )

            particles_group = file.create_group("particles")
            n_particles_total = int(solver.particles.n_particles_total)

            for name in (
                "position",
                "velocity",
                "vortex_strength",
                "core_radius",
                "particle_volume",
                "kinematic_viscosity",
                "eddy_viscosity",
                "effective_viscosity",
                "group_id",
                "vorticity",
            ):
                particles_group.create_dataset(
                    name,
                    data=cast_for_write(
                        getattr(solver.particles, f"{name}_cpu")(),
                        write_precision,
                    ),
                    **_COMPRESSION,
                )

            _BackupIO._write_optional_particle_fields(
                particles_group,
                solver,
                n_particles_total,
                write_precision,
            )

    @staticmethod
    def _write_xdmf(
        solver,
        backup_base: str,
        xdmf_file: str,
        time: float,
    ) -> None:
        """Write an XDMF descriptor using canonical field names."""
        n_particles_total = int(solver.particles.n_particles_total)
        hdf5_basename = os.path.basename(f"{backup_base}.h5")
        write_precision = getattr(solver, "write_precision", DEFAULT_WRITE_PRECISION)
        float_precision = storage_dtype(write_precision).itemsize
        optional_parts = [
            f"""
      <Attribute Name="zone_id" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Int" Format="HDF">
          {hdf5_basename}:/particles/zone_id
        </DataItem>
      </Attribute>"""
        ]

        optional = "\n".join(optional_parts)

        xdmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="vortex_particles" GridType="Uniform">
      <Topology TopologyType="Polyvertex" NumberOfElements="{n_particles_total}"/>

      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/position
        </DataItem>
      </Geometry>

      <Time Value="{time:.17g}"/>
      <Attribute Name="velocity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/velocity
        </DataItem>
      </Attribute>

      <Attribute Name="vortex_strength" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/vortex_strength
        </DataItem>
      </Attribute>

      <Attribute Name="vorticity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/vorticity
        </DataItem>
      </Attribute>

      <Attribute Name="core_radius" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/core_radius
        </DataItem>
      </Attribute>

      <Attribute Name="particle_volume" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/particle_volume
        </DataItem>
      </Attribute>

      <Attribute Name="kinematic_viscosity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/kinematic_viscosity
        </DataItem>
      </Attribute>

      <Attribute Name="eddy_viscosity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/eddy_viscosity
        </DataItem>
      </Attribute>

      <Attribute Name="effective_viscosity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/effective_viscosity
        </DataItem>
      </Attribute>

      <Attribute Name="group_id" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Int" Format="HDF">
          {hdf5_basename}:/particles/group_id
        </DataItem>
      </Attribute>
{optional}
    </Grid>
  </Domain>
</Xdmf>"""
        _atomic_write_text(xdmf_file, xdmf_content)

    @staticmethod
    def create_temporal_xdmf(
        backup_pattern: str,
        output_file: str | None = None,
    ) -> str:
        """Create an XDMF temporal collection from canonical HDF5 files."""
        hdf5_files = sorted(glob.glob(f"{backup_pattern}.h5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No backup files found matching {backup_pattern}.h5")

        if output_file is None:
            output_file = f"{backup_pattern.replace('*', 'series')}_temporal.xdmf"
        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        grids: list[str] = []
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as file:
                solver_group = file["solver"]
                particles_group = file["particles"]
                time = float(_read_attribute(solver_group, "time"))
                step = int(_read_attribute(solver_group, "step"))
                n_particles_total = _read_particle_count(solver_group)
                canonical_datasets = (
                    "position",
                    "velocity",
                    "vortex_strength",
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "effective_viscosity",
                    "group_id",
                    "vorticity",
                    "zone_id",
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                    "total_enstrophy",
                )
                stored = {
                    name: name if name in particles_group else None for name in canonical_datasets
                }
                float_precision = int(particles_group["position"].dtype.itemsize)

            hdf5_basename = os.path.basename(hdf5_file)

            def data_item(
                canonical: str,
                dimensions: str,
                *,
                number_type: str = "Float",
                stored: dict[str, str | None] = stored,
                hdf5_basename: str = hdf5_basename,
                float_precision: int = float_precision,
            ) -> str:
                stored_name = stored[canonical]
                if stored_name is None:
                    return ""
                precision = f' Precision="{float_precision}"' if number_type == "Float" else ""
                return (
                    f'<DataItem Dimensions="{dimensions}" '
                    f'NumberType="{number_type}"{precision} Format="HDF">'
                    f"{hdf5_basename}:/particles/{stored_name}"
                    "</DataItem>"
                )

            optional_parts: list[str] = []
            if stored["zone_id"] is not None:
                optional_parts.append(
                    f"""        <Attribute Name="zone_id" AttributeType="Scalar" Center="Node">
          {data_item("zone_id", str(n_particles_total), number_type="Int")}
        </Attribute>"""
                )
            optional_text = "\n".join(optional_parts)
            grids.append(
                f"""      <Grid Name="step_{step:06d}" GridType="Uniform">
        <Topology TopologyType="Polyvertex" NumberOfElements="{n_particles_total}"/>
        <Geometry GeometryType="XYZ">
          {data_item("position", f"{n_particles_total} 3")}
        </Geometry>
        <Time Value="{time:.17g}"/>
        <Attribute Name="velocity" AttributeType="Vector" Center="Node">
          {data_item("velocity", f"{n_particles_total} 3")}
        </Attribute>
        <Attribute Name="vortex_strength" AttributeType="Vector" Center="Node">
          {data_item("vortex_strength", f"{n_particles_total} 3")}
        </Attribute>
        <Attribute Name="vorticity" AttributeType="Vector" Center="Node">
          {data_item("vorticity", f"{n_particles_total} 3")}
        </Attribute>
{optional_text}
      </Grid>"""
            )

        content = (
            '<?xml version="1.0" ?>\n'
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
            '<Xdmf Version="3.0">\n'
            "  <Domain>\n"
            '    <Grid Name="vortex_particles_time_series" '
            'GridType="Collection" CollectionType="Temporal">\n'
            + "\n".join(grids)
            + "\n    </Grid>\n"
            "  </Domain>\n"
            "</Xdmf>\n"
        )
        _atomic_write_text(output_file, content)
        return output_file

    @staticmethod
    def _load_auxiliary_particle_fields(
        particles_group: h5py.Group,
    ) -> dict[str, np.ndarray | None]:
        """Load required auxiliary fields and optional filament lineage."""
        return {
            "zone_id": _read_dataset(
                particles_group,
                "zone_id",
            ),
            "effective_viscosity": _read_dataset(
                particles_group,
                "effective_viscosity",
            ),
            "filament_reference_vortex_strength": _read_dataset(
                particles_group,
                "filament_reference_vortex_strength",
                required=False,
            ),
            "filament_reference_length": _read_dataset(
                particles_group,
                "filament_reference_length",
                required=False,
            ),
        }

    @staticmethod
    def _load_numerical_data(
        solver,
        hdf5_file: str,
    ) -> None:
        """Load canonical HDF5 state without reducing precision."""
        with h5py.File(hdf5_file, "r") as file:
            solver_group = file["solver"]
            particles_group = file["particles"]

            solver.time = float(_read_attribute(solver_group, "time"))
            solver.step = int(_read_attribute(solver_group, "step"))
            solver.time_step_size = float(
                _read_attribute(
                    solver_group,
                    "time_step_size",
                )
            )
            solver._n_steps_since_dvh_diffusion = int(
                _read_attribute(solver_group, "n_steps_since_dvh_diffusion")
            )
            solver._is_particle_regeneration_pending = bool(
                _read_attribute(solver_group, "is_particle_regeneration_pending")
            )

            stabilization = _stabilization(solver)
            if stabilization is not None:
                stabilization.restore_diagnostics(
                    {
                        name: (value.item() if hasattr(value, "item") else value)
                        for name, value in solver_group.attrs.items()
                        if name in _STABILIZATION_DIAGNOSTIC_NAMES
                    }
                )

            if "divergence_relaxation_reference_moments" in solver_group:
                reference_array = np.asarray(
                    solver_group["divergence_relaxation_reference_moments"][:],
                    dtype=np.float64,
                )
                if reference_array.shape != (3, 3):
                    raise ValueError(
                        "Backup divergence-relaxation reference moments must have shape (3, 3)"
                    )
                if stabilization is not None:
                    stabilization.reference_moments = tuple(row.copy() for row in reference_array)

            n_particles_total = _read_particle_count(solver_group)
            if n_particles_total == 0:
                return

            position = _read_dataset(particles_group, "position")
            velocity = _read_dataset(particles_group, "velocity")
            vortex_strength = _read_dataset(
                particles_group,
                "vortex_strength",
            )
            core_radius = _read_dataset(
                particles_group,
                "core_radius",
            )
            particle_volume = _read_dataset(particles_group, "particle_volume")
            kinematic_viscosity = _read_dataset(
                particles_group,
                "kinematic_viscosity",
            )
            eddy_viscosity = _read_dataset(
                particles_group,
                "eddy_viscosity",
            )
            group_id = _read_dataset(
                particles_group,
                "group_id",
            )
            vorticity = _read_dataset(
                particles_group,
                "vorticity",
            )

            auxiliary = _BackupIO._load_auxiliary_particle_fields(particles_group)

            solver._loading_numerical_state = True
            try:
                solver.add_vortex_particles(
                    position=position,
                    velocity=velocity,
                    vortex_strength=vortex_strength,
                    core_radius=core_radius,
                    particle_volume=particle_volume,
                    kinematic_viscosity=kinematic_viscosity,
                    eddy_viscosity=eddy_viscosity,
                    group_id=group_id,
                    zone_id=auxiliary["zone_id"],
                )
            finally:
                solver._loading_numerical_state = False

            solver.particles.set_field("vorticity", vorticity)
            if solver.flow_model != "POTENTIAL":
                solver.stepper._update_velocity_gradients(announce=False)
            if auxiliary["effective_viscosity"] is not None:
                solver.particles.set_field(
                    "effective_viscosity",
                    auxiliary["effective_viscosity"],
                )

            saved_reference_vortex_strength = auxiliary["filament_reference_vortex_strength"]
            saved_reference_length = auxiliary["filament_reference_length"]
            if (
                stabilization is not None
                and saved_reference_vortex_strength is not None
                and saved_reference_length is not None
            ):
                stabilization.reference_vortex_strength = np.asarray(
                    saved_reference_vortex_strength,
                    dtype=np.float64,
                )
                stabilization.reference_lengths = np.asarray(
                    saved_reference_length,
                    dtype=np.float64,
                )
            elif solver.stabilization_config.filament_refinement.enabled:
                references = getattr(
                    stabilization,
                    "reference_vortex_strength",
                    None,
                )
                lengths = getattr(
                    stabilization,
                    "reference_lengths",
                    None,
                )
                if references is None or lengths is None or len(references) != n_particles_total:
                    raise ValueError(
                        "Backup has no filament-lineage state compatible with this refined cloud"
                    )

    @staticmethod
    def _validate_hdf5_structure(
        hdf5_file: str | Path,
    ) -> bool:
        """Return whether a backup has the minimum restart structure."""
        try:
            with h5py.File(hdf5_file, "r") as file:
                if set(file.keys()) != {"solver", "particles"}:
                    return False

                solver_group = file["solver"]
                required_solver_attributes = {
                    "backup_format_version",
                    "write_precision",
                    "freestream_velocity",
                    "time",
                    "step",
                    "time_step_size",
                    "n_steps_since_dvh_diffusion",
                    "is_particle_regeneration_pending",
                    "n_particles_total",
                }
                solver_attribute_names = set(solver_group.attrs.keys())
                if not required_solver_attributes <= solver_attribute_names:
                    return False
                if not solver_attribute_names <= (
                    required_solver_attributes | set(_STABILIZATION_DIAGNOSTIC_NAMES)
                ):
                    return False
                if set(solver_group.keys()) - {"divergence_relaxation_reference_moments"}:
                    return False
                format_version = str(solver_group.attrs.get("backup_format_version", ""))
                if format_version != _BACKUP_FORMAT_VERSION:
                    return False

                n_particles_total = _read_particle_count(solver_group)
                if n_particles_total < 0:
                    return False
                particles_group = file["particles"]

                required = {
                    "position",
                    "velocity",
                    "vortex_strength",
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "group_id",
                    "vorticity",
                    "effective_viscosity",
                    "zone_id",
                }
                if n_particles_total == 0:
                    return set(particles_group) == required
                optional = {
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                    "total_enstrophy",
                }
                particle_field_names = set(particles_group.keys())
                if not required <= particle_field_names:
                    return False
                if not particle_field_names <= required | optional:
                    return False
                filament_fields = {
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                }
                if len(particle_field_names & filament_fields) == 1:
                    return False
                vector_fields = (
                    "position",
                    "velocity",
                    "vortex_strength",
                    "vorticity",
                )
                scalar_fields = (
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "group_id",
                    "zone_id",
                    "effective_viscosity",
                )
                if any(
                    particles_group[name].shape != (n_particles_total, 3) for name in vector_fields
                ):
                    return False
                if any(
                    particles_group[name].shape != (n_particles_total,) for name in scalar_fields
                ):
                    return False
                return True
        except (OSError, KeyError, ValueError):
            return False
