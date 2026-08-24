"""Checkpoint/restart I/O for VPM simulations.

Checkpoints use the same canonical names as the live VPM state. Readers reject
every checkpoint format other than the current canonical contract.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from ..config.setup import VPMSetup
from .logging import Logging

_CHECKPOINT_FORMAT_VERSION = "7.0"
_STABILIZATION_DIAGNOSTIC_NAMES = (
    "n_stabilization_events",
    "last_stabilization_mechanism",
    "stabilization_vortex_strength_error",
    "stabilization_vortex_strength_growth",
    "stabilization_vorticity_growth",
    "max_stabilization_vorticity_growth",
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
        raise KeyError(f"Checkpoint is missing solver attribute {canonical_name!r}")
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
        raise KeyError(f"Checkpoint is missing particle field {canonical_name!r}")
    return group[canonical_name][:]


def _reshape_tensor(array: np.ndarray | None) -> np.ndarray | None:
    """Restore a flattened ``(N, 9)`` tensor dataset to ``(N, 3, 3)``."""
    if array is None:
        return None
    if array.ndim == 2 and array.shape[1] == 9:
        return array.reshape(-1, 3, 3)
    return array


class CheckpointManager:
    """Read and write VPM restart checkpoints."""

    @staticmethod
    def load_numerical_state(
        solver,
        hdf5_file: str | Path,
    ) -> None:
        """Replace ``solver`` state from an HDF5 checkpoint."""
        path = str(hdf5_file)
        if not CheckpointManager._validate_hdf5_structure(path):
            raise ValueError(f"Invalid VPM checkpoint: {path}")

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

        CheckpointManager._load_numerical_data(solver, path)

    @staticmethod
    def write_checkpoint(
        solver,
        checkpoint_path: str | Path,
        time: float | None = None,
        *,
        append_step: bool = True,
        verbose: bool = True,
    ) -> None:
        """Write HDF5 restart state and its XDMF descriptor."""
        try:
            time_value = float(solver.time if time is None else time)
            checkpoint_base = str(checkpoint_path)
            if append_step:
                checkpoint_base = f"{checkpoint_base}_{int(solver.step):06d}"

            hdf5_file = f"{checkpoint_base}.h5"
            xdmf_file = f"{checkpoint_base}.xdmf"
            Path(hdf5_file).parent.mkdir(parents=True, exist_ok=True)

            temporary_hdf5 = f"{hdf5_file}.tmp"
            try:
                if os.path.exists(temporary_hdf5):
                    os.remove(temporary_hdf5)
                CheckpointManager._write_numerical_data(
                    solver,
                    temporary_hdf5,
                    time_value,
                )
                os.replace(temporary_hdf5, hdf5_file)
            finally:
                if os.path.exists(temporary_hdf5):
                    os.remove(temporary_hdf5)

            CheckpointManager._write_xdmf(
                solver,
                checkpoint_base,
                xdmf_file,
                time_value,
            )

            if verbose:
                Logging.message(
                    f"[VPM][Checkpoint] status=written step={solver.step} "
                    f"time_s={time_value:.6e} particles={solver.particles.n_particles_total} "
                    f"path={hdf5_file!r}"
                )
        except Exception as exc:
            raise RuntimeError(f"Checkpoint write failed: {exc}") from exc

    @staticmethod
    def _write_optional_particle_fields(
        particles_group: h5py.Group,
        solver,
        n_particles_total: int,
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
                data=np.asarray(reference_vortex_strength, dtype=np.float64),
            )
            particles_group.create_dataset(
                "filament_reference_length",
                data=np.asarray(reference_lengths, dtype=np.float64),
            )

        particles_group.create_dataset(
            "zone_id",
            data=solver.particles.zone_id_cpu(),
        )

        velocity_gradient = solver.particles.velocity_gradient_cpu()
        if velocity_gradient.shape[0] != n_particles_total:
            raise ValueError(
                "velocity_gradient particle count does not match "
                f"n_particles_total ({velocity_gradient.shape[0]} != {n_particles_total})"
            )
        particles_group.create_dataset(
            "velocity_gradient",
            data=velocity_gradient.reshape(n_particles_total, 9),
        )

        strain_rate = solver.particles.strain_rate_cpu()
        if strain_rate.shape[0] != n_particles_total:
            raise ValueError(
                "strain_rate particle count does not match "
                f"n_particles_total ({strain_rate.shape[0]} != {n_particles_total})"
            )
        particles_group.create_dataset(
            "strain_rate",
            data=strain_rate.reshape(n_particles_total, 9),
        )

        freestream_velocity = np.asarray(
            solver.freestream_velocity,
            dtype=solver.np_dtype,
        )
        particles_group.create_dataset(
            "freestream_velocity",
            data=np.tile(freestream_velocity, (n_particles_total, 1)),
        )

        if hasattr(solver, "physics") and hasattr(solver.physics, "get_total_enstrophy"):
            total_enstrophy = solver.physics.get_total_enstrophy(
                solver.particles.position_cpu(),
                solver.particles.vortex_strength_cpu(),
                solver.particles.core_radius_cpu(),
            )
            particles_group.create_dataset(
                "total_enstrophy",
                data=total_enstrophy,
            )

    @staticmethod
    def _write_numerical_data(
        solver,
        hdf5_file: str,
        time: float,
    ) -> None:
        """Write canonical solver and particle state."""
        with h5py.File(hdf5_file, "w") as file:
            solver_group = file.create_group("solver")
            solver_group.attrs["checkpoint_format_version"] = _CHECKPOINT_FORMAT_VERSION
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
            if n_particles_total == 0:
                return

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
                    data=getattr(
                        solver.particles,
                        f"{name}_cpu",
                    )(),
                )

            particles_group.create_dataset(
                "vortex_strength_magnitude",
                data=np.linalg.norm(solver.particles.vortex_strength_cpu(), axis=1),
            )

            CheckpointManager._write_optional_particle_fields(
                particles_group,
                solver,
                n_particles_total,
            )

    @staticmethod
    def write_configuration(
        solver,
        configuration_file: str | Path,
    ) -> None:
        """Write setup and checkpoint metadata to JSON."""
        data = {
            "solver_setup": solver.setup.to_dict(),
            "checkpoint_metadata": {
                "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
                "original_compute_device": solver.setup.compute_device,
                "openonda_version": getattr(
                    solver,
                    "version",
                    "unknown",
                ),
                "n_particles_total": int(solver.particles.n_particles_total),
                "time": float(solver.time),
                "step": int(solver.step),
            },
        }
        _atomic_write_text(
            configuration_file,
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        )

    @staticmethod
    def _write_xdmf(
        solver,
        checkpoint_base: str,
        xdmf_file: str,
        time: float,
    ) -> None:
        """Write an XDMF descriptor using canonical field names."""
        n_particles_total = int(solver.particles.n_particles_total)
        hdf5_basename = os.path.basename(f"{checkpoint_base}.h5")
        float_precision = 8 if solver.precision == "f64" else 4
        optional = f"""
      <Attribute Name="zone_id" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Int" Format="HDF">
          {hdf5_basename}:/particles/zone_id
        </DataItem>
      </Attribute>

      <Attribute Name="velocity_gradient" AttributeType="Tensor" Center="Node">
        <DataItem Dimensions="{n_particles_total} 9" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/velocity_gradient
        </DataItem>
      </Attribute>

      <Attribute Name="strain_rate" AttributeType="Tensor" Center="Node">
        <DataItem Dimensions="{n_particles_total} 9" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/strain_rate
        </DataItem>
      </Attribute>"""

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

      <Attribute Name="freestream_velocity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/freestream_velocity
        </DataItem>
      </Attribute>

      <Attribute Name="vortex_strength" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{n_particles_total} 3" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/vortex_strength
        </DataItem>
      </Attribute>

      <Attribute Name="vortex_strength_magnitude" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_particles_total}" NumberType="Float" Precision="{float_precision}" Format="HDF">
          {hdf5_basename}:/particles/vortex_strength_magnitude
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
        checkpoint_pattern: str,
        output_file: str | None = None,
    ) -> str:
        """Create an XDMF temporal collection from canonical HDF5 files."""
        hdf5_files = sorted(glob.glob(f"{checkpoint_pattern}.h5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No checkpoint files found matching {checkpoint_pattern}.h5")

        if output_file is None:
            output_file = f"{checkpoint_pattern.replace('*', 'series')}_temporal.xdmf"
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
                    "vortex_strength_magnitude",
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "effective_viscosity",
                    "group_id",
                    "vorticity",
                    "velocity_gradient",
                    "strain_rate",
                    "zone_id",
                    "freestream_velocity",
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                    "total_enstrophy",
                )
                stored = {
                    name: name if name in particles_group else None for name in canonical_datasets
                }

            hdf5_basename = os.path.basename(hdf5_file)

            def data_item(
                canonical: str,
                dimensions: str,
                *,
                number_type: str = "Float",
                stored: dict[str, str | None] = stored,
                hdf5_basename: str = hdf5_basename,
            ) -> str:
                stored_name = stored[canonical]
                if stored_name is None:
                    return ""
                precision = ' Precision="4"' if number_type == "Float" else ""
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
            if stored["strain_rate"] is not None:
                optional_parts.append(
                    f"""        <Attribute Name="strain_rate" AttributeType="Tensor" Center="Node">
          {data_item("strain_rate", f"{n_particles_total} 9")}
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
        <Attribute Name="vortex_strength_magnitude" AttributeType="Scalar" Center="Node">
          {data_item("vortex_strength_magnitude", str(n_particles_total))}
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
    def load_configuration(
        configuration_file: str | Path,
    ) -> VPMSetup:
        """Load canonical JSON setup data."""
        with open(
            configuration_file,
            encoding="utf-8",
        ) as stream:
            data = json.load(stream)

        setup_data = data.get("solver_setup")
        if setup_data is None:
            raise ValueError("Checkpoint configuration is missing 'solver_setup'")
        return VPMSetup.from_dict(setup_data)

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
            "velocity_gradient": _reshape_tensor(
                _read_dataset(
                    particles_group,
                    "velocity_gradient",
                )
            ),
            "strain_rate": _reshape_tensor(
                _read_dataset(
                    particles_group,
                    "strain_rate",
                )
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
                        "Checkpoint divergence-relaxation reference moments must have shape (3, 3)"
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

            auxiliary = CheckpointManager._load_auxiliary_particle_fields(particles_group)

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
                    velocity_gradient=auxiliary["velocity_gradient"],
                    zone_id=auxiliary["zone_id"],
                )
            finally:
                solver._loading_numerical_state = False

            solver.particles.set_field("vorticity", vorticity)
            if auxiliary["strain_rate"] is not None:
                solver.particles.set_field(
                    "strain_rate",
                    auxiliary["strain_rate"],
                )
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
                        "Checkpoint has no filament-lineage "
                        "state compatible with this refined cloud"
                    )

    @staticmethod
    def _validate_hdf5_structure(
        hdf5_file: str | Path,
    ) -> bool:
        """Return whether a checkpoint has the minimum restart structure."""
        try:
            with h5py.File(hdf5_file, "r") as file:
                if set(file.keys()) != {"solver", "particles"}:
                    return False

                solver_group = file["solver"]
                required_solver_attributes = {
                    "checkpoint_format_version",
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
                format_version = str(solver_group.attrs.get("checkpoint_format_version", ""))
                if format_version != _CHECKPOINT_FORMAT_VERSION:
                    return False

                n_particles_total = _read_particle_count(solver_group)
                if n_particles_total < 0:
                    return False
                particles_group = file["particles"]
                if n_particles_total == 0:
                    return len(particles_group) == 0

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
                    "vortex_strength_magnitude",
                    "effective_viscosity",
                    "zone_id",
                    "freestream_velocity",
                    "velocity_gradient",
                    "strain_rate",
                }
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
                    "freestream_velocity",
                )
                scalar_fields = (
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "group_id",
                    "zone_id",
                    "vortex_strength_magnitude",
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
                for name in ("velocity_gradient", "strain_rate"):
                    if particles_group[name].shape != (n_particles_total, 9):
                        return False
                return True
        except (OSError, KeyError, ValueError):
            return False

    @staticmethod
    def validate_checkpoint(
        checkpoint_path: str | Path,
    ) -> bool:
        """Validate an HDF5 checkpoint base path or ``.h5`` file."""
        path = str(checkpoint_path)
        hdf5_file = path if path.endswith(".h5") else f"{path}.h5"
        return os.path.exists(hdf5_file) and CheckpointManager._validate_hdf5_structure(hdf5_file)
