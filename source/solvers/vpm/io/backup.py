"""Native VPM backups with one current numerical schema and canonical names."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, NoReturn

import h5py
import numpy as np

from source.solution_layout import collection_path, component_directory
from source.write_precision import DEFAULT_WRITE_PRECISION

from ..config.fingerprint import numerical_configuration
from .logging import Logging

# Restart data is numerical backup data, not visualization output. Bump the
# version whenever its layout changes so an older (possibly lossy) file is
# never accepted accidentally.
_BACKUP_FORMAT_VERSION = "10.1"
_COMPRESSION = {
    "chunks": True,
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
}
_STABILIZATION_DIAGNOSTIC_NAMES = (
    *(
        f"pedrizzetti_cumulative_{quantity}_transfer_{axis}"
        for quantity in ("vortex_strength", "linear_impulse", "angular_impulse")
        for axis in "xyz"
    ),
    "n_stabilization_events",
    "n_regularization_events",
    "regularization_cumulative_total_kinetic_energy_transfer",
    "regularization_cumulative_total_enstrophy_transfer",
    "last_stabilization_mechanism",
    "stabilization_vortex_strength_error",
    "stabilization_vortex_strength_growth",
    "stabilization_vorticity_growth",
    "max_stabilization_vorticity_growth",
    "lagrangian_cfl",
    "selective_eddy_viscosity_feedback_coefficient",
)
_FRAME_NAME = re.compile(r"vpm_(\d{6})\.vtu")


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


def _restart_dtype(solver: Any) -> np.dtype:
    """Return the floating-point dtype used by the numerical solver."""
    dtype = np.dtype(getattr(solver, "np_dtype", np.float32))
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"Solver compute dtype must be floating-point, got {dtype}")
    return dtype


def _cast_for_restart(values: Any, dtype: np.dtype) -> np.ndarray:
    """Copy restart values at compute precision without visualization rounding."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def _attribute_text(value: Any) -> str:
    """Normalise HDF5 string attributes for precise validation messages."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _numerical_configuration(solver: Any) -> dict[str, Any]:
    """Return the resolved configuration that determines VPM evolution."""
    return numerical_configuration(solver.setup)


def _canonical_configuration(configuration: dict[str, Any]) -> str:
    """Serialize a numerical configuration deterministically for a restart."""
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _capacity_sensitive_adaptation(configuration: dict[str, Any]) -> bool:
    """Unknown or adaptive allocation policies require an exact capacity."""
    stabilization = configuration.get("stabilization")
    if not isinstance(stabilization, dict):
        return True
    refinement = stabilization.get("filament_refinement")
    if not isinstance(refinement, dict):
        return True
    return bool(
        stabilization.get("regularization_interval_steps") or refinement.get("interval_steps")
    )


def _configuration_mismatches(
    expected: Any,
    found: Any,
    path: str = "",
) -> list[str]:
    """Return incompatible paths, allowing more storage for a fixed algorithm."""
    if isinstance(expected, dict) and isinstance(found, dict):
        paths: list[str] = []
        for key in sorted(set(expected) | set(found)):
            child_path = f"{path}.{key}" if path else key
            if child_path == "max_n_particles":
                old_capacity, new_capacity = found.get(key), expected.get(key)
                if (
                    type(old_capacity) is int
                    and type(new_capacity) is int
                    and new_capacity > old_capacity
                    and not any(
                        _capacity_sensitive_adaptation(configuration)
                        for configuration in (expected, found)
                    )
                ):
                    # These disabled operators cannot change behavior when
                    # extra particle storage becomes available. The saved
                    # checksum and every physical setting are still checked.
                    continue
            if key not in expected or key not in found:
                paths.append(child_path)
            else:
                paths.extend(_configuration_mismatches(expected[key], found[key], child_path))
        return paths
    if isinstance(expected, list) and isinstance(found, list):
        if len(expected) != len(found):
            return [path]
        paths = []
        for index, (expected_item, found_item) in enumerate(zip(expected, found, strict=True)):
            paths.extend(_configuration_mismatches(expected_item, found_item, f"{path}[{index}]"))
        return paths
    return [] if expected == found else [path]


class _BackupIO:
    """Read and write VPM restart backups."""

    @staticmethod
    def load(
        solver,
        hdf5_file: str | Path,
        *,
        time_step_size: float | None = None,
    ) -> None:
        """Replace ``solver`` state from an HDF5 backup.

        ``time_step_size`` is an explicit continuation override.  It permits
        only the numerical-configuration ``time_step_size`` field to differ;
        every other restart identity remains strict.  The complete checkpoint
        is restored first, including the accepted clock and optional VLM
        state, and the override is applied only after that restore succeeds.
        """
        path = str(hdf5_file)
        _BackupIO._validate_hdf5_structure(
            path,
            expected_float_dtype=_restart_dtype(solver),
            expected_configuration=_numerical_configuration(solver),
            allow_time_step_size_mismatch=time_step_size is not None,
        )

        vlm = getattr(solver, "vlm_solver", None)
        with h5py.File(path, "r") as file:
            solver_group = file["solver"]
            source_step = int(_read_attribute(solver_group, "step"))
            source_time = float(_read_attribute(solver_group, "time"))
            source_time_step_size = float(_read_attribute(solver_group, "time_step_size"))
            state = file["solver"].get("vlm")
            if vlm is not None:
                from ..boundary_elements.vlm.solver.restart import validate_vlm_restart

                validate_vlm_restart(vlm, state)
                vlm_time = float(state.attrs["time"])
                clock_tolerance = max(1.0e-10, abs(source_time) * 1.0e-12)
                if not np.isfinite(source_time) or not np.isfinite(vlm_time):
                    raise ValueError("VLM and solver restart clocks must be finite")
                if not np.isclose(vlm_time, source_time, rtol=0.0, atol=clock_tolerance):
                    raise ValueError(
                        "VLM restart time does not match solver accepted time: "
                        f"{vlm_time:.17g} != {source_time:.17g}"
                    )
            elif state is not None:
                raise ValueError("VLM backup requires a configured VLM solver")

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
        if time_step_size is not None:
            solver.time_step_size = float(time_step_size)
            solver._restart_provenance = {
                "kind": "explicit_changed_time_step_continuation",
                "source_checkpoint": str(Path(path).resolve()),
                "source": {
                    "accepted_step": source_step,
                    "accepted_time": source_time,
                    "time_step_size": source_time_step_size,
                },
                "continuation": {
                    "requested_time_step_size": float(time_step_size),
                    "accepted_step": int(solver.step),
                    "accepted_time": float(solver.time),
                },
            }
        else:
            solver._restart_provenance = None

    @staticmethod
    def save(
        solver,
        backup_path: str | Path,
        time: float | None = None,
        *,
        append_step: bool = True,
        verbose: bool = True,
    ) -> None:
        """Write HDF5 restart state and its VTK visualization frame."""
        try:
            time_value = float(solver.time if time is None else time)
            backup_base = str(backup_path)
            if append_step:
                backup_base = f"{backup_base}_{int(solver.step):06d}"

            hdf5_file = f"{backup_base}.h5"
            vtu_file = f"{backup_base}.vtu"
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

            _BackupIO._write_vtu(
                solver,
                vtu_file,
                time_value,
            )

            if verbose:
                Logging.record(
                    "backup written",
                    ("step", f"{solver.step:,}"),
                    ("time", f"{time_value:.6e}", "s"),
                    ("particles", f"{solver.particles.n_particles_total:,}"),
                    ("path", str(hdf5_file)),
                    flush=True,
                )
        except Exception as exc:
            raise RuntimeError(f"Backup write failed: {exc}") from exc

    @staticmethod
    def _write_optional_particle_fields(
        particles_group: h5py.Group,
        solver,
        n_particles_total: int,
        restart_dtype: np.dtype,
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
                data=_cast_for_restart(reference_vortex_strength, restart_dtype),
                **_COMPRESSION,
            )
            particles_group.create_dataset(
                "filament_reference_length",
                data=_cast_for_restart(reference_lengths, restart_dtype),
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
                data=_cast_for_restart(total_enstrophy, restart_dtype),
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
        restart_dtype = _restart_dtype(solver)
        with h5py.File(hdf5_file, "w") as file:
            solver_group = file.create_group("solver")
            solver_group.attrs["backup_format_version"] = _BACKUP_FORMAT_VERSION
            solver_group.attrs["write_precision"] = write_precision
            configuration = _canonical_configuration(_numerical_configuration(solver))
            solver_group.attrs["numerical_configuration"] = configuration
            solver_group.attrs["numerical_configuration_sha256"] = hashlib.sha256(
                configuration.encode("utf-8")
            ).hexdigest()
            solver_group.attrs["freestream_velocity"] = np.asarray(
                solver.freestream_velocity,
                dtype=restart_dtype,
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
                    dtype=restart_dtype,
                )
                if reference_array.shape != (3, 3):
                    raise ValueError(
                        "divergence-relaxation reference moments must have shape (3, 3)"
                    )
                solver_group.create_dataset(
                    "divergence_relaxation_reference_moments",
                    data=reference_array,
                )

            vlm = getattr(solver, "vlm_solver", None)
            if vlm is not None:
                from ..boundary_elements.vlm.solver.restart import write_vlm_restart

                write_vlm_restart(vlm, solver_group.create_group("vlm"))

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
                    data=_cast_for_restart(
                        # Derived fields can change at the same source-state
                        # revision. A checkpoint must read the actual device
                        # state, never a preceding diagnostic's CPU snapshot.
                        getattr(solver.particles, f"{name}_cpu")(use_cache=False),
                        restart_dtype,
                    ),
                    **_COMPRESSION,
                )

            _BackupIO._write_optional_particle_fields(
                particles_group,
                solver,
                n_particles_total,
                restart_dtype,
            )

    @staticmethod
    def _write_vtu(
        solver,
        vtu_file: str,
        time: float,
    ) -> None:
        """Write a VTK particle frame using canonical field names.

        One ``VTK_POLY_VERTEX`` cell spans the full particle cloud, matching
        the solver's Polyvertex particle topology. The points and arrays retain
        the compute dtype; the exact metadata keys ``time`` and ``TimeValue``
        let ParaView recover the physical clock from the standalone frame
        series. The frame is written atomically in appended-raw XML, so a
        reader never observes a partially written file.
        """
        from vtk import VTK_POLY_VERTEX, vtkCellArray, vtkPoints, vtkUnstructuredGrid
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        from source.vtk_output import write_vtk_dataset

        n_particles_total = int(solver.particles.n_particles_total)
        if not np.isfinite(time):
            raise ValueError("VPM frame time must be finite")
        position = np.ascontiguousarray(solver.particles.position_cpu(use_cache=False))
        if position.ndim != 2 or position.shape[1] != 3 or len(position) != n_particles_total:
            raise ValueError("VPM particle positions must have shape (n_particles_total, 3)")

        points = vtkPoints()
        points.SetData(numpy_to_vtk(position, deep=True))
        grid = vtkUnstructuredGrid()
        grid.SetPoints(points)
        if n_particles_total:
            cells = vtkCellArray()
            cells.SetData(
                numpy_to_vtkIdTypeArray(np.zeros(1, dtype=np.int64), deep=True),
                numpy_to_vtkIdTypeArray(
                    np.arange(n_particles_total, dtype=np.int64),
                    deep=True,
                ),
            )
            grid.SetCells(
                numpy_to_vtk(np.array([VTK_POLY_VERTEX], dtype=np.uint8), deep=True),
                cells,
            )

        def add_point_data(name: str, values: np.ndarray) -> None:
            array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
            array.SetName(name)
            grid.GetPointData().AddArray(array)

        for name in ("velocity", "vortex_strength", "vorticity"):
            add_point_data(name, solver.particles.__getattribute__(f"{name}_cpu")(use_cache=False))
        for name in (
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
            "group_id",
            "zone_id",
        ):
            add_point_data(name, solver.particles.__getattribute__(f"{name}_cpu")(use_cache=False))

        # VTK requires the exact metadata keys time and TimeValue to recover
        # the physical clock from the standalone frame series.
        for name in ("time", "TimeValue"):
            field = numpy_to_vtk(np.array([time], dtype=np.float64), deep=True)
            field.SetName(name)
            grid.GetFieldData().AddArray(field)
        write_vtk_dataset(grid, Path(vtu_file))

    @staticmethod
    def write_pvd(solution_directory: str | Path) -> Path:
        """Write the ParaView index for retained VPM particle frames.

        Parameters
        ----------
        solution_directory : str or pathlib.Path
            Case solution root. Canonical ``vpm_XXXXXX.h5`` and
            ``vpm_XXXXXX.vtu`` pairs are stored below ``vpm/``.

        Returns
        -------
        pathlib.Path
            Atomically replaced ``vpm.pvd`` collection path. Each entry
            references an immutable VTK frame by a relative filename below
            ``vpm/``.

        Raises
        ------
        FileNotFoundError
            If an indexed VTK frame has no matching HDF5 particle state.
        ValueError
            If frame steps or physical times are duplicate or nonmonotonic.

        Notes
        -----
        The atomically written native HDF5 clock is authoritative for the
        visualization series. Rebuilding the small index from retained frames
        makes scheduled output and coupled publication resume-safe without
        keeping a second in-memory clock.
        """
        solution = Path(solution_directory)
        directory = component_directory(solution, "vpm")
        frames: list[tuple[int, float, str]] = []
        for vtu in directory.glob("vpm_*.vtu"):
            match = _FRAME_NAME.fullmatch(vtu.name)
            if match is None:
                continue
            hdf5 = vtu.with_suffix(".h5")
            if not hdf5.is_file():
                raise FileNotFoundError(f"VPM frame {vtu.name} is missing {hdf5.name}")
            try:
                with h5py.File(hdf5, "r") as archive:
                    time = float(archive["solver"].attrs["time"])
            except (OSError, KeyError) as exc:
                raise ValueError(f"Invalid VPM frame {vtu.name}") from exc
            if not np.isfinite(time):
                raise ValueError(f"VPM frame {vtu.name} has non-finite time")
            frames.append((int(match.group(1)), time, f"vpm/{vtu.name}"))

        frames.sort()
        steps = [step for step, _, _ in frames]
        times = [time for _, time, _ in frames]
        if len(set(steps)) != len(steps) or any(
            current <= previous for previous, current in zip(times, times[1:], strict=False)
        ):
            raise ValueError("VPM particle frames are duplicate or nonmonotonic")

        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        lines.extend(
            f'    <DataSet timestep="{time:.17g}" file="{filename}"/>'
            for _, time, filename in frames
        )
        lines.extend(("  </Collection>", "</VTKFile>", ""))
        destination = collection_path(solution, "vpm")
        _atomic_write_text(destination, "\n".join(lines))
        return destination

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

            # Freestream is part of the advection state.  Restore it before any
            # velocity-derived fields are refreshed below, including for an
            # otherwise empty cloud.
            solver._set_freestream_velocity(
                np.asarray(
                    _read_attribute(solver_group, "freestream_velocity"),
                    dtype=_restart_dtype(solver),
                )
            )
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

            vlm = getattr(solver, "vlm_solver", None)
            if vlm is not None:
                from ..boundary_elements.vlm.solver.restart import restore_vlm_restart

                restore_vlm_restart(vlm, solver_group["vlm"])

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
                solver.stepper._update_velocity_and_gradients(announce=False)
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
        *,
        expected_float_dtype: np.dtype | None = None,
        expected_configuration: dict[str, Any] | None = None,
        allow_time_step_size_mismatch: bool = False,
    ) -> None:
        """Validate a restart file before it is allowed to mutate a solver.

        This is deliberately strict: a numerical restart must not silently
        coerce precision, truncate identifiers, or proceed with non-physical
        particle geometry.
        """
        path = str(hdf5_file)

        def invalid(reason: str) -> NoReturn:
            raise ValueError(f"Invalid VPM backup {path}: {reason}")

        try:
            with h5py.File(hdf5_file, "r") as file:
                if set(file.keys()) != {"solver", "particles"}:
                    invalid("top-level groups must be exactly {'solver', 'particles'}")

                solver_group = file["solver"]
                required_solver_attributes = {
                    "backup_format_version",
                    "write_precision",
                    "numerical_configuration",
                    "numerical_configuration_sha256",
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
                    missing = sorted(required_solver_attributes - solver_attribute_names)
                    invalid(f"missing solver attributes: {', '.join(missing)}")
                if not solver_attribute_names <= (
                    required_solver_attributes | set(_STABILIZATION_DIAGNOSTIC_NAMES)
                ):
                    unknown = sorted(
                        solver_attribute_names
                        - required_solver_attributes
                        - set(_STABILIZATION_DIAGNOSTIC_NAMES)
                    )
                    invalid(f"unknown solver attributes: {', '.join(unknown)}")
                if set(solver_group.keys()) - {"divergence_relaxation_reference_moments", "vlm"}:
                    invalid("contains unknown solver datasets")
                format_version = _attribute_text(
                    solver_group.attrs.get("backup_format_version", "")
                )
                if format_version != _BACKUP_FORMAT_VERSION:
                    invalid(
                        "unsupported backup format version "
                        f"{format_version!r}; expected {_BACKUP_FORMAT_VERSION!r}"
                    )

                configuration_text = _attribute_text(
                    _read_attribute(solver_group, "numerical_configuration")
                )
                configuration_hash = _attribute_text(
                    _read_attribute(solver_group, "numerical_configuration_sha256")
                )
                computed_hash = hashlib.sha256(configuration_text.encode("utf-8")).hexdigest()
                if configuration_hash != computed_hash:
                    invalid(
                        "numerical configuration fingerprint does not match its stored configuration"
                    )
                try:
                    stored_configuration = json.loads(configuration_text)
                except json.JSONDecodeError as exc:
                    invalid(f"numerical configuration is not valid JSON ({exc.msg})")
                if not isinstance(stored_configuration, dict):
                    invalid("numerical configuration must be a JSON object")
                if expected_configuration is not None:
                    mismatches = _configuration_mismatches(
                        expected_configuration,
                        stored_configuration,
                    )
                    if allow_time_step_size_mismatch:
                        mismatches = [
                            mismatch for mismatch in mismatches if mismatch != "time_step_size"
                        ]
                    if mismatches:
                        invalid("numerical configuration mismatch at " + ", ".join(mismatches))

                freestream_velocity = np.asarray(
                    _read_attribute(solver_group, "freestream_velocity")
                )
                if (
                    freestream_velocity.shape != (3,)
                    or not np.issubdtype(freestream_velocity.dtype, np.floating)
                    or not np.isfinite(freestream_velocity).all()
                ):
                    invalid(
                        "freestream_velocity must be a finite floating-point vector of shape (3,)"
                    )
                if (
                    expected_float_dtype is not None
                    and np.dtype(freestream_velocity.dtype) != expected_float_dtype
                ):
                    invalid(
                        "freestream_velocity has dtype "
                        f"{freestream_velocity.dtype}; expected solver compute dtype "
                        f"{expected_float_dtype}"
                    )

                for name in ("time", "time_step_size"):
                    value = _read_attribute(solver_group, name)
                    if not np.isscalar(value) or not np.isfinite(value):
                        invalid(f"solver attribute {name!r} must be finite")
                if float(_read_attribute(solver_group, "time_step_size")) <= 0.0:
                    invalid("solver attribute 'time_step_size' must be positive")
                for name in (
                    "step",
                    "n_steps_since_dvh_diffusion",
                    "n_particles_total",
                ):
                    value = _read_attribute(solver_group, name)
                    if not isinstance(value, int | np.integer) or int(value) < 0:
                        invalid(f"solver attribute {name!r} must be a non-negative integer")
                pending = _read_attribute(solver_group, "is_particle_regeneration_pending")
                if not isinstance(pending, bool | int | np.integer) or int(pending) not in (0, 1):
                    invalid("solver attribute 'is_particle_regeneration_pending' must be 0 or 1")

                if "divergence_relaxation_reference_moments" in solver_group:
                    moments = solver_group["divergence_relaxation_reference_moments"]
                    if moments.shape != (3, 3) or not np.issubdtype(moments.dtype, np.floating):
                        invalid(
                            "divergence-relaxation reference moments must be a floating (3, 3) array"
                        )
                    if (
                        expected_float_dtype is not None
                        and np.dtype(moments.dtype) != expected_float_dtype
                    ):
                        invalid(
                            "divergence-relaxation reference moments have dtype "
                            f"{moments.dtype}; expected solver compute dtype {expected_float_dtype}"
                        )
                    if not np.isfinite(moments[:]).all():
                        invalid("divergence-relaxation reference moments must be finite")

                n_particles_total = _read_particle_count(solver_group)
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
                optional = {
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                    "total_enstrophy",
                }
                particle_field_names = set(particles_group.keys())
                if not required <= particle_field_names:
                    missing = sorted(required - particle_field_names)
                    invalid(f"missing particle fields: {', '.join(missing)}")
                if not particle_field_names <= required | optional:
                    unknown = sorted(particle_field_names - required - optional)
                    invalid(f"unknown particle fields: {', '.join(unknown)}")
                filament_fields = {
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                }
                if len(particle_field_names & filament_fields) == 1:
                    invalid("filament-lineage fields must be stored together")
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
                    invalid("vector particle fields must have shape (n_particles_total, 3)")
                if any(
                    particles_group[name].shape != (n_particles_total,) for name in scalar_fields
                ):
                    invalid("scalar particle fields must have shape (n_particles_total,)")

                if filament_fields <= particle_field_names:
                    if particles_group["filament_reference_vortex_strength"].shape != (
                        n_particles_total,
                    ):
                        invalid(
                            "filament_reference_vortex_strength must have shape (n_particles_total,)"
                        )
                    if particles_group["filament_reference_length"].shape != (n_particles_total,):
                        invalid("filament_reference_length must have shape (n_particles_total,)")

                floating_fields = vector_fields + (
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "eddy_viscosity",
                    "effective_viscosity",
                )
                if filament_fields <= particle_field_names:
                    floating_fields += tuple(sorted(filament_fields))
                if "total_enstrophy" in particle_field_names:
                    floating_fields += ("total_enstrophy",)
                for name in floating_fields:
                    dataset = particles_group[name]
                    if not np.issubdtype(dataset.dtype, np.floating):
                        invalid(f"particle field {name!r} must use a floating-point dtype")
                    if (
                        expected_float_dtype is not None
                        and np.dtype(dataset.dtype) != expected_float_dtype
                    ):
                        invalid(
                            f"particle field {name!r} has dtype {dataset.dtype}; "
                            f"expected solver compute dtype {expected_float_dtype}"
                        )
                    if not np.isfinite(dataset[:]).all():
                        invalid(f"particle field {name!r} contains non-finite values")

                for name in ("group_id", "zone_id"):
                    if not np.issubdtype(particles_group[name].dtype, np.integer):
                        invalid(f"particle field {name!r} must use an integral dtype")

                if np.any(particles_group["core_radius"][:] <= 0.0):
                    invalid("particle field 'core_radius' must be strictly positive")
                if np.any(particles_group["particle_volume"][:] <= 0.0):
                    invalid("particle field 'particle_volume' must be strictly positive")
        except OSError as exc:
            invalid(f"cannot read HDF5 file ({exc})")
