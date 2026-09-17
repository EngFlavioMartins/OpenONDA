"""High-level incompressible FVM solver API."""

from contextlib import suppress
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from source.solution_layout import collection_path, component_directory

from ..config.case import FVMCase
from ..config.types import FVMSetup
from ..coupling import CouplerInterfaceMixin
from ..fields.mixed_velocity_boundary import (
    update_normal_velocity_tangential_gradient_boundary,
)
from ..io import logging, solver_io
from ..mesh import geometry
from ..sampling.executor import FVMSamplerExecutor
from ..solve import pimple_solver, simple_solver
from .parallel import ParallelContext
from .state import FieldState
from .time_step import event_aligned_time_step_size, maximum_courant_time_step_size


def _load_velocity_field(setup, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Initialise the velocity field from the Python configuration.

    Args:
        setup:    FVMSetup (may have ``initial_velocity``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Velocity array ``(n_total, 3)``.
    """
    del case_dir
    if setup.initial_velocity is None:
        raise ValueError("initial_velocity must be provided in FVMSetup")
    initial = np.asarray(setup.initial_velocity, dtype=np.float64)
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_velocity must be finite")
    if initial.shape == (3,):
        return np.tile(initial, (n_total, 1))
    if initial.ndim == 2 and initial.shape == (mesh_data["n_cells"], 3):
        result = np.zeros((n_total, 3), dtype=np.float64)
        result[: mesh_data["n_cells"]] = initial
        return result
    raise ValueError(f"initial_velocity must have shape (3,) or (n_cells, 3); got {initial.shape}")


def _load_kinematic_pressure_field(
    setup, case_dir: str, n_total: int, mesh_data: dict
) -> np.ndarray:
    """Initialise the pressure field from the Python configuration.

    Args:
        setup:    FVMSetup (may have ``initial_kinematic_pressure``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Pressure array ``(n_total,)``.
    """
    del case_dir, mesh_data
    if setup.initial_kinematic_pressure is None:
        raise ValueError("initial_kinematic_pressure must be provided in FVMSetup")
    initial = np.asarray(setup.initial_kinematic_pressure, dtype=np.float64)
    if initial.ndim != 0 or not np.isfinite(initial):
        raise ValueError("initial_kinematic_pressure must be a finite scalar")
    return np.full(n_total, float(initial), dtype=np.float64)


def _enforce_velocity_boundary_constraints(
    velocity: np.ndarray, boundaries: list, n_cells: int, mesh_data: dict, geo_data: dict
) -> None:
    """Enforce velocity boundary constraints on ghost cells after initialisation.

    Iterates over all boundary patches and sets the ghost-layer values
    in *velocity* according to each patch's boundary condition type (noSlip,
    fixedValue, zeroGradient, empty, etc.).

    Args:
        velocity:    Velocity array (mutated in place).
        boundaries: List of boundary patch dictionaries.
        n_elements: Number of interior elements.
        mesh_data:  Mesh dictionary.
        geo_data:   Geometry dictionary.
    """
    from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy

    for boundary in boundaries:
        boundary_condition_type = boundary.get("velocity_type")
        strategy = BOUNDARIES.strategy(boundary_condition_type, "velocity", "ghost")
        start = n_cells + (boundary["start_face"] - mesh_data["n_interior_faces"])
        end = start + boundary["n_faces"]
        if strategy is BoundaryStrategy.NO_SLIP:
            velocity[start:end] = 0.0
        elif (
            strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.FREESTREAM)
            and boundary.get("velocity_value_field") is not None
        ):
            velocity[start:end] = boundary["velocity_value_field"]
        elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.FREESTREAM) and (
            "velocity_value" in boundary
        ):
            velocity[start:end] = boundary["velocity_value"]
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.INLET_OUTLET,
        ):
            owners_b = mesh_data["owners"][
                boundary["start_face"] : boundary["start_face"] + boundary["n_faces"]
            ]
            velocity[start:end] = velocity[owners_b]
        elif strategy is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT:
            update_normal_velocity_tangential_gradient_boundary(
                velocity, boundary, mesh_data, geo_data
            )
        elif strategy is BoundaryStrategy.CYCLIC:
            faces = np.arange(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
            paired = mesh_data["boundary_neighbour_cell"][faces]
            if np.any(paired < 0):
                raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
            velocity[start:end] = velocity[paired]
        elif strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            owners_b = mesh_data["owners"][
                boundary["start_face"] : boundary["start_face"] + boundary["n_faces"]
            ]
            face_area_vector = geo_data["face_area_vector"][
                boundary["start_face"] : boundary["start_face"] + boundary["n_faces"]
            ]
            owner_velocity = velocity[owners_b]
            magnitudes = np.linalg.norm(face_area_vector, axis=1)
            valid = magnitudes > 1e-10
            projected = owner_velocity.copy()
            if np.any(valid):
                normals = face_area_vector[valid] / magnitudes[valid, np.newaxis]
                projected[valid] -= (
                    np.sum(owner_velocity[valid] * normals, axis=1)[:, np.newaxis] * normals
                )
            velocity[start:end] = projected


class FVMSolver(CouplerInterfaceMixin):
    """Run a constant-density, incompressible finite-volume simulation.

    The solver accepts the immutable high-level :class:`FVMCase` or the legacy
    mutable :class:`FVMSetup`. It materializes a face-based mesh, computes and
    caches geometry, reconstructs boundary ghost cells, assembles the selected
    SIMPLE/PISO/PIMPLE equations, and owns the accepted physical clock and
    BDF history. Numerical kernels may replace arrays internally; the public
    :attr:`state` view is republished so the primary fields stay synchronized.

    Attributes
    ----------
    case or config : FVMCase or FVMSetup
        Construction policy. ``config`` is the resolved low-level setup kept
        for compatibility; a public case is available as ``case`` when used.
    case_dir, solution_dir, samples_dir : pathlib.Path
        Case and framework-owned artifact roots.
    mesh_data : dict[str, object]
        Native connectivity, patch ranges, owner/neighbour indices, and counts.
    geo_data : dict[str, numpy.ndarray]
        Derived centres, volumes, face areas, interpolation weights, and wall
        distances. Lengths are m, areas m², volumes m³.
    velocity : numpy.ndarray
        Cell-centred velocity in m/s, including solver-owned boundary ghosts.
    kinematic_pressure : numpy.ndarray
        ``p/rho`` in m²/s², including boundary ghosts.
    volumetric_face_flux : numpy.ndarray
        ``phi = U_f · Sf`` in m³/s, positive owner-to-neighbour.
    time, time_step_size : float
        Accepted physical time and currently selected step duration in seconds.
    step : int
        Number of accepted time steps.
    state : FieldState
        Synchronized public view of the three primary fields.

    Notes
    -----
    Construction allocates solver arrays and may create output directories. It
    does not advance the solution. Use :meth:`run` for the framework-owned
    finite lifecycle or :meth:`advance`/the candidate APIs for interactive and
    coupled control. The solver is static-mesh only.
    """

    @property
    def topology(self):
        """Return the lazily constructed immutable mesh-topology facade.

        Returns
        -------
        MeshTopology
            Owner/neighbour, patch, and cell-face connectivity without copying
            the native mesh arrays. The view is cached for this solver.
        """
        if self._topology is None:
            from ..mesh.topology import MeshTopology

            self._topology = MeshTopology.from_mesh_data(self.mesh_data)
        return self._topology

    @property
    def geometry(self):
        """Return the lazily constructed read-only geometry facade.

        Returns
        -------
        MeshGeometry
            Cell/face centres, areas, volumes, interpolation weights, wall
            distances, and optional least-squares conditioning. Arrays use SI
            units and may share memory with ``geo_data``; write through the
            facade is prohibited.
        """
        if self._geometry is None:
            self._geometry = geometry.MeshGeometry.from_data(self.mesh_data, self.geo_data)
        return self._geometry

    def _invalidate_derived_fields(self) -> None:
        """Clear cached fields derived from the current primary solution.

        The cache contains quantities such as the velocity gradient, vorticity,
        and Courant number. It is invalid after any velocity, pressure, flux,
        mesh, or time-step mutation; the next accessor recomputes only what is
        requested. This operation changes cache state but not the primary
        numerical arrays.
        """
        if hasattr(self, "_derived_fields"):
            self._derived_fields.clear()

    def _publish_state(self) -> FieldState:
        """Publish the live arrays through one synchronized :class:`FieldState`.

        Numerical kernels are allowed to return replacement arrays.  This
        helper makes the state object and the solver attributes point at the
        same contiguous arrays after every such boundary, while preserving an
        already-issued state object's identity for interactive callers.
        """
        published = FieldState(
            self.velocity,
            self.kinematic_pressure,
            self.volumetric_face_flux,
        )
        current = getattr(self, "state", None)
        if current is None:
            self.state = published
        else:
            current.velocity = published.velocity
            current.kinematic_pressure = published.kinematic_pressure
            current.volumetric_face_flux = published.volumetric_face_flux
            self.state = current
        self.velocity = self.state.velocity
        self.kinematic_pressure = self.state.kinematic_pressure
        self.volumetric_face_flux = self.state.volumetric_face_flux
        if hasattr(self, "_state_revision"):
            self._state_revision += 1
        return self.state

    def _velocity_gradient(self):
        """Return the cached gradient for the current solved field state."""
        from ..fields import gradients

        gradient = self._derived_fields.get("velocity_gradient")
        if gradient is None:
            gradient = gradients._resolve_gradient_fn(self.geo_data)(
                self.velocity, self.mesh_data, self.geo_data
            )
            self._derived_fields["velocity_gradient"] = gradient
        return gradient

    def _courant_field(self, time_step_size: float):
        """Return the cached cell-local Courant field for one ``dt``.

        Parameters
        ----------
        time_step_size : float
            Candidate physical time step in seconds used with the current
            face fluxes and cell volumes.

        Returns
        -------
        numpy.ndarray, shape (n_cells,)
            Dimensionless local Courant numbers. The result is cached by
            ``dt`` and is recomputed after derived-field invalidation.
        """
        from ..fields import diagnostics

        key = ("courant", float(time_step_size))
        courant = self._derived_fields.get(key)
        if courant is None:
            courant = diagnostics.compute_courant_number(
                self.velocity,
                self.volumetric_face_flux,
                time_step_size,
                self.mesh_data,
                self.geo_data,
            )
            self._derived_fields[key] = courant
        return courant

    def _vorticity_field(self):
        """Return the cached cell-centred curl of the current velocity.

        Returns
        -------
        numpy.ndarray, shape (n_cells, 3)
            Vorticity in 1/s using the configured gradient convention. The
            result is a derived cache entry and does not mutate the velocity.
        """
        from ..fields import diagnostics

        vorticity = self._derived_fields.get("vorticity")
        if vorticity is None:
            vorticity = diagnostics.compute_vorticity(
                self.velocity,
                self.mesh_data,
                self.geo_data,
                gradient=self._velocity_gradient(),
            )
            self._derived_fields["vorticity"] = vorticity
        return vorticity

    def __init__(
        self,
        setup: FVMSetup | FVMCase,
        case_dir: str | None = None,
        solution_dir: str | None = None,
        samples_dir: str | None = None,
        mesh_data: dict[str, Any] | None = None,
        logger: Any | None = None,
    ):
        """Initialize an FVM solver and materialize its mesh/state.

        Parameters
        ----------
        setup : FVMCase or FVMSetup
            Preferred immutable case or legacy low-level setup. A case is
            converted to a setup copy; neither input object is mutated.
        case_dir : str or None
            Case root. For ``FVMCase`` this defaults to ``setup.directory``;
            otherwise it defaults to the current working directory.
        solution_dir, samples_dir : str or None
            Optional artifact roots. When omitted, resolved defaults are used
            by the selected construction path. Paths are created as needed.
        mesh_data : dict[str, object] or None
            Pre-materialized native mesh. If omitted for an ``FVMCase``, its
            mesh source is materialized. In distributed execution the required
            rank/collective ownership is enforced by the selected backend.
        logger : object or None
            Optional logger supplied by a legacy/coupled caller.

        Raises
        ------
        TypeError, ValueError, RuntimeError, FileNotFoundError
            If configuration, mesh topology/geometry, optional dependencies,
            or parallel execution contracts are invalid.

        Notes
        -----
        Construction computes geometry, initializes fields, builds the selected
        pressure--velocity algorithm, creates its output directories, and writes
        ``fvm_metadata.json`` in the resolved solution directory. It does not
        advance ``time`` or ``step``. All solver-owned arrays use SI units;
        boundary ghost rows are reconstructed after interior initialization.
        """
        from ..config.case import FVMCase
        from ..config.types import validate_fvm_setup

        public_case = setup if isinstance(setup, FVMCase) else None
        if public_case is not None:
            setup = public_case.to_setup()
            validate_fvm_setup(setup)
            import sys

            from openonda.runtime import RunConfig

            RunConfig(cpu_cores=setup.cores, parallel_mode="mpi").ensure_runtime(sys.argv[0])
            if case_dir is None:
                case_dir = str(public_case.directory)
            from ..factory import _runtime_setup

            setup = _runtime_setup(setup)
            validate_fvm_setup(setup)
            if mesh_data is None:
                from ..factory import _materialize_mesh
                from ..mesh.progress import mesh_stage, mesher_log_session

                mesh_source = public_case.mesh
                if isinstance(mesh_source, str | Path) and not Path(mesh_source).is_absolute():
                    mesh_source = Path(case_dir or os.getcwd()) / mesh_source
                materialize_here = True
                rank_is_root = True
                mesh_comm = None
                if public_case.cores > 1:
                    try:
                        from mpi4py import MPI

                        mesh_comm = MPI.COMM_WORLD
                        rank_is_root = MPI.COMM_WORLD.Get_rank() == 0
                        materialize_here = rank_is_root
                    except ImportError:
                        raise RuntimeError(
                            "FVMCase.cores > 1 requires mpi4py and an MPI launch"
                        ) from None
                requested_solution = (
                    Path(solution_dir)
                    if solution_dir is not None
                    else Path(case_dir or os.getcwd()) / "solution"
                )
                mesher_log_path = requested_solution.resolve() / "mesher.log"
                from source.simulation.parallel import collective_phase

                with (
                    collective_phase(mesh_comm, "FVM mesh materialization"),
                    mesher_log_session(
                        mesher_log_path if rank_is_root else None,
                        announce=rank_is_root,
                    ),
                    mesh_stage("mesh materialization") as materialization,
                ):
                    mesh_data = _materialize_mesh(mesh_source, is_root=materialize_here)
                    if mesh_data is not None:
                        materialization.details(
                            cells=mesh_data.get("n_cells"),
                            faces=mesh_data.get("n_faces"),
                            points=mesh_data.get("n_points"),
                        )
                if mesh_comm is not None and setup.execution.parallel_mode == "petsc_replicated":
                    mesh_data = mesh_comm.bcast(mesh_data, root=0)

            # The public case is immutable intent; give the numerical core a
            # private resolved snapshot so later mutation of a nested caller
            # object cannot change an already-admitted run.
            setup = deepcopy(setup)
            self.case = public_case
        elif not isinstance(setup, FVMSetup):
            raise TypeError("FVMSolver requires an FVMSetup or FVMCase")
        validate_fvm_setup(setup)

        # Keep the historical ``solver.setup`` identity for coupled callers,
        # but never let a mutable caller-owned setup alter an admitted run.
        # All numerical, output, and compatibility decisions below use this
        # detached snapshot; ``setup`` remains an informational compatibility
        # attribute only.
        self.setup = setup
        self._resolved_setup = deepcopy(setup)
        resolved_setup = self._resolved_setup
        # Runtime coupling may override molecular viscosity for subsequent
        # equations.  Keep that override separate from the admitted case so
        # the immutable numerical identity and the live physics state cannot
        # silently diverge through a nested setup mutation.
        self._kinematic_viscosity = float(resolved_setup.transport.kinematic_viscosity)
        # Capture immutable construction-time controls. The running solver owns
        # its evolving time state while these values remain fixed.
        self._time_config = resolved_setup.time
        self._output_schedule = resolved_setup.time.output_schedule
        self._backup_config = resolved_setup.backup
        self._samplers = tuple(resolved_setup.samplers or ())
        # Runtime-only sampler indexes belong to this solver, never to a
        # reusable sampler specification.
        self._sample_pvd_entries: dict[str, list[tuple[float, str]]] = {}
        self._sampler_schedules = {
            id(sampler): sampler.schedule
            for sampler in self._samplers
            if getattr(sampler, "schedule", None) is not None
        }
        self.case_dir = os.path.abspath(case_dir or os.getcwd())
        default_solution_name = "solution"
        self.solution_dir = os.path.abspath(
            solution_dir or os.path.join(self.case_dir, default_solution_name)
        )
        self.samples_dir = os.path.abspath(samples_dir or os.path.join(self.case_dir, "samples"))
        Path(self.solution_dir).mkdir(parents=True, exist_ok=True)
        # Sampling owns its directory.  An explicitly supplied destination is
        # also prepared for callers that intend to create products later; a
        # disabled/default sampler configuration does not leave an empty
        # ``samples/`` tree behind.
        if self._samplers or samples_dir is not None:
            Path(self.samples_dir).mkdir(parents=True, exist_ok=True)
        # These dictionaries intentionally contain heterogeneous mesh metadata
        # (arrays, counts, patch dictionaries, and parallel objects).
        self.mesh_data: Any
        self.geo_data: Any
        self.auto_write = True
        self.parallel = ParallelContext.create(resolved_setup.execution)
        self.logger = (
            logger
            if logger is not None
            else logging.Logging(
                self.case_dir,
                solution_dir=self.solution_dir,
                config=resolved_setup.logging,
                enabled=self.parallel.is_root,
            )
        )
        self._timer = logging.Timer()
        if public_case is not None:
            mesh_archive_error = None
            if self.parallel.is_root and mesh_data is not None:
                try:
                    from ..factory import _save_generated_mesh

                    _save_generated_mesh(
                        mesh_data,
                        Path(self.solution_dir),
                        resolved_setup.output,
                    )
                except BaseException as error:
                    mesh_archive_error = error
            self._collective_io_failure(mesh_archive_error, "mesh provenance output")
        from ..io.profiling import PerformanceProfiler

        self.profiler = PerformanceProfiler(
            self.case_dir,
            self.parallel,
            self.logger,
            solution_dir=self.solution_dir,
            solver=self,
        )
        self.logger.profiler = self.profiler
        self.operator_backend = resolved_setup.execution.operator_backend
        if resolved_setup.execution.linear_backend == "petsc":
            methods = {
                "momentum": resolved_setup.linear.momentum_solver
                or resolved_setup.linear.linear_solver,
                "pressure": resolved_setup.linear.pressure_solver
                or resolved_setup.linear.linear_solver,
            }
            invalid = {
                name: value
                for name, value in methods.items()
                if value not in {"bicgstab", "gmres", "cg", "amg"}
            }
            if invalid:
                raise ValueError(
                    "PETSc execution requires iterative momentum/pressure methods "
                    f"(bicgstab, gmres, cg, or pressure AMG); got {invalid!r}. Distributed "
                    "direct factorization is intentionally not assumed."
                )

        # Fail fast on typo'd / unsupported scheme or turbulence-model names
        # (otherwise the error only surfaces deep inside the first assembly).
        from types import SimpleNamespace

        from ..schemes import (
            validate_acceptance_limits,
            validate_solver_params,
            validate_turbulence,
        )

        validate_solver_params(
            SimpleNamespace(**resolved_setup.algorithm_params()), self._time_config
        )
        validate_turbulence(resolved_setup.turbulence)
        validate_acceptance_limits(resolved_setup.acceptance)
        if self.parallel.is_partitioned and resolved_setup.turbulence is not None:
            turbulence_name = resolved_setup.turbulence.model.lower()
            if resolved_setup.turbulence.dynamic or turbulence_name in {
                "dynamicsmagorinsky",
                "dynamic_smagorinsky",
            }:
                raise NotImplementedError(
                    "Dynamic Smagorinsky is not qualified for petsc_partitioned execution: "
                    "its Germano average must be reduced over owned cells globally."
                )
        if (
            resolved_setup.linear.pressure_nullspace_method == "petsc"
            and resolved_setup.execution.linear_backend != "petsc"
        ):
            raise ValueError(
                "pressure_nullspace_method='petsc' requires execution.linear_backend='petsc'"
            )
        if (
            self.parallel.is_partitioned
            and resolved_setup.linear.pressure_nullspace_method == "reference"
        ):
            raise ValueError(
                "petsc_partitioned requires pressure_nullspace_method='auto' or 'petsc'; "
                "a rank-local reference row is not a valid global pressure constraint"
            )
        if self.parallel.is_partitioned and resolved_setup.output.point_interpolation != "none":
            raise ValueError(
                "output.point_interpolation='boundary_weighted' is not qualified for "
                "petsc_partitioned execution: the partitioned writer drops the boundary "
                "ghost values the interpolation needs, and a rank's processor-interface "
                "faces are not physical boundaries. Run serially to write interpolated "
                "point data, or use ParaView's Cell Data to Point Data filter instead"
            )
        if (
            not np.isfinite(resolved_setup.transport.density)
            or resolved_setup.transport.density <= 0.0
        ):
            raise ValueError("Transport density must be finite and positive")
        if (
            not np.isfinite(resolved_setup.transport.kinematic_viscosity)
            or resolved_setup.transport.kinematic_viscosity <= 0.0
        ):
            raise ValueError("Kinematic viscosity must be finite and positive")
        # 0. UI Header
        self.logger.header("f64")
        self._timer.start("Total Initialization")

        # 1. Mesh Management
        from ..mesh.validation import (
            enforce_quality_thresholds,
            validate_geometry,
            validate_topology,
        )

        self._topology = None
        self._geometry = None
        gs = getattr(resolved_setup.schemes, "gradient_scheme", "gauss")
        self._timer.start("Geometry Compute")
        if self.parallel.is_partitioned:
            comm = self.parallel.comm
            assert comm is not None
            if any(boundary.velocity_type == "cyclic" for boundary in resolved_setup.boundaries):
                raise NotImplementedError(
                    "Partitioned cyclic patches require periodic partition adjacency, which is "
                    "not yet implemented"
                )
            if (
                resolved_setup.initial_velocity is None
                or resolved_setup.initial_kinematic_pressure is None
            ):
                raise ValueError(
                    "initial_velocity and initial_kinematic_pressure must be provided in FVMSetup"
                )
            quality = None
            preparation_error = None
            global_mesh = None
            global_geo = None
            global_hash = None
            if self.parallel.is_root:
                try:
                    if mesh_data is None:
                        raise ValueError(
                            "A solver-native mesh, mesh factory, or Gmsh .msh path is required"
                        )
                    self._timer.start("Mesh Set (In-Memory)")
                    global_mesh = mesh_data
                    self._timer.log(
                        "Mesh Set (In-Memory)",
                        sink=self.logger,
                    )
                    validate_topology(global_mesh)
                    global_geo = geometry.compute_mesh_geometry(
                        global_mesh,
                        gradient_scheme=gs,
                        compute_lsq=False,
                        logger=self.logger,
                        timer=self._timer,
                    )
                    quality = validate_geometry(global_mesh, global_geo)
                    enforce_quality_thresholds(quality, resolved_setup.mesh)
                    from ..io.backup import mesh_hash

                    global_hash = mesh_hash(global_mesh)
                except Exception as error:
                    preparation_error = {
                        "rank": self.parallel.rank,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
            preparation_error = self.parallel.bcast(preparation_error, root=0)
            if preparation_error is not None:
                raise RuntimeError(
                    "Partitioned mesh preparation failed: "
                    + json.dumps(preparation_error, sort_keys=True)
                )

            # Scatter would force rank zero to retain one complete localized
            # payload per rank.  Send one payload at a time instead; every
            # receiver participates in the following error broadcast before
            # it starts solver construction, so a late localization failure
            # cannot leave a peer in an incompatible collective.
            distribution_error = None
            local_payload = None
            payload = None
            received_payload = None
            if self.parallel.is_root:
                from ..mesh.partition import localize_mesh_and_geometry

                assert (
                    global_mesh is not None and global_geo is not None and global_hash is not None
                )
                # Send worker partitions first and build rank zero last.  More
                # importantly, drop each sent payload before constructing the
                # next one.  Keeping the previous payload alive during the
                # following localization made rank zero hold the global mesh
                # plus two complete local partitions at once.
                rank_order = [*range(1, self.parallel.size), 0]
                delivered: set[int] = set()
                for rank in rank_order:
                    try:
                        payload = localize_mesh_and_geometry(
                            global_mesh,
                            global_geo,
                            rank,
                            self.parallel.size,
                            include_visualization_ghosts=resolved_setup.output.ghost_layers == 1,
                        )
                        payload[0]["global_mesh_hash"] = global_hash
                    except Exception as error:
                        distribution_error = {
                            "rank": rank,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        if rank == 0:
                            local_payload = None
                        for destination in range(1, self.parallel.size):
                            if destination not in delivered:
                                comm.send((False, distribution_error), dest=destination, tag=9131)
                        break
                    if rank == 0:
                        local_payload = payload
                    else:
                        comm.send((True, payload), dest=rank, tag=9131)
                        delivered.add(rank)
                        payload = None
            else:
                received_ok, received_payload = comm.recv(source=0, tag=9131)
                if received_ok:
                    local_payload = received_payload
                else:
                    distribution_error = received_payload
            distribution_error = self.parallel.bcast(distribution_error, root=0)
            if distribution_error is not None:
                raise RuntimeError(
                    "Partitioned payload distribution failed: "
                    + json.dumps(distribution_error, sort_keys=True)
                )
            assert local_payload is not None
            self.mesh_data, self.geo_data, partition = local_payload
            self.mesh_quality = self.parallel.bcast(quality, root=0)
            self.parallel = self.parallel.with_partition(partition)
            self.mesh_data["_parallel_context"] = self.parallel

            # Partitioning is the last consumer of the global mesh/geometry.
            # Release every local alias before LSQ, matrix, and field storage is
            # allocated for rank zero; otherwise peak RAM includes both the
            # complete mesh and the fully initialized local solver.
            mesh_data = None
            global_mesh = None
            global_geo = None
            payload = None
            local_payload = None
            received_payload = None
            import gc

            gc.collect()
        else:
            if mesh_data is None:
                raise ValueError(
                    "A solver-native mesh, mesh factory, or Gmsh .msh path is required"
                )
            self._timer.start("Mesh Set (In-Memory)")
            self.mesh_data = mesh_data
            self._timer.log(
                "Mesh Set (In-Memory)",
                sink=self.logger,
            )
            validate_topology(self.mesh_data)
            self.geo_data = geometry.compute_mesh_geometry(
                self.mesh_data,
                gradient_scheme=gs,
                compute_lsq=False,
                logger=self.logger,
                timer=self._timer,
            )

        # Boundary configuration precedes immutable backend views because coupled
        # patches augment the operator topology and periodic geometry.
        self.boundaries = self.mesh_data["boundary"]
        self._setup_boundary_conditions()
        from ..mesh.coupled import configure_cyclic_boundaries
        from ..schemes import validate_boundary_conditions

        validate_boundary_conditions(self.boundaries)
        configure_cyclic_boundaries(self.mesh_data, self.geo_data)
        if gs == "lsq" and np.any(self.mesh_data["boundary_neighbour_cell"] >= 0):
            from ..fields.gradients import compute_lsq_geometry

            self.geo_data.update(compute_lsq_geometry(self.mesh_data, self.geo_data))

        # LSQ must be built after periodic topology is installed.  Non-cyclic
        # meshes also reach this point exactly once, rather than during base
        # geometry and again after boundary setup.
        if gs == "lsq" and "least_squares_normal_matrix_inverse" not in self.geo_data:
            from ..fields.gradients import compute_lsq_geometry

            self.geo_data.update(compute_lsq_geometry(self.mesh_data, self.geo_data))

        self.mesh_quality = validate_geometry(self.mesh_data, self.geo_data)
        enforce_quality_thresholds(self.mesh_quality, resolved_setup.mesh)

        from ..assemble.matrix_assembly import prepare_matrix_assembly

        prepare_matrix_assembly(self.mesh_data)

        self._timer.log(
            "Geometry Compute",
            sink=self.logger,
        )

        # 3. Component Setup
        self._initialize_fields()
        self.state = FieldState(self.velocity, self.kinematic_pressure, self.volumetric_face_flux)
        self._initialize_algorithm()
        self._initialize_turbulence()

        # Final housekeeping
        self.io = solver_io.SolverIO(self)
        self.vtk_exporter = None
        self.pvd_manager = None
        self._buffered_vtk_writer = None
        self.last_forces = None
        self.last_y_plus = None
        self.ibm = None
        self.forces_history_path = None
        self.max_courant_number = 0.0
        # Coupling / driver-split state
        self.registered_fields: dict[str, np.ndarray] = {}  # named volume fields (fvOptions)
        self._coupling_consistency_filter = None
        # Optional conservative field projection applied after the pressure-
        # velocity solve and before diagnostics, sampling, history commit, and
        # output. Benchmarks can use this to enforce a known invariant
        # subspace without bypassing the solver's normal bookkeeping.
        self._post_solve_state_callback = None
        self._n_committed_time_steps = 0  # number of committed time steps (BDF2 startup gate)
        self._accepted_time_step_size = self.time_step_size
        self._previous_time_step_size = self.time_step_size
        self._last_residuals = None
        self.last_diagnostics = None
        self._derived_fields: dict[object, np.ndarray] = {}
        self._state_revision = 0
        self._step_phase = "accepted"
        self._pending_step_size: float | None = None
        self._pending_acceptance_counters: dict[str, int] | None = None
        self._evolution_failure: BaseException | None = None
        self._closed = False
        self._run_started = False
        self._run_manifest_written = False
        self.run_status = "not_started"
        self.run_failure: BaseException | None = None
        self._initial_output_enabled = (
            True if public_case is None else bool(public_case.run.initial_output)
        )
        self._final_output_enabled = (
            True if public_case is None else bool(public_case.run.final_output)
        )
        self._n_consecutive_accepted_steps = {
            "max_continuity_error": 0,
            "max_equation_residual": 0,
            "max_courant_number": 0,
            "max_velocity_magnitude": 0,
        }

        initialization_time = self._timer.stop("Total Initialization")
        self.logger.log_solver_info(self, initialization_time)

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=self.volumetric_face_flux,
        )

        # Wall y+ is an optional scientific output.  Keep it opt-in so a case
        # with no configured samplers does not silently create ``samples/`` or
        # perform extra boundary work.  ``YPlusSampler`` remains available as
        # an explicit sampler in the public case configuration.
        self._default_yplus_sampler = None
        self.write_run_manifest(status="created")

    def _setup_boundary_conditions(self):
        """Map user-defined BoundaryConfig entries to internal mesh boundary data.

        Iterates over ``self.setup.boundaries`` and updates the
        corresponding entries in ``self.boundaries`` with the configured
        type and value for velocity, kinematic_pressure, and eddy_viscosity.
        Patches not found in the mesh
        trigger a warning.
        """
        for b_cfg in self._resolved_setup.boundaries:
            found = False
            for b_mesh in self.boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.velocity_value, dtype=np.float64)
                    if self.parallel.is_partitioned and velocity.ndim == 2:
                        ranges = self.mesh_data.get("global_boundary_ranges", {})
                        if b_cfg.name in ranges:
                            global_start, global_count = ranges[b_cfg.name]
                            if velocity.shape == (global_count, 3):
                                start = b_mesh["start_face"]
                                faces = self.mesh_data["global_face_id"][
                                    start : start + b_mesh["n_faces"]
                                ]
                                velocity = velocity[faces - global_start]
                    if velocity.shape not in {(3,), (b_mesh["n_faces"], 3)}:
                        raise ValueError(
                            f"Velocity value for patch {b_cfg.name!r} has shape {velocity.shape}; "
                            f"expected (3,) or {(b_mesh['n_faces'], 3)}"
                        )
                    b_mesh.update(
                        {
                            "velocity_type": b_cfg.velocity_type,
                            "pressure_type": b_cfg.pressure_type,
                            "kinematic_pressure_value": b_cfg.kinematic_pressure_value,
                            "eddy_viscosity_type": b_cfg.eddy_viscosity_type,
                            "eddy_viscosity_value": b_cfg.eddy_viscosity_value,
                        }
                    )
                    if b_cfg.mesh_type is not None:
                        b_mesh["type"] = b_cfg.mesh_type
                    else:
                        b_mesh.setdefault("type", "patch")
                    if b_cfg.neighbour_patch is not None:
                        b_mesh["neighbour_patch"] = b_cfg.neighbour_patch
                    if velocity.shape == (3,):
                        b_mesh["velocity_value"] = velocity
                        b_mesh.pop("velocity_value_field", None)
                    else:
                        b_mesh["velocity_value_field"] = velocity
                    found = True
                    break
            if not found:
                global_names = self.mesh_data.get("global_boundary_names", ())
                if not self.parallel.is_partitioned or b_cfg.name not in global_names:
                    raise ValueError(
                        f"Configured boundary {b_cfg.name!r} was not found in the mesh"
                    )

    def _initialize_fields(self):
        """Initialise velocity, kinematic pressure, and volumetric face flux.

        Loads or creates the initial fields, enforces boundary constraints
        on the velocity ghost layer, and computes the initial volumetric face
        flux ``volumetric_face_flux = velocity·Sf`` from the velocity field.
        """
        n_cells = self.mesh_data["n_cells"]
        n_total = self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"] + n_cells

        self.velocity = _load_velocity_field(
            self._resolved_setup, self.case_dir, n_total, self.mesh_data
        )
        self.kinematic_pressure = _load_kinematic_pressure_field(
            self._resolved_setup, self.case_dir, n_total, self.mesh_data
        )
        self.velocity_old = self.velocity.copy()
        # Second history level for BDF2 (u^{n-1}); ignored by BDF1.
        self.velocity_older = self.velocity.copy()

        _enforce_velocity_boundary_constraints(
            self.velocity, self.boundaries, n_cells, self.mesh_data, self.geo_data
        )
        self.parallel.exchange_halo(self.velocity[:n_cells])
        self.parallel.exchange_halo(self.kinematic_pressure[:n_cells])

        self._timer.start("Flux Init")
        from ..assemble import convection

        self.volumetric_face_flux = convection.compute_volumetric_face_flux(
            self.velocity, self.mesh_data, self.geo_data
        )
        # Flux history for the transient Rhie-Chow correction.
        # (``fvc::ddtCorr``), which needs flux and velocity at the same levels.
        self.volumetric_face_flux_old = self.volumetric_face_flux.copy()
        self.volumetric_face_flux_older = self.volumetric_face_flux.copy()
        self._timer.log("Flux Init", sink=self.logger)

    def _initialize_algorithm(self):
        """Initialise the numerical solver algorithm.

        Reads the algorithm type from ``self.setup.pimple.algorithm``
        and instantiates either a :class:`~solve.pimple_solver.PIMPLESolver`
        or :class:`~solve.simple_solver.SIMPLESolver`.

        Raises:
            ValueError: If the algorithm is not ``"SIMPLE"``, ``"PIMPLE"``,
                        or ``"PISO"``.
        """
        self._timer.start("Algorithm Init")
        params = dict(self._resolved_setup.algorithm_params())
        params["_linear_backend"] = self._resolved_setup.execution.linear_backend
        params["_operator_backend"] = self.operator_backend
        self.geo_data["_operator_backend"] = self.operator_backend
        params["_parallel_context"] = self.parallel
        params["_logger"] = self.logger
        params["_timer"] = self._timer
        algo = self._resolved_setup.pimple.algorithm.upper()

        if algo in ["PIMPLE", "PISO"]:
            self.algorithm = pimple_solver.PIMPLESolver(
                self.mesh_data, self.geo_data, self.boundaries, params
            )
        elif algo == "SIMPLE":
            self.algorithm = simple_solver.SIMPLESolver(
                self.mesh_data, self.geo_data, self.boundaries, params
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        self._timer.log("Algorithm Init", sink=self.logger)

    def set_initial_velocity(self, values: np.ndarray) -> None:
        """Set a cell-centred initial velocity and rebuild dependent state.

        Parameters
        ----------
        values : numpy.ndarray
            Finite interior velocity field with shape ``(n_cells, 3)`` in
            m/s. In partitioned mode this is the local owned-plus-halo view
            expected by the selected solver.

        Raises
        ------
        RuntimeError
            If a physical step has already been committed or the solver is
            terminal after a failed evolution.
        ValueError
            If the shape or values are not finite.

        Notes
        -----
        ``values`` contains one vector per interior cell. Boundary ghosts are
        reconstructed from the configured boundary conditions, and both BDF
        history levels and the face flux are reset to the resulting field. In
        partitioned execution, ``values`` is the rank-local owned-plus-halo
        field; owned values are exchanged so every halo is made consistent.
        This operation is only valid before the first time step is committed.
        """
        self._ensure_evolution_usable()
        if self._n_committed_time_steps or self.step or self._step_phase != "accepted":
            raise RuntimeError("Initial velocity can only be set before the first time step")

        n_cells = self.mesh_data["n_cells"]
        field = np.asarray(values, dtype=np.float64)
        if field.shape != (n_cells, 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Initial velocity must be finite with shape ({n_cells}, 3); got {field.shape}"
            )

        snapshot = {
            name: np.array(getattr(self, name), copy=True)
            for name in (
                "velocity",
                "kinematic_pressure",
                "velocity_old",
                "velocity_older",
                "volumetric_face_flux",
                "volumetric_face_flux_old",
                "volumetric_face_flux_older",
            )
        }
        revision = self._state_revision
        self._invalidate_derived_fields()
        try:
            self.velocity[:n_cells] = field
            if self.parallel.is_partitioned:
                self.parallel.exchange_halo(self.velocity[:n_cells])
            _enforce_velocity_boundary_constraints(
                self.velocity, self.boundaries, n_cells, self.mesh_data, self.geo_data
            )
            self.velocity_old[:] = self.velocity
            self.velocity_older[:] = self.velocity

            from ..assemble import convection
            from ..solve import simple_solver

            self.volumetric_face_flux = convection.compute_volumetric_face_flux(
                self.velocity, self.mesh_data, self.geo_data
            )
            self.volumetric_face_flux_old = self.volumetric_face_flux.copy()
            self.volumetric_face_flux_older = self.volumetric_face_flux.copy()
            simple_solver.update_scalar_boundaries(
                self.kinematic_pressure,
                self.mesh_data,
                self.boundaries,
                "kinematic_pressure",
                volumetric_face_flux=self.volumetric_face_flux,
            )
            self._publish_state()
        except BaseException:
            for name, values in snapshot.items():
                setattr(self, name, values)
            self._invalidate_derived_fields()
            self._publish_state()
            self._state_revision = revision
            raise

    def set_initial_state(self, velocity: np.ndarray, kinematic_pressure: np.ndarray) -> None:
        """Set a complete cell-centred initial state before the first step.

        Parameters
        ----------
        velocity : numpy.ndarray
            Interior velocity, shape ``(n_cells, 3)``, in m/s.
        kinematic_pressure : numpy.ndarray
            Interior ``p/rho`` field, shape ``(n_cells,)``, in m²/s².

        Raises
        ------
        RuntimeError
            If called after the first accepted step.
        ValueError
            If either array has the wrong shape or contains non-finite data.

        Notes
        -----
        This is intentionally narrower than backup loading: it supports
        deterministic manufactured/replay starts while retaining the solver's
        own boundary reconstruction, flux construction, and time-history
        ownership.
        """
        self._ensure_evolution_usable()
        if self._n_committed_time_steps or self.step or self._step_phase != "accepted":
            raise RuntimeError("Initial state can only be set before the first time step")
        n_cells = self.mesh_data["n_cells"]
        velocity = np.asarray(velocity, dtype=np.float64)
        kinematic_pressure = np.asarray(kinematic_pressure, dtype=np.float64)
        if velocity.shape != (n_cells, 3) or not np.all(np.isfinite(velocity)):
            raise ValueError(
                f"Initial velocity must be finite with shape ({n_cells}, 3); got {velocity.shape}"
            )
        if kinematic_pressure.shape != (n_cells,) or not np.all(np.isfinite(kinematic_pressure)):
            raise ValueError(
                "Initial kinematic_pressure must be finite with one value per interior cell; "
                f"got {kinematic_pressure.shape}, expected ({n_cells},)"
            )
        # Validate the complete proposed state before publishing either field.
        snapshot = {
            name: np.array(getattr(self, name), copy=True)
            for name in (
                "velocity",
                "kinematic_pressure",
                "velocity_old",
                "velocity_older",
                "volumetric_face_flux",
                "volumetric_face_flux_old",
                "volumetric_face_flux_older",
            )
        }
        revision = self._state_revision
        try:
            # Keep the complete setter transactional and publish exactly one
            # state revision.  Calling set_initial_velocity() here would
            # publish an intermediate pressure/flux state before the supplied
            # pressure field had been admitted.
            self._invalidate_derived_fields()
            self.velocity[:n_cells] = velocity
            if self.parallel.is_partitioned:
                self.parallel.exchange_halo(self.velocity[:n_cells])
            _enforce_velocity_boundary_constraints(
                self.velocity, self.boundaries, n_cells, self.mesh_data, self.geo_data
            )
            self.velocity_old[:] = self.velocity
            self.velocity_older[:] = self.velocity
            from ..assemble import convection

            self.kinematic_pressure[:n_cells] = kinematic_pressure
            from ..solve import simple_solver

            self.volumetric_face_flux = convection.compute_volumetric_face_flux(
                self.velocity, self.mesh_data, self.geo_data
            )
            self.volumetric_face_flux_old = self.volumetric_face_flux.copy()
            self.volumetric_face_flux_older = self.volumetric_face_flux.copy()
            simple_solver.update_scalar_boundaries(
                self.kinematic_pressure,
                self.mesh_data,
                self.boundaries,
                "kinematic_pressure",
                volumetric_face_flux=self.volumetric_face_flux,
            )
            self._invalidate_derived_fields()
            self._publish_state()
        except BaseException:
            for name, values in snapshot.items():
                setattr(self, name, values)
            self._invalidate_derived_fields()
            self._publish_state()
            self._state_revision = revision
            raise

    def set_post_solve_state_callback(self, callback) -> None:
        """Set a state projection called before accepted-step diagnostics.

        Parameters
        ----------
        callback : callable or None
            Function receiving this solver after the linear solve. It may
            mutate solver-owned cell fields/flux in place. ``None`` removes the
            callback.

        Notes
        -----
        The callback receives this solver and may update owned cell values and
        face fluxes in place. The solver then exchanges cell halos and rebuilds
        velocity and pressure boundary ghosts before evaluating continuity,
        forces, samplers, and output. Passing ``None`` disables the hook.
        """
        if callback is not None and not callable(callback):
            raise TypeError("post-solve state callback must be callable or None")
        self._post_solve_state_callback = callback

    def _initialize_turbulence(self):
        """Initialise the turbulence / LES model if configured.

        Uses :func:`..turbulence.create_model` to instantiate the model
        specified by ``self.setup.turbulence``.  Stores the result in
        ``self.turbulence`` and logs the model info.  Sets
        ``self.time`` and ``self.step`` to their initial values.
        """
        self.turbulence = None
        self.eddy_viscosity = None
        if (
            self._resolved_setup.turbulence
            and self._resolved_setup.turbulence.model.lower() != "none"
        ):
            from ..turbulence import create_model

            self.turbulence = create_model(
                self._resolved_setup.turbulence, self.mesh_data, self.geo_data
            )
            if self.turbulence is None:
                raise RuntimeError(
                    f"Turbulence model {self._resolved_setup.turbulence.model!r} returned no model"
                )

        # Sync state
        self.time = self._time_config.start_time
        self.step = 0
        self.time_step_size = self._time_config.time_step_size

    def compute_effective_viscosity(self):
        """Compute the effective viscosity (molecular + turbulent).

        Returns
        -------
        float or numpy.ndarray
            Molecular kinematic viscosity, or ``nu + nu_t`` per cell when a
            turbulence model is active, in m²/s. Partitioned results include
            the local halo layout.

        Raises
        ------
        FloatingPointError
            If the turbulence model returns a non-finite or negative eddy
            viscosity.

        Notes
        -----
        If a turbulence model is active, computes the subgrid eddy viscosity
        and returns ``kinematic_viscosity + eddy_viscosity``. Model failures propagate because silently
        switching a configured simulation to laminar flow is unsafe.

        Returns:
            Effective kinematic viscosity (scalar or per-element array).
        """
        if self.turbulence is not None:
            self.eddy_viscosity = self.turbulence.compute_eddy_viscosity(
                self.velocity, self.mesh_data, self.geo_data
            )
            self.parallel.exchange_halo(self.eddy_viscosity[: self.mesh_data["n_cells"]])
            if not np.all(np.isfinite(self.eddy_viscosity)) or np.any(self.eddy_viscosity < 0.0):
                raise FloatingPointError("Turbulence model returned invalid eddy viscosity")
            return self._kinematic_viscosity + self.eddy_viscosity
        return self._kinematic_viscosity

    def set_immersed_bodies(self, bodies, grid_spacing: float | None = None) -> "object":
        """Attach immersed bodies (discrete direct-forcing IBM) to the solver.

        Builds the interpolation/spreading operators (Pinelli et al. 2010,
        Constant et al. — see docs/literature/Constant2016.pdf) on the
        live mesh and hooks them into the PIMPLE momentum predictor.  Body
        forces are appended to ``samples/ibm_forces_history.csv`` every step.

        Args:
            bodies: One :class:`ImmersedBody` or a list of them.
            grid_spacing: Eulerian grid spacing near the bodies; inferred from the
                    mesh when ``None``.

        Returns:
            The constructed :class:`IBMForcing` (for direct inspection).

        Raises:
            ValueError: If the selected algorithm cannot host IBM forcing or
                the body/grid data are invalid.

        Side Effects:
            Installs the forcing object on the live algorithm and enables the
            per-step IBM force history sampler.
        """
        from ..immersed_boundary import IBMForcing

        if not hasattr(self.algorithm, "ibm"):
            raise ValueError(
                "Immersed boundaries require the PIMPLE/PISO algorithm "
                f"(configured: {self._resolved_setup.pimple.algorithm!r})."
            )
        body_list = [bodies] if hasattr(bodies, "prescribed_velocity") else list(bodies)
        if not body_list:
            raise ValueError("At least one immersed body is required")
        moving = [body.name for body in body_list if np.any(body.prescribed_velocity != 0.0)]
        if moving:
            raise NotImplementedError(
                "Moving immersed bodies require body-motion/ALE energy accounting, which is "
                f"not implemented; nonzero target velocity configured for {moving}"
            )
        self.ibm = IBMForcing(
            self.mesh_data,
            self.geo_data,
            body_list,
            grid_spacing=grid_spacing,
        )
        self.algorithm.ibm = self.ibm
        from ..sampling.forces import IBMForceSampler

        if not any(isinstance(s, IBMForceSampler) for s in self._samplers):
            self._default_ibm_sampler = IBMForceSampler()
        diag = self.ibm.diagnostics()
        self.logger.record(
            "immersed boundary",
            ("markers", diag["n_markers_total"]),
            ("grid spacing", diag["grid_spacing"], "m"),
            *(
                (f"{name}, marker-spacing ratio", ratio)
                for name, ratio in diag["marker_spacing_ratio_by_body"].items()
            ),
            ("kernel row sum, min", diag["min_kernel_row_sum"]),
            ("kernel row sum, max", diag["max_kernel_row_sum"]),
            ("quadrature residual, max", diag["max_quadrature_residual"]),
        )
        return self.ibm

    def _coupling_consistency_source(self):
        """Build the resolved-scale VPM-to-FVM relaxation source, if registered.

        The implicit part is ``Sp = rate``. The explicit target retains the
        current FVM high-pass fluctuation, so only scales representable by the
        VPM velocity field are relaxed:

        ``S = rate * (target + (U - filter(U))) - rate * U``.
        """
        rate_field = "couplingConsistencyRate"
        target_field = "couplingConsistencyTargetVelocity"
        rate = self.registered_fields.get(rate_field)
        target = self.registered_fields.get(target_field)
        if rate is None and target is None:
            return None, None
        if rate is None or target is None:
            raise RuntimeError(
                f"Incomplete coupling consistency source: {rate_field} and "
                f"{target_field} must be registered together"
            )
        n_cells = int(self.mesh_data["n_cells"])
        rate = np.asarray(rate, dtype=np.float64)[:n_cells]
        target = np.asarray(target, dtype=np.float64)[:n_cells]
        if rate.shape != (n_cells,) or target.shape != (n_cells, 3):
            raise RuntimeError("Coupling consistency fields have incompatible cell shapes")
        if np.any(rate < 0.0) or not np.all(np.isfinite(rate)) or not np.all(np.isfinite(target)):
            raise RuntimeError("Coupling consistency fields must be finite with non-negative rate")

        if self._coupling_consistency_filter is None:
            from ..fields.filters import CellBoxFilter

            self._coupling_consistency_filter = CellBoxFilter(
                self.mesh_data,
                self.geo_data,
                centre_weight="neighbour_sum",
            )
        velocity = np.asarray(self.velocity, dtype=np.float64)[:n_cells]
        high_pass_velocity = velocity - self._coupling_consistency_filter(velocity)
        return rate[:, np.newaxis] * (target + high_pass_velocity), rate

    def _ensure_evolution_usable(self) -> None:
        """Reject continuation after a numerical candidate has failed."""
        if self._evolution_failure is not None:
            raise RuntimeError(
                "FVMSolver is terminally invalid after a failed physical step; "
                "load a compatible accepted backup before continuing"
            ) from self._evolution_failure

    def _mark_evolution_failure(self, error: BaseException) -> None:
        """Latch the first physical-step failure and invalidate its candidate."""
        if self._evolution_failure is None:
            self._evolution_failure = error
        self._step_phase = "failed"
        self._pending_step_size = None
        self._pending_acceptance_counters = None
        self.run_status = "failed"
        self.run_failure = self._evolution_failure
        # Reporting must never replace the numerical failure.
        with suppress(Exception):
            self.logger.warning(f"FVM step failed: {type(error).__name__}: {error}")

    def _collective_io_failure(self, error: BaseException | None, operation: str) -> None:
        """Propagate a root/local output error before any rank continues.

        Numerical kernels already use their own collectives.  Filesystem and
        metadata publication is different: a root-only exception must still
        release every peer from the same stage, otherwise the next collective
        turns a useful I/O error into an MPI hang.
        """
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_parallel:
            if error is not None:
                raise error
            return
        local = None
        if error is not None:
            local = {
                "rank": int(parallel.rank),
                "type": type(error).__name__,
                "message": str(error),
            }
        failures = parallel.comm.allgather(local)
        failure = next((item for item in failures if item is not None), None)
        if failure is not None:
            raise RuntimeError(
                f"FVM collective {operation} failed on rank {failure['rank']} "
                f"({failure['type']}): {failure['message']}"
            )

    def _execute_sampler_event(self, event: str) -> None:
        """Dispatch one sampler event and envelope rank-local failures."""
        sampler_error = None
        try:
            FVMSamplerExecutor.execute(self, event=event)
        except BaseException as error:
            sampler_error = error
        self._collective_io_failure(sampler_error, f"{event} sampler output")

    def solve_pimple(self, time_step_size: float | None = None):
        """Solve one transient pressure--velocity candidate without advancing time.

        Parameters
        ----------
        time_step_size : float or None
            Candidate duration in seconds. ``None`` uses the solver-selected
            step. A repeated call while a candidate is pending must use the
            same value.

        Returns
        -------
        StepDiagnostics
            Structured residual, continuity, Courant, and acceptance data for
            the candidate state (the exact concrete mapping is retained for
            compatibility by the low-level solver).

        Raises
        ------
        RuntimeError
            If the setup is SIMPLE, a candidate is already in an incompatible
            phase, or an earlier physical step failed.
        ValueError
            If the requested duration is not finite and positive.

        Notes
        -----
        This is the coupler-facing half of the transaction. It may be called
        repeatedly for a boundary-condition/pressure Picard iteration; call
        :meth:`advance_time` exactly once after the candidate passes acceptance.
        The committed ``U_old`` is the transient reference on every repeated
        solve.
        """
        from ..fields import diagnostics

        self._ensure_evolution_usable()
        if not self._run_started and self.run_status == "not_started":
            self.run_status = "interactive"
        if self._resolved_setup.pimple.algorithm == "SIMPLE":
            raise RuntimeError(
                "SIMPLE is a steady algorithm; use solve_steady() instead of solve_pimple()"
            )
        if self._step_phase not in {"accepted", "candidate"}:
            raise RuntimeError("FVM physical step is not available for another solve")
        if self._step_phase == "candidate":
            expected_step_size = self._pending_step_size
            if expected_step_size is None:
                raise RuntimeError("FVM candidate step has no time-step token")
            if time_step_size is not None and not np.isclose(
                float(time_step_size), expected_step_size, rtol=0.0, atol=1.0e-14
            ):
                raise RuntimeError(
                    "Repeated FVM solves for one physical step must use the same time-step size"
                )
            step_time_step_size = expected_step_size
        else:
            step_time_step_size = (
                self.time_step_size if time_step_size is None else float(time_step_size)
            )
            if not np.isfinite(step_time_step_size) or step_time_step_size <= 0.0:
                raise ValueError("FVM time-step size must be finite and positive")
            self._pending_step_size = step_time_step_size
        self._accepted_time_step_size = float(step_time_step_size)

        # Diagnostics from the previously completed step cache full-mesh
        # Courant, velocity-gradient, and vorticity arrays.  None is valid once
        # a new solve begins, and retaining them through momentum/pressure
        # assembly adds roughly thirteen float64 values per local cell to the
        # transient peak.  Release them before turbulence and PIMPLE allocate
        # their workspaces; the solved state invalidation below remains the
        # guard for fields requested during a re-entrant/coupled solve.
        self._invalidate_derived_fields()

        try:
            self._timer.start("Effective viscosity")
            effective_viscosity = self.compute_effective_viscosity()
            self._timer.log(
                "Effective viscosity",
                sink=self.logger,
            )
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        # BDF2 needs u^{n-1}; available only once at least one step is committed.
        velocity_older_argument = self.velocity_older if self._n_committed_time_steps >= 1 else None
        try:
            source_explicit, source_implicit = self._coupling_consistency_source()
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        try:
            self.velocity, self.kinematic_pressure, self.volumetric_face_flux, residuals = (
                self.algorithm.step(
                    self.velocity,
                    self.kinematic_pressure,
                    self.volumetric_face_flux,
                    self.velocity_old,
                    step_time_step_size,
                    density=self._resolved_setup.transport.density,
                    kinematic_viscosity=effective_viscosity,
                    velocity_older=velocity_older_argument,
                    source_explicit=source_explicit,
                    source_implicit=source_implicit,
                    volumetric_face_flux_old=self.volumetric_face_flux_old,
                    volumetric_face_flux_older=(
                        self.volumetric_face_flux_older
                        if self._n_committed_time_steps >= 1
                        else None
                    ),
                    previous_time_step_size=(
                        self._previous_time_step_size if self._n_committed_time_steps >= 1 else None
                    ),
                )
            )
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        try:
            residuals = {str(name): float(value) for name, value in residuals.items()}
            if self._post_solve_state_callback is not None:
                self._post_solve_state_callback(self)
                n_cells = self.mesh_data["n_cells"]
                if self.parallel.is_partitioned:
                    self.parallel.exchange_halo(self.velocity[:n_cells])
                    self.parallel.exchange_halo(self.kinematic_pressure[:n_cells])
                _enforce_velocity_boundary_constraints(
                    self.velocity,
                    self.boundaries,
                    n_cells,
                    self.mesh_data,
                    self.geo_data,
                )
                simple_solver.update_scalar_boundaries(
                    self.kinematic_pressure,
                    self.mesh_data,
                    self.boundaries,
                    "kinematic_pressure",
                    volumetric_face_flux=self.volumetric_face_flux,
                )
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        try:
            self._invalidate_derived_fields()
            self._publish_state()
            self._last_residuals = residuals
            self.logger.convergence_info(residuals)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise

        ibm = getattr(self, "ibm", None)
        if ibm is not None:
            try:
                ibm.update_fictitious_fluid_momentum_rate(
                    self.velocity,
                    self.velocity_old,
                    step_time_step_size,
                )
            except BaseException as error:
                self._mark_evolution_failure(error)
                raise

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this makes loss of mass
        # conservation visible instead of silent.
        try:
            self._timer.start("Continuity diagnostics")
            continuity_error = diagnostics.compute_continuity_error(
                self.volumetric_face_flux, self.mesh_data, self.geo_data
            )
            cell_volume = self.geo_data["cell_volume"]
            n_owned = self.parallel.n_owned if self.parallel.is_partitioned else len(cell_volume)
            local_max = (
                float(np.max(np.abs(continuity_error[:n_owned]) / (cell_volume[:n_owned] + 1e-30)))
                if n_owned
                else 0.0
            )
            local_sum = float(np.sum(np.abs(continuity_error[:n_owned])))
            self.max_continuity_error = float(self.parallel.global_max(local_max))
            self.sum_absolute_continuity_error = float(self.parallel.global_sum(local_sum))
            self.logger.continuity_info(
                self.max_continuity_error,
                self.sum_absolute_continuity_error,
            )
            self._timer.log("Continuity diagnostics", sink=self.logger)

            self._timer.start("Acceptance checks")
            self.last_diagnostics = self._build_step_diagnostics(step_time_step_size, residuals)
            self._enforce_acceptance_limits(self.last_diagnostics)
            self._timer.log("Acceptance checks", sink=self.logger)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        self._step_phase = "candidate"
        return residuals

    def _build_step_diagnostics(self, step_time_step_size, residuals):
        """Build the backend-neutral health record for the current solved state."""
        from ..fields import diagnostics
        from ..solve.diagnostics import StepDiagnostics

        n_cells = self.mesh_data["n_cells"]
        n_owned = self.parallel.n_owned if self.parallel.is_partitioned else n_cells
        interior_velocity = np.asarray(self.velocity[:n_owned])
        interior_kinematic_pressure = np.asarray(self.kinematic_pressure[:n_owned])
        self.max_courant_number = self._measure_maximum_courant_number(step_time_step_size)
        local_nonfinite = int(
            np.count_nonzero(~np.isfinite(interior_velocity))
            + np.count_nonzero(~np.isfinite(interior_kinematic_pressure))
            + np.count_nonzero(~np.isfinite(self.volumetric_face_flux))
        )
        n_nonfinite_values = int(self.parallel.global_sum(local_nonfinite))
        min_eddy_viscosity = None
        max_eddy_viscosity = None
        if self.eddy_viscosity is not None:
            n_nonfinite_values += int(
                self.parallel.global_sum(
                    int(np.count_nonzero(~np.isfinite(self.eddy_viscosity[:n_owned])))
                )
            )
            local_eddy = np.asarray(self.eddy_viscosity[:n_owned])
            min_eddy_viscosity = float(
                self.parallel.global_min(float(np.nanmin(local_eddy)) if n_owned else float("inf"))
            )
            max_eddy_viscosity = float(
                self.parallel.global_max(float(np.nanmax(local_eddy)) if n_owned else float("-inf"))
            )
        n_interior = self.mesh_data["n_interior_faces"]
        linear_results = tuple(getattr(self.algorithm, "last_linear_results", ()))
        local_min_velocity = np.nanmin(interior_velocity, axis=0) if n_owned else np.full(3, np.inf)
        local_max_velocity = (
            np.nanmax(interior_velocity, axis=0) if n_owned else np.full(3, -np.inf)
        )
        min_velocity = np.asarray(
            [self.parallel.global_min(float(value)) for value in local_min_velocity]
        )
        max_velocity = np.asarray(
            [self.parallel.global_max(float(value)) for value in local_max_velocity]
        )
        if n_owned:
            finite_velocity = np.where(np.isfinite(interior_velocity), interior_velocity, 0.0)
            local_max_velocity_magnitude = float(np.max(np.linalg.norm(finite_velocity, axis=1)))
        else:
            local_max_velocity_magnitude = 0.0
        max_velocity_magnitude = float(self.parallel.global_max(local_max_velocity_magnitude))
        min_kinematic_pressure = float(
            self.parallel.global_min(
                float(np.nanmin(interior_kinematic_pressure)) if n_owned else float("inf")
            )
        )
        max_kinematic_pressure = float(
            self.parallel.global_max(
                float(np.nanmax(interior_kinematic_pressure)) if n_owned else float("-inf")
            )
        )
        local_kinetic_energy = (
            0.5
            * self._resolved_setup.transport.density
            * float(
                np.sum(
                    self.geo_data["cell_volume"][:n_owned]
                    * np.sum(interior_velocity * interior_velocity, axis=1)
                )
            )
        )
        local_enstrophy_integral = diagnostics.enstrophy_from_gradient(
            self._velocity_gradient(),
            self.geo_data["cell_volume"],
            n_owned,
        )
        projection_diagnostics = {}
        if self._post_solve_state_callback is not None:
            projection_diagnostics = {
                str(name): float(value)
                for name, value in getattr(
                    self._post_solve_state_callback,
                    "last_removed_maximum",
                    {},
                ).items()
            }
        return StepDiagnostics(
            algorithm=self._resolved_setup.pimple.algorithm.upper(),
            step=self.step + 1,
            time=self.time + step_time_step_size,
            time_step_size=float(step_time_step_size),
            residuals={key: float(value) for key, value in residuals.items()},
            outer_correctors=tuple(getattr(self.algorithm, "last_outer_diagnostics", ())),
            linear_solves=linear_results,
            max_continuity_error=self.max_continuity_error,
            sum_absolute_continuity_error=self.sum_absolute_continuity_error,
            net_boundary_volumetric_flux=float(
                self.parallel.global_sum(float(np.sum(self.volumetric_face_flux[n_interior:])))
            ),
            max_courant_number=self.max_courant_number,
            min_velocity=(
                float(min_velocity[0]),
                float(min_velocity[1]),
                float(min_velocity[2]),
            ),
            max_velocity=(
                float(max_velocity[0]),
                float(max_velocity[1]),
                float(max_velocity[2]),
            ),
            max_velocity_magnitude=max_velocity_magnitude,
            min_kinematic_pressure=min_kinematic_pressure,
            max_kinematic_pressure=max_kinematic_pressure,
            n_nonfinite_values=n_nonfinite_values,
            total_kinetic_energy=float(self.parallel.global_sum(local_kinetic_energy)),
            total_enstrophy=float(self.parallel.global_sum(local_enstrophy_integral)),
            min_eddy_viscosity=min_eddy_viscosity,
            max_eddy_viscosity=max_eddy_viscosity,
            state_projection=projection_diagnostics,
        )

    def _enforce_acceptance_limits(self, diagnostics) -> None:
        """Reject unhealthy solves using explicit immediate and sustained rules."""
        from dataclasses import replace

        if diagnostics.n_nonfinite_values:
            raise FloatingPointError(
                f"FVM step contains {diagnostics.n_nonfinite_values} non-finite field values"
            )
        failed = [result for result in diagnostics.linear_solves if not result.converged]
        if failed:
            raise RuntimeError(f"FVM step contains {len(failed)} failed linear solve(s)")
        if diagnostics.min_eddy_viscosity is not None and diagnostics.min_eddy_viscosity < 0.0:
            raise FloatingPointError("FVM step contains negative turbulent viscosity")

        limits = self._resolved_setup.acceptance
        max_velocity = getattr(diagnostics, "max_velocity_magnitude", None)
        if max_velocity is None:
            # Compatibility for manually constructed diagnostic records.  New
            # solver records always carry the true cell-wise maximum norm.
            max_velocity = max(
                float(np.linalg.norm(diagnostics.min_velocity)),
                float(np.linalg.norm(diagnostics.max_velocity)),
            )
        max_velocity = float(max_velocity)
        metrics = {
            "max_continuity_error": diagnostics.max_continuity_error,
            "max_equation_residual": max(
                diagnostics.residuals.get("velocity", 0.0),
                diagnostics.residuals.get("kinematic_pressure", 0.0),
            ),
            "max_courant_number": diagnostics.max_courant_number,
            "max_velocity_magnitude": max_velocity,
        }
        warnings = []
        trial_counters = dict(self._n_consecutive_accepted_steps)
        for name, value in metrics.items():
            warning = getattr(limits, f"{name}_warning")
            abort = getattr(limits, f"{name}_abort")
            if warning is not None and value > warning:
                warnings.append(f"{name}={value:.6g} exceeds warning threshold {warning:.6g}")
            if abort is not None and value > abort:
                trial_counters[name] += 1
            else:
                trial_counters[name] = 0
            if trial_counters[name] >= limits.sustained_steps:
                raise RuntimeError(
                    f"FVM acceptance limits rejected the step: {name}={value:.6g} "
                    f"exceeded {abort:.6g} for {trial_counters[name]} "
                    "consecutive accepted step(s)"
                )
        self._pending_acceptance_counters = trial_counters
        self.last_diagnostics = replace(diagnostics, warnings=tuple(warnings))
        self.logger.warnings_info(tuple(warnings))

    def _measure_maximum_courant_number(self, time_step_size: float) -> float:
        """Return the global maximum CFL for the current field state."""
        n_owned = (
            self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
        )
        courant_number_field = self._courant_field(time_step_size)
        local_maximum = float(np.max(courant_number_field[:n_owned])) if n_owned > 0 else 0.0
        return float(self.parallel.global_max(local_maximum))

    def _select_time_step_size(self) -> float:
        """Select one solver-owned step and cap it at the run horizon.

        Fixed stepping reads the immutable construction value; automatic
        stepping applies the configured maximum-Courant control.
        """
        control = self._time_config.adjustment
        time_until_event = None
        if control is None:
            selected = float(self.time_step_size)
        else:
            current_courant_number = self._measure_maximum_courant_number(self.time_step_size)
            selected = maximum_courant_time_step_size(
                self.time_step_size,
                current_courant_number,
                control,
            )

            # Find the nearest deadline first, then distribute the remaining
            # interval over CFL-limited steps. Last-step clipping alone leaves
            # tiny remainders and pressure spikes at otherwise harmless outputs.
            for schedule in self._run_event_schedules():
                next_event_time = schedule.next_time_after(self.time)
                if next_event_time is not None:
                    distance = next_event_time - self.time
                    time_until_event = (
                        distance if time_until_event is None else min(time_until_event, distance)
                    )

        # Lifecycle-managed runs land exactly on their physical horizon.
        remaining = float(self._time_config.end_time - self.time)
        tolerance = max(1.0e-14, abs(self._time_config.end_time) * 1.0e-12)
        if remaining > tolerance:
            time_until_event = (
                remaining if time_until_event is None else min(time_until_event, remaining)
            )
        if time_until_event is not None:
            selected = (
                min(selected, time_until_event)
                if control is None
                else event_aligned_time_step_size(selected, time_until_event)
            )
        self.time_step_size = selected
        return selected

    def _run_event_schedules(self) -> tuple:
        """Return construction-time schedules that constrain adaptive steps."""
        schedules = [self.logger.schedule]
        if self.auto_write:
            schedules.append(self._output_schedule)
        if self._backup_config.schedule is not None:
            schedules.append(self._backup_config.schedule)
        for sampler in self._samplers:
            schedule = self._sampler_schedules.get(id(sampler))
            if schedule is not None:
                schedules.append(schedule)
        return tuple(schedules)

    def solve_steady(self):
        """Run the solver-owned steady SIMPLE loop without advancing time.

        Returns
        -------
        bool
            Whether the configured SIMPLE loop converged. The steady iteration
            count is exposed as `step` for output identity; `time` remains at
            the configured start time.

        Raises
        ------
        RuntimeError
            If the configured algorithm is not SIMPLE, a candidate is pending,
            or the solver has become terminal after a failed evolution.
        FloatingPointError
            If SIMPLE returns a non-finite or shape-incompatible state.

        Notes
        -----
        SIMPLE uses its dedicated steady iteration plan rather than the
        transient candidate/commit path.  ``time`` therefore remains at the
        configured start time; ``step`` is the steady iteration count used for
        output identity and reporting.
        """
        if self._resolved_setup.pimple.algorithm != "SIMPLE":
            raise RuntimeError("solve_steady() requires algorithm='SIMPLE'")
        self._ensure_evolution_usable()
        if self._step_phase != "accepted":
            raise RuntimeError("Cannot start a steady solve with a pending FVM candidate")

        effective_viscosity = self.compute_effective_viscosity()
        result = self.algorithm.solve(
            self.velocity,
            self.kinematic_pressure,
            density=self._resolved_setup.transport.density,
            kinematic_viscosity=effective_viscosity,
        )
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("SIMPLE steady solve must return four state values")
        candidate_velocity, candidate_pressure, candidate_flux, converged = result
        candidate_velocity = np.asarray(candidate_velocity, dtype=np.float64)
        candidate_pressure = np.asarray(candidate_pressure, dtype=np.float64)
        candidate_flux = np.asarray(candidate_flux, dtype=np.float64)
        if (
            candidate_velocity.shape != self.velocity.shape
            or candidate_pressure.shape != self.kinematic_pressure.shape
            or candidate_flux.shape != self.volumetric_face_flux.shape
            or not np.all(np.isfinite(candidate_velocity))
            or not np.all(np.isfinite(candidate_pressure))
            or not np.all(np.isfinite(candidate_flux))
        ):
            raise FloatingPointError("SIMPLE steady solve returned an invalid candidate state")

        self.velocity[:] = candidate_velocity
        self.kinematic_pressure[:] = candidate_pressure
        self.volumetric_face_flux[:] = candidate_flux
        self.parallel.exchange_halo(self.velocity[: self.mesh_data["n_cells"]])
        self.parallel.exchange_halo(self.kinematic_pressure[: self.mesh_data["n_cells"]])
        self.velocity_old[:] = self.velocity
        self.velocity_older[:] = self.velocity
        self.volumetric_face_flux_old[:] = self.volumetric_face_flux
        self.volumetric_face_flux_older[:] = self.volumetric_face_flux
        self._invalidate_derived_fields()
        self._publish_state()

        iterations = len(getattr(self.algorithm, "residuals", ()))
        self.steady_iterations = int(iterations)
        self.steady_converged = bool(converged)
        record_steady_iterations = getattr(self.logger, "steady_iterations", None)
        if record_steady_iterations is not None:
            record_steady_iterations(self.steady_iterations)
        self.step = self.steady_iterations
        self._n_committed_time_steps = self.steady_iterations
        self._accepted_time_step_size = self.time_step_size
        self._previous_time_step_size = self.time_step_size
        self.last_steady_residuals = tuple(getattr(self.algorithm, "residuals", ()))
        self._step_phase = "accepted"
        self._pending_step_size = None
        self._pending_acceptance_counters = None
        self._run_manifest_written = False
        if not self._run_started:
            self.run_status = "complete" if converged else "not_converged"
            self.write_run_manifest(status=self.run_status)
        return self.steady_converged

    def run(self) -> None:
        """Run from the current clock to the configured end time.

        The finite lifecycle writes initial output, executes steady SIMPLE or
        transient accepted steps, writes final output/backups, refreshes solver
        metadata, and closes owned resources. It may be called only once.

        The lifecycle writes the initial state, advances only accepted steps,
        and delegates periodic output and sampling to the solver-owned output
        controller.

        Raises
        ------
        RuntimeError, FloatingPointError
            If a numerical solve, acceptance gate, output writer, or finalizer
            fails. The primary failure is re-raised after collective cleanup.
        """
        if getattr(self, "_run_started", False):
            raise RuntimeError("FVMSolver.run() may be called only once")
        self._run_started = True
        self.run_status = "failed"
        self.run_failure: BaseException | None = None
        primary_failure: BaseException | None = None
        try:
            self._ensure_evolution_usable()
            if self.auto_write and self._initial_output_enabled:
                self.write_vtk()
            if self._initial_output_enabled:
                self._execute_sampler_event("initial")
            if self._resolved_setup.pimple.algorithm == "SIMPLE":
                converged = self.solve_steady()
                if self.auto_write and self._final_output_enabled:
                    self.write_vtk()
                if self._final_output_enabled:
                    self._execute_sampler_event("final")
                if self._backup_config.write_at_end:
                    self._save_automatic_backup()
                self.run_status = "complete" if converged else "not_converged"
            else:
                end_time = float(self._time_config.end_time)
                tolerance = max(1.0e-14, abs(end_time) * 1.0e-12)
                while self.time < end_time - tolerance:
                    self.advance()
                if (
                    self.auto_write
                    and self._final_output_enabled
                    and getattr(self, "_last_vtk_state", None)
                    != (self.step, self.time, self._state_revision)
                ):
                    self.write_vtk()
                if self._final_output_enabled:
                    self._execute_sampler_event("final")
                if self._backup_config.write_at_end and self._step_phase == "accepted":
                    self._save_automatic_backup()
                self.run_status = "complete"
        except BaseException as error:
            primary_failure = error
            self.run_failure = error
        finally:

            def finalize_collectively(operation: str, callback) -> None:
                """Run one finalizer on every rank before propagating errors."""
                nonlocal primary_failure
                local_error = None
                try:
                    callback()
                except BaseException as error:
                    local_error = error
                try:
                    self._collective_io_failure(local_error, operation)
                except BaseException as error:
                    if primary_failure is None:
                        primary_failure = error
                        self.run_failure = error
                        self.run_status = "failed"

            # Every rank participates in each stage, including root-owned
            # flush/close/metadata work.  This prevents one rank from entering
            # the next MPI operation after a peer failed during finalization.
            finalize_collectively("final output flush", self.flush_output)
            finalize_collectively(
                "solver close",
                lambda: self.close(status=self.run_status, failure=primary_failure),
            )
            finalize_collectively(
                "solver metadata",
                lambda: self.write_run_manifest(status=self.run_status),
            )
        if primary_failure is not None:
            raise primary_failure.with_traceback(primary_failure.__traceback__)

    def advance(self) -> None:
        """Solve and commit one configured FVM time step.

        The method selects a fixed or maximum-Courant-limited duration, solves
        a candidate with :meth:`solve_pimple`, checks acceptance limits, then
        calls :meth:`advance_time`. On success `step` and `time` advance and
        diagnostics/scheduled output may be written. On failure no candidate is
        committed and the solver becomes terminally unusable when the physical
        kernels may have partially mutated state.

        Raises
        ------
        RuntimeError, ValueError, FloatingPointError
            For an invalid lifecycle phase, failed linear/physical solve, or
            rejected candidate.
        """
        self._ensure_evolution_usable()
        if not self._run_started and self.run_status == "not_started":
            self.run_status = "interactive"
        if self._resolved_setup.pimple.algorithm == "SIMPLE":
            raise RuntimeError(
                "SIMPLE is a steady algorithm; use solve_steady() instead of advance()"
            )
        if self._step_phase != "accepted":
            raise RuntimeError("A candidate FVM step is pending; call advance_time() to commit it")
        step_time_step_size = self._select_time_step_size()
        step_number = self.step + 1
        self.profiler.begin_step(
            step=step_number,
            time=self.time + step_time_step_size,
            time_step_size=step_time_step_size,
        )
        timer_name = f"Step {step_number}"
        self._timer.start(timer_name)
        self.logger.step_begin(step_number, self.time + step_time_step_size, step_time_step_size)
        primary_failure: BaseException | None = None
        try:
            self.solve_pimple(step_time_step_size)

            adjustment = self._time_config.adjustment
            self.logger.courant_info(
                self.max_courant_number,
                adjustment.maximum if adjustment is not None else None,
            )

            self.advance_time()
        except BaseException as error:
            primary_failure = error
            raise
        finally:
            elapsed = self._timer.stop(timer_name)
            try:
                self.logger.step_end(elapsed, accepted=primary_failure is None)
            except BaseException:
                if primary_failure is None:
                    raise
            try:
                self.profiler.finish_step(
                    elapsed,
                    getattr(self.algorithm, "last_linear_results", ()),
                )
            except BaseException:
                if primary_failure is None:
                    raise

    def advance_time(self, *, defer_output: bool = False) -> None:
        """Commit the solved candidate and advance the accepted FVM clock.

        This coupler-facing method rolls the BDF2 velocity/flux history,
        increments `step` and `time`, writes step diagnostics, dispatches
        samplers, and applies visualization/restart schedules. It requires a
        successful candidate from :meth:`solve_pimple`; calling it twice or
        before a solve raises `RuntimeError`. The field arrays are mutated in
        place and the resulting state is the only state eligible for restart.
        """
        if self._step_phase != "candidate":
            raise RuntimeError("Cannot commit an FVM step before a successful solve")
        self._ensure_evolution_usable()
        pending_counters = self._pending_acceptance_counters
        if pending_counters is None:
            raise RuntimeError("FVM candidate has not passed acceptance checks")
        self._timer.start("Field history commit")

        # Roll the BDF time-history ring: U_old_old <- u^n, U_old <- u^{n+1}.
        self.velocity_older[:] = self.velocity_old[:]
        self.velocity_old[:] = self.velocity[:]
        self.volumetric_face_flux_older[:] = self.volumetric_face_flux_old[:]
        self.volumetric_face_flux_old[:] = self.volumetric_face_flux[:]
        self._n_committed_time_steps += 1
        self.step += 1
        self.time += self._accepted_time_step_size
        self._run_manifest_written = False
        step_time_step_size = self._accepted_time_step_size
        self._previous_time_step_size = step_time_step_size
        self._n_consecutive_accepted_steps = pending_counters
        self._pending_acceptance_counters = None
        self._pending_step_size = None
        self._step_phase = "accepted"
        self._timer.log("Field history commit", sink=self.logger)

        if not defer_output:
            self.write_accepted_step_output()

    def write_accepted_step_output(self) -> None:
        """Publish diagnostics and due output for the current accepted state.

        Couplers may defer provisional substeps and call this once after the
        interface converges. Their sampling schedules must align with exchange
        times; this method does not replay output for earlier substeps.
        """
        if self._step_phase != "accepted":
            raise RuntimeError("Output requires an accepted FVM state")
        step_time_step_size = self._accepted_time_step_size
        self._timer.start("Diagnostics file")
        diagnostics_error = None
        try:
            self.io.write_step_diagnostics()
        except BaseException as error:
            diagnostics_error = error
        self._collective_io_failure(diagnostics_error, "diagnostics output")
        self._timer.log("Diagnostics file", sink=self.logger)

        # Samplers decide their own cadence; the executor runs after every
        # accepted step and every sampler checks whether it is due.  Force,
        # IBM force and y+ output all flow through this single path.
        self._timer.start("Samplers")
        self._execute_sampler_event("accepted")
        self._timer.log("Samplers", sink=self.logger)

        self._timer.start("Turbulence statistics")
        if (
            self.turbulence
            and self.eddy_viscosity is not None
            and self.logger.should_report(self.step, self.time, step_time_step_size)
        ):
            n_owned = (
                self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
            )
            owned_eddy_viscosity = self.eddy_viscosity[:n_owned]
            assert self.last_diagnostics is not None
            min_eddy_viscosity = self.last_diagnostics.min_eddy_viscosity
            max_eddy_viscosity = self.last_diagnostics.max_eddy_viscosity
            sum_eddy_viscosity = float(
                self.parallel.global_sum(float(np.sum(owned_eddy_viscosity)))
            )
            n_eddy_viscosity_values = int(self.parallel.global_sum(int(n_owned)))
            if self.parallel.is_root:
                self.logger.turbulence_info(
                    self.eddy_viscosity,
                    self._kinematic_viscosity,
                    statistics=(
                        min_eddy_viscosity,
                        max_eddy_viscosity,
                        sum_eddy_viscosity / n_eddy_viscosity_values,
                    ),
                )
        self._timer.log("Turbulence statistics", sink=self.logger)

        # Visualization cadence is deterministic from accepted step/time state;
        # physical-time events have already constrained adaptive step selection.
        self._timer.start("Visualization output")
        if self.auto_write and self._output_schedule.is_due(
            self.step, self.time, step_time_step_size
        ):
            self.write_vtk()
        self._timer.log("Visualization output", sink=self.logger)

        self._timer.start("Restart backup")
        backup_schedule = self._backup_config.schedule
        scheduled_backup = backup_schedule is not None and backup_schedule.is_due(
            self.step, self.time, step_time_step_size
        )
        end_tolerance = max(1.0e-14, abs(self._time_config.end_time) * 1.0e-12)
        final_backup = self._backup_config.write_at_end and (
            self.time >= self._time_config.end_time - end_tolerance
        )
        if scheduled_backup or final_backup:
            self._save_automatic_backup()
        self._timer.log("Restart backup", sink=self.logger)

        if not self._run_started and self.time >= self._time_config.end_time - end_tolerance:
            self.run_status = "complete"
            self.write_run_manifest(status=self.run_status)

        # Courant/gradient/vorticity caches describe this accepted state and
        # remain valid until ``solve_pimple`` starts the next mutation.  In a
        # coupled step the endpoint gradient is consumed immediately by the
        # vorticity handoff; dropping it here forced an identical full-mesh
        # reconstruction and global gather.  ``solve_pimple`` clears the cache
        # before its next assembly, retaining the former peak-memory behaviour.

    def _save_automatic_backup(self) -> None:
        """Write each configured checkpoint state once, including run finalization."""
        path = self._backup_config.path
        if not os.path.isabs(path):
            path = os.path.join(self.solution_dir, path)
        state = (self.step, self.time, self._state_revision, path)
        if getattr(self, "_automatic_backup_state", None) != state:
            self.save_state(path)
            self._automatic_backup_state = state

    def save_state(self, path) -> str:
        """Flush output and atomically save the complete accepted FVM restart.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file/directory accepted by the selected serial or
            partitioned backup writer. Relative paths follow the configured
            solution-directory convention.

        Returns
        -------
        str
            The resolved/broadcast restart path.

        Notes
        -----
        The backup includes mesh/configuration identity, primary fields, and
        all time-history data needed by the selected time scheme. Output is
        flushed first, so an asynchronous writer failure prevents a false
        successful checkpoint. A candidate state cannot be saved.
        """
        self._ensure_evolution_usable()
        if self._step_phase != "accepted":
            raise RuntimeError("Cannot save a restart while an uncommitted FVM candidate exists")
        flush_error = None
        try:
            self.flush_output()
        except BaseException as error:
            flush_error = error
        self._collective_io_failure(flush_error, "output flush before restart")
        if self.parallel.is_partitioned:
            from ..io.partitioned import save_partitioned_solver_backup

            saved_path = str(save_partitioned_solver_backup(self, path))
        else:
            from ..io.backup import save_backup

            saved = None
            save_error = None
            if self.parallel.is_root:
                try:
                    saved = save_backup(self, path)
                except BaseException as error:
                    save_error = error
            self._collective_io_failure(save_error, "restart backup")
            saved_path = str(
                self.parallel.bcast(str(saved) if saved is not None else str(path), root=0)
            )
        log_error = None
        if self.parallel.is_root:
            try:
                self.logger.output_info(f"Restart backup written: {saved_path}")
            except BaseException as error:
                log_error = error
        self._collective_io_failure(log_error, "restart-backup logging")
        return saved_path

    def write_run_manifest(self, path=None, *, status: str | None = None) -> str:
        """Write universal FVM configuration and accepted-state metadata.

        Parameters
        ----------
        path : str or pathlib.Path or None
            Destination; defaults to ``fvm_metadata.json`` in the configured
            solution/backup directory.
        status : str or None
            Optional terminal/interactive status recorded in the metadata.

        Returns
        -------
        str
            Destination path, broadcast to all ranks.

        Raises
        ------
        RuntimeError
            If metadata writing fails on any participating rank.

        Side Effects
        ------------
        Creates the destination directory if needed and atomically replaces
        the metadata file on the root rank.
        """
        from ..io.manifest import write_manifest

        destination = path or os.path.join(self.solution_dir, "fvm_metadata.json")
        written = None
        manifest_error = None
        if self.parallel.is_root:
            try:
                written = write_manifest(self, destination, status=status)
            except BaseException as error:
                manifest_error = error
        self._collective_io_failure(manifest_error, "solver metadata")
        self._run_manifest_written = True
        return str(
            self.parallel.bcast(str(written) if written is not None else str(destination), root=0)
        )

    def load_state(self, path, *, allow_config_change: bool = False) -> None:
        """Restore a compatible accepted restart and reconcile output history.

        Parameters
        ----------
        path : str or pathlib.Path
            Restart written by :meth:`save_state` or a compatible legacy writer.
        allow_config_change : bool, default=False
            Permit explicitly classified configuration differences. Mesh and
            shape identity remain mandatory; use this only when the changed
            policy is known to be restart-safe.

        Raises
        ------
        ValueError, RuntimeError
            If the backup is corrupt, incompatible, or output reconciliation
            fails collectively.
        """
        flush_error = None
        try:
            self.flush_output()
        except BaseException as error:
            flush_error = error
        self._collective_io_failure(flush_error, "output flush before restart")
        if self.parallel.is_partitioned:
            from ..io.partitioned import load_partitioned_solver_backup

            load_partitioned_solver_backup(self, path, allow_config_change=allow_config_change)
        else:
            from ..io.backup import load_backup

            load_backup(self, path, allow_config_change=allow_config_change)

        rewind_error = None
        if self.parallel.is_root:
            try:
                self.io.rewind_histories(self.time)
            except Exception as error:
                rewind_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        rewind_error = self.parallel.bcast(rewind_error, root=0)
        if rewind_error is not None:
            raise RuntimeError(
                "FVM restart output reconciliation failed: "
                f"{rewind_error['type']}: {rewind_error['message']}"
            )
        self.parallel.barrier()

    def write_vtk(self, filename: str | None = None) -> None:
        """Collectively publish the current accepted state as VTK output.

        Parameters
        ----------
        filename : str or None
            Optional output path. With `None`, the case name and accepted step
            generate a `.vtu`/`.pvtu` file below the solution directory.

        Notes
        -----
        The output is cell-centred and includes velocity, kinematic pressure,
        Courant number, vorticity, and active eddy viscosity. Asynchronous
        writes are queued; call :meth:`flush_output` to surface writer errors.
        In MPI execution all ranks must enter this method collectively.

        Root-only replicated output still has a collective error envelope:
        workers wait at the same stage and receive the root's failure instead
        of entering a later MPI operation with a misleading success state.
        """
        local_error = None
        try:
            if self.parallel.is_root or self.parallel.is_partitioned:
                self._write_vtk_local(filename)
        except BaseException as error:
            local_error = error
        self._collective_io_failure(local_error, "visualization output")
        if filename is None:
            self._last_vtk_state = (self.step, self.time, self._state_revision)

    def _write_vtk_local(self, filename: str | None = None) -> None:
        """Export the current simulation state to a ``.vtu`` file with PVD time-series support.

        Writes ``velocity``, ``kinematic_pressure``, ``courant_number``, and
        ``vorticity`` fields. If turbulence is active, also writes
        ``eddy_viscosity``. These names are the canonical cross-solver output
        schema; compact kernel symbols are not serialized.
        Includes physical ``cell_volume`` and ``cell_equivalent_size`` plus
        nominal ``cell_size`` and ``refinement_level`` when the mesh has them.
        Updates the PVD collection file for time-series visualisation.

        Args:
            filename: Optional output path. If ``None``, auto-generates
                ``solution/fvm/fvm_<step>.vtu`` and indexes it in
                ``solution/fvm.pvd``.
        """
        if not self.parallel.is_root and not self.parallel.is_partitioned:
            return
        sol_dir = Path(self.solution_dir)
        frame_dir = component_directory(sol_dir, "fvm")
        if filename is None:
            frame_dir.mkdir(parents=True, exist_ok=True)
            filename = str(frame_dir / f"fvm_{self.step:06d}.vtu")

        fields = {
            "velocity": self.velocity,
            "kinematic_pressure": self.kinematic_pressure,
            "courant_number": self._courant_field(self._accepted_time_step_size),
            "vorticity": self._vorticity_field(),
        }
        from ..io.vtk_exporter import mesh_cell_fields

        fields.update(mesh_cell_fields(self.mesh_data, self.geo_data["cell_volume"]))
        if self.eddy_viscosity is not None:
            fields["eddy_viscosity"] = self.eddy_viscosity

        if self.parallel.is_partitioned:
            from ..io.partitioned import write_partition_vtu
            from ..io.vtk_exporter import VTKExporter

            n_local = self.mesh_data["n_cells"]
            stem = Path(filename).stem
            if self.vtk_exporter is None:
                export_mesh = self.mesh_data.get("_visualization_mesh", self.mesh_data)
                self.vtk_exporter = VTKExporter(export_mesh, self._resolved_setup.output)
            collection = write_partition_vtu(
                Path(filename).parent,
                stem,
                self.mesh_data,
                self.parallel.partition,
                {name: np.asarray(values)[:n_local] for name, values in fields.items()},
                self.parallel.comm,
                output=self._resolved_setup.output,
                exporter=self.vtk_exporter,
            )
            if self.parallel.is_root:
                from ..io.vtk_exporter import PVDManager

                pvd_file = str(collection_path(sol_dir, "fvm"))
                if self.pvd_manager is None:
                    self.pvd_manager = PVDManager(pvd_file)
                self.pvd_manager.add_step(self.time, str(collection))
                self.logger.output_info(f"Output written: {stem}.pvtu")
            return

        asynchronous = (
            self._resolved_setup.output.asynchronous
            or self._resolved_setup.execution.output_mode == "threaded"
        )
        if asynchronous:
            if self._buffered_vtk_writer is None:
                from ..io.async_output import BufferedVTKWriter

                pvd_file = str(collection_path(sol_dir, "fvm"))
                self._buffered_vtk_writer = BufferedVTKWriter(
                    self.mesh_data,
                    pvd_file,
                    self._resolved_setup.output,
                )
            self._buffered_vtk_writer.submit(filename, self.time, fields)
            action = "queued"
        else:
            if self.vtk_exporter is None:
                from ..io.vtk_exporter import VTKExporter

                self.vtk_exporter = VTKExporter(self.mesh_data, self._resolved_setup.output)
            if self.pvd_manager is None:
                from ..io.vtk_exporter import PVDManager

                pvd_file = str(collection_path(sol_dir, "fvm"))
                self.pvd_manager = PVDManager(pvd_file)
            self.vtk_exporter.export(filename, fields)
            self.pvd_manager.add_step(self.time, filename)
            action = "written"

        self.logger.output_info(f"Output {action}: {os.path.basename(filename)}")

    def flush_output(self) -> None:
        """Wait for buffered visualization output and surface writer failures.

        This method has no numerical effect. It is a lifecycle barrier for
        asynchronous VTK/log writers and should be called before a restart,
        metadata, process exit, or external reader consumes the files.
        """
        if self._buffered_vtk_writer is not None:
            self._buffered_vtk_writer.flush()
        self.logger.flush()

    def close(self, *, status: str | None = None, failure=None) -> None:
        """Finish solver-owned writers, algorithm resources, profiler, and logger.

        Parameters
        ----------
        status : str or None
            Optional terminal status passed to the logger and metadata hooks.
        failure : BaseException or None
            Optional primary failure summary for final logging.

        `close()` is idempotent after successful cleanup and does not change the
        accepted fields. A cleanup failure is raised so callers cannot mistake
        an incomplete output close for a clean run.
        """
        if self._closed:
            return
        primary_failure: BaseException | None = None
        try:
            if self._buffered_vtk_writer is not None:
                self._buffered_vtk_writer.close()
        except BaseException as error:
            primary_failure = error
        try:
            algorithm_close = getattr(self.algorithm, "close", None)
            if algorithm_close is not None:
                algorithm_close()
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        try:
            self.profiler.close()
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        if status is None:
            status = getattr(self, "run_status", "not_started")
        if not getattr(self, "_run_started", False) and not getattr(
            self, "_run_manifest_written", False
        ):
            try:
                manifest_status = "failed" if primary_failure is not None else status
                self.write_run_manifest(status=manifest_status)
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        try:
            close_logger = self.logger.close
            try:
                import inspect

                parameters = inspect.signature(close_logger).parameters.values()
                accepts_keywords = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    or parameter.name in {"status", "failure"}
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                accepts_keywords = True
            if accepts_keywords:
                close_logger(status=status, failure=failure)
            else:
                close_logger()
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        if primary_failure is not None:
            raise primary_failure
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        status = "failed" if _exc_value is not None else None
        self.close(status=status, failure=_exc_value)

    def evaluate(self, callback, *args, **kwargs):
        """Run an analysis once against a complete, detached field snapshot.

        Calls ``callback(snapshot, *args, **kwargs)`` and returns its result.
        The library gathers interior cells and physical boundary faces, owns
        output, and propagates failures. Callbacks contain ordinary analysis
        code; no MPI calls or rank checks are needed. Only explicitly requested
        analyses gather full fields; normal solver steps remain distributed.
        """
        from ..io.analysis import evaluate

        return evaluate(self, callback, *args, **kwargs)

    def write_csv(self, filename, rows, *, columns, append=False) -> None:
        """Write a table once; relative filenames use the solution directory."""
        from ..io.analysis import write_csv

        write_csv(self, filename, rows, columns=columns, append=append)

    def info(self) -> None:
        """Print a summary of the current solver state.

        Displays case name, flow time, time step, cell count, and
        active algorithm.
        """
        self.logger.solver_state(self)
