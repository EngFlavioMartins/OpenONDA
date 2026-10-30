"""Incompressible finite-volume solver."""

from collections.abc import Iterable, Sequence
from contextlib import suppress
import gc
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core.output_events import OutputEvent
from openonda.core.simulation.parallel import collective_phase
from openonda.core.solution_layout import collection_path, component_directory
from openonda.core.status import RunStatus
from openonda.runtime import RunConfig

from ..config.case import FVMCase
from ..config.configuration import FVMConfiguration
from ..coupling import CouplerInterfaceMixin
from ..fields.mixed_velocity_boundary import (
    update_mixed_velocity_boundary,
)
from ..io import logging, solver_io
from ..mesh import geometry
from ..sampling.executor import execute_samplers
from ..solve import pimple_solver, simple_solver
from .parallel import ParallelContext
from .state import FieldState
from .time_step import event_aligned_time_step_size, maximum_courant_time_step_size

if TYPE_CHECKING:
    from ..diagnostics.step import StepDiagnostics
    from ..mesh.topology import MeshTopology


class AcceptanceLimitError(RuntimeError):
    """A candidate FVM step exceeded a configured :class:`RunAcceptanceLimits` threshold."""


def _initial_velocity_field(
    configuration: FVMConfiguration, n_total: int, mesh_data: dict
) -> np.ndarray:
    """Return the initial velocity of every cell and boundary ghost, shape ``(n_total, 3)``."""
    initial = np.asarray(configuration.initial_conditions.velocity, dtype=np.float64)
    if initial.shape == (3,):
        return np.tile(initial, (n_total, 1))
    if initial.shape == (mesh_data["n_cells"], 3):
        result = np.zeros((n_total, 3), dtype=np.float64)
        result[: mesh_data["n_cells"]] = initial
        return result
    raise ValueError(
        f"InitialFields.velocity must have shape (3,) or ({mesh_data['n_cells']}, 3); "
        f"got {initial.shape}"
    )


def _enforce_velocity_boundary_constraints(
    velocity: np.ndarray, boundaries: list, n_cells: int, mesh_data: dict, geo_data: dict
) -> None:
    """Enforce velocity boundary constraints on ghost cells after initialisation.

    Iterates over all boundary patches and sets the ghost-layer values
    in *velocity* according to each patch's boundary condition type (no_slip,
    fixed_value, zero_gradient, empty, etc.).

    Args:
        velocity:    Velocity array (mutated in place).
        boundaries: List of boundary patch dictionaries.
        n_cells: Number of interior cells.
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
        elif strategy is BoundaryStrategy.MIXED_VELOCITY:
            update_mixed_velocity_boundary(velocity, boundary, mesh_data, geo_data)
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
    """Constant-density, incompressible finite-volume solver.

    The solver constructs the face-based mesh of an :class:`FVMCase`, computes
    its geometry, reconstructs boundary ghost cells, assembles the selected
    SIMPLE, PISO or PIMPLE equations and owns the accepted physical clock and
    time history. Numerical kernels may replace arrays internally; the public
    :attr:`state` view is republished so the primary fields stay synchronized.

    Attributes
    ----------
    case : FVMCase
        Case definition.
    case_dir, solution_dir, samples_dir : pathlib.Path
        Case root and its output directories.
    mesh_data : dict[str, object]
        Native connectivity, patch ranges, owner/neighbour indices and counts.
    geo_data : dict[str, numpy.ndarray]
        Cell centres, volumes, face areas, interpolation weights and wall
        distances in m, m² and m³.
    velocity : numpy.ndarray
        Cell-centred velocity in m/s, including boundary ghosts.
    kinematic_pressure : numpy.ndarray
        ``p/rho`` in m²/s², including boundary ghosts.
    volumetric_face_flux : numpy.ndarray
        ``phi = U_f · S_f`` in m³/s, positive from owner to neighbour.
    time, time_step_size : float
        Accepted physical time and selected time-step size in s.
    step : int
        Number of accepted time steps.
    state : FieldState
        Synchronized view of the three primary fields.

    Notes
    -----
    Use :meth:`run` for the complete run, or :meth:`advance` (or
    :meth:`solve_step` followed by :meth:`accept_step`) for interactive and
    coupled stepping. The solver is a context manager. A
    :class:`~openonda.fvm.motion.MeshMotion` in the case moves the whole mesh
    rigidly (arbitrary Lagrangian-Eulerian form); see :attr:`mesh_motion`.
    """

    @property
    def topology(self) -> "MeshTopology":
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
    def geometry(self) -> geometry.MeshGeometry:
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
        self._derived_fields.clear()

    def _publish_state(self) -> FieldState:
        """Publish the live arrays through one synchronized :class:`FieldState`.

        Numerical kernels are allowed to return replacement arrays.  This
        helper makes the state object and the solver attributes point at the
        same contiguous arrays after every such boundary, while preserving an
        already-issued state object reference for interactive callers.
        """
        published = FieldState(
            self.velocity,
            self.kinematic_pressure,
            self.volumetric_face_flux,
        )
        current = self.state
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
        self._state_revision += 1
        return self.state

    def _velocity_gradient(self) -> np.ndarray:
        """Return the cached gradient for the current solved field state."""
        from ..fields import gradients

        gradient = self._derived_fields.get("velocity_gradient")
        if gradient is None:
            gradient = gradients._resolve_gradient_fn(self.geo_data)(
                self.velocity, self.mesh_data, self.geo_data
            )
            self._derived_fields["velocity_gradient"] = gradient
        return gradient

    def _courant_field(self, time_step_size: float) -> np.ndarray:
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
        from ..diagnostics import fields as diagnostics

        key = ("courant", float(time_step_size))
        courant = self._derived_fields.get(key)
        if courant is None:
            # On a moving mesh the transport Courant number uses the flux
            # relative to the moving faces.
            courant = diagnostics.compute_courant_number(
                self.volumetric_face_flux
                if self.mesh_motion is None
                else self.volumetric_face_flux - self.mesh_motion.mesh_face_flux,
                time_step_size,
                self.mesh_data,
                self.geo_data,
            )
            self._derived_fields[key] = courant
        return courant

    def _vorticity_field(self) -> np.ndarray:
        """Return the cached cell-centred curl of the current velocity.

        Returns
        -------
        numpy.ndarray, shape (n_cells, 3)
            Vorticity in 1/s using the configured gradient convention. The
            result is a derived cache entry and does not mutate the velocity.
        """
        from ..diagnostics import fields as diagnostics

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

    def __init__(self, case: FVMCase) -> None:
        """Construct the mesh, geometry, fields and algorithm of ``case``.

        Parameters
        ----------
        case : FVMCase
            Case definition. Under MPI every rank constructs the solver with
            the same case.

        Raises
        ------
        TypeError, ValueError, RuntimeError, FileNotFoundError
            If the case, mesh topology or geometry, an optional dependency or
            the parallel execution settings are invalid.

        Notes
        -----
        Construction creates ``solution/``, stores a generated mesh as
        ``assets/<name>_mesh.npz`` and writes ``solution/fvm_metadata.json``.
        It does not advance ``time`` or ``step``.
        """
        if not isinstance(case, FVMCase):
            raise TypeError("FVMSolver requires an FVMCase")
        self.case = case
        configuration = FVMConfiguration.from_case(case)
        self._configuration = configuration
        numerics = configuration.numerics
        RunConfig(cpu_cores=case.cores, parallel_mode="mpi").ensure_runtime(sys.argv[0])
        initialization_start = perf_counter()
        self.case_dir = case.directory.resolve()
        self.solution_dir = case.solution_directory.resolve()
        self.samples_dir = case.samples_directory.resolve()
        mesh_data = self._construct_mesh()

        # Runtime coupling may override the molecular viscosity of later
        # equations; the case value stays unchanged.
        self._kinematic_viscosity = float(numerics.transport.kinematic_viscosity)
        self.state: FieldState | None = None
        self._state_revision = 0
        self._derived_fields: dict[object, np.ndarray] = {}
        self._samplers = configuration.samplers
        self._sample_collections: dict[str, list[tuple[float, str]]] = {}
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        if self._samplers:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
        # Heterogeneous mesh metadata: arrays, counts, patch dictionaries and
        # parallel objects.
        self.mesh_data: Any
        self.geo_data: Any
        self.auto_write = True
        self.parallel = ParallelContext.create(numerics.execution)
        self.logger = logging.Logging(
            self.solution_dir / "fvm.log",
            configuration.logging,
            enabled=self.parallel.is_root,
        )
        from openonda.core.profiling import Profiler

        self.profiler = Profiler(
            enabled=configuration.logging.detailed_timing,
            parallel=self.parallel,
            output_path=self.solution_dir / "performance.jsonl",
        )
        self.operator_backend = configuration.numerics.execution.operator_backend
        if configuration.numerics.execution.linear_backend == "petsc":
            methods = {
                "momentum": configuration.numerics.linear.momentum_solver
                or configuration.numerics.linear.linear_solver,
                "pressure": configuration.numerics.linear.pressure_solver
                or configuration.numerics.linear.linear_solver,
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

        if (
            configuration.numerics.linear.pressure_nullspace_method == "petsc"
            and configuration.numerics.execution.linear_backend != "petsc"
        ):
            raise ValueError(
                "pressure_nullspace_method='petsc' requires execution.linear_backend='petsc'"
            )
        if (
            self.parallel.is_partitioned
            and configuration.numerics.linear.pressure_nullspace_method == "reference"
        ):
            raise ValueError(
                "petsc_partitioned requires pressure_nullspace_method='auto' or 'petsc'; "
                "a rank-local reference row is not a valid global pressure constraint"
            )
        if self.parallel.is_partitioned and configuration.output.point_interpolation != "none":
            raise ValueError(
                "output.point_interpolation='boundary_weighted' is not qualified for "
                "petsc_partitioned execution: the partitioned writer drops the boundary "
                "ghost values the interpolation needs, and a rank's processor-interface "
                "faces are not physical boundaries. Run serially to write interpolated "
                "point data, or use ParaView's Cell Data to Point Data filter instead"
            )
        self.logger.run_header("f64")

        from ..mesh.validation import (
            enforce_quality_thresholds,
            validate_geometry,
            validate_topology,
        )

        self._topology = None
        self._geometry = None
        gs = configuration.numerics.schemes.gradient_scheme
        if self.parallel.is_partitioned:
            comm = self.parallel.comm
            assert comm is not None
            if any(boundary.velocity_type == "cyclic" for boundary in configuration.boundaries):
                raise NotImplementedError(
                    "Partitioned cyclic patches require periodic partition adjacency, which is "
                    "not yet implemented"
                )
            quality = None
            preparation_error = None
            global_mesh = None
            global_geo = None
            global_hash = None
            if self.parallel.is_root:
                try:
                    global_mesh = mesh_data
                    validate_topology(global_mesh)
                    global_geo = geometry.compute_mesh_geometry(
                        global_mesh,
                        gradient_scheme=gs,
                        compute_lsq=False,
                    )
                    quality = validate_geometry(global_mesh, global_geo)
                    enforce_quality_thresholds(quality, configuration.mesh_quality)
                    from ..io.backup import mesh_digest

                    global_hash = mesh_digest(global_mesh)
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
            # mesh partition per rank.  Send one partition at a time instead; every
            # receiver participates in the following error broadcast before
            # it starts solver construction, so a late localization failure
            # cannot leave a peer in an incompatible collective.
            distribution_error = None
            local_partition_data = None
            partition_data = None
            received_partition_data = None
            if self.parallel.is_root:
                from ..mesh.partition import localize_mesh_and_geometry

                assert (
                    global_mesh is not None and global_geo is not None and global_hash is not None
                )
                # Send worker partitions first and build rank zero last. Drop
                # each sent mesh partition before localizing the next partition so
                # rank zero holds at most one local mesh partition beside the global mesh.
                rank_order = [*range(1, self.parallel.size), 0]
                delivered: set[int] = set()
                for rank in rank_order:
                    try:
                        partition_data = localize_mesh_and_geometry(
                            global_mesh,
                            global_geo,
                            rank,
                            self.parallel.size,
                            include_visualization_ghosts=configuration.output.ghost_layers == 1,
                        )
                        partition_data[0]["global_mesh_hash"] = global_hash
                    except Exception as error:
                        distribution_error = {
                            "rank": rank,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        if rank == 0:
                            local_partition_data = None
                        for destination in range(1, self.parallel.size):
                            if destination not in delivered:
                                comm.send((False, distribution_error), dest=destination, tag=9131)
                        break
                    if rank == 0:
                        local_partition_data = partition_data
                    else:
                        comm.send((True, partition_data), dest=rank, tag=9131)
                        delivered.add(rank)
                        partition_data = None
            else:
                received_ok, received_partition_data = comm.recv(source=0, tag=9131)
                if received_ok:
                    local_partition_data = received_partition_data
                else:
                    distribution_error = received_partition_data
            distribution_error = self.parallel.bcast(distribution_error, root=0)
            if distribution_error is not None:
                raise RuntimeError(
                    "Mesh partition distribution failed: "
                    + json.dumps(distribution_error, sort_keys=True)
                )
            assert local_partition_data is not None
            self.mesh_data, self.geo_data, partition = local_partition_data
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
            partition_data = None
            local_partition_data = None
            received_partition_data = None
            gc.collect()
        else:
            self.mesh_data = mesh_data
            validate_topology(self.mesh_data)
            self.geo_data = geometry.compute_mesh_geometry(
                self.mesh_data,
                gradient_scheme=gs,
                compute_lsq=False,
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
        enforce_quality_thresholds(self.mesh_quality, configuration.mesh_quality)

        from ..assemble.matrix_assembly import prepare_matrix_assembly

        prepare_matrix_assembly(self.mesh_data)

        self._initialize_fields()
        self.state = FieldState(self.velocity, self.kinematic_pressure, self.volumetric_face_flux)
        self._initialize_algorithm()
        self._initialize_turbulence()
        self._initialize_mesh_motion()

        # Final housekeeping
        self.io = solver_io.SolverIO(self)
        self.vtk_exporter = None
        self._frame_collection: list[tuple[float, str]] | None = None
        self._buffered_vtk_writer = None
        self.max_courant_number = 0.0
        # Coupling / driver-split state
        self.registered_fields: dict[str, np.ndarray] = {}  # named volume fields (fvOptions)
        self._outer_boundary_patch_ids_by_rank: dict[str, Any] = {}
        self._coupling_cell_ids_by_rank: dict[str, Any] = {}
        self._coupling_consistency_filter = None
        # Optional conservative field projection applied after the pressure-
        # velocity solve and before diagnostics, sampling, history commit, and
        # output. Benchmarks can use this to enforce a known invariant
        # subspace without bypassing the solver's normal bookkeeping.
        self._post_solve_state_callback = None
        self._n_committed_time_steps = 0  # number of committed time steps (BDF2 startup check)
        self._accepted_time_step_size = self.time_step_size
        self._previous_time_step_size = self.time_step_size
        self.last_diagnostics = None
        self._last_backup_state: tuple[int, float, int] | None = None
        self._last_vtk_state: tuple[int, float, int] | None = None
        self._restart_loaded = False
        self._initial_run_selected = False
        self._initial_rewound_histories: set[Path] = set()
        self._velocity_boundary_values: dict[int, np.ndarray] = {}
        self.steady_iterations = 0
        self.steady_converged = False
        self._step_phase = "accepted"
        self._pending_step_size: float | None = None
        self._pending_acceptance_counters: dict[str, int] | None = None
        self._evolution_failure: BaseException | None = None
        self._closed = False
        self._run_started = False
        self._run_metadata_written = False
        self.run_status = RunStatus.not_started
        self.run_failure: BaseException | None = None
        self._n_consecutive_accepted_steps = {
            "max_continuity_error": 0,
            "max_equation_residual": 0,
            "max_courant_number": 0,
            "max_velocity_magnitude": 0,
        }

        self.logger.solver_configuration(self, perf_counter() - initialization_start)

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=self.volumetric_face_flux,
        )
        self.write_run_metadata(status=RunStatus.created)

    def _construct_mesh(self) -> dict[str, Any] | None:
        """Build or load the case mesh on the rank that needs it.

        Rank 0 builds the mesh and stores it as ``assets/<name>_mesh.npz``
        with its ParaView view; a replicated run broadcasts it to every rank.
        Progress is written to ``solution/mesher.log``.
        """
        from ..mesh.construction import build_or_load_mesh, save_mesh
        from ..mesh.progress import mesh_stage, mesher_log_session

        source = self.case.mesh
        if isinstance(source, str | Path) and not Path(source).is_absolute():
            source = self.case_dir / source
        is_root = True
        comm = None
        if self.case.cores > 1:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            is_root = comm.Get_rank() == 0
        mesh_data = None
        with (
            collective_phase(comm, "FVM mesh construction"),
            mesher_log_session(
                self.solution_dir / "mesher.log" if is_root else None, announce=is_root
            ),
        ):
            if is_root:
                with mesh_stage("mesh construction") as construction:
                    mesh_data = build_or_load_mesh(source, cache_path=self.case.mesh_path)
                    construction.details(
                        cells=mesh_data.get("n_cells"),
                        faces=mesh_data.get("n_faces"),
                        points=mesh_data.get("n_points"),
                    )
                # Stored before validation, so a rejected mesh can be inspected.
                save_mesh(mesh_data, self.case.mesh_path, self._configuration.output)
        if comm is not None and (
            self._configuration.numerics.execution.parallel_mode == "petsc_replicated"
        ):
            mesh_data = comm.bcast(mesh_data, root=0)
        return mesh_data

    def _setup_boundary_conditions(self) -> None:
        """Map user-defined BoundaryConfig entries to internal mesh boundary data.

        Iterates over the configured boundaries and updates the
        corresponding entries in ``self.boundaries`` with the configured
        type and value for velocity, kinematic_pressure, and eddy_viscosity.
        A configured patch that is not in the mesh raises ``ValueError``.
        """
        for b_cfg in self._configuration.boundaries:
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

    def _initialize_fields(self) -> None:
        """Initialise velocity, kinematic pressure, and volumetric face flux.

        Loads or creates the initial fields, enforces boundary constraints
        on the velocity ghost layer, and computes the initial volumetric face
        flux ``volumetric_face_flux = velocity·Sf`` from the velocity field.
        """
        n_cells = self.mesh_data["n_cells"]
        n_total = self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"] + n_cells

        self.velocity = _initial_velocity_field(self._configuration, n_total, self.mesh_data)
        self.kinematic_pressure = np.full(
            n_total,
            self._configuration.initial_conditions.kinematic_pressure,
            dtype=np.float64,
        )
        self.velocity_old = self.velocity.copy()
        # Second history level for BDF2 (u^{n-1}); ignored by BDF1.
        self.velocity_older = self.velocity.copy()

        _enforce_velocity_boundary_constraints(
            self.velocity, self.boundaries, n_cells, self.mesh_data, self.geo_data
        )
        self.parallel.exchange_halo(self.velocity[:n_cells])
        self.parallel.exchange_halo(self.kinematic_pressure[:n_cells])

        from ..assemble import convection

        self.volumetric_face_flux = convection.compute_volumetric_face_flux(
            self.velocity, self.mesh_data, self.geo_data
        )
        # Flux history for the transient Rhie-Chow correction.
        # (``fvc::ddtCorr``), which needs flux and velocity at the same levels.
        self.volumetric_face_flux_old = self.volumetric_face_flux.copy()
        self.volumetric_face_flux_older = self.volumetric_face_flux.copy()

    def _initialize_mesh_motion(self) -> None:
        """Place a moving mesh at its start pose and impose its boundary motion."""
        self.mesh_motion = None
        motion = self.case.mesh_motion
        if motion is None:
            return
        from ..motion.rigid import RigidMeshMotion

        previous_surface = np.array(self.geo_data["face_area_vector"])
        self.mesh_motion = RigidMeshMotion(motion, self)
        self._carry_face_fluxes(previous_surface)
        self.mesh_motion.update_boundaries(self)
        self.volumetric_face_flux_old[:] = self.volumetric_face_flux
        self.volumetric_face_flux_older[:] = self.volumetric_face_flux
        self._invalidate_derived_fields()
        self._publish_state()

    def _carry_face_fluxes(
        self, previous_surface: np.ndarray, *, include_current: bool = True
    ) -> None:
        """Carry the stored face fluxes onto faces rotated by the mesh motion."""
        if np.array_equal(previous_surface, self.geo_data["face_area_vector"]):
            return
        levels = [
            (self.volumetric_face_flux_old, self.velocity_old),
            (self.volumetric_face_flux_older, self.velocity_older),
        ]
        if include_current:
            levels.append((self.volumetric_face_flux, self.velocity))
        for flux, velocity in levels:
            self.mesh_motion.reexpress_flux(flux, velocity, previous_surface)

    def move_mesh(self, time_step_size: float) -> None:
        """Place a moving mesh at the end ``time + time_step_size`` of the next step.

        :meth:`solve_step` calls this itself. A coupling driver calls it
        first when boundary data must be evaluated on the end-of-step
        geometry. Repeated calls for the same step do nothing; the pose is
        always advanced from the accepted pose, so a different step size
        replaces an earlier placement. A static mesh is unaffected.
        """
        if self.mesh_motion is None:
            return
        target = self.time + float(time_step_size)
        if self.mesh_motion.time == target:
            return
        previous_mesh_flux = self.mesh_motion.mesh_face_flux.copy()
        previous_surface = self.mesh_motion.move_to(target)
        # The convecting flux starts from the flux relative to the mesh, a
        # scalar carried with the faces: it is exact for fluid moving with the
        # mesh, for still fluid around a steadily rotating mesh, and for
        # steady translation. The history fluxes keep their Rhie-Chow part.
        self.volumetric_face_flux += self.mesh_motion.mesh_face_flux - previous_mesh_flux
        self._carry_face_fluxes(previous_surface, include_current=False)
        self.mesh_motion.update_boundaries(self)
        self._invalidate_derived_fields()
        self._publish_state()

    def _initialize_algorithm(self) -> None:
        """Create the PIMPLE/PISO or steady SIMPLE pressure--velocity algorithm."""
        params = self._configuration.algorithm_parameters()
        params["_linear_backend"] = self._configuration.numerics.execution.linear_backend
        params["_operator_backend"] = self.operator_backend
        self.geo_data["_operator_backend"] = self.operator_backend
        params["_parallel_context"] = self.parallel
        params["_logger"] = self.logger
        params["_profiler"] = self.profiler
        if self._configuration.numerics.pimple.algorithm == "simple":
            self.algorithm = simple_solver.SIMPLESolver(
                self.mesh_data, self.geo_data, self.boundaries, params
            )
        else:
            self.algorithm = pimple_solver.PIMPLESolver(
                self.mesh_data, self.geo_data, self.boundaries, params
            )

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
        rank assignment.
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
            # Keep the complete setter validated and publish exactly one
            # state revision.  Calling set_initial_velocity() here would
            # publish an intermediate pressure/flux state before the supplied
            # pressure field had been validated.
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

    def _initialize_turbulence(self) -> None:
        """Create the configured LES model and set the initial clock."""
        from ..turbulence import create_model

        self.turbulence = create_model(
            self._configuration.numerics.turbulence, self.mesh_data, self.geo_data
        )
        self.eddy_viscosity = None
        self.time = self._configuration.run.start_time
        self.step = 0
        self.time_step_size = self._configuration.numerics.time_step_size

    def compute_effective_viscosity(self) -> float | np.ndarray:
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

    def _coupling_consistency_source(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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

            self._coupling_consistency_filter = CellBoxFilter(self.mesh_data, self.geo_data)
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
        self.run_status = RunStatus.failed
        self.run_failure = self._evolution_failure
        # Reporting must never replace the numerical failure.
        with suppress(Exception):
            self.logger.warning(f"FVM step failed: {type(error).__name__}: {error}")

    def _collective_io_failure(self, error: BaseException | None, operation: str) -> None:
        """Propagate a root/local output error before any rank continues.

        Numerical kernels already use their own collectives.  Filesystem and
        metadata output writing is different: a root-only exception must still
        release every peer from the same stage, otherwise the next collective
        turns a useful I/O error into an MPI hang.
        """
        parallel = self.__dict__.get("parallel")
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
            ) from error

    def _execute_sampler_event(self, event: OutputEvent) -> None:
        """Dispatch one sampler event and envelope rank-local failures."""
        sampler_error = None
        try:
            execute_samplers(self, event)
        except BaseException as error:
            sampler_error = error
        self._collective_io_failure(sampler_error, f"{event} sampler output")

    def solve_step(self, time_step_size: float | None = None) -> dict[str, float]:
        """Solve one transient pressure--velocity candidate without advancing time.

        Parameters
        ----------
        time_step_size : float or None
            Candidate duration in seconds. ``None`` uses the solver-selected
            step. A repeated call while a candidate is pending must use the
            same value.

        Returns
        -------
        dict[str, float]
            Final equation residuals of the candidate. The complete record
            is :attr:`last_diagnostics`.

        Raises
        ------
        RuntimeError
            If the algorithm is ``simple``, a candidate is in an incompatible
            phase, or an earlier step failed.
        AcceptanceLimitError
            If the candidate exceeds a configured acceptance limit.
        ValueError
            If the requested duration is not finite and positive.

        Notes
        -----
        A coupler may call this repeatedly for one step (an interface
        iteration); call :meth:`accept_step` once the candidate is accepted.
        The accepted ``velocity_old`` is the time-derivative reference of
        every repeated solve.
        """
        from ..diagnostics import fields as diagnostics

        self._ensure_evolution_usable()
        if self.run_status in {RunStatus.not_started, RunStatus.created}:
            self.run_status = RunStatus.running
        if self._configuration.numerics.pimple.algorithm == "simple":
            raise RuntimeError("The 'simple' algorithm is steady; use solve_steady()")
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
        try:
            self.move_mesh(step_time_step_size)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise

        # Diagnostics from the previously completed step cache full-mesh
        # Courant, velocity-gradient, and vorticity arrays.  None is valid once
        # a new solve begins, and retaining them through momentum/pressure
        # assembly adds roughly thirteen float64 values per local cell to the
        # transient peak.  Release them before turbulence and PIMPLE allocate
        # their workspaces; the solved state invalidation below remains the
        # guard for fields requested during a re-entrant/coupled solve.
        self._invalidate_derived_fields()

        try:
            with self.profiler.section("effective viscosity"):
                effective_viscosity = self.compute_effective_viscosity()
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
                    density=self._configuration.numerics.transport.density,
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
                    mesh_face_flux=(
                        None if self.mesh_motion is None else self.mesh_motion.mesh_face_flux
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
            self.logger.convergence(residuals)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this makes loss of mass
        # conservation visible instead of silent.
        try:
            continuity_start = perf_counter()
            continuity_error = diagnostics.compute_continuity_error(
                self.volumetric_face_flux, self.mesh_data
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
            self.logger.continuity(
                self.max_continuity_error,
                self.sum_absolute_continuity_error,
            )
            self.profiler.record("continuity diagnostics", perf_counter() - continuity_start)

            with self.profiler.section("acceptance checks"):
                self.last_diagnostics = self._build_step_diagnostics(step_time_step_size, residuals)
                self._enforce_acceptance_limits(self.last_diagnostics)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        self._step_phase = "candidate"
        return residuals

    def _build_step_diagnostics(
        self, step_time_step_size: float, residuals: dict[str, object]
    ) -> "StepDiagnostics":
        """Build the backend-neutral field and convergence record for the current solved state."""
        from ..diagnostics import fields as diagnostics
        from ..diagnostics.step import StepDiagnostics

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
            * self._configuration.numerics.transport.density
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
            algorithm=self._configuration.numerics.pimple.algorithm,
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

    def _enforce_acceptance_limits(self, diagnostics: "StepDiagnostics") -> None:
        """Reject nonfinite or unconverged solves using explicit immediate and sustained rules."""
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

        limits = self._configuration.numerics.acceptance
        max_velocity = float(diagnostics.max_velocity_magnitude)
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
                raise AcceptanceLimitError(
                    f"FVM acceptance limits rejected the step: {name}={value:.6g} "
                    f"exceeded {abort:.6g} for {trial_counters[name]} "
                    "consecutive accepted step(s)"
                )
        self._pending_acceptance_counters = trial_counters
        self.last_diagnostics = replace(diagnostics, warnings=tuple(warnings))
        self.logger.warnings(tuple(warnings))

    def _measure_maximum_courant_number(self, time_step_size: float) -> float:
        """Return the global maximum CFL for the current field state."""
        n_owned = (
            self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
        )
        courant_number_field = self._courant_field(time_step_size)
        local_maximum = float(np.max(courant_number_field[:n_owned])) if n_owned > 0 else 0.0
        return float(self.parallel.global_max(local_maximum))

    def _select_time_step_size(
        self,
        additional_event_schedules: Sequence = (),
        *,
        stop_time: float | None = None,
    ) -> float:
        """Select one FVM step and stop at the next event or driver deadline.

        Fixed stepping reads the immutable construction value; automatic
        stepping applies the configured maximum-Courant control. A driver can
        provide ``stop_time`` to end its current interval exactly.
        """
        control = self._configuration.numerics.time_step_control
        time_until_event = None
        selected = self._courant_time_step_limit()

        # Adaptive steps are distributed over time deadlines. Fixed steps are
        # shortened only when a deadline falls before the configured step.
        for schedule in (*self._run_event_schedules(), *additional_event_schedules):
            next_event_time = schedule.next_time_after(self.time)
            if next_event_time is not None:
                distance = next_event_time - self.time
                time_until_event = (
                    distance if time_until_event is None else min(time_until_event, distance)
                )

        # Run-managed runs land exactly on their physical horizon.
        end_time = float(self._configuration.run.end_time)
        if stop_time is not None:
            stop_time = float(stop_time)
            if not math.isfinite(stop_time) or stop_time <= self.time:
                raise ValueError("stop_time must be finite and later than the accepted FVM time")
            end_time = min(end_time, stop_time)
        remaining = float(end_time - self.time)
        tolerance = max(1.0e-14, abs(end_time) * 1.0e-12)
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

    def courant_time_step_limit(self) -> float:
        """Return the current CFL-limited step ceiling without applying output events."""
        if self._step_phase != "accepted":
            raise RuntimeError("A candidate FVM step is pending; accept or restore it first")
        return self._courant_time_step_limit()

    def _courant_time_step_limit(self) -> float:
        """Return the current CFL ceiling from the stored flow state."""
        control = self._configuration.numerics.time_step_control
        if control is None:
            return float(self._configuration.numerics.time_step_size)
        current_courant_number = self._measure_maximum_courant_number(self.time_step_size)
        return maximum_courant_time_step_size(
            self.time_step_size,
            current_courant_number,
            control,
        )

    def select_next_time_step_size(
        self,
        *,
        additional_event_schedules: Sequence = (),
        stop_time: float | None = None,
    ) -> float:
        """Select the next CFL-limited step, aligned with local and driver events.

        A coupled driver may pass schedules owned by another solver and a
        ``stop_time`` for the end of its current interval. The returned value
        is stored as the current proposed step and can be passed to
        :meth:`advance`.
        """
        if self._step_phase != "accepted":
            raise RuntimeError("A candidate FVM step is pending; accept or restore it first")
        return self._select_time_step_size(additional_event_schedules, stop_time=stop_time)

    def _run_event_schedules(self) -> tuple:
        """Return construction-time schedules that constrain adaptive steps."""
        schedules = [self._configuration.logging.schedule]
        if self.auto_write and self._configuration.output.schedule is not None:
            schedules.append(self._configuration.output.schedule)
        if self._configuration.backup.schedule is not None:
            schedules.append(self._configuration.backup.schedule)
        schedules.extend(sampler.schedule for sampler in self._samplers)
        return tuple(schedules)

    def solve_steady(self) -> bool:
        """Run the steady SIMPLE iteration without advancing time.

        Returns
        -------
        bool
            Whether the steady loop converged. The steady iteration
            count is exposed as `step` for output numbering; `time` remains at
            the configured start time.

        Raises
        ------
        RuntimeError
            If the configured algorithm is not ``simple``, a candidate is pending,
            or the solver has become terminal after a failed evolution.
        FloatingPointError
            If the iteration returns a non-finite or shape-incompatible state.

        Notes
        -----
        The steady algorithm uses its dedicated steady iteration plan rather than the
        transient candidate/commit path.  ``time`` therefore remains at the
        configured start time; ``step`` is the steady iteration count used for
        output numbering and reporting.
        """
        if self._configuration.numerics.pimple.algorithm != "simple":
            raise RuntimeError("solve_steady() requires the 'simple' algorithm")
        self._ensure_evolution_usable()
        if self._step_phase != "accepted":
            raise RuntimeError("Cannot start a steady solve with a pending FVM candidate")

        effective_viscosity = self.compute_effective_viscosity()
        result = self.algorithm.solve(
            self.velocity,
            self.kinematic_pressure,
            density=self._configuration.numerics.transport.density,
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
        self.logger.steady_iterations(self.steady_iterations)
        self.step = self.steady_iterations
        self._n_committed_time_steps = self.steady_iterations
        self._accepted_time_step_size = self.time_step_size
        self._previous_time_step_size = self.time_step_size
        self._step_phase = "accepted"
        self._pending_step_size = None
        self._pending_acceptance_counters = None
        self._run_metadata_written = False
        if not self._run_started:
            self.run_status = RunStatus.completed if converged else RunStatus.not_converged
            self.write_run_metadata(status=self.run_status)
        return self.steady_converged

    def apply_velocity_boundaries(self, time: float) -> None:
        """Impose the case velocity histories at physical time ``time`` in s.

        Collective under MPI. A history with ``normal_only`` prescribes the
        normal velocity with a zero tangential normal gradient; a configured
        slip patch returns to slip when its normal velocity reaches zero.
        """
        configured = {item.name: item for item in self._configuration.boundaries}
        for index, condition in enumerate(self._configuration.velocity_boundaries):
            velocity = condition.velocity.at(time)
            previous = self._velocity_boundary_values.get(index)
            if previous is not None and np.array_equal(previous, velocity):
                continue
            for name in condition.patches:
                normals = self.gather_boundary_face_normal(name)
                if condition.normal_only:
                    normal_velocity = normals @ velocity
                    self.set_mixed_velocity_boundary_condition(
                        normal_velocity, np.zeros_like(normals), name
                    )
                    restore_slip = configured[name].velocity_type == "slip" and bool(
                        np.all(np.abs(normal_velocity) <= 1e-12)
                    )
                    restore_slip = self.parallel.bcast(restore_slip, root=0)
                    if restore_slip:
                        boundary = self._optional_patch(name)
                        if boundary is not None:
                            boundary["velocity_type"] = "slip"
                            for key in (
                                "normal_velocity_field",
                                "tangential_velocity_normal_gradient_field",
                                "max_removed_normal_gradient_normal_component",
                            ):
                                boundary.pop(key, None)
                else:
                    self.set_dirichlet_velocity_boundary_condition(
                        np.tile(velocity, (len(normals), 1)), name
                    )
            self._velocity_boundary_values[index] = velocity

    def _backup_path(self, step: int) -> Path:
        """Return the numbered backup of ``step`` in ``solution/fvm``."""
        name = f"fvm_{step:06d}" if self.parallel.is_partitioned else f"fvm_{step:06d}.npz"
        return component_directory(self.solution_dir, "fvm") / name

    def start_from(self, selection: str | Path = "latest") -> bool:
        """Select the initial state or a backup before the first step.

        Parameters
        ----------
        selection : {'latest', 'initial'} or path, default='latest'
            ``latest`` loads the backup with the highest step in
            ``solution/fvm`` and starts from the initial state when there is
            none; ``initial`` deletes the earlier output series and starts
            from the initial state; a path loads that backup.

        Returns
        -------
        bool
            Whether a backup was loaded.
        """
        from openonda.core.restart import reset_run_outputs, select_backup

        if selection == "initial":
            state_error = None
            if (
                self.step
                or self.time != self._configuration.run.start_time
                or self._n_committed_time_steps
                or self._step_phase != "accepted"
                or self._restart_loaded
            ):
                state_error = RuntimeError(
                    "start_from('initial') requires a new FVM solver before advancing"
                )
            self._collective_io_failure(state_error, "initial state selection")
            error = None
            if self.parallel.is_root:
                try:
                    reset_run_outputs(
                        self.solution_dir,
                        kind="fvm",
                        samples_dir=self.samples_dir,
                        sample_names={sampler.name for sampler in self._samplers},
                    )
                    self.io.rewind_histories(-1.0e100)
                except Exception as exc:
                    error = exc
            self._collective_io_failure(error, "initial output reset")
            self._initial_run_selected = True
            return False

        selected = None
        error = None
        if self.parallel.is_root:
            try:
                selected = select_backup(selection, directory=self.solution_dir, kind="fvm")
            except Exception as exc:
                error = exc
        self._collective_io_failure(error, "restart selection")
        selected = self.parallel.bcast(selected, root=0)
        if selected is None:
            return self.start_from("initial")
        self.load_backup(selected)
        self.logger.info(f"Continuing from {selected}: step {self.step}, time {self.time:.17g}")
        return True

    def run(self, *, start_from: str | Path | None = None) -> None:
        """Run the case to ``RunPlan.end_time``.

        Parameters
        ----------
        start_from : {'latest', 'initial'}, path or None, default=None
            Initial-state selection passed to :meth:`start_from`; ``None``
            keeps the current in-memory state.

        Notes
        -----
        The run writes the initial output, advances accepted steps (or runs
        the steady iteration), writes the final output and backup, records
        the run status in ``solution/fvm_metadata.json`` and closes the
        solver. It may be called once.

        Raises
        ------
        RuntimeError, FloatingPointError
            If a solve, an acceptance check, an output writer or a finalizer
            fails; the first failure is raised after collective cleanup.
        """
        if self._run_started:
            raise RuntimeError("FVMSolver.run() may be called only once")
        self._run_started = True
        self.run_status = RunStatus.running
        self.run_failure = None
        primary_failure: BaseException | None = None
        run_plan = self._configuration.run
        try:
            if start_from is not None:
                self.start_from(start_from)
            self._ensure_evolution_usable()
            restored = self._restart_loaded
            self.apply_velocity_boundaries(self.time)
            self.write_initial_output()
            if start_from is not None and not restored:
                self._save_scheduled_backup()
            if self._configuration.numerics.pimple.algorithm == "simple":
                converged = self.solve_steady()
                final_status = RunStatus.completed if converged else RunStatus.not_converged
            else:
                end_time = run_plan.end_time
                tolerance = max(1.0e-14, abs(end_time) * 1.0e-12)
                while self.time < end_time - tolerance:
                    self.advance()
                final_status = RunStatus.completed
            self.write_final_output()
            if (
                self._configuration.backup.write_at_end or start_from is not None
            ) and self._step_phase == "accepted":
                self._save_scheduled_backup()
            self.run_status = final_status
        except BaseException as error:
            primary_failure = error
            self.run_failure = error
            self.run_status = RunStatus.failed
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
                        self.run_status = RunStatus.failed

            # Every rank enters each stage, so a failure on one rank cannot
            # leave a peer waiting in the next MPI operation.
            finalize_collectively("final output flush", self.flush_output)
            finalize_collectively(
                "solver metadata", lambda: self.write_run_metadata(status=self.run_status)
            )
            finalize_collectively("solver close", lambda: self.close(failure=primary_failure))
        if primary_failure is not None:
            raise primary_failure.with_traceback(primary_failure.__traceback__)

    def advance(
        self,
        *,
        defer_output: bool = False,
        time: float | None = None,
        time_step_size: float | None = None,
    ) -> None:
        """Solve and accept one time step.

        The step size is the fixed or maximum-Courant size, shortened to land
        on output events and the end time. The candidate from
        :meth:`solve_step` is checked against the acceptance limits and
        accepted with :meth:`accept_step`.

        Parameters
        ----------
        defer_output : bool, default=False
            Leave the step output to a later :meth:`write_step_output`.
        time : float or None, default=None
            Accepted physical time in s imposed by a driver that owns the
            clock; it must equal ``time + time_step_size`` to round-off.
        time_step_size : float or None, default=None
            Step selected by a driver after calling
            :meth:`select_next_time_step_size`. ``None`` lets the FVM select
            its own step.

        Raises
        ------
        RuntimeError, ValueError, FloatingPointError, AcceptanceLimitError
            For an invalid phase, a failed solve or a rejected candidate.
        """
        self._ensure_evolution_usable()
        if self._configuration.numerics.pimple.algorithm == "simple":
            raise RuntimeError("The 'simple' algorithm is steady; use solve_steady()")
        if self._step_phase != "accepted":
            raise RuntimeError("A candidate FVM step is pending; call accept_step() first")
        if time_step_size is None:
            step_time_step_size = self._select_time_step_size()
        else:
            step_time_step_size = float(time_step_size)
            if not np.isfinite(step_time_step_size) or step_time_step_size <= 0.0:
                raise ValueError("time_step_size must be finite and positive")
            self.time_step_size = step_time_step_size
        self.apply_velocity_boundaries(self.time + step_time_step_size)
        step_number = self.step + 1
        self.profiler.begin_step(step_number, self.time, step_time_step_size)
        self.logger.begin_step(step_number, self.time + step_time_step_size, step_time_step_size)
        primary_failure: BaseException | None = None
        try:
            self.solve_step(step_time_step_size)
            control = self._configuration.numerics.time_step_control
            self.logger.courant_number(
                self.max_courant_number, control.maximum if control is not None else None
            )
            self.accept_step(defer_output=defer_output, time=time)
        except BaseException as error:
            primary_failure = error
            raise
        finally:
            self._finish_step_timing(primary_failure)

    def _finish_step_timing(self, primary_failure: BaseException | None) -> None:
        """Close the step record of the profiler and the log."""
        try:
            iterations: dict[str, float] = {}
            for result in getattr(self.algorithm, "last_linear_results", ()):
                key = f"{result.equation or 'linear'} iterations"
                iterations[key] = iterations.get(key, 0.0) + float(result.iterations)
            record = self.profiler.end_step(details=iterations)
            self.logger.end_step(self.profiler.step_seconds, accepted=primary_failure is None)
            if record is not None:
                self.logger.profile(record)
        except BaseException:
            if primary_failure is None:
                raise

    def accept_step(self, *, defer_output: bool = False, time: float | None = None) -> None:
        """Accept the solved candidate and advance the clock.

        Rolls the BDF time history of velocity and face flux, increments
        ``step`` and ``time`` and, unless deferred, writes the step output.
        It requires a candidate from :meth:`solve_step`.

        Parameters
        ----------
        defer_output : bool, default=False
            Leave the step output to a later :meth:`write_step_output`.
        time : float or None, default=None
            Accepted physical time in s imposed by a driver that owns the
            clock, such as the FVM--VPM coupler. It must equal
            ``time + time_step_size`` to round-off; ``None`` sums them.
        """
        if self._step_phase != "candidate":
            raise RuntimeError("Cannot accept an FVM step before a successful solve")
        self._ensure_evolution_usable()
        pending_counters = self._pending_acceptance_counters
        if pending_counters is None:
            raise RuntimeError("FVM candidate has not passed acceptance checks")
        summed_time = self.time + self._accepted_time_step_size
        if time is not None and not math.isclose(
            time, summed_time, rel_tol=1.0e-12, abs_tol=1.0e-9 * self._accepted_time_step_size
        ):
            raise ValueError(
                f"Imposed FVM time {time!r} s does not follow {self.time!r} s "
                f"by one time step of {self._accepted_time_step_size!r} s"
            )
        with self.profiler.section("field history"):
            # Roll the BDF history: older <- old (u^n-1), old <- u^n+1.
            self.velocity_older[:] = self.velocity_old[:]
            self.velocity_old[:] = self.velocity[:]
            self.volumetric_face_flux_older[:] = self.volumetric_face_flux_old[:]
            self.volumetric_face_flux_old[:] = self.volumetric_face_flux[:]
            self._n_committed_time_steps += 1
            self.step += 1
            self.time = summed_time if time is None else float(time)
            if self.mesh_motion is not None:
                self.mesh_motion.accept(self.time)
            self._run_metadata_written = False
            self._previous_time_step_size = self._accepted_time_step_size
            self._n_consecutive_accepted_steps = pending_counters
            self._pending_acceptance_counters = None
            self._pending_step_size = None
            self._step_phase = "accepted"

        if not defer_output:
            self.write_step_output()

    def write_initial_output(self) -> None:
        """Write the initial frame and the samplers with ``initial=True``.

        :meth:`run` calls this before the first step; a driver that advances
        the solver itself, such as the FVM--VPM coupler, calls it once at the
        start of a fresh run. Both follow ``RunPlan.initial_output``. A state
        restored from a backup already has its initial output, so the call
        does nothing then. Collective under MPI.
        """
        if self._restart_loaded or not self._configuration.run.initial_output:
            return
        if self.auto_write:
            self.write_vtk()
        self._execute_sampler_event(OutputEvent.initial)

    def write_final_output(self) -> None:
        """Write the final frame and the final-only samplers of the accepted state.

        :meth:`run` calls this after the last step; a driver that advances
        the solver itself calls it once at the end of the run. Both follow
        ``RunPlan.final_output``; a frame already written for this state is
        not repeated. Collective under MPI.
        """
        if self._step_phase != "accepted":
            raise RuntimeError("Output requires an accepted FVM state")
        if not self._configuration.run.final_output:
            return
        if self.auto_write and self._last_vtk_state != (self.step, self.time, self._state_revision):
            self.write_vtk()
        self._execute_sampler_event(OutputEvent.final)

    def write_step_output(self) -> None:
        """Write the diagnostics and the due output of the accepted state.

        A coupler may defer output during provisional boundary-coupling
        iterations and call this once the interface converges. Provisional
        iteration states are not written.
        """
        if self._step_phase != "accepted":
            raise RuntimeError("Output requires an accepted FVM state")
        step_time_step_size = self._accepted_time_step_size
        with self.profiler.section("diagnostics output"):
            diagnostics_error = None
            try:
                self.io.write_step_diagnostics()
            except BaseException as error:
                diagnostics_error = error
            self._collective_io_failure(diagnostics_error, "diagnostics output")

        with self.profiler.section("samplers"):
            self._execute_sampler_event(OutputEvent.accepted_step)

        if (
            self.turbulence is not None
            and self.eddy_viscosity is not None
            and self._configuration.logging.schedule.is_due(
                self.step, self.time, step_time_step_size
            )
        ):
            n_owned = (
                self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
            )
            sum_eddy_viscosity = float(
                self.parallel.global_sum(float(np.sum(self.eddy_viscosity[:n_owned])))
            )
            n_eddy_viscosity_values = int(self.parallel.global_sum(int(n_owned)))
            assert self.last_diagnostics is not None
            self.logger.turbulence(
                self.last_diagnostics.min_eddy_viscosity,
                self.last_diagnostics.max_eddy_viscosity,
                sum_eddy_viscosity / n_eddy_viscosity_values,
                self._kinematic_viscosity,
            )

        # Physical-time events have already constrained adaptive steps.
        output_schedule = self._configuration.output.schedule
        if (
            self.auto_write
            and output_schedule is not None
            and output_schedule.is_due(self.step, self.time, step_time_step_size)
        ):
            with self.profiler.section("visualization output"):
                self.write_vtk()

        backup = self._configuration.backup
        end_time = self._configuration.run.end_time
        end_tolerance = max(1.0e-14, abs(end_time) * 1.0e-12)
        at_end = self.time >= end_time - end_tolerance
        if (
            backup.schedule is not None
            and backup.schedule.is_due(self.step, self.time, step_time_step_size)
        ) or (backup.write_at_end and at_end):
            with self.profiler.section("restart backup"):
                self._save_scheduled_backup()

        if not self._run_started and at_end:
            self.run_status = RunStatus.completed
            self.write_run_metadata(status=self.run_status)
        # The Courant, gradient and vorticity caches describe this accepted
        # state and stay valid until solve_step() starts the next step; a
        # coupler reads the gradient immediately.

    def _save_scheduled_backup(self) -> None:
        """Write a backup of the accepted state unless it already has one."""
        state = (self.step, self.time, self._state_revision)
        if self._last_backup_state != state:
            self.save_backup()
            self._last_backup_state = state

    @property
    def state_revision(self) -> int:
        """Counter increased whenever the published fields change."""
        return self._state_revision

    def save_backup(self, path: str | Path | None = None) -> Path:
        """Flush output and atomically save the accepted state for a restart.

        Parameters
        ----------
        path : str or pathlib.Path or None
            Destination; ``None`` writes ``solution/fvm/fvm_{step:06d}.npz``
            (the directory ``solution/fvm/fvm_{step:06d}`` in a partitioned
            run).

        Returns
        -------
        pathlib.Path
            Path of the backup on every rank.

        Notes
        -----
        The backup holds the configuration and mesh digests, the primary
        fields and the time history of the time scheme. A candidate state
        cannot be saved.
        """
        with collective_phase(self.parallel.comm, "FVM backup accepted state"):
            self._ensure_evolution_usable()
            if self._step_phase != "accepted":
                raise RuntimeError("Cannot save a restart while an unaccepted FVM candidate exists")
        destination = self._backup_path(self.step) if path is None else Path(path)
        flush_error = None
        try:
            self.flush_output()
        except BaseException as error:
            flush_error = error
        self._collective_io_failure(flush_error, "output flush before restart")
        if self.parallel.is_partitioned:
            from ..io.partitioned import save_partitioned_solver_backup

            saved = Path(save_partitioned_solver_backup(self, destination))
        else:
            from ..io.backup import save_backup

            save_error = None
            if self.parallel.is_root:
                try:
                    save_backup(self, destination)
                except BaseException as error:
                    save_error = error
            self._collective_io_failure(save_error, "restart backup")
            saved = destination
        # A restart from this backup starts without reused preconditioners.
        self.algorithm.rebuild_preconditioners()
        self.logger.output(f"Restart backup written: {saved}")
        return saved

    def load_backup(
        self,
        path: str | Path,
        *,
        allow_time_step_control_change: bool = False,
    ) -> None:
        """Restore a backup of this case and rewind the output histories.

        Parameters
        ----------
        path : str or pathlib.Path
            Backup written by :meth:`save_backup` with the same mesh and
            numerical configuration.
        allow_time_step_control_change : bool, default=False
            Allow a fixed-step coupled backup to continue with an adaptive
            FVM time-step controller, provided every other restart setting
            matches the legacy backup digest.

        Raises
        ------
        ValueError, RuntimeError
            If the backup is corrupt or incompatible, or the output rewind
            fails on any rank.
        """
        flush_error = None
        try:
            self.flush_output()
        except BaseException as error:
            flush_error = error
        self._collective_io_failure(flush_error, "output flush before restart")
        if self.parallel.is_partitioned:
            from ..io.partitioned import load_partitioned_solver_backup

            load_partitioned_solver_backup(
                self, path, allow_time_step_control_change=allow_time_step_control_change
            )
        else:
            from ..io.backup import load_backup

            load_backup(self, path, allow_time_step_control_change=allow_time_step_control_change)

        rewind_error = None
        if self.parallel.is_root:
            try:
                self.io.rewind_histories(self.time)
            except Exception as error:
                rewind_error = error
        self._collective_io_failure(rewind_error, "restart output rewind")
        self._restart_loaded = True
        self._last_backup_state = (self.step, self.time, self._state_revision)

    def write_run_metadata(self, *, status: RunStatus | str) -> Path:
        """Write ``solution/fvm_metadata.json`` with the configuration and state.

        Parameters
        ----------
        status : RunStatus or str
            Run status recorded in the metadata.

        Returns
        -------
        pathlib.Path
            Path of the metadata file on every rank.
        """
        from openonda.core.metadata import write_run_metadata

        destination = self.solution_dir / "fvm_metadata.json"
        metadata_error = None
        if self.parallel.is_root:
            try:
                n_points = self.mesh_data.get("n_points")
                if n_points is None:
                    n_points = len(self.mesh_data["vertex_position"])
                write_run_metadata(
                    destination,
                    solver="fvm",
                    case_name=self._configuration.case_name,
                    status=RunStatus(status),
                    configuration=self._configuration,
                    sections={
                        "mesh": {
                            "n_cells": int(self.mesh_data["n_cells"]),
                            "n_faces": int(self.mesh_data["n_faces"]),
                            "n_points": int(n_points),
                            "mesh_generation": self.mesh_data.get("mesh_generation"),
                        },
                        "state": {"step": int(self.step), "time": float(self.time)},
                    },
                )
            except BaseException as error:
                metadata_error = error
        self._collective_io_failure(metadata_error, "solver metadata")
        self._run_metadata_written = True
        return destination

    def write_vtk(self) -> None:
        """Write the accepted state as a visualization frame.

        Collective under MPI. The frame ``solution/fvm/fvm_{step:06d}.vtu``
        (``.pvtu`` when partitioned) holds the cell-centred velocity,
        kinematic pressure, Courant number, vorticity, cell volume and
        size, and the eddy viscosity of an LES; it is listed in
        ``solution/fvm.pvd``. Asynchronous writes are queued; call
        :meth:`flush_output` to surface writer errors.
        """
        local_error = None
        try:
            if self.parallel.is_root or self.parallel.is_partitioned:
                self._write_vtk_local()
        except BaseException as error:
            local_error = error
        self._collective_io_failure(local_error, "visualization output")
        self._last_vtk_state = (self.step, self.time, self._state_revision)

    def _write_vtk_local(self) -> None:
        """Write this rank's part of the visualization frame."""
        from openonda.core.vtk_output import read_collection, write_collection

        from ..io.vtk_exporter import VTKExporter, mesh_cell_fields

        frame_dir = component_directory(self.solution_dir, "fvm")
        frame_dir.mkdir(parents=True, exist_ok=True)
        filename = frame_dir / f"fvm_{self.step:06d}.vtu"
        collection = collection_path(self.solution_dir, "fvm")
        output = self._configuration.output

        fields = {
            "velocity": self.velocity,
            "kinematic_pressure": self.kinematic_pressure,
            "courant_number": self._courant_field(self._accepted_time_step_size),
            "vorticity": self._vorticity_field(),
        }
        fields.update(mesh_cell_fields(self.mesh_data, self.geo_data["cell_volume"]))
        if self.eddy_viscosity is not None:
            fields["eddy_viscosity"] = self.eddy_viscosity

        if self.parallel.is_partitioned:
            from ..io.partitioned import write_partition_vtu

            n_local = self.mesh_data["n_cells"]
            if self.vtk_exporter is None:
                export_mesh = self.mesh_data.get("_visualization_mesh", self.mesh_data)
                self.vtk_exporter = VTKExporter(export_mesh, output)
            frame = write_partition_vtu(
                frame_dir,
                filename.stem,
                self.mesh_data,
                self.parallel.partition,
                {name: np.asarray(values)[:n_local] for name, values in fields.items()},
                self.parallel.comm,
                output=output,
                exporter=self.vtk_exporter,
            )
            if self.parallel.is_root:
                self._add_frame(collection, frame, read_collection, write_collection)
                self.logger.output(f"Output written: {Path(frame).name}")
            return

        if output.asynchronous or self._configuration.numerics.execution.output_mode == "threaded":
            if self._buffered_vtk_writer is None:
                from ..io.async_output import BufferedVTKWriter

                self._buffered_vtk_writer = BufferedVTKWriter(self.mesh_data, collection, output)
            self._buffered_vtk_writer.submit(filename, self.time, fields)
            action = "queued"
        else:
            if self.vtk_exporter is None:
                self.vtk_exporter = VTKExporter(self.mesh_data, output)
            self.vtk_exporter.export(str(filename), fields)
            self._add_frame(collection, filename, read_collection, write_collection)
            action = "written"
        self.logger.output(f"Output {action}: {filename.name}")

    def _add_frame(self, collection: Path, frame: str | Path, read, write) -> None:
        """List one frame in the visualization collection at the accepted time."""
        if self._frame_collection is None:
            self._frame_collection = read(collection)
        relative = Path(frame).resolve().relative_to(collection.parent.resolve()).as_posix()
        self._frame_collection = [
            entry for entry in self._frame_collection if entry[1] != relative
        ] + [(self.time, relative)]
        write(collection, self._frame_collection)

    def flush_output(self) -> None:
        """Wait for queued visualization frames and surface writer errors."""
        if self._buffered_vtk_writer is not None:
            self._buffered_vtk_writer.flush()
        self.logger.flush()

    def close(self, *, failure: BaseException | None = None) -> None:
        """Close the output writers, the linear-algebra resources and the log.

        Parameters
        ----------
        failure : BaseException or None
            Error that ended the run, reported in the closing log record.

        ``close`` is idempotent and does not change the fields. A cleanup
        error is raised so an incomplete close is never taken for a clean one.
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
            self.algorithm.close()
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        if failure is not None and self.run_status not in {
            RunStatus.failed,
            RunStatus.unstable,
        }:
            self.run_status = RunStatus.failed
        if not self._run_metadata_written:
            try:
                self.write_run_metadata(
                    status=RunStatus.failed if primary_failure is not None else self.run_status
                )
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        try:
            self.logger.close(status=self.run_status, failure=failure, profiler=self.profiler)
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        if primary_failure is not None:
            raise primary_failure
        self._closed = True

    def __enter__(self) -> "FVMSolver":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(failure=exc_value)

    def evaluate(self, callback, *args: object, **kwargs: object) -> object:
        """Run an analysis once on a complete, detached snapshot of the fields.

        Calls ``callback(snapshot, *args, **kwargs)`` and returns its result.
        The interior cells and physical boundary faces are gathered; the
        callback needs no MPI calls or rank checks.
        """
        from ..io.analysis import evaluate

        return evaluate(self, callback, *args, **kwargs)

    def write_csv(
        self,
        filename: str | Path,
        rows: Iterable[Sequence[object]],
        *,
        columns: Sequence[str],
        append: bool = False,
    ) -> None:
        """Write a table once; a relative file name is below ``solution/``."""
        from ..io.analysis import write_csv

        write_csv(self, filename, rows, columns=columns, append=append)

    def restore_history(self, filename: str | Path) -> bool:
        """Rewind a case CSV history to the accepted time.

        Use after :meth:`start_from` in a custom time loop, before appending
        rows. A relative path is below ``solution/``. Rows after the accepted
        time are deleted. Collective under MPI.

        Returns
        -------
        bool
            Whether the history already holds the accepted time.
        """
        import csv

        from openonda.core.file_output import TIME_TOLERANCE, rewind_csv

        error = None
        current = False
        if self.parallel.is_root:
            try:
                path = Path(filename)
                if not path.is_absolute():
                    path = self.solution_dir / path
                if self._initial_run_selected:
                    if path not in self._initial_rewound_histories:
                        rewind_csv(path, -1.0e100)
                        self._initial_rewound_histories.add(path)
                else:
                    rewind_csv(path, self.time)
                if path.is_file():
                    with path.open(newline="", encoding="utf-8") as stream:
                        current = any(
                            abs(float(row["time"]) - self.time) <= TIME_TOLERANCE
                            for row in csv.DictReader(stream)
                        )
            except Exception as exc:
                error = exc
        self._collective_io_failure(error, "history time alignment")
        return self.parallel.bcast(current, root=0)
