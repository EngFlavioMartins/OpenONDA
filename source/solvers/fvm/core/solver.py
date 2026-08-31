"""High-level incompressible FVM solver API."""

import json
import os
from typing import Any

import numpy as np

from ..config.types import FVMSetup
from ..coupling import CouplerInterfaceMixin
from ..io import logging, solver_io
from ..mesh import geometry
from ..sampling.executor import FVMSamplerExecutor
from ..solve import pimple_solver, simple_solver
from .parallel import ParallelContext
from .state import FieldState


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
    del case_dir, mesh_data
    if setup.initial_velocity is None:
        raise ValueError("initial_velocity must be provided in FVMSetup")
    initial = np.asarray(setup.initial_velocity, dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_velocity must be a finite three-component vector")
    return np.tile(initial, (n_total, 1))


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
    """Finite Volume Method (FVM) simulator for incompressible flow.

    Provides a high-level Python API for managing unstructured mesh CFD simulations.
    Supports PIMPLE/SIMPLE algorithms, Smagorinsky turbulence models, and VTK/PVD export.

    Attributes:
        config (FVMSetup): Simulation configuration object.
        case_dir (str): Root directory for simulation outputs and logs.
        mesh_data (Dict[str, Any]): Mesh connectivity and naming data.
        geo_data (Dict[str, Any]): Computed geometric properties (volumes, areas, etc.).
        velocity (np.ndarray): Velocity field [m/s] (includes ghost boundary cells).
        kinematic_pressure (np.ndarray): Kinematic pressure field [m^2/s^2].
        volumetric_face_flux (np.ndarray): Volumetric face flux ``velocity·Sf`` [m³/s].
            from owner to neighbour on interior faces.
        time (float): Current physical time in the simulation.
        step (int): Current time step index.
        auto_write (bool): If True, automatically writes results based on writeInterval.
    """

    @property
    def topology(self):
        """Build the immutable topology view only for consumers that request it."""
        if self._topology is None:
            from ..mesh.topology import MeshTopology

            self._topology = MeshTopology.from_mesh_data(self.mesh_data)
        return self._topology

    @property
    def geometry(self):
        """Build the typed geometry facade lazily without duplicating solver state."""
        if self._geometry is None:
            self._geometry = geometry.MeshGeometry.from_data(self.mesh_data, self.geo_data)
        return self._geometry

    def _invalidate_derived_fields(self) -> None:
        self._derived_fields.clear()

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
        setup: FVMSetup,
        case_dir: str | None = None,
        solution_dir: str | None = None,
        samples_dir: str | None = None,
        mesh_data: dict[str, Any] | None = None,
    ):
        """Initializes the FVM solver instance.

        Args:
            setup: FVMSetup object containing all simulation and time parameters.
            case_dir: Root directory for the case. Defaults to current working directory.
            solution_dir: Optional solver-output directory. Defaults to ``case_dir/solution``.
            samples_dir: Optional sampler-output directory. Defaults to ``case_dir/samples``.
            mesh_data: Solver-native mesh dictionary. Required on the root rank.
        """
        self.setup = setup
        self.case_dir = os.path.abspath(case_dir or os.getcwd())
        self.solution_dir = os.path.abspath(solution_dir or os.path.join(self.case_dir, "solution"))
        self.samples_dir = os.path.abspath(samples_dir or os.path.join(self.case_dir, "samples"))
        # These dictionaries intentionally contain heterogeneous mesh metadata
        # (arrays, counts, patch dictionaries, and parallel objects).
        self.mesh_data: Any
        self.geo_data: Any
        self.auto_write = True
        self.parallel = ParallelContext.create(self.setup.execution)
        self.logger = logging.Logging(
            self.case_dir,
            solution_dir=self.solution_dir,
            config=self.setup.logging,
            enabled=self.parallel.is_root,
        )
        from ..io.profiling import PerformanceProfiler

        self.profiler = PerformanceProfiler(
            self.case_dir,
            self.parallel,
            self.logger,
            solution_dir=self.solution_dir,
            solver=self,
        )
        self.logger.profiler = self.profiler
        self.operator_backend = self.setup.execution.operator_backend
        if self.setup.execution.linear_backend == "petsc":
            methods = {
                "momentum": self.setup.linear.momentum_solver or self.setup.linear.linear_solver,
                "pressure": self.setup.linear.pressure_solver or self.setup.linear.linear_solver,
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

        validate_solver_params(SimpleNamespace(**self.setup.algorithm_params()), self.setup.time)
        validate_turbulence(self.setup.turbulence)
        validate_acceptance_limits(self.setup.acceptance)
        if self.parallel.is_partitioned and self.setup.turbulence is not None:
            turbulence_name = self.setup.turbulence.model.lower()
            if self.setup.turbulence.dynamic or turbulence_name in {
                "dynamicsmagorinsky",
                "dynamic_smagorinsky",
            }:
                raise NotImplementedError(
                    "Dynamic Smagorinsky is not qualified for petsc_partitioned execution: "
                    "its Germano average must be reduced over owned cells globally."
                )
        if (
            self.setup.linear.pressure_nullspace_method == "petsc"
            and self.setup.execution.linear_backend != "petsc"
        ):
            raise ValueError(
                "pressure_nullspace_method='petsc' requires execution.linear_backend='petsc'"
            )
        if (
            self.parallel.is_partitioned
            and self.setup.linear.pressure_nullspace_method == "reference"
        ):
            raise ValueError(
                "petsc_partitioned requires pressure_nullspace_method='auto' or 'petsc'; "
                "a rank-local reference row is not a valid global pressure constraint"
            )
        if self.parallel.is_partitioned and self.setup.output.point_interpolation != "none":
            raise ValueError(
                "output.point_interpolation='boundary_weighted' is not qualified for "
                "petsc_partitioned execution: the partitioned writer drops the boundary "
                "ghost values the interpolation needs, and a rank's processor-interface "
                "faces are not physical boundaries. Run serially to write interpolated "
                "point data, or use ParaView's Cell Data to Point Data filter instead"
            )
        if not np.isfinite(self.setup.transport.density) or self.setup.transport.density <= 0.0:
            raise ValueError("Transport density must be finite and positive")
        if (
            not np.isfinite(self.setup.transport.kinematic_viscosity)
            or self.setup.transport.kinematic_viscosity <= 0.0
        ):
            raise ValueError("Kinematic viscosity must be finite and positive")
        if self.setup.dynamic_mesh.method != "static":
            raise NotImplementedError(
                "Dynamic meshes are not supported by the incompressible solver yet: "
                "the ALE mesh-flux terms required for conservative motion are not implemented."
            )

        # 0. UI Header
        self.logger.header("f64")
        logging.Timer.start("Total Initialization")

        # 1. Mesh Management
        from ..mesh.validation import (
            enforce_quality_thresholds,
            validate_geometry,
            validate_topology,
        )

        self._topology = None
        self._geometry = None
        gs = getattr(self.setup.schemes, "gradient_scheme", "gauss")
        logging.Timer.start("Geometry Compute")
        if self.parallel.is_partitioned:
            comm = self.parallel.comm
            assert comm is not None
            if any(boundary.velocity_type == "cyclic" for boundary in self.setup.boundaries):
                raise NotImplementedError(
                    "Partitioned cyclic patches require periodic partition adjacency, which is "
                    "not yet implemented"
                )
            if self.setup.initial_velocity is None or self.setup.initial_kinematic_pressure is None:
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
                    logging.Timer.start("Mesh Set (In-Memory)")
                    global_mesh = mesh_data
                    logging.Timer.log(
                        "Mesh Set (In-Memory)",
                        sink=self.logger,
                    )
                    validate_topology(global_mesh)
                    global_geo = geometry.compute_mesh_geometry(
                        global_mesh,
                        gradient_scheme=gs,
                        compute_lsq=False,
                        logger=self.logger,
                    )
                    quality = validate_geometry(global_mesh, global_geo)
                    enforce_quality_thresholds(quality, self.setup.mesh)
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
                            include_visualization_ghosts=self.setup.output.ghost_layers == 1,
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
            logging.Timer.start("Mesh Set (In-Memory)")
            self.mesh_data = mesh_data
            logging.Timer.log(
                "Mesh Set (In-Memory)",
                sink=self.logger,
            )
            validate_topology(self.mesh_data)
            self.geo_data = geometry.compute_mesh_geometry(
                self.mesh_data,
                gradient_scheme=gs,
                compute_lsq=False,
                logger=self.logger,
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
        enforce_quality_thresholds(self.mesh_quality, self.setup.mesh)

        from ..assemble.matrix_assembly import prepare_matrix_assembly

        prepare_matrix_assembly(self.mesh_data)

        logging.Timer.log(
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
        self._time_since_last_write = 0.0
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
        self._last_residuals = None
        self.last_diagnostics = None
        self._derived_fields: dict[object, np.ndarray] = {}
        self._n_consecutive_accepted_steps = {
            "max_continuity_error": 0,
            "max_equation_residual": 0,
            "max_courant_number": 0,
            "max_velocity_magnitude": 0,
        }

        initialization_time = logging.Timer.stop("Total Initialization")
        self.logger.log_solver_info(self, initialization_time)

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=self.volumetric_face_flux,
        )

        # Wall y+ is a per-step diagnostic decoupled from any force cadence.
        # Unless the user configured a YPlusSampler explicitly, run a default
        # that keeps ``last_y_plus`` fresh on every accepted step.
        self._default_yplus_sampler = None
        from ..sampling.forces import YPlusSampler

        if not any(isinstance(s, YPlusSampler) for s in (self.setup.samplers or ())):
            self._default_yplus_sampler = YPlusSampler(patch_names=None)

    def _setup_boundary_conditions(self):
        """Map user-defined BoundaryConfig entries to internal mesh boundary data.

        Iterates over ``self.setup.boundaries`` and updates the
        corresponding entries in ``self.boundaries`` with the configured
        type and value for velocity, kinematic_pressure, and eddy_viscosity.
        Patches not found in the mesh
        trigger a warning.
        """
        for b_cfg in self.setup.boundaries:
            found = False
            for b_mesh in self.boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.velocity_value, dtype=np.float64)
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

        self.velocity = _load_velocity_field(self.setup, self.case_dir, n_total, self.mesh_data)
        self.kinematic_pressure = _load_kinematic_pressure_field(
            self.setup, self.case_dir, n_total, self.mesh_data
        )
        self.velocity_old = self.velocity.copy()
        # Second history level for BDF2 (u^{n-1}); ignored by BDF1.
        self.velocity_older = self.velocity.copy()

        _enforce_velocity_boundary_constraints(
            self.velocity, self.boundaries, n_cells, self.mesh_data, self.geo_data
        )
        self.parallel.exchange_halo(self.velocity[:n_cells])
        self.parallel.exchange_halo(self.kinematic_pressure[:n_cells])

        logging.Timer.start("Flux Init")
        from ..assemble import convection

        self.volumetric_face_flux = convection.compute_volumetric_face_flux(
            self.velocity, self.mesh_data, self.geo_data
        )
        # Flux history for the transient Rhie-Chow correction.
        # (``fvc::ddtCorr``), which needs flux and velocity at the same levels.
        self.volumetric_face_flux_old = self.volumetric_face_flux.copy()
        self.volumetric_face_flux_older = self.volumetric_face_flux.copy()
        logging.Timer.log("Flux Init", sink=self.logger)

    def _initialize_algorithm(self):
        """Initialise the numerical solver algorithm.

        Reads the algorithm type from ``self.setup.pimple.algorithm``
        and instantiates either a :class:`~solve.pimple_solver.PIMPLESolver`
        or :class:`~solve.simple_solver.SIMPLESolver`.

        Raises:
            ValueError: If the algorithm is not ``"SIMPLE"``, ``"PIMPLE"``,
                        or ``"PISO"``.
        """
        logging.Timer.start("Algorithm Init")
        params = dict(self.setup.algorithm_params())
        params["_linear_backend"] = self.setup.execution.linear_backend
        params["_operator_backend"] = self.operator_backend
        params["_parallel_context"] = self.parallel
        params["_logger"] = self.logger
        algo = self.setup.pimple.algorithm.upper()

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
        logging.Timer.log("Algorithm Init", sink=self.logger)

    def set_initial_velocity(self, values: np.ndarray) -> None:
        """Set a cell-centred initial velocity and rebuild dependent state.

        ``values`` contains one vector per interior cell. Boundary ghosts are
        reconstructed from the configured boundary conditions, and both BDF
        history levels and the face flux are reset to the resulting field. In
        partitioned execution, ``values`` is the rank-local owned-plus-halo
        field; owned values are exchanged so every halo is made consistent.
        This operation is only valid before the first time step is committed.
        """
        if self._n_committed_time_steps or self.step:
            raise RuntimeError("Initial velocity can only be set before the first time step")

        n_cells = self.mesh_data["n_cells"]
        field = np.asarray(values, dtype=np.float64)
        if field.shape != (n_cells, 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Initial velocity must be finite with shape ({n_cells}, 3); got {field.shape}"
            )

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
        self.state = FieldState(self.velocity, self.kinematic_pressure, self.volumetric_face_flux)
        simple_solver.update_scalar_boundaries(
            self.kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=self.volumetric_face_flux,
        )

    def set_initial_state(self, velocity: np.ndarray, kinematic_pressure: np.ndarray) -> None:
        """Set a complete cell-centred initial state before the first step.

        This is intentionally narrower than backup loading: it supports
        deterministic manufactured/replay starts while retaining the solver's
        own boundary reconstruction, flux construction, and time-history
        ownership.
        """
        self.set_initial_velocity(velocity)
        n_cells = self.mesh_data["n_cells"]
        kinematic_pressure = np.asarray(kinematic_pressure, dtype=np.float64).reshape(-1)
        if kinematic_pressure.shape != (n_cells,) or not np.all(np.isfinite(kinematic_pressure)):
            raise ValueError(
                "Initial kinematic_pressure must be finite with one value per interior cell; "
                f"got {kinematic_pressure.shape}, expected ({n_cells},)"
            )
        self.kinematic_pressure[:n_cells] = kinematic_pressure
        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=self.volumetric_face_flux,
        )
        self.state = FieldState(self.velocity, self.kinematic_pressure, self.volumetric_face_flux)

    def set_post_solve_state_callback(self, callback) -> None:
        """Set a state projection called before accepted-step diagnostics.

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
        if self.setup.turbulence and self.setup.turbulence.model.lower() != "none":
            from ..turbulence import create_model

            self.turbulence = create_model(self.setup.turbulence, self.mesh_data, self.geo_data)
            if self.turbulence is None:
                raise RuntimeError(
                    f"Turbulence model {self.setup.turbulence.model!r} returned no model"
                )

        # Sync state
        self.time = self.setup.time.start_time
        self.step = 0
        self.time_step_size = self.setup.time.time_step_size

    def compute_effective_viscosity(self):
        """Compute the effective viscosity (molecular + turbulent).

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
            return self.setup.transport.kinematic_viscosity + self.eddy_viscosity
        return self.setup.transport.kinematic_viscosity

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
        """
        from ..immersed_boundary import IBMForcing

        if not hasattr(self.algorithm, "ibm"):
            raise ValueError(
                "Immersed boundaries require the PIMPLE/PISO algorithm "
                f"(configured: {self.setup.pimple.algorithm!r})."
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

        if not any(isinstance(s, IBMForceSampler) for s in (self.setup.samplers or ())):
            self._default_ibm_sampler = IBMForceSampler()
        diag = self.ibm.diagnostics()
        self.logger.info(
            f"component=immersed_boundary n_markers_total={diag['n_markers_total']} "
            f"grid_spacing_m={diag['grid_spacing']:.4g} "
            "marker_spacing_ratio_by_body={"
            f"{','.join(f'{key}:{value:.3f}' for key, value in diag['marker_spacing_ratio_by_body'].items())}"
            "} "
            f"min_kernel_row_sum={diag['min_kernel_row_sum']:.3f} "
            f"max_kernel_row_sum={diag['max_kernel_row_sum']:.3f} "
            f"max_quadrature_residual={diag['max_quadrature_residual']:.2e}"
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

    def solve_pimple(self, time_step_size: float | None = None):
        """Solve the pressure–velocity system at the current time level WITHOUT
        advancing the clock (coupler-facing method).

        Re-callable within a step: the coupler's VPM boundary-condition↔pressure Picard loop
        calls this repeatedly with the boundary condition re-imposed between
        solves, then a single :meth:`advance_time`.  The committed previous level
        ``U_old`` is the transient reference on every call.
        """
        from ..fields import diagnostics

        step_time_step_size = time_step_size if time_step_size is not None else self.time_step_size
        self._accepted_time_step_size = step_time_step_size

        # Diagnostics from the previously completed step cache full-mesh
        # Courant, velocity-gradient, and vorticity arrays.  None is valid once
        # a new solve begins, and retaining them through momentum/pressure
        # assembly adds roughly thirteen float64 values per local cell to the
        # transient peak.  Release them before turbulence and PIMPLE allocate
        # their workspaces; the solved state invalidation below remains the
        # guard for fields requested during a re-entrant/coupled solve.
        self._invalidate_derived_fields()

        logging.Timer.start("Effective viscosity")
        effective_viscosity = self.compute_effective_viscosity()
        logging.Timer.log(
            "Effective viscosity",
            sink=self.logger,
        )
        # BDF2 needs u^{n-1}; available only once at least one step is committed.
        velocity_older_argument = self.velocity_older if self._n_committed_time_steps >= 1 else None
        source_explicit, source_implicit = self._coupling_consistency_source()
        self.velocity, self.kinematic_pressure, self.volumetric_face_flux, residuals = (
            self.algorithm.step(
                self.velocity,
                self.kinematic_pressure,
                self.volumetric_face_flux,
                self.velocity_old,
                step_time_step_size,
                density=self.setup.transport.density,
                kinematic_viscosity=effective_viscosity,
                velocity_older=velocity_older_argument,
                source_explicit=source_explicit,
                source_implicit=source_implicit,
                volumetric_face_flux_old=self.volumetric_face_flux_old,
                volumetric_face_flux_older=(
                    self.volumetric_face_flux_older if self._n_committed_time_steps >= 1 else None
                ),
            )
        )
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
        self._invalidate_derived_fields()
        self.state = FieldState(self.velocity, self.kinematic_pressure, self.volumetric_face_flux)
        self._last_residuals = residuals
        self.logger.convergence_info(residuals)

        ibm = getattr(self, "ibm", None)
        if ibm is not None:
            ibm.update_fictitious_fluid_momentum_rate(
                self.velocity,
                self.velocity_old,
                step_time_step_size,
            )

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this makes loss of mass
        # conservation visible instead of silent.
        logging.Timer.start("Continuity diagnostics")
        continuity_error = diagnostics.compute_continuity_error(
            self.volumetric_face_flux, self.mesh_data, self.geo_data
        )
        cell_volume = self.geo_data["cell_volume"]
        n_owned = self.parallel.n_owned if self.parallel.is_partitioned else len(cell_volume)
        local_max = float(
            np.max(np.abs(continuity_error[:n_owned]) / (cell_volume[:n_owned] + 1e-30))
        )
        local_sum = float(np.sum(np.abs(continuity_error[:n_owned])))
        self.max_continuity_error = float(self.parallel.global_max(local_max))
        self.sum_absolute_continuity_error = float(self.parallel.global_sum(local_sum))
        self.logger.continuity_info(self.max_continuity_error, self.sum_absolute_continuity_error)
        logging.Timer.log("Continuity diagnostics", sink=self.logger)

        logging.Timer.start("Acceptance checks")
        self.last_diagnostics = self._build_step_diagnostics(step_time_step_size, residuals)
        self._enforce_acceptance_limits(self.last_diagnostics)
        logging.Timer.log("Acceptance checks", sink=self.logger)
        return residuals

    def _build_step_diagnostics(self, step_time_step_size, residuals):
        """Build the backend-neutral health record for the current solved state."""
        from ..fields import diagnostics
        from ..solve.diagnostics import StepDiagnostics

        n_cells = self.mesh_data["n_cells"]
        n_owned = self.parallel.n_owned if self.parallel.is_partitioned else n_cells
        interior_velocity = np.asarray(self.velocity[:n_owned])
        interior_kinematic_pressure = np.asarray(self.kinematic_pressure[:n_owned])
        courant_number_field = self._courant_field(step_time_step_size)
        self.max_courant_number = float(
            self.parallel.global_max(float(np.max(courant_number_field[:n_owned])))
        )
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
            min_eddy_viscosity = float(
                self.parallel.global_min(float(np.nanmin(self.eddy_viscosity[:n_owned])))
            )
            max_eddy_viscosity = float(
                self.parallel.global_max(float(np.nanmax(self.eddy_viscosity[:n_owned])))
            )
        n_interior = self.mesh_data["n_interior_faces"]
        linear_results = tuple(getattr(self.algorithm, "last_linear_results", ()))
        min_velocity = np.asarray(
            [
                self.parallel.global_min(float(value))
                for value in np.nanmin(interior_velocity, axis=0)
            ]
        )
        max_velocity = np.asarray(
            [
                self.parallel.global_max(float(value))
                for value in np.nanmax(interior_velocity, axis=0)
            ]
        )
        min_kinematic_pressure = float(
            self.parallel.global_min(float(np.nanmin(interior_kinematic_pressure)))
        )
        max_kinematic_pressure = float(
            self.parallel.global_max(float(np.nanmax(interior_kinematic_pressure)))
        )
        local_kinetic_energy = (
            0.5
            * self.setup.transport.density
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
            algorithm=self.setup.pimple.algorithm.upper(),
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
            min_velocity=tuple(float(value) for value in min_velocity),
            max_velocity=tuple(float(value) for value in max_velocity),
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

        limits = self.setup.acceptance
        max_velocity = max(
            float(np.linalg.norm(diagnostics.min_velocity)),
            float(np.linalg.norm(diagnostics.max_velocity)),
        )
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
        for name, value in metrics.items():
            warning = getattr(limits, f"{name}_warning")
            abort = getattr(limits, f"{name}_abort")
            if warning is not None and value > warning:
                warnings.append(f"{name}={value:.6g} exceeds warning threshold {warning:.6g}")
            if abort is not None and value > abort:
                self._n_consecutive_accepted_steps[name] += 1
            else:
                self._n_consecutive_accepted_steps[name] = 0
            if self._n_consecutive_accepted_steps[name] >= limits.sustained_steps:
                raise RuntimeError(
                    f"FVM acceptance limits rejected the step: {name}={value:.6g} "
                    f"exceeded {abort:.6g} for {self._n_consecutive_accepted_steps[name]} "
                    "consecutive solve(s)"
                )
        self.last_diagnostics = replace(diagnostics, warnings=tuple(warnings))
        self.logger.warnings_info(tuple(warnings))

    def advance(self, time_step_size: float | None = None) -> None:
        """Advance the simulation by one full time step (= solve_pimple + advance_time).

        Args:
            time_step_size: Optional override for the time step size [s].
        """
        from ..fields import diagnostics

        # --- CFL-based adaptive time-step adjustment (before step) ---
        cfg_time = self.setup.time
        if (
            time_step_size is None
            and cfg_time.adjust_time_step
            and self.max_courant_number > 0
            and self.step > 1
        ):
            ratio = cfg_time.max_courant_number / max(self.max_courant_number, 1e-8)
            ratio = min(ratio, cfg_time.time_step_size_adjust_coeff)
            self.time_step_size = np.clip(
                self.time_step_size * ratio,
                cfg_time.min_time_step_size,
                cfg_time.max_time_step_size,
            )

        step_time_step_size = time_step_size if time_step_size is not None else self.time_step_size
        self.profiler.begin_step(
            step=self.step + 1,
            time=self.time + step_time_step_size,
            time_step_size=step_time_step_size,
        )
        logging.Timer.start(f"Step {self.step + 1}")
        self.logger.step_begin(self.step + 1, self.time + step_time_step_size, step_time_step_size)

        self.solve_pimple(step_time_step_size)

        # Compute CFL after step (for next step's time-step adjustment)
        if cfg_time.adjust_time_step:
            courant_number_field = diagnostics.compute_courant_number(
                self.velocity,
                self.volumetric_face_flux,
                step_time_step_size,
                self.mesh_data,
                self.geo_data,
            )
            n_owned = (
                self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
            )
            self.max_courant_number = float(
                self.parallel.global_max(float(np.max(courant_number_field[:n_owned])))
            )
        self.logger.courant_info(
            self.max_courant_number,
            cfg_time.max_courant_number if cfg_time.adjust_time_step else None,
        )

        self.advance_time()

        elapsed = logging.Timer.stop(f"Step {self.step}")
        self.logger.step_end(elapsed)
        self.profiler.finish_step(
            elapsed,
            getattr(self.algorithm, "last_linear_results", ()),
        )

    def advance_time(self) -> None:
        """Commit the solved field as the new time level and advance the clock
        (coupler-facing method): roll the BDF history, increment step/time, then
        run per-step force logging and output control."""
        # Roll the BDF time-history ring: U_old_old <- u^n, U_old <- u^{n+1}.
        logging.Timer.start("Field history commit")
        self.velocity_older[:] = self.velocity_old[:]
        self.velocity_old[:] = self.velocity[:]
        self.volumetric_face_flux_older[:] = self.volumetric_face_flux_old[:]
        self.volumetric_face_flux_old[:] = self.volumetric_face_flux[:]
        self._n_committed_time_steps += 1
        self.step += 1
        self.time += self._accepted_time_step_size
        step_time_step_size = self._accepted_time_step_size
        cfg_time = self.setup.time
        logging.Timer.log("Field history commit", sink=self.logger)

        logging.Timer.start("Diagnostics file")
        self.io.write_step_diagnostics()
        logging.Timer.log("Diagnostics file", sink=self.logger)

        # Samplers decide their own cadence; the executor runs after every
        # accepted step and every sampler checks whether it is due.  Force,
        # IBM force and y+ output all flow through this single path.
        logging.Timer.start("Samplers")
        FVMSamplerExecutor.execute(self)
        logging.Timer.log("Samplers", sink=self.logger)

        logging.Timer.start("Turbulence statistics")
        if self.turbulence and self.eddy_viscosity is not None:
            n_owned = (
                self.parallel.n_owned if self.parallel.is_partitioned else self.mesh_data["n_cells"]
            )
            owned_eddy_viscosity = self.eddy_viscosity[:n_owned]
            min_eddy_viscosity = float(
                self.parallel.global_min(float(np.min(owned_eddy_viscosity)))
            )
            max_eddy_viscosity = float(
                self.parallel.global_max(float(np.max(owned_eddy_viscosity)))
            )
            sum_eddy_viscosity = float(
                self.parallel.global_sum(float(np.sum(owned_eddy_viscosity)))
            )
            n_eddy_viscosity_values = int(self.parallel.global_sum(int(n_owned)))
            if self.parallel.is_root:
                self.logger.turbulence_info(
                    self.eddy_viscosity,
                    self.setup.transport.kinematic_viscosity,
                    statistics=(
                        min_eddy_viscosity,
                        max_eddy_viscosity,
                        sum_eddy_viscosity / n_eddy_viscosity_values,
                    ),
                )
        logging.Timer.log("Turbulence statistics", sink=self.logger)

        # Output control — time-based if write_interval_time is set, else step-based
        logging.Timer.start("Visualization output")
        if (self.parallel.is_root or self.parallel.is_partitioned) and self.auto_write:
            wrt_time = cfg_time.output_interval_time
            if wrt_time is not None:
                self._time_since_last_write += step_time_step_size
                if self._time_since_last_write >= wrt_time:
                    self.write_vtk()
                    self._time_since_last_write = 0.0
            else:
                if self.step % cfg_time.output_interval_steps == 0:
                    self.write_vtk()
        logging.Timer.log("Visualization output", sink=self.logger)

        # Courant/gradient/vorticity caches describe this accepted state and
        # remain valid until ``solve_pimple`` starts the next mutation.  In a
        # coupled step the endpoint gradient is consumed immediately by the
        # vorticity handoff; dropping it here forced an identical full-mesh
        # reconstruction and global gather.  ``solve_pimple`` clears the cache
        # before its next assembly, retaining the former peak-memory behaviour.

    def save_state(self, path) -> str:
        """Atomically save a versioned restart containing the complete time state."""
        self.flush_output()
        if self.parallel.is_partitioned:
            from ..io.partitioned import save_partitioned_solver_backup

            return str(save_partitioned_solver_backup(self, path))
        from ..io.backup import save_backup

        saved = None
        if self.parallel.is_root:
            saved = save_backup(self, path)
        self.parallel.barrier()
        return str(saved if saved is not None else path)

    def write_run_manifest(self, path=None) -> str:
        """Write source, dependency, backend, mesh, and configuration identity."""
        from ..io.manifest import write_manifest

        destination = path or os.path.join(self.solution_dir, "run_manifest.json")
        written = None
        if self.parallel.is_root:
            written = write_manifest(self, destination)
        self.parallel.barrier()
        return str(written if written is not None else destination)

    def load_state(self, path, *, allow_config_change: bool = False) -> None:
        """Restore a compatible restart, rejecting mismatched meshes or configs."""
        self.flush_output()
        if self.parallel.is_partitioned:
            from ..io.partitioned import load_partitioned_solver_backup

            load_partitioned_solver_backup(self, path, allow_config_change=allow_config_change)
            self.io.rewind_histories(self.time)
            self.parallel.barrier()
            return
        from ..io.backup import load_backup

        self.parallel.barrier()
        load_backup(self, path, allow_config_change=allow_config_change)
        self.io.rewind_histories(self.time)
        self.parallel.barrier()

    def write_vtk(self, filename: str | None = None) -> None:
        """Export the current simulation state to a ``.vtu`` file with PVD time-series support.

        Writes ``velocity``, ``kinematic_pressure``, ``courant_number``, and
        ``vorticity`` fields. If turbulence is active, also writes
        ``eddy_viscosity``. These names are the canonical cross-solver output
        schema; compact kernel symbols are not serialized.
        Updates the PVD collection file for time-series visualisation.

        Args:
            filename: Optional output path.  If ``None``, auto-generates
                      ``solution/{case_name}_{step:06d}.vtu``.
        """
        if not self.parallel.is_root and not self.parallel.is_partitioned:
            return
        sol_dir = self.solution_dir
        if filename is None:
            os.makedirs(sol_dir, exist_ok=True)
            # Use case_name and sequential numbering: case_name_000000.vtu
            filename = os.path.join(sol_dir, f"{self.setup.case_name}_{self.step:06d}.vtu")

        fields = {
            "velocity": self.velocity,
            "kinematic_pressure": self.kinematic_pressure,
            "courant_number": self._courant_field(self._accepted_time_step_size),
            "vorticity": self._vorticity_field(),
        }
        if self.eddy_viscosity is not None:
            fields["eddy_viscosity"] = self.eddy_viscosity

        if self.parallel.is_partitioned:
            from pathlib import Path

            from ..io.partitioned import write_partition_vtu
            from ..io.vtk_exporter import VTKExporter

            n_local = self.mesh_data["n_cells"]
            stem = Path(filename).stem
            if self.vtk_exporter is None:
                export_mesh = self.mesh_data.get("_visualization_mesh", self.mesh_data)
                self.vtk_exporter = VTKExporter(export_mesh, self.setup.output)
            collection = write_partition_vtu(
                Path(filename).parent,
                stem,
                self.mesh_data,
                self.parallel.partition,
                {name: np.asarray(values)[:n_local] for name, values in fields.items()},
                self.parallel.comm,
                output=self.setup.output,
                exporter=self.vtk_exporter,
            )
            if self.parallel.is_root:
                from ..io.vtk_exporter import PVDManager

                pvd_file = os.path.join(sol_dir, f"{self.setup.case_name}.pvd")
                if self.pvd_manager is None:
                    self.pvd_manager = PVDManager(pvd_file)
                self.pvd_manager.add_step(self.time, str(collection))
                self.logger.output_info(f"Output written: {stem}.pvtu")
            return

        asynchronous = (
            self.setup.output.asynchronous or self.setup.execution.output_mode == "threaded"
        )
        if asynchronous:
            if self._buffered_vtk_writer is None:
                from ..io.async_output import BufferedVTKWriter

                pvd_file = os.path.join(sol_dir, f"{self.setup.case_name}.pvd")
                self._buffered_vtk_writer = BufferedVTKWriter(
                    self.mesh_data,
                    pvd_file,
                    self.setup.output,
                )
            self._buffered_vtk_writer.submit(filename, self.time, fields)
            action = "queued"
        else:
            if self.vtk_exporter is None:
                from ..io.vtk_exporter import VTKExporter

                self.vtk_exporter = VTKExporter(self.mesh_data, self.setup.output)
            if self.pvd_manager is None:
                from ..io.vtk_exporter import PVDManager

                pvd_file = os.path.join(sol_dir, f"{self.setup.case_name}.pvd")
                self.pvd_manager = PVDManager(pvd_file)
            self.vtk_exporter.export(filename, fields)
            self.pvd_manager.add_step(self.time, filename)
            action = "written"

        self.logger.output_info(f"Output {action}: {os.path.basename(filename)}")

    def flush_output(self) -> None:
        """Wait for buffered visualization output and surface writer failures."""
        if self._buffered_vtk_writer is not None:
            self._buffered_vtk_writer.flush()
        self.logger.flush()

    def close(self) -> None:
        """Finish background output resources owned by the solver."""
        if self._buffered_vtk_writer is not None:
            self._buffered_vtk_writer.close()
        algorithm_close = getattr(self.algorithm, "close", None)
        if algorithm_close is not None:
            algorithm_close()
        self.profiler.close()
        self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def info(self) -> None:
        """Print a summary of the current solver state.

        Displays case name, flow time, time step, cell count, and
        active algorithm.
        """
        self.logger.solver_state(self)
