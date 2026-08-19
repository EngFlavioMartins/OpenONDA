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


def _load_velocity_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Initialise the velocity field from the Python configuration.

    Args:
        config:   FVMSetup (may have ``initial_velocity``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Velocity array ``(n_total, 3)``.
    """
    del case_dir, mesh_data
    if config.initial_velocity is None:
        raise ValueError("initial_velocity must be provided in FVMSetup")
    initial = np.asarray(config.initial_velocity, dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_velocity must be a finite three-component vector")
    return np.tile(initial, (n_total, 1))


def _load_pressure_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Initialise the pressure field from the Python configuration.

    Args:
        config:   FVMSetup (may have ``initial_p``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Pressure array ``(n_total,)``.
    """
    del case_dir, mesh_data
    if config.initial_p is None:
        raise ValueError("initial_p must be provided in FVMSetup")
    initial = np.asarray(config.initial_p, dtype=np.float64)
    if initial.ndim != 0 or not np.isfinite(initial):
        raise ValueError("initial_p must be a finite scalar")
    return np.full(n_total, float(initial), dtype=np.float64)


def _enforce_u_boundary_constraints(
    U: np.ndarray, boundaries: list, n_elements: int, mesh_data: dict, geo_data: dict
) -> None:
    """Enforce velocity boundary constraints on ghost cells after initialisation.

    Iterates over all boundary patches and sets the ghost-layer values
    in *U* according to each patch's boundary condition type (noSlip,
    fixedValue, zeroGradient, empty, etc.).

    Args:
        U:          Velocity array (mutated in place).
        boundaries: List of boundary patch dictionaries.
        n_elements: Number of interior elements.
        mesh_data:  Mesh dictionary.
        geo_data:   Geometry dictionary.
    """
    from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy

    for boundary in boundaries:
        bc_type = boundary.get("bc_type_velocity")
        strategy = BOUNDARIES.strategy(bc_type, "U", "ghost")
        start = n_elements + (boundary["startFace"] - mesh_data["n_interior_faces"])
        end = start + boundary["nFaces"]
        if strategy is BoundaryStrategy.NO_SLIP:
            U[start:end] = 0.0
        elif (
            strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.FREESTREAM)
            and boundary.get("value_velocity_field") is not None
        ):
            U[start:end] = boundary["value_velocity_field"]
        elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.FREESTREAM) and (
            "value_velocity" in boundary
        ):
            U[start:end] = boundary["value_velocity"]
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.INLET_OUTLET,
        ):
            owners_b = mesh_data["owners"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            U[start:end] = U[owners_b]
        elif strategy is BoundaryStrategy.CYCLIC:
            faces = np.arange(boundary["startFace"], boundary["startFace"] + boundary["nFaces"])
            paired = mesh_data["boundary_neighbours"][faces]
            if np.any(paired < 0):
                raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
            U[start:end] = U[paired]
        elif strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            owners_b = mesh_data["owners"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            face_sf = geo_data["face_sf"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            owner_velocity = U[owners_b]
            magnitudes = np.linalg.norm(face_sf, axis=1)
            valid = magnitudes > 1e-10
            projected = owner_velocity.copy()
            if np.any(valid):
                normals = face_sf[valid] / magnitudes[valid, np.newaxis]
                projected[valid] -= (
                    np.sum(owner_velocity[valid] * normals, axis=1)[:, np.newaxis] * normals
                )
            U[start:end] = projected


class FVMSolver(CouplerInterfaceMixin):
    """Finite Volume Method (FVM) simulator for incompressible flow.

    Provides a high-level Python API for managing unstructured mesh CFD simulations.
    Supports PIMPLE/SIMPLE algorithms, Smagorinsky turbulence models, and VTK/PVD export.

    Attributes:
        config (FVMSetup): Simulation configuration object.
        case_dir (str): Root directory for simulation outputs and logs.
        mesh_data (Dict[str, Any]): Mesh connectivity and naming data.
        geo_data (Dict[str, Any]): Computed geometric properties (volumes, areas, etc.).
        U (np.ndarray): Velocity field [m/s] (includes ghost boundary cells).
        p (np.ndarray): Kinematic pressure field [m^2/s^2] (includes ghost boundary cells).
        phi (np.ndarray): Volumetric face flux ``U·Sf`` [m³/s], positive
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
                self.U, self.mesh_data, self.geo_data
            )
            self._derived_fields["velocity_gradient"] = gradient
        return gradient

    def _courant_field(self, time_step_size: float):
        from ..fields import diagnostics

        key = ("courant", float(time_step_size))
        courant = self._derived_fields.get(key)
        if courant is None:
            courant = diagnostics.compute_courant_number(
                self.U, self.phi, time_step_size, self.mesh_data, self.geo_data
            )
            self._derived_fields[key] = courant
        return courant

    def _vorticity_field(self):
        from ..fields import diagnostics

        vorticity = self._derived_fields.get("vorticity")
        if vorticity is None:
            vorticity = diagnostics.compute_vorticity(
                self.U,
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
        mesh_data: dict[str, Any] | None = None,
    ):
        """Initializes the FVM solver instance.

        Args:
            setup: FVMSetup object containing all simulation and time parameters.
            case_dir: Root directory for the case. Defaults to current working directory.
            mesh_data: Solver-native mesh dictionary. Required on the root rank.
        """
        self.config = setup
        self.case_dir = os.path.abspath(case_dir or os.getcwd())
        # These dictionaries intentionally contain heterogeneous mesh metadata
        # (arrays, counts, patch dictionaries, and parallel objects).
        self.mesh_data: Any
        self.geo_data: Any
        self.auto_write = True
        self.parallel = ParallelContext.create(self.config.execution)
        self.logger = logging.Logging(
            self.case_dir,
            config=self.config.logging,
            enabled=self.parallel.is_root,
        )
        from ..io.profiling import PerformanceProfiler

        self.profiler = PerformanceProfiler(
            self.case_dir,
            self.parallel,
            self.logger,
            solver=self,
        )
        self.logger.profiler = self.profiler
        self.operator_backend = self.config.execution.operator_backend
        if self.config.execution.linear_backend == "petsc":
            methods = {
                "momentum": self.config.linear.momentum_solver or self.config.linear.linear_solver,
                "pressure": self.config.linear.pressure_solver or self.config.linear.linear_solver,
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
            validate_acceptance_policy,
            validate_solver_params,
            validate_turbulence,
        )

        validate_solver_params(SimpleNamespace(**self.config.algorithm_params()), self.config.time)
        validate_turbulence(self.config.turbulence)
        validate_acceptance_policy(self.config.acceptance)
        if self.parallel.is_partitioned and self.config.turbulence is not None:
            turbulence_name = self.config.turbulence.model.lower()
            if self.config.turbulence.dynamic or turbulence_name in {
                "dynamicsmagorinsky",
                "dynamic_smagorinsky",
            }:
                raise NotImplementedError(
                    "Dynamic Smagorinsky is not qualified for petsc_partitioned execution: "
                    "its Germano average must be reduced over owned cells globally."
                )
        if (
            self.config.linear.pressure_nullspace_policy == "petsc"
            and self.config.execution.linear_backend != "petsc"
        ):
            raise ValueError(
                "pressure_nullspace_policy='petsc' requires execution.linear_backend='petsc'"
            )
        if (
            self.parallel.is_partitioned
            and self.config.linear.pressure_nullspace_policy == "reference"
        ):
            raise ValueError(
                "petsc_partitioned requires pressure_nullspace_policy='auto' or 'petsc'; "
                "a rank-local reference row is not a valid global pressure constraint"
            )
        if self.parallel.is_partitioned and self.config.output.point_interpolation != "none":
            raise ValueError(
                "output.point_interpolation='boundary_weighted' is not qualified for "
                "petsc_partitioned execution: the partitioned writer drops the boundary "
                "ghost values the interpolation needs, and a rank's processor-interface "
                "faces are not physical boundaries. Run serially to write interpolated "
                "point data, or use ParaView's Cell Data to Point Data filter instead"
            )
        if not np.isfinite(self.config.transport.density) or self.config.transport.density <= 0.0:
            raise ValueError("Transport density must be finite and positive")
        if not np.isfinite(self.config.transport.nu) or self.config.transport.nu <= 0.0:
            raise ValueError("Kinematic viscosity must be finite and positive")
        if self.config.dynamic_mesh.method != "static":
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
        gs = getattr(self.config.schemes, "gradient_scheme", "gauss")
        logging.Timer.start("Geometry Compute")
        if self.parallel.is_partitioned:
            comm = self.parallel.comm
            assert comm is not None
            if any(boundary.type_velocity == "cyclic" for boundary in self.config.boundaries):
                raise NotImplementedError(
                    "Partitioned cyclic patches require periodic partition adjacency, which is "
                    "not yet implemented"
                )
            if self.config.initial_velocity is None or self.config.initial_p is None:
                raise ValueError("initial_velocity and initial_p must be provided in FVMSetup")
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
                    enforce_quality_thresholds(quality, self.config.mesh)
                    from ..io.checkpoint import mesh_hash

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
                            include_visualization_ghosts=self.config.output.ghost_layers == 1,
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
        if gs == "lsq" and np.any(self.mesh_data["boundary_neighbours"] >= 0):
            from ..fields.gradients import compute_lsq_geometry

            self.geo_data.update(compute_lsq_geometry(self.mesh_data, self.geo_data))

        # LSQ must be built after periodic topology is installed.  Non-cyclic
        # meshes also reach this point exactly once, rather than during base
        # geometry and again after boundary setup.
        if gs == "lsq" and "lsq_M_inv" not in self.geo_data:
            from ..fields.gradients import compute_lsq_geometry

            self.geo_data.update(compute_lsq_geometry(self.mesh_data, self.geo_data))

        self.mesh_quality = validate_geometry(self.mesh_data, self.geo_data)
        enforce_quality_thresholds(self.mesh_quality, self.config.mesh)

        from ..assemble.matrix_assembly import prepare_matrix_assembly

        prepare_matrix_assembly(self.mesh_data)

        logging.Timer.log(
            "Geometry Compute",
            sink=self.logger,
        )

        # 3. Component Setup
        self._initialize_fields()
        self.state = FieldState(self.U, self.p, self.phi)
        self._initialize_algorithm()
        self._initialize_turbulence()

        # Final housekeeping
        self.io = solver_io.SolverIO(self)
        self.vtk_exporter = None
        self.pvd_manager = None
        self._buffered_vtk_writer = None
        self.last_forces = None
        self.last_yplus = None
        self.ibm = None
        self.forces_history_path = None
        self.cfl_max = 0.0
        self._time_since_last_write = 0.0
        # Coupling / driver-split state
        self.registered_fields: dict[str, np.ndarray] = {}  # named volume fields (fvOptions)
        # Blending-zone relaxation acts on the resolved scales only (see _blending_source).
        # Set False to recover the plain S = λ(Utarget − U).
        self.blending_scale_selective = True
        self._blending_filter = None  # lazy CellBoxFilter, built on first blending-zone solve
        self._n_committed = 0  # number of committed time steps (BDF2 startup gate)
        self._current_time_step_size = self.time_step_size
        self._last_residuals = None
        self.last_diagnostics = None
        self._derived_fields: dict[object, np.ndarray] = {}
        self._acceptance_counts = {
            "continuity": 0,
            "residual": 0,
            "cfl": 0,
            "velocity": 0,
        }

        initialization_time = logging.Timer.stop("Total Initialization")
        self.logger.log_solver_info(self, initialization_time)

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.p, self.mesh_data, self.boundaries, "p", face_flux=self.phi
        )

        # Wall y+ is a per-step diagnostic decoupled from any force cadence.
        # Unless the user configured a YPlusSampler explicitly, run a default
        # that keeps ``last_yplus`` fresh on every accepted step.
        self._default_yplus_sampler = None
        from ..sampling.forces import YPlusSampler

        if not any(isinstance(s, YPlusSampler) for s in (self.config.samplers or ())):
            self._default_yplus_sampler = YPlusSampler(patch_names=None)

    def _setup_boundary_conditions(self):
        """Map user-defined BoundaryConfig entries to internal mesh boundary data.

        Iterates over ``self.config.boundaries`` and updates the
        corresponding entries in ``self.boundaries`` with the configured
        type and value for U, p, and nut.  Patches not found in the mesh
        trigger a warning.
        """
        for b_cfg in self.config.boundaries:
            found = False
            for b_mesh in self.boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.value_velocity, dtype=np.float64)
                    if velocity.shape not in {(3,), (b_mesh["nFaces"], 3)}:
                        raise ValueError(
                            f"Velocity value for patch {b_cfg.name!r} has shape {velocity.shape}; "
                            f"expected (3,) or {(b_mesh['nFaces'], 3)}"
                        )
                    b_mesh.update(
                        {
                            "bc_type_velocity": b_cfg.type_velocity,
                            "bc_type_p": b_cfg.type_p,
                            "value_p": b_cfg.value_p,
                            "bc_type_nut": b_cfg.type_nut,
                            "value_nut": b_cfg.value_nut,
                        }
                    )
                    if b_cfg.mesh_type is not None:
                        b_mesh["type"] = b_cfg.mesh_type
                    else:
                        b_mesh.setdefault("type", "patch")
                    if b_cfg.neighbour_patch is not None:
                        b_mesh["neighbourPatch"] = b_cfg.neighbour_patch
                    if velocity.shape == (3,):
                        b_mesh["value_velocity"] = velocity
                        b_mesh.pop("value_velocity_field", None)
                    else:
                        b_mesh["value_velocity_field"] = velocity
                    found = True
                    break
            if not found:
                global_names = self.mesh_data.get("global_boundary_names", ())
                if not self.parallel.is_partitioned or b_cfg.name not in global_names:
                    raise ValueError(
                        f"Configured boundary {b_cfg.name!r} was not found in the mesh"
                    )

    def _initialize_fields(self):
        """Initialise velocity (U), pressure (p), and flux (phi) fields.

        Loads or creates the initial fields, enforces boundary constraints
        on the velocity ghost layer, and computes the initial volumetric face
        flux ``phi = U·Sf`` from the velocity field.
        """
        n_elements = self.mesh_data["n_elements"]
        n_total = self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"] + n_elements

        self.U = _load_velocity_field(self.config, self.case_dir, n_total, self.mesh_data)
        self.p = _load_pressure_field(self.config, self.case_dir, n_total, self.mesh_data)
        self.U_old = self.U.copy()
        # Second history level for BDF2 (u^{n-1}); ignored by BDF1.
        self.U_old_old = self.U.copy()

        _enforce_u_boundary_constraints(
            self.U, self.boundaries, n_elements, self.mesh_data, self.geo_data
        )
        self.parallel.exchange_halo(self.U[:n_elements])
        self.parallel.exchange_halo(self.p[:n_elements])

        logging.Timer.start("Flux Init")
        from ..assemble import convection

        self.phi = convection.compute_volumetric_face_flux(self.U, self.mesh_data, self.geo_data)
        # Flux history for the transient Rhie-Chow correction.
        # (``fvc::ddtCorr``), which needs phi and U at the same time levels.
        self.phi_old = self.phi.copy()
        self.phi_old_old = self.phi.copy()
        logging.Timer.log("Flux Init", sink=self.logger)

    def _initialize_algorithm(self):
        """Initialise the numerical solver algorithm.

        Reads the algorithm type from ``self.config.pimple.algorithm``
        and instantiates either a :class:`~solve.pimple_solver.PIMPLESolver`
        or :class:`~solve.simple_solver.SIMPLESolver`.

        Raises:
            ValueError: If the algorithm is not ``"SIMPLE"``, ``"PIMPLE"``,
                        or ``"PISO"``.
        """
        logging.Timer.start("Algorithm Init")
        params = dict(self.config.algorithm_params())
        params["_linear_backend"] = self.config.execution.linear_backend
        params["_operator_backend"] = self.operator_backend
        params["_parallel_context"] = self.parallel
        params["_logger"] = self.logger
        algo = self.config.pimple.algorithm.upper()

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
        if self._n_committed or self.step:
            raise RuntimeError("Initial velocity can only be set before the first time step")

        n_elements = self.mesh_data["n_elements"]
        field = np.asarray(values, dtype=np.float64)
        if field.shape != (n_elements, 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Initial velocity must be finite with shape ({n_elements}, 3); got {field.shape}"
            )

        self.U[:n_elements] = field
        if self.parallel.is_partitioned:
            self.parallel.exchange_halo(self.U[:n_elements])
        _enforce_u_boundary_constraints(
            self.U, self.boundaries, n_elements, self.mesh_data, self.geo_data
        )
        self.U_old[:] = self.U
        self.U_old_old[:] = self.U

        from ..assemble import convection
        from ..solve import simple_solver

        self.phi = convection.compute_volumetric_face_flux(self.U, self.mesh_data, self.geo_data)
        self.phi_old = self.phi.copy()
        self.phi_old_old = self.phi.copy()
        self.state = FieldState(self.U, self.p, self.phi)
        simple_solver.update_scalar_boundaries(
            self.p, self.mesh_data, self.boundaries, "p", face_flux=self.phi
        )

    def set_initial_state(self, velocity: np.ndarray, pressure: np.ndarray) -> None:
        """Set a complete cell-centred initial state before the first step.

        This is intentionally narrower than checkpoint loading: it supports
        deterministic manufactured/replay starts while retaining the solver's
        own boundary reconstruction, flux construction, and time-history
        ownership.
        """
        self.set_initial_velocity(velocity)
        n_elements = self.mesh_data["n_elements"]
        pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
        if pressure.shape != (n_elements,) or not np.all(np.isfinite(pressure)):
            raise ValueError(
                "Initial pressure must be finite with one value per interior cell; "
                f"got {pressure.shape}, expected ({n_elements},)"
            )
        self.p[:n_elements] = pressure
        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(
            self.p, self.mesh_data, self.boundaries, "p", face_flux=self.phi
        )
        self.state = FieldState(self.U, self.p, self.phi)

    def _initialize_turbulence(self):
        """Initialise the turbulence / LES model if configured.

        Uses :func:`..turbulence.create_model` to instantiate the model
        specified by ``self.config.turbulence``.  Stores the result in
        ``self.turbulence`` and logs the model info.  Sets
        ``self.time`` and ``self.step`` to their initial values.
        """
        self.turbulence = None
        self.nut = None
        if self.config.turbulence and self.config.turbulence.model.lower() != "none":
            from ..turbulence import create_model

            self.turbulence = create_model(self.config.turbulence, self.mesh_data, self.geo_data)
            if self.turbulence is None:
                raise RuntimeError(
                    f"Turbulence model {self.config.turbulence.model!r} returned no model"
                )

        # Sync state
        self.time = self.config.time.start_time
        self.step = 0
        self.time_step_size = self.config.time.time_step_size

    def compute_effective_viscosity(self):
        """Compute the effective viscosity (molecular + turbulent).

        If a turbulence model is active, computes the subgrid eddy viscosity
        and returns ``nu + nut``. Model failures propagate because silently
        switching a configured simulation to laminar flow is unsafe.

        Returns:
            Effective kinematic viscosity (scalar or per-element array).
        """
        if self.turbulence is not None:
            self.nut = self.turbulence.compute_nut(self.U, self.mesh_data, self.geo_data)
            self.parallel.exchange_halo(self.nut[: self.mesh_data["n_elements"]])
            if not np.all(np.isfinite(self.nut)) or np.any(self.nut < 0.0):
                raise FloatingPointError("Turbulence model returned invalid eddy viscosity")
            return self.config.transport.nu + self.nut
        return self.config.transport.nu

    def set_immersed_bodies(self, bodies, h: float | None = None) -> "object":
        """Attach immersed bodies (discrete direct-forcing IBM) to the solver.

        Builds the interpolation/spreading operators (Pinelli et al. 2010,
        Constant et al. — see docs/literature/Constant2016.pdf) on the
        live mesh and hooks them into the PIMPLE momentum predictor.  Body
        forces are appended to ``samples/ibm_forces_history.csv`` every step.

        Args:
            bodies: One :class:`ImmersedBody` or a list of them.
            h:      Eulerian grid spacing near the bodies; inferred from the
                    mesh when ``None``.

        Returns:
            The constructed :class:`IBMForcing` (for direct inspection).
        """
        from ..immersed_boundary import IBMForcing

        if not hasattr(self.algorithm, "ibm"):
            raise ValueError(
                "Immersed boundaries require the PIMPLE/PISO algorithm "
                f"(configured: {self.config.pimple.algorithm!r})."
            )
        body_list = [bodies] if hasattr(bodies, "U_target") else list(bodies)
        if not body_list:
            raise ValueError("At least one immersed body is required")
        moving = [body.name for body in body_list if np.any(body.U_target != 0.0)]
        if moving:
            raise NotImplementedError(
                "Moving immersed bodies require body-motion/ALE energy accounting, which is "
                f"not implemented; nonzero target velocity configured for {moving}"
            )
        self.ibm = IBMForcing(self.mesh_data, self.geo_data, body_list, h=h)
        self.algorithm.ibm = self.ibm
        from ..sampling.forces import IBMForceSampler

        if not any(isinstance(s, IBMForceSampler) for s in (self.config.samplers or ())):
            self.config.samplers = (*self.config.samplers, IBMForceSampler())
        diag = self.ibm.diagnostics()
        self.logger.info(
            f"Immersed boundary: {diag['n_markers']} markers, h={diag['h']:.4g}, "
            f"alpha={ {k: round(v, 3) for k, v in diag['alpha'].items()} }, "
            f"kernel sums [{diag['kernel_row_sum_min']:.3f}, "
            f"{diag['kernel_row_sum_max']:.3f}], "
            f"quadrature residual {diag['quadrature_residual']:.2e}"
        )
        return self.ibm

    def _blending_source(self):
        """Build the (Su, Sp) volumetric momentum source S = λ(Utarget − Ḡ) from
        registered coupling fields, or (None, None) if not set.

        ``lambdaRelax`` (volScalarField) and ``Utarget`` (volVectorField) are
        pushed by the coupler (source/coupler/core/helpers/fvm_blending_zone.py).
        Sp = λ goes on the momentum diagonal (fvm::Sp) and Su on the RHS, so the
        cell velocity relaxes toward the VPM target in the blending zone while the
        FVM core (λ = 0) is untouched.

        SCALE-SELECTIVE.  ``Utarget`` is the Biot–Savart velocity of Gaussian
        blobs of core σ on a lattice of spacing h ≈ σ, so it carries no
        information below ~2σ.  Relaxing the full velocity toward it therefore
        destroys FVM structure the target could never have represented — measured
        on the coupled cubeFlow case, the blending zone erased 95% of any FVM–VPM
        disagreement per transit and the vorticity reaching the coupling face fell
        to 0.51 of its value in the FVM core.  Relaxing only the resolved part,

            S = λ(Utarget − G∗U) = λ(Utarget + (U − G∗U)) − λ·U

        leaves the sub-filter fluctuation (U − G∗U) untouched while still pulling
        the resolved field onto the VPM BC.  Sp is unchanged, so the implicit
        diagonal and its dominance are unchanged; the added explicit term is the
        high-pass part of the current iterate, which is small next to U — a
        deferred correction, not a new stiff term.

        Set ``blending_scale_selective = False`` on the solver to recover the plain
        S = λ(Utarget − U) (the A/B control).
        """
        lam = self.registered_fields.get("lambdaRelax")
        ut = self.registered_fields.get("Utarget")
        if lam is None and ut is None:
            return None, None
        if lam is None or ut is None:
            raise RuntimeError(
                "Incomplete blending source: lambdaRelax and Utarget must be registered together"
            )
        n = self.mesh_data["n_elements"]
        lam = np.asarray(lam, dtype=np.float64)[:n]
        ut = np.asarray(ut, dtype=np.float64)[:n]

        if not getattr(self, "blending_scale_selective", True):
            return lam[:, np.newaxis] * ut, lam

        # One pass only: its width follows the LOCAL cell size, which near a
        # coupling face is already >= 2σ on a graded mesh, and it needs no halo
        # exchange because owned rows read only their own one-ring (ghost U is
        # refreshed by the previous solve).  A second pass would widen the filter
        # but requires self.parallel.exchange_halo() on the intermediate.
        # centre_weight="neighbour_sum": the residual (U − G∗U) is what this term
        # preserves, and the plain box filter has a NEGATIVE grid-scale response
        # (−5/7 on a hex interior), which would make that residual larger than U
        # and flip the source's sign — anti-damping the very modes it is meant to
        # leave untouched.  The neighbour-sum centre weight gives gain 1 at DC and
        # exactly 0 at the grid scale, so the retained fraction stays in [0, 1].
        if getattr(self, "_blending_filter", None) is None:
            from ..fields.filters import CellBoxFilter

            self._blending_filter = CellBoxFilter(
                self.mesh_data, self.geo_data, centre_weight="neighbour_sum"
            )
        u = np.asarray(self.U, dtype=np.float64)[:n]
        u_high = u - self._blending_filter(u)
        return lam[:, np.newaxis] * (ut + u_high), lam

    def solve_pimple(self, time_step_size: float | None = None):
        """Solve the pressure–velocity system at the current time level WITHOUT
        advancing the clock (coupler-facing method).

        Re-callable within a step: the coupler's VPM-BC↔pressure Picard loop
        calls this repeatedly with the boundary condition re-imposed between
        solves, then a single :meth:`advance_time`.  The committed previous level
        ``U_old`` is the transient reference on every call.
        """
        from ..fields import diagnostics

        step_time_step_size = time_step_size if time_step_size is not None else self.time_step_size
        self._current_time_step_size = step_time_step_size

        # Diagnostics from the previously completed step cache full-mesh
        # Courant, velocity-gradient, and vorticity arrays.  None is valid once
        # a new solve begins, and retaining them through momentum/pressure
        # assembly adds roughly thirteen float64 values per local cell to the
        # transient peak.  Release them before turbulence and PIMPLE allocate
        # their workspaces; the solved state invalidation below remains the
        # guard for fields requested during a re-entrant/coupled solve.
        self._invalidate_derived_fields()

        logging.Timer.start("Effective viscosity")
        nu_eff = self.compute_effective_viscosity()
        logging.Timer.log(
            "Effective viscosity",
            sink=self.logger,
        )
        # BDF2 needs u^{n-1}; available only once at least one step is committed.
        u_old_old_arg = self.U_old_old if self._n_committed >= 1 else None
        logging.Timer.start("Blending source")
        src_exp, src_imp = self._blending_source()
        logging.Timer.log("Blending source", sink=self.logger)

        self.U, self.p, self.phi, residuals = self.algorithm.step(
            self.U,
            self.p,
            self.phi,
            self.U_old,
            step_time_step_size,
            rho=self.config.transport.density,
            nu=nu_eff,
            U_old_old=u_old_old_arg,
            source_explicit=src_exp,
            source_implicit=src_imp,
            phi_old=self.phi_old,
            phi_old_old=self.phi_old_old if self._n_committed >= 1 else None,
        )
        self._invalidate_derived_fields()
        self.state = FieldState(self.U, self.p, self.phi)
        self._last_residuals = residuals
        self.logger.convergence_info(residuals)

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this makes loss of mass
        # conservation visible instead of silent.
        logging.Timer.start("Continuity diagnostics")
        cont = diagnostics.compute_continuity_error(self.phi, self.mesh_data, self.geo_data)
        vol = self.geo_data["element_volumes"]
        n_owned = self.parallel.n_owned if self.parallel.is_partitioned else len(vol)
        local_max = float(np.max(np.abs(cont[:n_owned]) / (vol[:n_owned] + 1e-30)))
        local_sum = float(np.sum(np.abs(cont[:n_owned])))
        self.continuity_max = float(self.parallel.global_max(local_max))
        self.continuity_sum = float(self.parallel.global_sum(local_sum))
        self.logger.continuity_info(self.continuity_max, self.continuity_sum)
        logging.Timer.log("Continuity diagnostics", sink=self.logger)

        logging.Timer.start("Acceptance checks")
        self.last_diagnostics = self._build_step_diagnostics(step_time_step_size, residuals)
        self._enforce_acceptance_policy(self.last_diagnostics)
        logging.Timer.log("Acceptance checks", sink=self.logger)
        return residuals

    def _build_step_diagnostics(self, step_time_step_size, residuals):
        """Build the backend-neutral health record for the current solved state."""
        from ..fields import diagnostics
        from ..solve.contracts import StepDiagnostics

        n = self.mesh_data["n_elements"]
        n_owned = self.parallel.n_owned if self.parallel.is_partitioned else n
        interior_u = np.asarray(self.U[:n_owned])
        interior_p = np.asarray(self.p[:n_owned])
        cfl = self._courant_field(step_time_step_size)
        self.cfl_max = float(self.parallel.global_max(float(np.max(cfl[:n_owned]))))
        local_nonfinite = int(
            np.count_nonzero(~np.isfinite(interior_u))
            + np.count_nonzero(~np.isfinite(interior_p))
            + np.count_nonzero(~np.isfinite(self.phi))
        )
        nonfinite_count = int(self.parallel.global_sum(local_nonfinite))
        turbulence_min = None
        turbulence_max = None
        if self.nut is not None:
            nonfinite_count += int(
                self.parallel.global_sum(int(np.count_nonzero(~np.isfinite(self.nut[:n_owned]))))
            )
            turbulence_min = float(self.parallel.global_min(float(np.nanmin(self.nut[:n_owned]))))
            turbulence_max = float(self.parallel.global_max(float(np.nanmax(self.nut[:n_owned]))))
        n_interior = self.mesh_data["n_interior_faces"]
        linear_results = tuple(getattr(self.algorithm, "last_linear_results", ()))
        velocity_min = np.asarray(
            [self.parallel.global_min(float(value)) for value in np.nanmin(interior_u, axis=0)]
        )
        velocity_max = np.asarray(
            [self.parallel.global_max(float(value)) for value in np.nanmax(interior_u, axis=0)]
        )
        pressure_min = float(self.parallel.global_min(float(np.nanmin(interior_p))))
        pressure_max = float(self.parallel.global_max(float(np.nanmax(interior_p))))
        local_ke = (
            0.5
            * self.config.transport.density
            * float(
                np.sum(
                    self.geo_data["element_volumes"][:n_owned]
                    * np.sum(interior_u * interior_u, axis=1)
                )
            )
        )
        local_enstrophy = diagnostics.enstrophy_from_gradient(
            self._velocity_gradient(),
            self.geo_data["element_volumes"],
            n_owned,
        )
        return StepDiagnostics(
            algorithm=self.config.pimple.algorithm.upper(),
            step=self.step + 1,
            time=self.time + step_time_step_size,
            time_step_size=float(step_time_step_size),
            residuals={key: float(value) for key, value in residuals.items()},
            outer_correctors=tuple(getattr(self.algorithm, "last_outer_diagnostics", ())),
            linear_solves=linear_results,
            continuity_max=self.continuity_max,
            continuity_sum=self.continuity_sum,
            boundary_mass_balance=float(
                self.parallel.global_sum(float(np.sum(self.phi[n_interior:])))
            ),
            cfl_max=self.cfl_max,
            velocity_min=tuple(float(value) for value in velocity_min),
            velocity_max=tuple(float(value) for value in velocity_max),
            pressure_min=pressure_min,
            pressure_max=pressure_max,
            nonfinite_count=nonfinite_count,
            kinetic_energy=float(self.parallel.global_sum(local_ke)),
            enstrophy=float(self.parallel.global_sum(local_enstrophy)),
            turbulence_min=turbulence_min,
            turbulence_max=turbulence_max,
        )

    def _enforce_acceptance_policy(self, diagnostics) -> None:
        """Reject unhealthy solves using explicit immediate and sustained rules."""
        from dataclasses import replace

        if diagnostics.nonfinite_count:
            raise FloatingPointError(
                f"FVM step contains {diagnostics.nonfinite_count} non-finite field values"
            )
        failed = [result for result in diagnostics.linear_solves if not result.converged]
        if failed:
            raise RuntimeError(f"FVM step contains {len(failed)} failed linear solve(s)")
        if diagnostics.turbulence_min is not None and diagnostics.turbulence_min < 0.0:
            raise FloatingPointError("FVM step contains negative turbulent viscosity")

        policy = self.config.acceptance
        velocity_max = max(
            float(np.linalg.norm(diagnostics.velocity_min)),
            float(np.linalg.norm(diagnostics.velocity_max)),
        )
        metrics = {
            "continuity": diagnostics.continuity_max,
            "residual": max(
                diagnostics.residuals.get("U", 0.0),
                diagnostics.residuals.get("p", 0.0),
            ),
            "cfl": diagnostics.cfl_max,
            "velocity": velocity_max,
        }
        warnings = []
        for name, value in metrics.items():
            warning = getattr(policy, f"{name}_warning")
            abort = getattr(policy, f"{name}_abort")
            if warning is not None and value > warning:
                warnings.append(f"{name}={value:.6g} exceeds warning threshold {warning:.6g}")
            if abort is not None and value > abort:
                self._acceptance_counts[name] += 1
            else:
                self._acceptance_counts[name] = 0
            if self._acceptance_counts[name] >= policy.sustained_steps:
                raise RuntimeError(
                    f"FVM acceptance policy rejected the step: {name}={value:.6g} "
                    f"exceeded {abort:.6g} for {self._acceptance_counts[name]} "
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

        # --- CFL-based adaptive dt adjustment (before step) ---
        cfg_time = self.config.time
        if (
            time_step_size is None
            and cfg_time.adjust_timestep
            and self.cfl_max > 0
            and self.step > 1
        ):
            ratio = cfg_time.max_cfl / max(self.cfl_max, 1e-8)
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

        # Compute CFL after step (for next step's dt adjustment)
        if cfg_time.adjust_timestep:
            Co_field = diagnostics.compute_courant_number(
                self.U, self.phi, step_time_step_size, self.mesh_data, self.geo_data
            )
            n_owned = (
                self.parallel.n_owned
                if self.parallel.is_partitioned
                else self.mesh_data["n_elements"]
            )
            self.cfl_max = float(self.parallel.global_max(float(np.max(Co_field[:n_owned]))))
        self.logger.courant_info(
            self.cfl_max,
            cfg_time.max_cfl if cfg_time.adjust_timestep else None,
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
        self.U_old_old[:] = self.U_old[:]
        self.U_old[:] = self.U[:]
        self.phi_old_old[:] = self.phi_old[:]
        self.phi_old[:] = self.phi[:]
        self._n_committed += 1
        self.step += 1
        self.time += self._current_time_step_size
        step_time_step_size = self._current_time_step_size
        cfg_time = self.config.time
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
        if self.turbulence and self.nut is not None:
            n_owned = (
                self.parallel.n_owned
                if self.parallel.is_partitioned
                else self.mesh_data["n_elements"]
            )
            owned_nut = self.nut[:n_owned]
            nut_minimum = float(self.parallel.global_min(float(np.min(owned_nut))))
            nut_maximum = float(self.parallel.global_max(float(np.max(owned_nut))))
            nut_sum = float(self.parallel.global_sum(float(np.sum(owned_nut))))
            nut_count = int(self.parallel.global_sum(int(n_owned)))
            if self.parallel.is_root:
                self.logger.turbulence_info(
                    self.nut,
                    self.config.transport.nu,
                    statistics=(nut_minimum, nut_maximum, nut_sum / nut_count),
                )
        logging.Timer.log("Turbulence statistics", sink=self.logger)

        # Output control — time-based if write_interval_time is set, else step-based
        logging.Timer.start("Visualization output")
        if (self.parallel.is_root or self.parallel.is_partitioned) and self.auto_write:
            wrt_time = cfg_time.write_interval_time
            if wrt_time is not None:
                self._time_since_last_write += step_time_step_size
                if self._time_since_last_write >= wrt_time:
                    self.write_vtk()
                    self._time_since_last_write = 0.0
            else:
                if self.step % cfg_time.write_interval == 0:
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
            from ..io.partitioned import save_partitioned_solver_checkpoint

            return str(save_partitioned_solver_checkpoint(self, path))
        from ..io.checkpoint import save_checkpoint

        saved = None
        if self.parallel.is_root:
            saved = save_checkpoint(self, path)
        self.parallel.barrier()
        return str(saved if saved is not None else path)

    def write_run_manifest(self, path=None) -> str:
        """Write source, dependency, backend, mesh, and configuration identity."""
        from ..io.manifest import write_manifest

        destination = path or os.path.join(self.case_dir, "solution", "run_manifest.json")
        written = None
        if self.parallel.is_root:
            written = write_manifest(self, destination)
        self.parallel.barrier()
        return str(written if written is not None else destination)

    def load_state(self, path, *, allow_config_change: bool = False) -> None:
        """Restore a compatible restart, rejecting mismatched meshes or configs."""
        self.flush_output()
        if self.parallel.is_partitioned:
            from ..io.partitioned import load_partitioned_solver_checkpoint

            load_partitioned_solver_checkpoint(self, path, allow_config_change=allow_config_change)
            self.io.rewind_histories(self.time)
            self.parallel.barrier()
            return
        from ..io.checkpoint import load_checkpoint

        self.parallel.barrier()
        load_checkpoint(self, path, allow_config_change=allow_config_change)
        self.io.rewind_histories(self.time)
        self.parallel.barrier()

    def write_vtk(self, filename: str | None = None) -> None:
        """Export the current simulation state to a ``.vtu`` file with PVD time-series support.

        Writes velocity (U), pressure (p), Courant number (Co), and
        vorticity fields.  If turbulence is active, also writes ``nut``.
        Updates the PVD collection file for time-series visualisation.

        Args:
            filename: Optional output path.  If ``None``, auto-generates
                      ``solution/{case_name}_{step:06d}.vtu``.
        """
        if not self.parallel.is_root and not self.parallel.is_partitioned:
            return
        sol_dir = os.path.join(self.case_dir, "solution")
        if filename is None:
            os.makedirs(sol_dir, exist_ok=True)
            # Use case_name and sequential numbering: case_name_000000.vtu
            filename = os.path.join(sol_dir, f"{self.config.case_name}_{self.step:06d}.vtu")

        fields = {
            "U": self.U,
            "p": self.p,
            "Co": self._courant_field(self._current_time_step_size),
            "vorticity": self._vorticity_field(),
        }
        if self.nut is not None:
            fields["nut"] = self.nut

        if self.parallel.is_partitioned:
            from pathlib import Path

            from ..io.partitioned import write_partition_vtu
            from ..io.vtk_exporter import VTKExporter

            n_local = self.mesh_data["n_elements"]
            stem = Path(filename).stem
            if self.vtk_exporter is None:
                export_mesh = self.mesh_data.get("_visualization_mesh", self.mesh_data)
                self.vtk_exporter = VTKExporter(export_mesh, self.config.output)
            collection = write_partition_vtu(
                Path(filename).parent,
                stem,
                self.mesh_data,
                self.parallel.partition,
                {name: np.asarray(values)[:n_local] for name, values in fields.items()},
                self.parallel.comm,
                output=self.config.output,
                exporter=self.vtk_exporter,
            )
            if self.parallel.is_root:
                from ..io.vtk_exporter import PVDManager

                pvd_file = os.path.join(sol_dir, f"{self.config.case_name}.pvd")
                if self.pvd_manager is None:
                    self.pvd_manager = PVDManager(pvd_file)
                self.pvd_manager.add_step(self.time, str(collection))
                self.logger.output_info(f"Output written: {stem}.pvtu")
            return

        asynchronous = (
            self.config.output.asynchronous or self.config.execution.output_mode == "threaded"
        )
        if asynchronous:
            if self._buffered_vtk_writer is None:
                from ..io.async_output import BufferedVTKWriter

                pvd_file = os.path.join(sol_dir, f"{self.config.case_name}.pvd")
                self._buffered_vtk_writer = BufferedVTKWriter(
                    self.mesh_data,
                    pvd_file,
                    self.config.output,
                )
            self._buffered_vtk_writer.submit(filename, self.time, fields)
            action = "queued"
        else:
            if self.vtk_exporter is None:
                from ..io.vtk_exporter import VTKExporter

                self.vtk_exporter = VTKExporter(self.mesh_data, self.config.output)
            if self.pvd_manager is None:
                from ..io.vtk_exporter import PVDManager

                pvd_file = os.path.join(sol_dir, f"{self.config.case_name}.pvd")
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
