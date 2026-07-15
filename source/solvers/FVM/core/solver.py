#!/usr/bin/env python3
"""Finite Volume Method (FVM) Solver Core.

Main entry point for the FVM solver, providing a unified Python API
consistent with the VPM solver.

Author: OpenONDA Team
Date: January 2026
"""

import csv
import os
import sys
from typing import Any

import numpy as np

from ..config.types import FVMConfig
from ..coupling import OFWInterfaceMixin
from ..io import logging, solver_io
from ..mesh import geometry, mesh_io
from ..solve import pimple_solver, simple_solver
from .parallel import ParallelContext


def _load_velocity_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Load or initialise the velocity field.

    If *config.initial_U* is set, tiles it across all cells.  Otherwise
    reads the ``0/U`` OpenFOAM file.  Parsing errors are fatal so a case
    cannot silently start from a different field.

    Args:
        config:   FVMConfig (may have ``initial_U``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Velocity array ``(n_total, 3)``.
    """
    from ..fields import field_io

    if config.initial_U is not None:
        initial = np.asarray(config.initial_U, dtype=np.float64)
        if initial.shape != (3,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_U must be a finite three-component vector")
        return np.tile(initial, (n_total, 1))
    U_data = field_io.read_field("U", os.path.join(case_dir, "0"), mesh_data)
    values = np.asarray(U_data["phi"], dtype=np.float64)
    if values.shape != (n_total, 3):
        raise ValueError(f"Initial U has shape {values.shape}; expected {(n_total, 3)}")
    return values


def _load_pressure_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Load or initialise the pressure field.

    If *config.initial_p* is set, fills all cells with that value.
    Otherwise reads the ``0/p`` OpenFOAM file.  Parsing errors are fatal.

    Args:
        config:   FVMConfig (may have ``initial_p``).
        case_dir: Case root directory.
        n_total:  Total number of elements (interior + boundary ghosts).
        mesh_data: Mesh dictionary.

    Returns:
        Pressure array ``(n_total,)``.
    """
    from ..fields import field_io

    if config.initial_p is not None:
        initial = np.asarray(config.initial_p, dtype=np.float64)
        if initial.ndim != 0 or not np.isfinite(initial):
            raise ValueError("initial_p must be a finite scalar")
        return np.full(n_total, float(initial), dtype=np.float64)
    p_data = field_io.read_field("p", os.path.join(case_dir, "0"), mesh_data)
    values = np.asarray(p_data["phi"], dtype=np.float64)
    if values.shape != (n_total,):
        raise ValueError(f"Initial p has shape {values.shape}; expected {(n_total,)}")
    return values


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
    from ..solve.simple_solver import _remove_normal_component

    for boundary in boundaries:
        bc_type = boundary.get("bc_type_U")
        start = n_elements + (boundary["startFace"] - mesh_data["n_interior_faces"])
        end = start + boundary["nFaces"]
        if bc_type == "noSlip":
            U[start:end] = 0.0
        elif bc_type in ["fixedValue", "freestream"] and boundary.get("value_U_field") is not None:
            U[start:end] = boundary["value_U_field"]
        elif bc_type in ["fixedValue", "freestream"] and "value_U" in boundary:
            U[start:end] = boundary["value_U"]
        elif bc_type in ("zeroGradient", "inletOutlet", "directionMixed"):
            owners_b = mesh_data["owners"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            U[start:end] = U[owners_b]
        elif bc_type == "empty":
            owners_b = mesh_data["owners"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            face_sf = geo_data["face_sf"][
                boundary["startFace"] : boundary["startFace"] + boundary["nFaces"]
            ]
            for i in range(len(owners_b)):
                U[start + i] = _remove_normal_component(U[owners_b[i]], face_sf[i])


class Solver(OFWInterfaceMixin):
    """Finite Volume Method (FVM) simulator for incompressible flow.

    Provides a high-level Python API for managing unstructured mesh CFD simulations.
    Supports PIMPLE/SIMPLE algorithms, Smagorinsky turbulence models, and VTK/PVD export.

    Attributes:
        config (FVMConfig): Simulation configuration object.
        case_dir (str): Root directory for simulation outputs and logs.
        mesh_data (Dict[str, Any]): Mesh connectivity and naming data.
        geo_data (Dict[str, Any]): Computed geometric properties (volumes, areas, etc.).
        U (np.ndarray): Velocity field [m/s] (includes ghost boundary cells).
        p (np.ndarray): Kinematic pressure field [m^2/s^2] (includes ghost boundary cells).
        phi (np.ndarray): Mass flow rate flux on faces [m^3/s].
        flow_time (float): Current physical time in the simulation.
        time_step (int): Current time step index.
        auto_write (bool): If True, automatically writes results based on writeInterval.
    """

    def __init__(
        self,
        config: FVMConfig,
        case_dir: str | None = None,
        mesh_data: dict[str, Any] | None = None,
    ):
        """Initializes the FVM solver instance.

        Args:
            config: FVMConfig object containing all simulation and time parameters.
            case_dir: Root directory for the case. Defaults to current working directory.
            mesh_data: Optional pre-loaded mesh dictionary. If None, loaded from disk.
        """
        self.config = config
        self.case_dir = os.path.abspath(case_dir or os.getcwd())
        self.auto_write = True
        self.parallel = ParallelContext.create(self.config.execution)
        if self.config.execution.linear_backend == "petsc":
            methods = {
                "momentum": self.config.solver.momentum_solver or self.config.solver.linear_solver,
                "pressure": self.config.solver.pressure_solver or self.config.solver.linear_solver,
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
        from ..schemes import (
            validate_acceptance_policy,
            validate_solver_params,
            validate_turbulence,
        )

        validate_solver_params(self.config.solver, self.config.time)
        validate_turbulence(self.config.turbulence)
        validate_acceptance_policy(self.config.acceptance)
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
        logging.print_openonda_header()
        logging.Timer.start("Total Initialization")

        # 1. Mesh Management
        if mesh_data is not None:
            self.mesh_data = mesh_data
            logging.Timer.log("  Mesh Set (In-Memory)")
        else:
            logging.Timer.start("  Mesh Load (Disk)")
            self.mesh_data = mesh_io.load_poly_mesh(self.case_dir)
            logging.Timer.log("  Mesh Load (Disk)")

        from ..mesh.validation import enforce_quality_thresholds, validate_mesh

        validate_mesh(self.mesh_data)

        # 2. Geometry Computation
        logging.Timer.start("  Geometry Compute")
        from ..mesh import cache

        self.cache = cache.WeightCache()
        gs = getattr(self.config.solver, "gradient_scheme", "gauss")
        self.geo_data = geometry.compute_mesh_geometry(self.mesh_data, gradient_scheme=gs)
        self.mesh_quality = validate_mesh(self.mesh_data, self.geo_data)
        enforce_quality_thresholds(self.mesh_quality, self.config.mesh)

        from ..mesh.topology import MeshTopology

        self.topology = MeshTopology.from_mesh_data(self.mesh_data)
        self.geometry = geometry.MeshGeometry.from_data(self.mesh_data, self.geo_data)

        self.cache.set_static_weights(self.cache.from_mesh_geometry(self.geo_data).static_weights)
        logging.Timer.log("  Geometry Compute")

        # 3. Component Setup
        self.boundaries = self.mesh_data["boundary"]
        self._setup_boundary_conditions()
        from ..schemes import validate_boundary_conditions

        validate_boundary_conditions(self.boundaries)
        self._initialize_fields()
        self._initialize_algorithm()
        self._initialize_turbulence()

        # Final housekeeping
        self.io = solver_io.SolverIO(self)
        self.vtk_exporter = None
        self.pvd_manager = None
        self.last_forces = None
        self.ibm = None
        self.forces_history_path = None
        self._force_log_counter = 0
        self.cfl_max = 0.0
        self._time_since_last_write = 0.0
        # Coupling / driver-split state
        self.registered_fields: dict[str, np.ndarray] = {}  # named volume fields (fvOptions)
        self._n_committed = 0  # number of committed time steps (BDF2 startup gate)
        self._current_dt = self.dt
        self._last_residuals = None
        self.last_diagnostics = None
        self._acceptance_counts = {
            "continuity": 0,
            "residual": 0,
            "cfl": 0,
            "velocity": 0,
        }

        logging.Logging.solver_summary(self)
        logging.Timer.log("Total Initialization")
        print()
        sys.stdout.flush()

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(self.p, self.mesh_data, self.boundaries, "p")

    @classmethod
    def from_case(cls, case_dir: str, **overrides):
        """Build a Solver from an OpenFOAM case directory.

        Assembles an :class:`FVMConfig` from the case's ``system``/``constant``/``0``
        dictionaries (reusing the ``from_*`` loaders) so the FVM is constructable
        the same way the coupler builds the OFW backend (``backend(case_dir)``).
        Required case dictionaries and initial fields must exist and parse
        successfully. ``overrides`` may replace known top-level
        :class:`FVMConfig` fields.
        """
        from ..config.types import (
            BoundaryConfig,
            FVMConfig,
            SolverParams,
            TimeConfig,
            TransportConfig,
            TurbulenceConfig,
        )

        case_dir = os.path.abspath(case_dir)

        required = (
            os.path.join("system", "controlDict"),
            os.path.join("system", "fvSolution"),
            os.path.join("system", "fvSchemes"),
            os.path.join("constant", "transportProperties"),
            os.path.join("0", "U"),
            os.path.join("0", "p"),
        )
        missing = [
            relative
            for relative in required
            if not os.path.isfile(os.path.join(case_dir, relative))
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete FVM case {case_dir!r}; missing required files: {', '.join(missing)}"
            )

        turbulence_path = os.path.join(case_dir, "constant", "turbulenceProperties")

        cfg = FVMConfig(
            case_name=os.path.basename(case_dir.rstrip("/")) or "case",
            time=TimeConfig.from_control_dict(case_dir),
            solver=SolverParams.from_system(case_dir),
            transport=TransportConfig.from_foam_file(case_dir),
            turbulence=(
                TurbulenceConfig.from_foam_file(turbulence_path)
                if os.path.isfile(turbulence_path)
                else None
            ),
            boundaries=BoundaryConfig.load_from_time_dir(case_dir, "0"),
            initial_U=None,
            initial_p=None,
        )
        valid_overrides = set(cfg.__dataclass_fields__)
        unknown = sorted(set(overrides) - valid_overrides)
        if unknown:
            raise TypeError(f"Unknown FVMConfig override(s): {', '.join(unknown)}")
        for key, val in overrides.items():
            setattr(cfg, key, val)
        return cls(cfg, case_dir=case_dir)

    def _setup_boundary_conditions(self):
        """Map user-defined BoundaryConfig entries to internal mesh boundary data.

        Iterates over ``self.config.boundaries`` and updates the
        corresponding entries in ``self.boundaries`` with the configured
        type and value for U, p, and nut.  Patches not found in the mesh
        trigger a warning.
        """
        print("\nBoundary Conditions Setup:")
        for b_cfg in self.config.boundaries:
            found = False
            for b_mesh in self.boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.value_U, dtype=np.float64)
                    if velocity.shape not in {(3,), (b_mesh["nFaces"], 3)}:
                        raise ValueError(
                            f"Velocity value for patch {b_cfg.name!r} has shape {velocity.shape}; "
                            f"expected (3,) or {(b_mesh['nFaces'], 3)}"
                        )
                    b_mesh.update(
                        {
                            "type": b_mesh.get("type", b_cfg.type_U),
                            "bc_type_U": b_cfg.type_U,
                            "bc_type_p": b_cfg.type_p,
                            "value_p": b_cfg.value_p,
                            "bc_type_nut": b_cfg.type_nut,
                            "value_nut": b_cfg.value_nut,
                        }
                    )
                    if velocity.shape == (3,):
                        b_mesh["value_U"] = velocity
                        b_mesh.pop("value_U_field", None)
                    else:
                        b_mesh["value_U_field"] = velocity
                    print(f"  {b_cfg.name:<15} : {b_cfg.type_U:<12} U={b_cfg.value_U}")
                    found = True
                    break
            if not found:
                raise ValueError(f"Configured boundary {b_cfg.name!r} was not found in the mesh")

    def _initialize_fields(self):
        """Initialise velocity (U), pressure (p), and flux (phi) fields.

        Loads or creates the initial fields, enforces boundary constraints
        on the velocity ghost layer, and computes the initial mass-flow
        rate ``phi`` from the velocity field.
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

        # 3. Flux (phi)
        logging.Timer.start("  Flux Init")
        from ..assemble import convection

        self.phi = convection.compute_mass_flow_rate(self.U, self.mesh_data, self.geo_data)
        logging.Timer.log("  Flux Init")

    def _initialize_algorithm(self):
        """Initialise the numerical solver algorithm.

        Reads the algorithm type from ``self.config.solver.algorithm``
        and instantiates either a :class:`~solve.pimple_solver.PIMPLESolver`
        or :class:`~solve.simple_solver.SIMPLESolver`.

        Raises:
            ValueError: If the algorithm is not ``"SIMPLE"``, ``"PIMPLE"``,
                        or ``"PISO"``.
        """
        logging.Timer.start("  Algorithm Init")
        if hasattr(self.config.solver, "to_dict"):
            params = self.config.solver.to_dict()  # type: ignore[union-attr]
        else:
            params = vars(self.config.solver)
        params = dict(params)
        params["_linear_backend"] = self.config.execution.linear_backend
        params["_parallel_context"] = self.parallel
        algo = self.config.solver.algorithm.upper()

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
        logging.Timer.log("  Algorithm Init")

    def _initialize_turbulence(self):
        """Initialise the turbulence / LES model if configured.

        Uses :func:`..turbulence.create_model` to instantiate the model
        specified by ``self.config.turbulence``.  Stores the result in
        ``self.turbulence`` and logs the model info.  Sets
        ``self.flow_time`` and ``self.time_step`` to their initial values.
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
            info = self.turbulence.get_filter_info()
            print(f"Turbulence model: {info['model']} (coeff={info['Cs']:.3g})")

        # Sync state
        self.flow_time = self.config.time.start_time
        self.time_step = 0
        self.dt = self.config.time.delta_t

    def _effective_viscosity(self):
        """Compute the effective viscosity (molecular + turbulent).

        If a turbulence model is active, computes the subgrid eddy viscosity
        and returns ``nu + nut``. Model failures propagate because silently
        switching a configured simulation to laminar flow is unsafe.

        Returns:
            Effective kinematic viscosity (scalar or per-element array).
        """
        if self.turbulence is not None:
            self.nut = self.turbulence.compute_nut(self.U, self.mesh_data, self.geo_data)
            if not np.all(np.isfinite(self.nut)) or np.any(self.nut < 0.0):
                raise FloatingPointError("Turbulence model returned invalid eddy viscosity")
            return self.config.transport.nu + self.nut
        return self.config.transport.nu

    def set_immersed_bodies(self, bodies, h: float | None = None) -> "object":
        """Attach immersed bodies (discrete direct-forcing IBM) to the solver.

        Builds the interpolation/spreading operators (Pinelli et al. 2010,
        Constant et al. — see docs/literature/Constant2016.pdf) on the
        live mesh and hooks them into the PIMPLE momentum predictor.  Body
        forces are appended to ``solution/ibm_forces_history.csv`` every step.

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
                f"(configured: {self.config.solver.algorithm!r})."
            )
        self.ibm = IBMForcing(self.mesh_data, self.geo_data, bodies, h=h)
        self.algorithm.ibm = self.ibm
        diag = self.ibm.diagnostics()
        print(
            f"Immersed boundary: {diag['n_markers']} markers, h={diag['h']:.4g}, "
            f"alpha={ {k: round(v, 3) for k, v in diag['alpha'].items()} }, "
            f"kernel sums [{diag['kernel_row_sum_min']:.3f}, "
            f"{diag['kernel_row_sum_max']:.3f}], "
            f"quadrature residual {diag['quadrature_residual']:.2e}"
        )
        return self.ibm

    def _log_ibm_forces(self, step_dt: float) -> None:
        """Append per-body IBM forces (and Cd/Cl) to solution/ibm_forces_history.csv."""
        rho = self.config.transport.density
        forces = self.ibm.body_forces(rho=rho)
        ref_U = getattr(self.config.solver, "ref_velocity", 1.0)
        ref_area = getattr(self.config.solver, "ref_area", 1.0)
        q = 0.5 * rho * ref_U**2 * ref_area

        sol_dir = os.path.join(self.case_dir, "solution")
        os.makedirs(sol_dir, exist_ok=True)
        # No-slip quality monitor: marker slip of the *final* velocity field
        # (the predictor slip stored by compute_force overestimates it).
        slip = self.ibm.slip_error(self.U)

        csv_path = os.path.join(sol_dir, "ibm_forces_history.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(
                    ["time", "step", "dt", "body", "Fx", "Fy", "Fz", "Cd", "Cl", "slip"]
                )
            for name, F in forces.items():
                writer.writerow(
                    [
                        self.flow_time,
                        self.time_step,
                        step_dt,
                        name,
                        F[0],
                        F[1],
                        F[2],
                        F[0] / q,
                        F[1] / q,
                        slip,
                    ]
                )
        msg = "  IBM forces:"
        for name, F in forces.items():
            msg += f" {name}: Cd={F[0] / q:.4f} Cl={F[1] / q:.4f}"
        msg += f" | slip={slip:.2e}"
        print(msg)

    def _fringe_source(self):
        """Build the (Su, Sp) volumetric momentum source S = λ(Utarget − U) from
        registered coupling fields, or (None, None) if not set.

        ``lambdaRelax`` (volScalarField) and ``Utarget`` (volVectorField) are
        pushed by the coupler (source/coupler/core/helpers/fvm_fringe.py).
        Sp = λ goes on the momentum diagonal (fvm::Sp), Su = λ·Utarget on the
        RHS, so the cell velocity relaxes toward the VPM target in the fringe
        band while the FVM core (λ = 0) is untouched.
        """
        lam = self.registered_fields.get("lambdaRelax")
        ut = self.registered_fields.get("Utarget")
        if lam is None and ut is None:
            return None, None
        if lam is None or ut is None:
            raise RuntimeError(
                "Incomplete fringe source: lambdaRelax and Utarget must be registered together"
            )
        n = self.mesh_data["n_elements"]
        lam = np.asarray(lam, dtype=np.float64)[:n]
        ut = np.asarray(ut, dtype=np.float64)[:n]
        return lam[:, np.newaxis] * ut, lam

    def solve_pimple(self, dt: float | None = None):
        """Solve the pressure–velocity system at the current time level WITHOUT
        advancing the clock (OFW-contract method).

        Re-callable within a step: the coupler's donor-BC↔pressure Picard loop
        calls this repeatedly with the boundary condition re-imposed between
        solves, then a single :meth:`advance_time`.  The committed previous level
        ``U_old`` is the transient reference on every call.
        """
        from ..fields import diagnostics

        step_dt = dt if dt is not None else self.dt
        self._current_dt = step_dt

        nu_eff = self._effective_viscosity()
        # BDF2 needs u^{n-1}; available only once at least one step is committed.
        u_old_old_arg = self.U_old_old if self._n_committed >= 1 else None
        src_exp, src_imp = self._fringe_source()

        self.U, self.p, self.phi, residuals = self.algorithm.step(
            self.U,
            self.p,
            self.phi,
            self.U_old,
            step_dt,
            rho=self.config.transport.density,
            nu=nu_eff,
            U_old_old=u_old_old_arg,
            source_explicit=src_exp,
            source_implicit=src_imp,
        )
        self._last_residuals = residuals
        if self.parallel.is_root:
            logging.Logging.convergence_info(residuals)

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this makes loss of mass
        # conservation visible instead of silent.
        cont = diagnostics.compute_continuity_error(self.phi, self.mesh_data, self.geo_data)
        vol = self.geo_data["element_volumes"]
        self.continuity_max = float(np.max(np.abs(cont) / (vol + 1e-30)))
        self.continuity_sum = float(np.sum(np.abs(cont)))
        if self.parallel.is_root:
            print(
                f"  continuity: max|div U| = {self.continuity_max:.3e} 1/s, "
                f"sum|imbalance| = {self.continuity_sum:.3e} m3/s"
            )

        self.last_diagnostics = self._build_step_diagnostics(step_dt, residuals)
        self._enforce_acceptance_policy(self.last_diagnostics)
        return residuals

    def _build_step_diagnostics(self, step_dt, residuals):
        """Build the backend-neutral health record for the current solved state."""
        from ..fields import diagnostics
        from ..solve.contracts import StepDiagnostics

        n = self.mesh_data["n_elements"]
        interior_u = np.asarray(self.U[:n])
        interior_p = np.asarray(self.p[:n])
        cfl = diagnostics.compute_courant_number(
            self.U, self.phi, step_dt, self.mesh_data, self.geo_data
        )
        self.cfl_max = float(np.max(cfl))
        nonfinite_count = int(
            np.count_nonzero(~np.isfinite(interior_u))
            + np.count_nonzero(~np.isfinite(interior_p))
            + np.count_nonzero(~np.isfinite(self.phi))
        )
        turbulence_min = None
        turbulence_max = None
        if self.nut is not None:
            nonfinite_count += int(np.count_nonzero(~np.isfinite(self.nut)))
            turbulence_min = float(np.nanmin(self.nut))
            turbulence_max = float(np.nanmax(self.nut))
        n_interior = self.mesh_data["n_interior_faces"]
        linear_results = tuple(getattr(self.algorithm, "last_linear_results", ()))
        return StepDiagnostics(
            algorithm=self.config.solver.algorithm.upper(),
            step=self.time_step + 1,
            time=self.flow_time + step_dt,
            dt=float(step_dt),
            residuals={key: float(value) for key, value in residuals.items()},
            outer_correctors=tuple(getattr(self.algorithm, "last_outer_diagnostics", ())),
            linear_solves=linear_results,
            continuity_max=self.continuity_max,
            continuity_sum=self.continuity_sum,
            boundary_mass_balance=float(np.sum(self.phi[n_interior:])),
            cfl_max=self.cfl_max,
            velocity_min=tuple(float(value) for value in np.nanmin(interior_u, axis=0)),
            velocity_max=tuple(float(value) for value in np.nanmax(interior_u, axis=0)),
            pressure_min=float(np.nanmin(interior_p)),
            pressure_max=float(np.nanmax(interior_p)),
            nonfinite_count=nonfinite_count,
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

    def evolve(self, dt: float | None = None) -> None:
        """Advance the simulation by one full time step (= solve_pimple + advance_time).

        Args:
            dt: Optional override for the time step size [s].
        """
        from ..fields import diagnostics

        # --- CFL-based adaptive dt adjustment (before step) ---
        cfg_time = self.config.time
        if dt is None and cfg_time.adjust_timestep and self.cfl_max > 0 and self.time_step > 1:
            ratio = cfg_time.max_cfl / max(self.cfl_max, 1e-8)
            ratio = min(ratio, cfg_time.dt_adjust_coeff)
            self.dt = np.clip(
                self.dt * ratio,
                cfg_time.min_delta_t,
                cfg_time.max_delta_t,
            )

        step_dt = dt if dt is not None else self.dt
        logging.Timer.start(f"  Step {self.time_step + 1}")
        logging.Logging.step_info(self.flow_time + step_dt, self.time_step + 1, step_dt)

        self.solve_pimple(step_dt)

        # Compute CFL after step (for next step's dt adjustment)
        if cfg_time.adjust_timestep:
            Co_field = diagnostics.compute_courant_number(
                self.U, self.phi, step_dt, self.mesh_data, self.geo_data
            )
            self.cfl_max = float(np.max(Co_field))
            print(
                f"  max Co = {self.cfl_max:.3f}  dt = {step_dt:.6f} s  (target Co <= {cfg_time.max_cfl})"
            )
            sys.stdout.flush()

        self.advance_time()

        elapsed = logging.Timer.log(f"  Step {self.time_step}")
        print(f"\nStep completed in {elapsed:.3f} s")
        sys.stdout.flush()

    def advance_time(self) -> None:
        """Commit the solved field as the new time level and advance the clock
        (OFW-contract method): roll the BDF history, increment step/time, then
        run per-step force logging and output control."""
        from ..fields import diagnostics

        # Roll the BDF time-history ring: U_old_old <- u^n, U_old <- u^{n+1}.
        self.U_old_old[:] = self.U_old[:]
        self.U_old[:] = self.U[:]
        self._n_committed += 1
        self.time_step += 1
        self.flow_time += self._current_dt
        step_dt = self._current_dt
        cfg_time = self.config.time
        self.io.write_step_diagnostics()

        # y+ and Turbulence info
        patch_names = getattr(self.config.solver, "yplus_patches", None)
        if self.parallel.is_root:
            yplus_stats = diagnostics.compute_y_plus(
                self.U,
                self.config.transport.nu,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
                patch_names=patch_names,
            )
            logging.Logging.yplus_info(yplus_stats)

        # Force computation and logging
        force_interval = getattr(self.config.solver, "force_log_interval", None)
        if force_interval is None:
            force_interval = cfg_time.write_interval

        self._force_log_counter += 1
        if self.parallel.is_root and (
            self._force_log_counter % force_interval == 0 or self._force_log_counter == 1
        ):
            ref_U = getattr(self.config.solver, "ref_velocity", 1.0)
            ref_area = getattr(self.config.solver, "ref_area", 1.0)
            ref_length = getattr(self.config.solver, "ref_length", 1.0)
            mu = self.config.transport.nu * self.config.transport.density
            rho = self.config.transport.density

            patches = getattr(self.config.solver, "force_patches", None)

            forces = diagnostics.compute_surface_forces(
                self.U,
                self.p,
                mu,
                rho,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
                patch_names=patches,
                ref_U=ref_U,
                ref_area=ref_area,
                ref_length=ref_length,
                moment_centre=getattr(self.config.solver, "moment_centre", [0, 0, 0]),
            )
            self.last_forces = forces

            sol_dir = os.path.join(self.case_dir, "solution")
            os.makedirs(sol_dir, exist_ok=True)
            csv_path = os.path.join(sol_dir, "forces_history.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(
                        [
                            "time",
                            "step",
                            "dt",
                            "patch",
                            "Fpx",
                            "Fpy",
                            "Fpz",
                            "Fvx",
                            "Fvy",
                            "Fvz",
                            "Ftx",
                            "Fty",
                            "Ftz",
                            "Cd",
                            "Cl",
                            "Cz",
                            "Cm",
                        ]
                    )
                for pname, fdata in forces.items():
                    Fp = fdata.get("Fp", [0, 0, 0])
                    Fv = fdata.get("Fv", [0, 0, 0])
                    Ft = fdata.get("Ftot", [0, 0, 0])
                    C = fdata.get("coeffs", {})
                    writer.writerow(
                        [
                            self.flow_time,
                            self.time_step,
                            step_dt,
                            pname,
                            Fp[0],
                            Fp[1],
                            Fp[2],
                            Fv[0],
                            Fv[1],
                            Fv[2],
                            Ft[0],
                            Ft[1],
                            Ft[2],
                            C.get("Cd"),
                            C.get("Cl"),
                            C.get("Cz"),
                            C.get("Cm"),
                        ]
                    )
            log_msg = f"  Forces logged: {len(forces)} patch(es)"
            for pname, fdata in forces.items():
                C = fdata.get("coeffs", {})
                log_msg += f" | {pname}: Cd={C.get('Cd', 0):.4f} Cl={C.get('Cl', 0):.4f}"
            print(log_msg)
            sys.stdout.flush()

        if self.parallel.is_root and self.ibm is not None:
            self._log_ibm_forces(step_dt)

        if self.parallel.is_root and self.turbulence and self.nut is not None:
            logging.Logging.turbulence_info(self.nut, self.config.transport.nu)

        # Output control — time-based if write_interval_time is set, else step-based
        if self.parallel.is_root and self.auto_write:
            wrt_time = cfg_time.write_interval_time
            if wrt_time is not None:
                self._time_since_last_write += step_dt
                if self._time_since_last_write >= wrt_time:
                    self.write_vtk()
                    self._time_since_last_write = 0.0
            else:
                if self.time_step % cfg_time.write_interval == 0:
                    self.write_vtk()

    def save_state(self, path) -> str:
        """Atomically save a versioned restart containing the complete time state."""
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

    def load_state(self, path) -> None:
        """Restore a compatible restart, rejecting mismatched meshes or configs."""
        from ..io.checkpoint import load_checkpoint

        self.parallel.barrier()
        load_checkpoint(self, path)
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
        if not self.parallel.is_root:
            return
        sol_dir = os.path.join(self.case_dir, "solution")
        if self.vtk_exporter is None:
            from ..io.vtk_exporter import VTKExporter

            self.vtk_exporter = VTKExporter(self.mesh_data)

        if self.pvd_manager is None:
            from ..io.vtk_exporter import PVDManager

            pvd_file = os.path.join(sol_dir, f"{self.config.case_name}.pvd")
            self.pvd_manager = PVDManager(pvd_file)

        if filename is None:
            os.makedirs(sol_dir, exist_ok=True)
            # Use case_name and sequential numbering: case_name_000000.vtu
            filename = os.path.join(sol_dir, f"{self.config.case_name}_{self.time_step:06d}.vtu")

        from ..fields import diagnostics

        fields = {
            "U": self.U,
            "p": self.p,
            "Co": diagnostics.compute_courant_number(
                self.U, self.phi, self.dt, self.mesh_data, self.geo_data
            ),
            "vorticity": diagnostics.compute_vorticity(self.U, self.mesh_data, self.geo_data),
        }
        if self.nut is not None:
            fields["nut"] = self.nut

        self.vtk_exporter.export(filename, fields)
        self.pvd_manager.add_step(self.flow_time, filename)

        print(f"  Output written: {os.path.basename(filename)}")
        sys.stdout.flush()

    def info(self) -> None:
        """Print a summary of the current solver state.

        Displays case name, flow time, time step, cell count, and
        active algorithm.
        """
        print("-" * 40)
        print("FVM Solver Information")
        print(f"  Case      : {self.config.case_name}")
        print(f"  Time      : {self.flow_time:.5f}")
        print(f"  Step      : {self.time_step}")
        print(f"  Cells     : {self.mesh_data['n_elements']}")
        print(f"  Algorithm : {self.config.solver.algorithm}")
        print("-" * 40)
