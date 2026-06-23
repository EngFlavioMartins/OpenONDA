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
from ..io import logging, solver_io
from ..mesh import geometry, mesh_io
from ..solve import pimple_solver, simple_solver


def _load_velocity_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Load or initialise the velocity field."""
    from ..fields import field_io

    if config.initial_U is not None:
        return np.tile(np.array(config.initial_U), (n_total, 1)).astype(np.float64)
    try:
        U_data = field_io.read_field("U", os.path.join(case_dir, "0"), mesh_data)
        return U_data["phi"].astype(np.float64)
    except Exception:
        return np.zeros((n_total, 3), dtype=np.float64)


def _load_pressure_field(config, case_dir: str, n_total: int, mesh_data: dict) -> np.ndarray:
    """Load or initialise the pressure field."""
    from ..fields import field_io

    if config.initial_p is not None:
        return np.full(n_total, config.initial_p, dtype=np.float64)
    try:
        p_data = field_io.read_field("p", os.path.join(case_dir, "0"), mesh_data)
        return p_data["phi"].astype(np.float64)
    except Exception:
        return np.zeros(n_total, dtype=np.float64)


def _enforce_u_boundary_constraints(
    U: np.ndarray, boundaries: list, n_elements: int, mesh_data: dict, geo_data: dict
) -> None:
    """Enforce velocity boundary constraints on ghost cells after initialisation."""
    from ..solve.simple_solver import _remove_normal_component

    for boundary in boundaries:
        bc_type = boundary.get("bc_type_U")
        start = n_elements + (boundary["startFace"] - mesh_data["n_interior_faces"])
        end = start + boundary["nFaces"]
        if bc_type == "noSlip":
            U[start:end] = 0.0
        elif bc_type in ["fixedValue", "freestream"] and "value_U" in boundary:
            U[start:end] = boundary["value_U"]
        elif bc_type in ("zeroGradient", "inletOutlet"):
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


class Solver:
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

        # 2. Geometry Computation
        logging.Timer.start("  Geometry Compute")
        from ..mesh import cache

        self.cache = cache.WeightCache(is_dynamic=(config.dynamic_mesh.method != "static"))
        gs = getattr(self.config.solver, "gradient_scheme", "gauss")
        self.geo_data = geometry.compute_mesh_geometry(self.mesh_data, gradient_scheme=gs)

        if not self.cache.is_dynamic:
            self.cache.set_static_weights(
                self.cache.from_mesh_geometry(self.geo_data).static_weights
            )
        else:
            self._precompute_dynamic_weights()
        logging.Timer.log("  Geometry Compute")

        # 3. Component Setup
        self.boundaries = self.mesh_data["boundary"]
        self._setup_boundary_conditions()
        self._initialize_fields()
        self._initialize_algorithm()
        self._initialize_turbulence()

        # Final housekeeping
        self.io = solver_io.SolverIO(self)
        self.vtk_exporter = None
        self.pvd_manager = None
        self.last_forces = None
        self.forces_history_path = None
        self._force_log_counter = 0
        self.cfl_max = 0.0
        self._time_since_last_write = 0.0

        logging.Logging.solver_summary(self)
        logging.Timer.log("Total Initialization")
        print()
        sys.stdout.flush()

        from ..solve import simple_solver

        simple_solver.update_scalar_boundaries(self.p, self.mesh_data, self.boundaries, "p")

    def _setup_boundary_conditions(self):
        """Maps user-defined BoundaryConfig to internal mesh boundary data."""
        print("\nBoundary Conditions Setup:")
        for b_cfg in self.config.boundaries:
            found = False
            for b_mesh in self.boundaries:
                if b_mesh["name"] == b_cfg.name:
                    b_mesh.update(
                        {
                            "type": b_cfg.type_U,
                            "bc_type_U": b_cfg.type_U,
                            "value_U": np.array(b_cfg.value_U),
                            "bc_type_p": b_cfg.type_p,
                            "value_p": b_cfg.value_p,
                            "bc_type_nut": b_cfg.type_nut,
                            "value_nut": b_cfg.value_nut,
                        }
                    )
                    print(f"  {b_cfg.name:<15} : {b_cfg.type_U:<12} U={b_cfg.value_U}")
                    found = True
                    break
            if not found:
                print(f"  Warning: Boundary '{b_cfg.name}' not found in mesh.")

    def _initialize_fields(self):
        """Initializes velocity (U), pressure (p), and flux (phi) fields."""
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
        """Initializes the numerical solver algorithm (PIMPLE or SIMPLE)."""
        logging.Timer.start("  Algorithm Init")
        if hasattr(self.config.solver, "to_dict"):
            params = self.config.solver.to_dict()  # type: ignore[union-attr]
        else:
            params = vars(self.config.solver)
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
        """Initializes the turbulence model if configured."""
        self.turbulence = None
        self.nut = None
        if self.config.turbulence and self.config.turbulence.model.lower() != "none":
            try:
                from ..turbulence import create_model

                self.turbulence = create_model(
                    self.config.turbulence, self.mesh_data, self.geo_data
                )
                if self.turbulence is not None:
                    info = self.turbulence.get_filter_info()
                    print(f"Turbulence model: {info['model']} (coeff={info['Cs']:.3g})")
            except Exception as e:
                print(f"Warning: Failed to initialize turbulence: {e}")

        # Sync state
        self.flow_time = self.config.time.start_time
        self.time_step = 0
        self.dt = self.config.time.delta_t

    def _precompute_dynamic_weights(self):
        """Pre-compute geometric weights for all steps if mesh motion is predictable."""
        logging.Timer.start("  Precompute Dynamic Weights")
        config = self.config
        dt = config.time.delta_t
        n_steps = int((config.time.end_time - config.time.start_time) / dt)
        orig_points = self.mesh_data["points"].copy()

        for step in range(n_steps + 1):
            t = config.time.start_time + step * dt
            if config.dynamic_mesh.method == "rigidMotion":
                translation = np.array(config.dynamic_mesh.velocity) * t
                self.mesh_data["points"] = orig_points + translation

            geo = geometry.compute_mesh_geometry(self.mesh_data)
            keys_to_cache = [
                "face_sf",
                "face_areas",
                "element_volumes",
                "face_weights",
                "face_cf_vector",
                "face_cf",
                "face_ff",
                "wall_dist",
                "wall_dist_limited",
            ]
            weights = {k: geo[k].copy() for k in keys_to_cache if k in geo}
            self.cache.add_dynamic_step(weights)

        self.mesh_data["points"] = orig_points
        logging.Timer.log("  Precompute Dynamic Weights")

    def evolve(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step.

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

        step_dt = dt or self.dt
        self.time_step += 1
        self.flow_time += step_dt

        logging.Timer.start(f"  Step {self.time_step}")
        logging.Logging.step_info(self.flow_time, self.time_step, step_dt)

        if self.cache.is_dynamic:
            cached = self.cache.get_weights(self.time_step)
            if cached:
                self.geo_data.update(cached)

        # Roll the time-history ring: U_old_old <- u^{n-1}, U_old <- u^n.
        # BDF2 needs u^{n-1}; pass it only from the 2nd step onward so the first
        # step self-starts with BDF1 (standard BDF2 startup).
        self.U_old_old[:] = self.U_old[:]
        self.U_old[:] = self.U[:]
        u_old_old_arg = self.U_old_old if self.time_step >= 2 else None

        # Determine effective viscosity
        if self.turbulence is not None:
            try:
                self.nut = self.turbulence.compute_nut(self.U, self.mesh_data, self.geo_data)
                nu_eff = self.config.transport.nu + self.nut
            except Exception as e:
                print(f"Warning: nut computation failed: {e}")
                nu_eff = self.config.transport.nu
        else:
            nu_eff = self.config.transport.nu

        # Solve step
        self.U, self.p, self.phi, residuals = self.algorithm.step(
            self.U,
            self.p,
            self.phi,
            self.U_old,
            step_dt,
            rho=self.config.transport.density,
            nu=nu_eff,
            U_old_old=u_old_old_arg,
        )

        logging.Logging.convergence_info(residuals)

        # Continuity (incompressibility) diagnostic: a divergence-free solution
        # has ~0 net flux per cell.  Surfacing this each step makes loss of
        # mass conservation visible instead of silent.
        cont = diagnostics.compute_continuity_error(self.phi, self.mesh_data, self.geo_data)
        vol = self.geo_data["element_volumes"]
        self.continuity_max = float(np.max(np.abs(cont) / (vol + 1e-30)))
        self.continuity_sum = float(np.sum(np.abs(cont)))
        print(
            f"  continuity: max|div U| = {self.continuity_max:.3e} 1/s, "
            f"sum|imbalance| = {self.continuity_sum:.3e} m3/s"
        )

        # Surface divergence loudly instead of hiding it behind a velocity clip.
        if not np.all(np.isfinite(self.U[: self.mesh_data["n_elements"]])):
            print(
                "  WARNING: non-finite velocity detected — the solution is "
                "diverging (try a smaller dt, more correctors, or a bounded "
                "convection scheme)."
            )

        # Compute CFL after step (for next step's dt adjustment)
        if cfg_time.adjust_timestep:
            Co_field = diagnostics.compute_courant_number(
                self.U, self.phi, step_dt, self.mesh_data, self.geo_data
            )
            self.cfl_max = float(np.max(Co_field))
            print(f"  max Co = {self.cfl_max:.3f}  dt = {step_dt:.6f} s  (target Co <= {cfg_time.max_cfl})")
            sys.stdout.flush()

        # y+ and Turbulence info
        patch_names = getattr(self.config.solver, "yplus_patches", None)
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
        force_interval = getattr(self.config.solver, 'force_log_interval', None)
        if force_interval is None:
            force_interval = cfg_time.write_interval

        self._force_log_counter += 1
        if self._force_log_counter % force_interval == 0 or self._force_log_counter == 1:
            ref_U = getattr(self.config.solver, 'ref_velocity', 1.0)
            ref_area = getattr(self.config.solver, 'ref_area', 1.0)
            ref_length = getattr(self.config.solver, 'ref_length', 1.0)
            mu = self.config.transport.nu * self.config.transport.density
            rho = self.config.transport.density

            patches = getattr(self.config.solver, 'force_patches', None)

            forces = diagnostics.compute_surface_forces(
                self.U, self.p, mu, rho, self.mesh_data, self.geo_data,
                self.boundaries, patch_names=patches,
                ref_U=ref_U, ref_area=ref_area, ref_length=ref_length,
                moment_centre=getattr(self.config.solver, 'moment_centre', [0, 0, 0])
            )
            self.last_forces = forces

            sol_dir = os.path.join(self.case_dir, "solution")
            os.makedirs(sol_dir, exist_ok=True)
            csv_path = os.path.join(sol_dir, "forces_history.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(["time", "step", "dt", "patch", "Fpx", "Fpy", "Fpz",
                                     "Fvx", "Fvy", "Fvz", "Ftx", "Fty", "Ftz",
                                     "Cd", "Cl", "Cz", "Cm"])
                for pname, fdata in forces.items():
                    Fp = fdata.get("Fp", [0, 0, 0])
                    Fv = fdata.get("Fv", [0, 0, 0])
                    Ft = fdata.get("Ftot", [0, 0, 0])
                    C = fdata.get("coeffs", {})
                    writer.writerow([
                        self.flow_time, self.time_step, step_dt, pname,
                        Fp[0], Fp[1], Fp[2],
                        Fv[0], Fv[1], Fv[2],
                        Ft[0], Ft[1], Ft[2],
                        C.get("Cd"), C.get("Cl"), C.get("Cz"), C.get("Cm")
                    ])
            log_msg = f"  Forces logged: {len(forces)} patch(es)"
            for pname, fdata in forces.items():
                C = fdata.get("coeffs", {})
                log_msg += f" | {pname}: Cd={C.get('Cd', 0):.4f} Cl={C.get('Cl', 0):.4f}"
            print(log_msg)
            sys.stdout.flush()

        if self.turbulence and self.nut is not None:
            logging.Logging.turbulence_info(self.nut, self.config.transport.nu)

        # Output control — time-based if write_interval_time is set, else step-based
        if self.auto_write:
            wrt_time = cfg_time.write_interval_time
            if wrt_time is not None:
                self._time_since_last_write += step_dt
                if self._time_since_last_write >= wrt_time:
                    self.write_vtk()
                    self._time_since_last_write = 0.0
            else:
                if self.time_step % cfg_time.write_interval == 0:
                    self.write_vtk()

        elapsed = logging.Timer.log(f"  Step {self.time_step}")
        print(f"\nStep completed in {elapsed:.3f} s")
        sys.stdout.flush()

    def write_vtk(self, filename: str | None = None) -> None:
        """Exports the current simulation state to VTK format with PVD support."""
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
        """Prints high-level solver state information."""
        print("-" * 40)
        print("FVM Solver Information")
        print(f"  Case      : {self.config.case_name}")
        print(f"  Time      : {self.flow_time:.5f}")
        print(f"  Step      : {self.time_step}")
        print(f"  Cells     : {self.mesh_data['n_elements']}")
        print(f"  Algorithm : {self.config.solver.algorithm}")
        print("-" * 40)
