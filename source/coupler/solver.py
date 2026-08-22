"""FVM–VPM coupling driver with VPM boundary conditions and compatible handoff."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

try:
    from mpi4py import MPI as _MPI

    _mpi4py_comm = _MPI.COMM_WORLD
except ImportError:
    _mpi4py_comm = None

import numpy as np

from source.coupler.boundary import (
    advance_fvm,
    evaluate_vpm_boundary,
    initialize_vpm_boundary_history,
    resynchronize_vpm_boundary,
)
from source.coupler.checkpoint import (
    load_coupled_state,
    save_coupled_state,
)
from source.coupler.config.types import CouplerSetup
from source.coupler.pressure_reference import PressureReference
from source.coupler.reporting import (
    OutputRedirector,
    configure_logging,
    flush_log,
    record_step,
    write_run_metadata,
)
from source.coupler.vorticity_transfer import VorticityTransfer

if TYPE_CHECKING:
    from source.solvers.FVM import FVMSolver
    from source.solvers.VPM import VPMSolver

logger = logging.getLogger("coupler")


def _world_rank() -> int:
    """Return the launcher rank for OpenMPI, MPICH/PMI, or MVAPICH."""
    for name in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    if _mpi4py_comm is not None:
        return int(_mpi4py_comm.Get_rank())
    return 0


def _vpm_solver_info(vpm_solver) -> str:
    from source.solvers.VPM.io.logging import Logging

    return Logging.solver_info(vpm_solver)


class FVMVPMCoupler:
    """
    FVM-VPM coupler for boundary exchange and compatible-curl wake handoff.
    """

    def __init__(self, fvm_solver, vpm_solver, coupler_setup: CouplerSetup):
        """Build a coupler from externally configured FVM and VPM solvers.

        The FVM solver is required on every rank. The VPM solver is required
        only on rank zero.
        """
        if fvm_solver is None:
            raise ValueError(
                "FVMVPMCoupler requires an injected fvm_solver on every rank. "
                "Build it with fvm_solver(case_dir) (all ranks) and the VPM on "
                "the master, then FVMVPMCoupler(fvm_solver, vpm_solver, "
                "coupler_setup)."
            )
        self.setup = coupler_setup
        self.case_dir = Path(fvm_solver.case_dir).expanduser().absolute()

        self._injected_fvm = fvm_solver
        self._injected_vpm = vpm_solver

        self._mpi_rank = _world_rank()
        self._is_master = self._mpi_rank == 0

        self.solution_dir = self.case_dir / "solution"
        if self._is_master:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            configure_logging(self.solution_dir, logger)

        if self._is_master:
            self.vpm_redirector = OutputRedirector(
                logfile=str(self.solution_dir / "vpm.log"), append=True
            )
        else:
            self.vpm_redirector = OutputRedirector()  # no-op

        self.vpm_solver: VPMSolver | None = None
        self.fvm_solver: FVMSolver | None = None
        self.vorticity_transfer: VorticityTransfer | None = None
        self._velocity_bc_prev: np.ndarray | None = None
        self._normal_velocity_bc_prev: np.ndarray | None = None
        self._normal_velocity_bc_next: np.ndarray | None = None
        self._tangential_gradient_bc_prev: np.ndarray | None = None
        self._tangential_gradient_bc_next: np.ndarray | None = None
        self._pressure_gradient_bc_prev: np.ndarray | None = None
        self._pressure_gradient_bc_next: np.ndarray | None = None
        self._pressure_velocity_snapshot: np.ndarray | None = None
        self._velocity_global_buffer: np.ndarray | None = None
        self._velocity_gradient_global_buffer: np.ndarray | None = None
        self._last_vpm_bc_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self.pressure_reference: PressureReference | None = None
        self.coupling_diagnostics: list[dict] = []
        self._last_transfer_result = None

        self.fvm_time_step_size: float | None = None
        self.vpm_time_step_size: float | None = None
        self.end_time: float | None = None
        self.kinematic_viscosity: float | None = None
        self.density: float | None = None
        self.fvm_box: np.ndarray | None = None
        self.fvm_substeps = 1
        self.freestream_velocity = np.array(coupler_setup.freestream_velocity, dtype=np.float64)

    @staticmethod
    def _validate_vpm(vpm, cfg: CouplerSetup, box: np.ndarray, nu: float) -> None:
        """Validate the injected VPM against the coupling discretization."""
        vsc = vpm.setup.viscous
        scheme = vsc.scheme.upper()
        regen = vsc.core_radius_ratio
        if (
            scheme in {"DVH", "GBD"}
            and regen is not None
            and abs(float(regen) - float(cfg.vpm_core_radius_ratio)) > 1e-9
        ):
            raise ValueError(
                "VPM regen core_radius_ratio must match the coupler vpm_core_radius_ratio"
            )
        dom = vpm.setup.domain_bounds
        if dom is not None:
            contains = (
                dom[0] <= box[0]
                and dom[1] >= box[1]
                and dom[2] <= box[2]
                and dom[3] >= box[3]
                and dom[4] <= box[4]
                and dom[5] >= box[5]
            )
            if not contains:
                raise ValueError(
                    f"Injected VPM domain {tuple(dom)} does not contain the FVM "
                    f"box {tuple(box)}. The near-body particles the coupler injects "
                    "would be removed by the VPM's out-of-bounds cull every step. "
                    "Widen domain_bounds (or the VPM stabilization "
                    "remove_particles_by_bounds) to enclose fvm_box."
                )
        vpm_nu = vsc.kinematic_viscosity
        if vpm_nu is not None and abs(float(vpm_nu) - nu) > 1e-12:
            raise ValueError(
                f"Incompatible kinematic viscosity: VPM viscous.kinematic_viscosity="
                f"{float(vpm_nu):g} but the Eulerian solver uses nu={float(nu):g}. "
                "The two solvers must model the same fluid."
            )
        bg = np.asarray(vpm.freestream_velocity, dtype=np.float64)
        if not np.allclose(bg, np.asarray(cfg.freestream_velocity), atol=1e-9):
            raise ValueError(
                f"Incompatible freestream: VPM freestream_velocity {tuple(bg)} "
                f"!= coupling freestream_velocity {tuple(cfg.freestream_velocity)}. "
                "The VPM far-field and "
                "the VPM advection frame must agree."
            )

    @staticmethod
    def _derive_fvm_substeps(vpm_time_step_size: float, fvm_time_step_size: float) -> int:
        """Return the integer FVM sub-cycle count implied by solver time steps."""
        if fvm_time_step_size <= 0.0:
            raise ValueError(f"FVM time step must be positive, got {fvm_time_step_size!r}.")
        if vpm_time_step_size <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {vpm_time_step_size!r}.")
        ratio = vpm_time_step_size / fvm_time_step_size
        fvm_substeps = max(1, int(round(ratio)))
        if not np.isclose(ratio, fvm_substeps, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "The VPM time step must be an integer multiple of the FVM time "
                f"step for sub-cycling. Got vpm_dt={vpm_time_step_size:.12g}, "
                f"fvm_dt={fvm_time_step_size:.12g}, ratio={ratio:.12g}."
            )
        return fvm_substeps

    @staticmethod
    def _derive_coupling_step_count(end_time: float, vpm_time_step_size: float) -> int:
        """Return the number of VPM/coupling intervals for a given end time.

        The end time need not be an exact multiple of the VPM step size; the
        count is rounded to the nearest integer, landing on the closest
        coupling-step boundary.

        Args:
            end_time: Requested simulation end time.
            vpm_dt:   VPM (coupling) time-step size.

        Returns:
            Integer number of coupling steps.
        """
        if vpm_time_step_size <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {vpm_time_step_size!r}.")
        if end_time < 0.0:
            raise ValueError(f"Coupling end time must be non-negative, got {end_time!r}.")
        return max(0, int(round(end_time / vpm_time_step_size)))

    def _derive_fvm_box(self) -> np.ndarray:
        """Bounds of the coupling patch, from the injected solver's geometry.

        The patch faces lie exactly on the six box planes, so the per-axis
        min/max of the face centroids reproduce the box bounds to round-off.
        Collective (all ranks) — the face-geometry getter gathers globally.
        """
        assert self.fvm_solver is not None
        fc = np.asarray(
            self.fvm_solver.get_boundary_face_centre_coordinates(self.setup.coupling_patch),
            dtype=np.float64,
        ).reshape(-1, 3)
        box = None
        error = None
        collective = _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1
        if self._is_master or not collective:
            if fc.shape[0] == 0:
                error = (
                    f"Coupling patch {self.setup.coupling_patch!r} has no faces on the "
                    "injected Eulerian solver."
                )
            else:
                box = np.array(
                    [
                        fc[:, 0].min(),
                        fc[:, 0].max(),
                        fc[:, 1].min(),
                        fc[:, 1].max(),
                        fc[:, 2].min(),
                        fc[:, 2].max(),
                    ]
                )
        if collective:
            error, box = _mpi4py_comm.bcast((error, box) if self._is_master else None, root=0)
        if error is not None:
            raise ValueError(error)
        return np.asarray(box, dtype=np.float64)

    def _read_fvm_state(self) -> None:
        """Read fluid properties, time integration, and domain from the FVM."""
        assert self.fvm_solver is not None
        fvm_cfg = self.fvm_solver.setup
        self.fvm_time_step_size = float(fvm_cfg.time.time_step_size)
        self.end_time = float(fvm_cfg.time.end_time)
        self.kinematic_viscosity = float(fvm_cfg.transport.kinematic_viscosity)
        self.density = float(fvm_cfg.transport.density)
        self.fvm_box = self._derive_fvm_box()
        self.setup.validate_transfer_region_box(self.fvm_box)

    def initialize(self) -> None:
        """Adopt the injected solvers, derive sub-cycling, and build coupling
        components.

        The injected FVM solver's configuration owns the FVM step; the
        injected VPM solver's ``time_step_size`` configures the
        coupling/VPM step.  After both are known, the coupler derives
        ``fvm_substeps = round(vpm_dt / fvm_dt)`` internally.

        Idempotent: a second call is a no-op (the coupling components are built
        exactly once), so ``initialize`` then ``run``/``solve`` is safe.
        """
        if self.vorticity_transfer is not None:
            return  # already initialized
        cfg = self.setup

        self.fvm_solver = self._injected_fvm

        world_size = 1
        if _mpi4py_comm is not None:
            world_size = int(_mpi4py_comm.Get_size())
        else:
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size > 1 and int(self.fvm_solver.n_procs()) == 1:
            raise RuntimeError(
                f"Launched under MPI (world size {world_size}) but the injected "
                "Eulerian solver is serial (n_procs() == 1). Configure a parallel "
                "solver or launch one process."
            )

        self._read_fvm_state()

        candidate_vpm = self._injected_vpm
        inactive_rank = bool(getattr(candidate_vpm, "_openonda_inactive_rank", False))
        self.vpm_solver = candidate_vpm if self._is_master and not inactive_rank else None
        if self._is_master:
            if self.vpm_solver is None:
                raise ValueError(
                    "vpm_solver is None on the master rank. "
                    "Build the VPM on the master (the Coupler's internal VPM-owner rank)."
                )
            assert self.fvm_box is not None and self.kinematic_viscosity is not None
            self._validate_vpm(self.vpm_solver, cfg, self.fvm_box, self.kinematic_viscosity)
            logger.info("[Init] Using injected VPM solver.")

        assert self.fvm_time_step_size is not None
        vpm_time_step_size = self.fvm_time_step_size
        if self._is_master:
            assert self.vpm_solver is not None
            vpm_time_step_size = float(self.vpm_solver.time_step_size)
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            vpm_time_step_size = float(
                _mpi4py_comm.bcast(vpm_time_step_size if self._is_master else None, root=0)
            )

        self.vpm_time_step_size = vpm_time_step_size
        self.fvm_substeps = self._derive_fvm_substeps(
            self.vpm_time_step_size, self.fvm_time_step_size
        )
        if self._is_master:
            logger.info(
                "[Init] Time steps: fvm_dt=%.4e s, vpm_dt=%.4e s, fvm_substeps=%d.",
                self.fvm_time_step_size,
                self.vpm_time_step_size,
                self.fvm_substeps,
            )

        self.vorticity_transfer = VorticityTransfer(self)
        self.vorticity_transfer.setup(self.fvm_solver)
        if self._is_master and self.vorticity_transfer._body_bounds is not None:
            assert self.vpm_solver is not None
            self.vpm_solver.physics.configure_body_box(self.vorticity_transfer._body_bounds)
            logger.info(
                "[Init] VPM grid diffusion body mask enabled for box %s.",
                self.vorticity_transfer._body_bounds.tolist(),
            )
        if self._is_master:
            anchor = self.vorticity_transfer._lattice_anchor
            if (
                anchor is None
                and self.vorticity_transfer._cell_centers is not None
                and len(self.vorticity_transfer._cell_centers) > 0
            ):
                anchor = self.vorticity_transfer._cell_centers[0]
            if anchor is not None:
                assert self.vpm_solver is not None
                self.vpm_solver.physics.configure_grid_lattice_anchor(
                    anchor, self.setup.vpm_particle_spacing
                )
                logger.info("[Init] VPM diffusion lattice aligned with the transfer lattice.")

        assert self.fvm_box is not None
        self.pressure_reference = PressureReference(
            self.fvm_solver,
            fvm_box=self.fvm_box,
            freestream_velocity=self.freestream_velocity,
            particle_spacing=self.setup.vpm_particle_spacing,
            boundary_mode=self.setup.boundary_condition_mode,
            enabled=self.setup.pressure_anchor_to_freestream,
            is_master=self._is_master,
            comm=_mpi4py_comm,
        )
        self.pressure_reference.prepare()

        if self._is_master:
            logger.info("[Init] Impulsive start: zero VPM particles.")
            with self.vpm_redirector:
                print(_vpm_solver_info(self.vpm_solver))
                sys.stdout.flush()
            write_run_metadata(self)
            print("Initialization complete.\n")

        self._initialize_run_state()

    def run(
        self,
        start_step: int = 0,
        restart_from=None,
    ) -> None:
        """Initialize and run the coupling loop."""
        if self.vorticity_transfer is None:
            self.initialize()
        if restart_from is not None:
            if start_step:
                raise ValueError("start_step and restart_from are mutually exclusive")
            start_step = self.load_state(restart_from)
        self.solve(start_step=start_step)

    def solve(self, start_step: int = 0) -> None:
        """Run the FVM--VPM coupling loop."""
        face_geometry, n_steps = self._prepare_run()
        initialize_vpm_boundary_history(self, *face_geometry)
        assert self.vpm_time_step_size is not None
        for step in range(1 + start_step, n_steps + 1):
            time_end = step * self.vpm_time_step_size
            vpm_time = self._advance_vpm(step, time_end)
            u_previous, u_next, boundary_time = evaluate_vpm_boundary(self, *face_geometry)
            fvm_time = advance_fvm(self, *face_geometry, u_previous, u_next)
            # A pressure datum shift changes neither the incompressible
            # solution nor closed-body pressure forces.  Keep the solver's
            # native null-space datum in the numerical loop; presentation code
            # can apply a reported offset to an output copy when needed.
            transfer_result, transfer_time = self._transfer_vorticity_to_vpm(*face_geometry)
            resynchronize_vpm_boundary(self, *face_geometry)
            self._last_transfer_result = transfer_result
            record_step(
                self,
                step,
                time_end,
                (vpm_time, boundary_time, fvm_time, transfer_time),
                transfer_result,
                logger=logger,
                comm=_mpi4py_comm,
            )
        if self._is_master:
            flush_log(logger)

    def _initialize_run_state(self) -> None:
        self._step_transfer_stats: dict[str, float | int] | None = None
        self.coupling_diagnostics = []

    def _prepare_run(self):
        """Validate a run and collect the immutable interface geometry."""
        if self.vorticity_transfer is None:
            raise RuntimeError(
                "solve() called before initialize(); call coupler.initialize() "
                "first, or use coupler.run() which does both."
            )
        assert self._is_master == (self.vpm_solver is not None)
        assert self.fvm_solver is not None

        assert self.end_time is not None and self.vpm_time_step_size is not None
        n_steps = self._derive_coupling_step_count(self.end_time, self.vpm_time_step_size)
        patch = self.setup.coupling_patch
        if self._is_master:
            logger.info("=" * 60)
            logger.info("FVM-VPM COUPLED SOLVER")
            logger.info("=" * 60)

        face_centers = np.asarray(
            self.fvm_solver.get_boundary_face_centre_coordinates(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_normals = np.asarray(
            self.fvm_solver.get_boundary_face_normals(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_areas = np.asarray(
            self.fvm_solver.get_boundary_face_areas(patch), dtype=np.float64
        ).ravel()
        self._n_steps = n_steps
        return (face_centers, face_normals, face_areas), n_steps

    def _advance_vpm(self, step: int, time_end: float) -> float:
        t0 = time.perf_counter()
        if self._is_master:
            assert self.vpm_solver is not None
            with self.vpm_redirector:
                self.vpm_solver.set_freestream_velocity(self.setup.freestream_velocity)
            print()
            print("-" * 60)
            print(f"STEP {step}/{self._n_steps}  (t={time_end:.3f}s)")

            with self.vpm_redirector:
                self.vpm_solver.advance()
            self.vpm_solver.synchronize()
        return time.perf_counter() - t0

    def _transfer_vorticity_to_vpm(
        self,
        face_centers: np.ndarray,
        _face_normals: np.ndarray,
        _face_areas: np.ndarray,
    ):
        """Fetch the FVM velocity trace and transfer it to the particle lattice."""
        t_transfer = time.perf_counter()
        velocity_global = self._get_velocity_field_buffer()
        gradient_global = self._get_velocity_gradient_field_buffer()
        transfer_result = None
        if self._is_master:
            assert self.vpm_solver is not None
            assert self.vorticity_transfer is not None
            vpm = self.vpm_solver
            transfer = self.vorticity_transfer
            n_before = vpm.particles.n_particles
            sum_before = (
                float(np.sum(np.linalg.norm(np.asarray(vpm.particle_vortex_strength), axis=1)))
                if n_before > 0
                else 0.0
            )
            transfer_result = transfer.transfer(
                vpm,
                velocity=velocity_global,
                velocity_gradient=gradient_global,
            )
            n_after = vpm.particles.n_particles
            sum_after = (
                float(np.sum(np.linalg.norm(np.asarray(vpm.particle_vortex_strength), axis=1)))
                if n_after > 0
                else 0.0
            )
            self._step_transfer_stats = {
                "n_before": n_before,
                "n_after": n_after,
                "sum_before": sum_before,
                "sum_after": sum_after,
                "face_count": len(face_centers),
            }
        return transfer_result, time.perf_counter() - t_transfer

    def _get_velocity_field_buffer(self) -> np.ndarray:
        assert self.fvm_solver is not None
        if self._velocity_global_buffer is None:
            self._velocity_global_buffer = np.ascontiguousarray(
                self.fvm_solver.get_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm_solver.get_velocity_field_into(self._velocity_global_buffer)
        return self._velocity_global_buffer

    def _get_velocity_gradient_field_buffer(self) -> np.ndarray:
        assert self.fvm_solver is not None
        if self._velocity_gradient_global_buffer is None:
            self._velocity_gradient_global_buffer = np.ascontiguousarray(
                self.fvm_solver.get_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm_solver.get_velocity_gradient_field_into(self._velocity_gradient_global_buffer)
        return self._velocity_gradient_global_buffer

    def save_state(self, directory, *, coupling_step: int | None = None) -> Path:
        """Write a complete coupled checkpoint."""
        return save_coupled_state(self, directory, coupling_step=coupling_step)

    def load_state(self, directory) -> int:
        """Restore both solvers and the VPM boundary history."""
        return load_coupled_state(self, directory, comm=_mpi4py_comm)
