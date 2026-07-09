"""
FVM-VPM Coupled Solver — overset/Chimera-style four-step loop.

0. VPM       : advance the particle cloud to t^{n+1} first (donor time level).
1. BC (donor): full-cloud Biot-Savart (all particles, no exterior masking)
   evaluated at the FVM boundary face centres, then projected onto the
   discretely solenoidal subspace (uniform normal shift ε/A_tot) so that
   Σ u·S = 0 to machine precision — the Gresho-Sani compatibility condition
   for the pure-Neumann pressure problem.  Direction-agnostic: no face is
   treated as inflow or outflow.
   With ``donor_interior_source="fvm"`` the trace is instead split per
   sub-step into an interpolated exterior-particle term and a LIVE
   FVM-interior Biot-Savart term (Weymouth–Lauber-consistent; see
   ``_run_fvm_substeps_live_interior``).
2. FVM       : impose Dirichlet-U on all faces; the pressure closure is
   fixedFluxPressure (0/p), whose gradient the solver's constrainPressure
   computes with exact discrete weights — compatibility by construction.
   The fringe relaxation (fvm_fringe.py) blends the FVM interior toward the
   VPM field in a buffer band so the interior match is smooth.
3. INJECT    : conservative continuous hand-off (η-blended M4' remesh) of the
   FVM near-field vorticity into the particle cloud (continuous_overlap.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

try:
    from mpi4py import MPI as _MPI
    _mpi4py_comm = _MPI.COMM_WORLD
except ImportError:
    _mpi4py_comm = None

import numpy as np
import taichi as ti

from source.coupler.config.types import CouplerSetup
from source.coupler.core.helpers.continuous_overlap import (
    ContinuousOverlapInjector,
    cosine_eta,
)
from source.coupler.core.helpers.fvm_velocity_blend import FVMVelocityBlend
from source.coupler.core.helpers.output_redirector import OutputRedirector
from source.coupler.core.helpers.setup import SetupHandler
from source.solvers.OFW.fvm_solver import fvm_solver  # type hint for the injected FVM
from source.solvers.VPM import Solver as VPM_Solver  # type hint for the injected VPM
from source.solvers.VPM.io.logging import Logging

logger = logging.getLogger("coupler")

def flush_log():
    for handler in logger.handlers:
        handler.flush()

class _DisableSIGFPE:
    """Context manager that disables OpenFOAM's SIGFPE handler."""

    def __enter__(self):
        self._old_handler = signal.signal(signal.SIGFPE, signal.SIG_IGN)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGFPE, self._old_handler)
        return False

class FVMVPMCoupler:
    """
    FVM-VPM coupler: the four-step overset loop with fringe relaxation.
    """

    def __init__(self, vpm_solver, fvm_solver, coupler_setup: CouplerSetup):
        """Build the coupler around externally-constructed sub-solvers::

            coupler = FVMVPMCoupler(vpm_solver, fvm_solver, coupler_setup)

        The caller builds each sub-solver with its OWN native API
        (``VPM_Solver(SolverConfig(...))``, ``fvm_solver(case_dir)`` /
        ``FVM.Solver.from_case(...)``) and injects the instances;
        :class:`CouplerSetup` carries ONLY the coupling parameters — the
        interface box, the hand-off/donor/fringe knobs, the shared physical
        quantities (``u_inf``, ``nu``, particle spacing ``h``) and the FVM
        sub-step ``dt``.  The VPM's physics (viscous scheme, stretching,
        advection, turbulence, treecode θ, stabilization, kernel, precision,
        domain bounds, max_particles) lives in its ``SolverConfig``; the FVM's
        in its case dictionaries.  The sub-cycle count is derived internally
        from the configured solver time steps.

        * ``fvm_solver`` — required on EVERY MPI rank (collective solve).
        * ``vpm_solver`` — the GPU particle solver on the master rank; ``None``
          on non-master ranks (build the GPU VPM on the master only, gated by
          :meth:`is_master_rank`).

        The FVM case must be prepared (:meth:`prepare_case`) before the FVM
        solver reads it — see :meth:`from_solvers` for the canonical order.
        """
        if fvm_solver is None:
            raise ValueError(
                "FVMVPMCoupler requires an injected fvm_solver on every rank. "
                "Build it with fvm_solver(case_dir) (all ranks) and the VPM on "
                "the master, then FVMVPMCoupler(vpm_solver, fvm_solver, "
                "coupler_setup)."
            )
        self.coupler_setup = coupler_setup
        self.config = coupler_setup
        self.case_dir = Path(".").absolute()

        # Injected sub-solvers.  The VPM may be None on non-master ranks.
        self._injected_fvm = fvm_solver
        self._injected_vpm = vpm_solver

        # In parallel (mpirun), every process runs this constructor.
        # OMPI_COMM_WORLD_RANK is set by mpirun before Python starts.
        # Rank 0 owns the VPM/GPU and all IO; non-master ranks only participate
        # in the collective FVM solve.
        self._mpi_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        self._is_master = (self._mpi_rank == 0)

        self.solution_dir = self.case_dir / "solution"
        if self._is_master:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            self._configure_logging()

        # Non-master ranks redirect to /dev/null (no VPM output there anyway).
        # Redirectors capture the solvers' C-level stdout into their own logs.
        if self._is_master:
            self.vpm_redirector = OutputRedirector(
                logfile=str(self.solution_dir / "vpm.log"), append=True
            )
            self.ofw_redirector = OutputRedirector(
                logfile=str(self.solution_dir / "ofw.log"), append=True
            )
        else:
            self.vpm_redirector = OutputRedirector()  # no-op
            self.ofw_redirector = OutputRedirector()  # no-op

        # Solvers (built in initialize())
        self.vpm: VPM_Solver | None = None
        self.ofw: fvm_solver | None = None
        self.injector: ContinuousOverlapInjector | None = None
        self.fringe = None
        self.vel_blend: FVMVelocityBlend | None = None
        self._u_bc_prev: np.ndarray | None = None  # donor BC carried between sub-cycles
        self._omega_bc_prev: np.ndarray | None = None  # donor ω BC carried between sub-cycles
        self._last_omega_donor: np.ndarray | None = None  # ω at faces from last _donor_velocity call
        self._omega_global_buffer: np.ndarray | None = None
        self._bc_omega_tree = None  # lazy cKDTree on injector._cell_centers (mixed BC)

        # ── Multi-rate time stepping (FVM sub-cycling) ───────────────────────
        # ``coupler_setup.dt`` starts as dt_fvm.  During initialize(), the
        # injected VPM time step becomes authoritative for dt_vpm and the
        # integer sub-cycle count is derived from dt_vpm / dt_fvm.
        self.dt_fvm = float(coupler_setup.dt)
        self.period_multiplier = 1
        self.dt_vpm = self.dt_fvm
        self.dt = self.dt_vpm
        self.u_inf = np.array(coupler_setup.u_inf, dtype=np.float64)

    # =========================================================
    # Construction helpers (dependency injection)
    # =========================================================

    @classmethod
    def from_solvers(
        cls,
        coupler_setup: CouplerSetup,
        *,
        fvm_solver,
        vpm_solver=None,
    ) -> FVMVPMCoupler:
        """Keyword-argument constructor (``coupler_setup`` first, solvers named).

        Equivalent to ``FVMVPMCoupler(vpm_solver, fvm_solver, coupler_setup)``.
        The FVM case must already reflect ``coupler_setup`` before the FVM
        solver reads it, so the canonical flow is::

            vpm = VPM_Solver(SolverConfig(...)) if is_master else None
            FVMVPMCoupler.prepare_case(coupler_setup, vpm_solver=vpm)  # all ranks
            fvm = fvm_solver(case_dir)                                 # all ranks
            coupler = FVMVPMCoupler.from_solvers(
                coupler_setup, fvm_solver=fvm, vpm_solver=vpm)
            coupler.run()
        """
        return cls(vpm_solver, fvm_solver, coupler_setup)

    @staticmethod
    def _validate_injected_vpm(vpm, cfg: CouplerSetup) -> None:
        """Fail-fast consistency checks between an injected VPM and the coupling
        config — the safety net that makes dependency injection safe without
        restricting which native VPM options the caller sets.

        Hard errors on show-stoppers (VPM removal domain must contain the FVM
        box, else injected near-body particles get culled every step); warnings
        on likely-unintended mismatches (background velocity, hand-off radius).
        """
        box = cfg.fvm_box
        # Hand-off radius: DVH/GBD regen rebuilds every particle with
        # σ = regen_radius_ratio·h, but the Beale correction deconvolves
        # assuming σ = overlap_radius_ratio·h.  A mismatch costs ~4× in-box
        # velocity error.  The coupler can no longer silently sync it (the VPM
        # is already built), so warn loudly — set regen_radius_ratio on the
        # VPM's ViscousConfig to equal cfg.overlap_radius_ratio.
        vsc = getattr(getattr(vpm, "config", None), "viscous", None)
        regen = getattr(vsc, "regen_radius_ratio", None)
        if regen is not None and abs(float(regen) - float(cfg.overlap_radius_ratio)) > 1e-9:
            logger.warning(
                "[Init] Injected VPM viscous regen_radius_ratio=%.2f != "
                "overlap_radius_ratio=%.2f. The Beale deconvolution assumes the "
                "hand-off radius; set the VPM ViscousConfig.regen_radius_ratio = "
                "%.2f to match (measured ~4× in-box velocity error otherwise).",
                float(regen), float(cfg.overlap_radius_ratio), float(cfg.overlap_radius_ratio),
            )
        dom = getattr(vpm, "vpm_domain_bounds", None)
        if dom is None:
            dom = getattr(getattr(vpm, "config", None), "vpm_domain_bounds", None)
        if dom is not None and len(dom) == 6:
            contains = (
                dom[0] <= box[0] and dom[1] >= box[1]
                and dom[2] <= box[2] and dom[3] >= box[3]
                and dom[4] <= box[4] and dom[5] >= box[5]
            )
            if not contains:
                raise ValueError(
                    f"Injected VPM domain {tuple(dom)} does not contain the FVM "
                    f"box {tuple(box)}. The near-body particles the coupler injects "
                    "would be removed by the VPM's out-of-bounds cull every step. "
                    "Widen vpm_domain_bounds (or the VPM stabilization "
                    "remove_particles_by_bounds) to enclose fvm_box."
                )
        bg = getattr(vpm, "background_velocity", None)
        if bg is not None:
            bg = np.asarray(bg, dtype=np.float64).reshape(-1)
            if bg.size == 3 and not np.allclose(bg, np.asarray(cfg.u_inf), atol=1e-9):
                logger.warning(
                    "[Init] Injected VPM background_velocity %s != config.u_inf %s; "
                    "the donor freestream and the VPM advection freestream disagree.",
                    tuple(bg), tuple(cfg.u_inf),
                )

    @staticmethod
    def is_master_rank() -> bool:
        """True on the rank that owns the GPU VPM and all IO (rank 0).

        Lets an injection setup script build the VPM on the master only without
        hard-coding the ``OMPI_COMM_WORLD_RANK`` lookup."""
        return int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0

    @staticmethod
    def prepare_case(
        coupler_setup: CouplerSetup,
        *,
        vpm_solver=None,
        restart: bool = False,
    ) -> None:
        """Write the FVM case dictionaries from ``coupler_setup`` (deltaT,
        nu, coupling-patch BC type, initial field) BEFORE the FVM solver is
        built in injection mode.  Idempotent; master-rank only (guards inside).

        If ``vpm_solver`` is provided on the master rank, the write cadence is
        derived from the VPM/FVM time-step ratio before the FVM wrapper reads
        controlDict.  initialize() recomputes the same sub-cycle count after
        both solvers are attached."""
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
            return
        period_multiplier = 1
        if vpm_solver is not None:
            vpm_dt = FVMVPMCoupler._get_vpm_time_step(vpm_solver)
            period_multiplier = FVMVPMCoupler._derive_period_multiplier(
                vpm_dt, float(coupler_setup.dt)
            )
        SetupHandler(coupler_setup).prepare_directories(
            period_multiplier,
            float(coupler_setup.dt),
            restart=restart,
        )


    def _configure_logging(self) -> None:
        """Send coupler diagnostics to solution/coupler.log AND the console.

        Without an explicit handler the 'coupler' logger drops all INFO
        records (Python's last-resort handler only shows WARNING+), losing
        the per-step diagnostics (donor flux residual, injection balance).
        """
        if logger.handlers:
            return  # already configured (e.g. by an embedding application)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_h = logging.FileHandler(self.solution_dir / "coupler.log", mode="w")
        file_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(file_h)

        console_h = logging.StreamHandler(sys.stdout)
        console_h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_h)

    def _write_run_metadata(self) -> None:
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "case_dir": str(self.case_dir),
            **self.config.to_dict(),
            "dt_fvm": float(self.dt_fvm),
            "dt_vpm": float(self.dt_vpm),
            "injection_period": int(self.period_multiplier),
            "backup_period": self.config.backup_period,
            "log_period": self.config.log_period,
            "period_multiplier": float(self.period_multiplier),
        }
        try:
            out_path = self.solution_dir / "run_metadata.json"
            out_path.write_text(json.dumps(metadata, indent=2))
        except Exception:
            logger.warning("[Init] Failed to write run_metadata.json", exc_info=True)

    @staticmethod
    def _get_vpm_time_step(vpm) -> float:
        get_vpm_dt = getattr(vpm, "get_time_step_size", None)
        return float(get_vpm_dt() if callable(get_vpm_dt) else vpm.time_step_size)

    @staticmethod
    def _derive_period_multiplier(dt_vpm: float, dt_fvm: float) -> int:
        """Return the integer FVM sub-cycle count implied by solver time steps."""
        if dt_fvm <= 0.0:
            raise ValueError(f"FVM time step must be positive, got {dt_fvm!r}.")
        if dt_vpm <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {dt_vpm!r}.")
        ratio = dt_vpm / dt_fvm
        period_multiplier = max(1, int(round(ratio)))
        if not np.isclose(ratio, period_multiplier, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "The VPM time step must be an integer multiple of the FVM time "
                f"step for sub-cycling. Got dt_vpm={dt_vpm:.12g}, "
                f"dt_fvm={dt_fvm:.12g}, ratio={ratio:.12g}."
            )
        return period_multiplier

    def initialize(self) -> None:
        """Adopt the injected solvers, derive sub-cycling, and build coupling
        components.

        ``CouplerSetup.dt`` configures the FVM step.  The injected VPM solver's
        ``time_step_size`` configures the coupling/VPM step.  After both
        solvers have their time steps set, the coupler derives
        ``period_multiplier = round(dt_vpm / dt_fvm)`` internally.

        Idempotent: a second call is a no-op (the coupling components are built
        exactly once), so ``initialize`` then ``run``/``solve`` is safe.
        """
        if self.injector is not None:
            return  # already initialized
        cfg = self.config

        # Adopt the injected VPM (None on non-master).  Validation checks it is
        # mutually consistent with the coupling config (domain ⊇ box, hand-off
        # radius, freestream) — the VPM is already built, so the coupler can no
        # longer silently fix a mismatch; it fails/warns instead.
        self.vpm = self._injected_vpm if self._is_master else None
        if self._is_master:
            if self.vpm is None:
                raise ValueError(
                    "from_solvers: vpm_solver is None on the master rank. "
                    "Build the VPM on the master (FVMVPMCoupler.is_master_rank())."
                )
            self._validate_injected_vpm(self.vpm, cfg)
            logger.info("[Init] Using injected VPM solver.")

        # Adopt the injected FVM (all ranks), built by the caller from a case
        # prepared via ``prepare_case``.  The runtime setters stamp dt_fvm and
        # nu so the setup remains the single source of truth even if the case
        # dictionaries differed.
        self.ofw = self._injected_fvm
        self.ofw.set_time_step(self.dt_fvm)
        self.ofw.set_kinematic_viscosity(cfg.nu)

        # Derive sub-cycling only after both solvers have been configured.
        vpm_dt = float(self.dt_fvm)
        if self._is_master:
            vpm_dt = self._get_vpm_time_step(self.vpm)
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            vpm_dt = float(_mpi4py_comm.bcast(vpm_dt if self._is_master else None, root=0))

        self.dt_vpm = vpm_dt
        self.period_multiplier = self._derive_period_multiplier(self.dt_vpm, self.dt_fvm)
        self.dt = self.dt_vpm
        if self._is_master:
            logger.info(
                "[Init] Time steps: dt_fvm=%.4e s, dt_vpm=%.4e s, "
                "period_multiplier=%d.",
                self.dt_fvm,
                self.dt_vpm,
                self.period_multiplier,
            )
            SetupHandler(cfg).update_controldict(
                self.period_multiplier, self.dt_fvm, restart=False
            )

        # Downstream coupling components (injector hand-off buffer, fringe) size
        # themselves on the inter-hand-off interval = dt_vpm; make cfg.dt report
        # the coupling step from here on.  The FVM alone uses dt_fvm (above).
        cfg.dt = self.dt_vpm

        # Build injector
        self.injector = ContinuousOverlapInjector(self)
        self.injector.setup(self.ofw)

        # Build velocity forcing.  FVMVelocityBlend is built on ALL ranks because
        # vel_blend.update() triggers collective OFW getters; only the VPM wiring
        # is rank-0-only.
        if cfg.overlap_velocity_forcing:
            self.vel_blend = FVMVelocityBlend(cfg, self.ofw)
            self.vel_blend.update()  # collective: snapshot initial 0/U field
            if self._is_master:
                self.vpm.set_velocity_override(self.vel_blend)
            if self._is_master:
                logger.info("[VelBlend] velocity forcing ENABLED (overlap_velocity_forcing=True)")
        else:
            if self._is_master:
                logger.info("[VelBlend] velocity forcing DISABLED (overlap_velocity_forcing=False)")

        # Build fringe — all ranks (collective getters + scatter in __init__).
        from source.coupler.core.helpers.fvm_fringe import FringeFields

        self.fringe = FringeFields(cfg, self.vpm, self.ofw)

        if self._is_master:
            logger.info("[Init] Impulsive start: zero VPM particles.")
            with self.vpm_redirector:
                print(Logging.solver_info(self.vpm))
                sys.stdout.flush()
            self._write_run_metadata()
            print("Initialization complete.\n")

    def run(self, start_step: int = 0) -> None:
        """Initialize (if not already) and run the coupling loop.

        Convenience wrapper = :meth:`initialize` + :meth:`solve`.  Callers who
        want the explicit build→initialize→solve flow can call those two
        directly; ``initialize`` is idempotent so ``run`` after a manual
        ``initialize`` will not rebuild.
        """
        if self.injector is None:  # not yet initialized
            self.initialize()
        self.solve(start_step=start_step)

    def solve(self, start_step: int = 0) -> None:
        """Run the four-step coupling loop (requires :meth:`initialize` first)."""
        if self.injector is None:
            raise RuntimeError(
                "solve() called before initialize(); call coupler.initialize() "
                "first, or use coupler.run() which does both."
            )
        assert self._is_master == (self.vpm is not None)
        assert self.ofw is not None

        # MPI-safety gate.  The Robin BC setter uses pstreamScatterDoubles for
        # both velocity and vorticity arrays, making it parallel-safe on all
        # ranks.  Non-master ranks always pass zero arrays so the C++ scatter
        # overwrites them with rank-0's data.  Every call to _fvm_step must
        # invoke the SAME C++ setter on all ranks (Robin or Dirichlet) to keep
        # the MPI collectives in sync.
        n_procs = self.ofw.n_procs()

        n_steps = int(self.config.t_end / self.dt)
        patch = self.config.patch_name

        if self._is_master:
            logger.info("=" * 60)
            logger.info("FVM-VPM COUPLED SOLVER")
            logger.info("=" * 60)

        # ── All ranks: pre-fetch boundary geometry (collective gather) ────────
        face_centers = np.asarray(
            self.ofw.get_boundary_face_center_coordinates(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_normals = np.asarray(
            self.ofw.get_boundary_face_normals(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_areas = np.asarray(
            self.ofw.get_boundary_face_areas(patch), dtype=np.float64
        ).ravel()

        for step in range(1 + start_step, n_steps + 1):
            t_end = step * self.dt

            # ─────────────────────────────────────────────────────────────────
            # STEP 0 — ADVANCE VPM (rank 0 only)
            # ─────────────────────────────────────────────────────────────────
            t0 = time.time()
            if self._is_master:
                with self.vpm_redirector:
                    self.vpm.set_background_velocity(self.config.u_inf)
                print()
                print("─" * 60)
                print(f"STEP {step}/{n_steps}  (t={t_end:.3f}s)")

                with _DisableSIGFPE(), self.vpm_redirector:
                    self.vpm.update_state()
                ti.sync()

                if self.vel_blend is not None and self.vpm.particles.number_of_particles > 0:
                    pos_np = np.asarray(self.vpm.particles_positions, dtype=np.float64).reshape(-1, 3)
                    vel_np = np.asarray(self.vpm.particles_velocities, dtype=np.float64).reshape(-1, 3)
                    self.vel_blend.log_diagnostics(pos_np, vel_np, float(np.linalg.norm(self.u_inf)))

            t0 = time.time() - t0

            # ─────────────────────────────────────────────────────────────────
            # STEP 1b — FRINGE: scatter lambda + Utarget (ALL ranks, collective)
            # ─────────────────────────────────────────────────────────────────
            t1b = time.time()
            self.fringe.update_target()
            t1b = time.time() - t1b

            # ─────────────────────────────────────────────────────────────────
            # STEP 1 — Donor BC (rank 0 computes; C++ scatter distributes)
            # ─────────────────────────────────────────────────────────────────
            t1 = time.time()
            if self._is_master:
                if getattr(self.config, "donor_interior_source", "particles") == "fvm":
                    u_bc_next = self._donor_exterior_velocity(face_centers)
                else:
                    u_bc_next = self._donor_velocity(face_centers, face_normals, face_areas)
                omega_bc_next = self._last_omega_donor
                if self._u_bc_prev is None:
                    self._u_bc_prev = u_bc_next.copy()
                if self._omega_bc_prev is None:
                    self._omega_bc_prev = (
                        omega_bc_next.copy() if omega_bc_next is not None else None
                    )
            else:
                u_bc_next = np.zeros_like(face_centers)
                _mixed = getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed"
                omega_bc_next = np.zeros_like(face_centers) if _mixed else None
                if self._u_bc_prev is None:
                    self._u_bc_prev = np.zeros_like(face_centers)
                if self._omega_bc_prev is None and _mixed:
                    self._omega_bc_prev = np.zeros_like(face_centers)
            t1 = time.time() - t1

            # ─────────────────────────────────────────────────────────────────
            # STEP 2 — FVM sub-cycle (ALL ranks, collective scatter + solve)
            # ─────────────────────────────────────────────────────────────────
            t2 = time.time()
            self._run_fvm_substeps(
                patch, face_centers, face_normals, face_areas,
                self._u_bc_prev, u_bc_next,
                self._omega_bc_prev, omega_bc_next,
            )
            if self._is_master:
                self._u_bc_prev = u_bc_next
                self._omega_bc_prev = omega_bc_next
            elif getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed":
                self._u_bc_prev = u_bc_next
                self._omega_bc_prev = omega_bc_next

            # Refresh FVM velocity snapshot: ALL ranks (collective getter).
            if self.vel_blend is not None:
                self.vel_blend.update()
            t2 = time.time() - t2

            # ─────────────────────────────────────────────────────────────────
            # STEP 3 — PRE-FETCH vorticity (ALL ranks, collective getter),
            #          then INJECT (rank 0 only, uses pre-fetched omega)
            # ─────────────────────────────────────────────────────────────────
            t3 = time.time()
            omega_global = self._get_vorticity_field_buffer()

            if self._is_master:
                eta_fn = self._build_eta_fn()
                n_before = self.vpm.particles.number_of_particles
                if n_before > 0:
                    sum_before = float(np.sum(
                        np.linalg.norm(np.asarray(self.vpm.particles_circulation), axis=1)
                    ))
                else:
                    sum_before = 0.0

                self.injector.inject(self.ofw, self.vpm, eta_fn=eta_fn, omega=omega_global)

                n_after = self.vpm.particles.number_of_particles
                sum_after = 0.0
                if n_after > 0:
                    sum_after = float(np.sum(
                        np.linalg.norm(np.asarray(self.vpm.particles_circulation), axis=1)
                    ))

                logger.info(
                    "     [Inject] N_before=%d  N_after=%d  |Γ|_before=%.4e  |Γ|_after=%.4e",
                    n_before, n_after, sum_before, sum_after,
                )
                print()
                print(f"[Step {step:4d}] t={t_end:.3f}s | Particles: {n_after}")
                print(
                    f"     Timing: VPM={t0:.2f}s | BC={t1:.2f}s | Fringe={t1b:.2f}s | "
                    f"FVM={t2:.2f}s | Inject={t3:.2f}s"
                )
                sys.stdout.flush()
                flush_log()
            t3 = time.time() - t3

            # Barrier: sync all ranks after inject so non-master cannot exit
            # while rank 0 is still inside inject.  mpi4py MPI.Barrier is used
            # because Foam::UPstream::barrier is a no-op in the dummy Pstream.
            if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
                _mpi4py_comm.Barrier()

        if self._is_master:
            flush_log()

# =========================================================
    # Neumann vorticity helper (mixed / Robin BC)
# =========================================================

    def _fvm_face_vorticity(self, face_centers: np.ndarray) -> np.ndarray | None:
        """Interpolate the cached FVM vorticity to boundary face centres.

        This is the correct Neumann target for the mixed BC:

          ∂u_t/∂n = ω_FVM × n̂   (Billuart 2023 Eq. 12)

        Using the FVM's own cell vorticity (nearest-cell interpolation)
        rather than the full particle-cloud vorticity eliminates two
        sources of noise that caused spurious VPM wake contamination:

        1. **Source inconsistency** (split-donor path): the velocity BC
           uses exterior particles + live FVM-interior Biot–Savart, while
           the particle vorticity includes in-box particles that lag one
           full coupling window. Using FVM vorticity for both halves of
           the Robin condition removes this mismatch.

        2. **Gaussian smoothing bias**: particle vorticity is filtered by
           the σ = 1.5h core, systematically under-representing gradients
           at scales < 2σ. Near the boundary this biases the Neumann
           gradient down, causing the FVM to generate compensating
           spurious vorticity every step. FVM cell vorticity has no such
           filter.

        Returns ``None`` on the first coupling step when ``_omega_global_buffer``
        has not yet been populated; callers should then fall back to the
        particle vorticity as a one-time bootstrap.

        The buffer ``_omega_global_buffer`` is populated by
        ``_get_vorticity_field_buffer()`` at the end of step 3 every cycle,
        so the values here are always from the most-recently solved FVM step
        (one coupling-window lag, identical to the lag already present in the
        velocity BC from the particle side).
        """
        if self._omega_global_buffer is None:
            return None  # first step: no FVM solve has occurred yet

        assert self.injector is not None
        centers = self.injector._cell_centers
        if centers is None or centers.shape[0] == 0:
            return None

        # Build the kD-tree once; the mesh never changes during a run.
        if self._bc_omega_tree is None:
            from scipy.spatial import cKDTree
            self._bc_omega_tree = cKDTree(np.asarray(centers, dtype=np.float64))

        _, idx = self._bc_omega_tree.query(
            np.ascontiguousarray(face_centers, dtype=np.float64), k=1
        )
        omega = np.ascontiguousarray(self._omega_global_buffer[idx], dtype=np.float64)
        logger.info(
            "     [Donor] Neumann ω from FVM nearest-cell  n_face=%d  "
            "|ω|_max=%.3e s⁻¹",
            len(face_centers),
            float(np.max(np.linalg.norm(omega, axis=1))) if len(omega) else 0.0,
        )
        return omega

# =========================================================
    # STEP 1 — Donor velocity BC (overset-style, full particle cloud)
# =========================================================

    def _donor_velocity(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        exterior_mask: np.ndarray | None = None,
        add_fvm_interior: bool = False,
        fvm_omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Donor velocity at boundary face centres (overset/Chimera-style).

        Standard donor path — ``exterior_mask=None, add_fvm_interior=False``:
        U_donor = U_inf + BiotSavart(**all** particles) at face centres.  The
        full VPM cloud acts as the background donor mesh (no exterior masking).

        Weymouth–Lauber coupled path — ``exterior_mask`` selects the particles
        *outside* the FVM box and ``add_fvm_interior=True`` adds the Biot–Savart
        velocity induced by the FVM *interior* vorticity field:
        U_donor = U_inf + BiotSavart(exterior particles) + BiotSavart(FVM ω).
        This replaces the (stale, smoothed) in-box particle representation of the
        interior with the freshly-solved sharp FVM field, closing the BC↔pressure
        coupling.  No vorticity is dropped — the exterior wake is fully kept (the
        contrast with the failed near-face exclusion, which deleted it).

        The full analytic field is solenoidal; the residual quadrature flux is
        projected out (uniform normal shift ε/A_tot, Gresho–Sani compatibility).
        """
        n = len(face_centers)
        assert self.vpm is not None
        if self.vpm.particles.number_of_particles == 0:
            u_donor = np.tile(self.u_inf, (n, 1)).astype(np.float64)
            # No particles → no VPM vorticity at the boundary.
            self._last_omega_donor = np.zeros((n, 3), dtype=np.float64)
            if add_fvm_interior:
                u_donor = u_donor + self._fvm_interior_induced_velocity(
                    face_centers, omega=fvm_omega
                )
            else:
                return u_donor
        else:
            n_particles = self.vpm.particles.number_of_particles
            if exterior_mask is None:
                logger.info(
                    "     [Donor] particles=%d  (full field, no exterior masking)",
                    n_particles,
                )
            else:
                logger.info(
                    "     [Donor] exterior particles=%d/%d + FVM-interior BS "
                    "(coupled BC)",
                    int(exterior_mask.sum()),
                    n_particles,
                )

            u_donor = self.vpm.compute_target_velocities(
                face_centers,
                include_freestream=True,
                zone_mask=exterior_mask,  # None means all particles.
            )
            # Neumann target for the mixed/Robin BC (Billuart 2023 Eq. 12:
            # ∂u_t/∂n = ω × n̂).  Use the FVM's own cell vorticity (sharp,
            # unsmoothed, no core-mollification bias) rather than the full
            # particle cloud (which includes smoothed, lagged in-box particles
            # that generate inconsistent Neumann gradients near the boundary).
            # Falls back to particle vorticity on the first step only.
            _omega_fvm = self._fvm_face_vorticity(face_centers)
            self._last_omega_donor = (
                _omega_fvm
                if _omega_fvm is not None
                else np.asarray(
                    self.vpm.compute_target_vorticities(face_centers), dtype=np.float64
                )
            )
            if add_fvm_interior:
                u_donor = u_donor + self._fvm_interior_induced_velocity(
                    face_centers, omega=fvm_omega
                )

# =========================================================
        # The Biot-Savart donor is analytically solenoidal (∮u·n dA = 0), but
        # treecode quadrature, finite face-point sampling, and Gaussian core
        # truncation leave a small residual ε = Σ_f u_f·S_f ≠ 0.  With all
        # faces fixedValue there is no adjustable patch, so OpenFOAM cannot
        # call adjustPhi.  The entire defect is instead absorbed by pRefCell=0
        # (the min-corner cell), producing a pressure monopole there → spurious
        # vorticity at (x,y,z)_min that regenerates every step.
        #
        # Fix: project u_donor onto the discretely solenoidal subspace via the
        # unique minimum-L²(∂Ω) correction — a uniform normal shift ε/A_tot.
        # This is not a tuning knob; it is the Gresho-Sani compatibility
        # condition for the pure-Neumann pressure problem.  The correction is
        # O(quadrature error), vanishes under mesh/treecode refinement, and
        # carries no inflow/outflow assumption (works for any flow direction).
        normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
        areas = np.asarray(face_areas, dtype=np.float64).ravel()
        flux_residual_raw = 0.0
        if len(areas) > 0:
            u_normal = np.einsum("ij,ij->i", u_donor, normals)
            flux_residual_raw = float(np.dot(u_normal, areas))
            total_area = float(np.sum(areas))

            # Minimal-L² projection: subtract uniform normal δu = ε/A_tot
            if total_area > 0.0:
                delta_u_n = flux_residual_raw / total_area  # scalar [m/s]
                u_donor = u_donor - delta_u_n * normals

            # Recompute post-projection residual for logging
            u_normal_post = np.einsum("ij,ij->i", u_donor, normals)
            flux_residual_post = float(np.dot(u_normal_post, areas))

            u_inf_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
            rel_flux_raw = abs(flux_residual_raw) / (u_inf_mag * total_area + 1e-30)
            rel_flux_post = abs(flux_residual_post) / (u_inf_mag * total_area + 1e-30)
            logger.info(
                "     [Donor] Flux residual: raw=%.3e m³/s (%.2e×U∞A)  "
                "post-projection=%.3e m³/s (%.2e×U∞A)  δu_n=%.3e m/s",
                flux_residual_raw,
                rel_flux_raw,
                flux_residual_post,
                rel_flux_post,
                delta_u_n if total_area > 0.0 else 0.0,
            )

        # Deficit probe on the OUTFLOW face (direction-agnostic: derived from
        # u_inf, not hard-wired to +x).
        self._log_outflow_deficit(face_centers, u_donor)

        return u_donor

    def _outflow_axis_sign(self) -> tuple[int, float]:
        """(axis, sign) of the box face most aligned with the freestream.

        Makes the flow-direction diagnostics work for any inlet/outlet
        placement; the coupling physics is already direction-agnostic."""
        u = np.asarray(self.u_inf, dtype=np.float64).reshape(-1)
        if u.size != 3 or not np.any(u != 0.0):
            return 0, +1.0
        axis = int(np.argmax(np.abs(u)))
        return axis, float(np.sign(u[axis]))

    def _log_outflow_deficit(self, face_centers: np.ndarray, u_field: np.ndarray) -> None:
        """Log the streamwise-velocity deficit on the outflow face (the face the
        freestream points toward), normalised by |U∞|.  Direction-agnostic."""
        axis, sign = self._outflow_axis_sign()
        u_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
        box = self.config.fvm_box
        face_lo, face_hi = box[2 * axis], box[2 * axis + 1]
        if sign >= 0:
            mask = face_centers[:, axis] >= face_hi - 1e-6
        else:
            mask = face_centers[:, axis] <= face_lo + 1e-6
        if not mask.any():
            return
        # Streamwise component = projection of u onto the freestream direction.
        u_stream = (u_field[mask] @ (np.asarray(self.u_inf) / u_mag))
        logger.info(
            "     [Donor deficit outflow axis=%d sign=%+d] u_s/U∞ min=%.3f "
            "mean=%.3f max=%.3f  n_face=%d",
            axis, int(sign), u_stream.min() / u_mag, u_stream.mean() / u_mag,
            u_stream.max() / u_mag, int(mask.sum()),
        )

    def _donor_exterior_velocity(self, face_centers: np.ndarray) -> np.ndarray:
        """Exterior-wake donor endpoint: U∞ + BS(particles OUTSIDE the box).

        Used by ``donor_interior_source="fvm"``.  Deliberately UNPROJECTED and
        WITHOUT the interior term: the FVM-interior Biot–Savart is added live
        at every sub-step and the solenoidal projection is applied to the
        assembled trace there (projecting the endpoint too would double-count
        the compatibility shift).

        Exterior sources are at least a buffer width from every face and
        evolve on the slow VPM timescale, so linearly interpolating THIS term
        across the window is far more defensible than interpolating the total
        trace (which smears the fast near-field).

        For ``donor_bc_mode="mixed"``, the Neumann vorticity target uses the
        FVM's own cell vorticity (via :meth:`_fvm_face_vorticity`), consistent
        with the FVM-interior split used for the velocity.  Using the full
        particle cloud here would pair a velocity that excludes in-box particles
        with a vorticity that includes them, producing a Neumann gradient
        inconsistent with the imposed normal velocity.
        """
        n = len(face_centers)
        assert self.vpm is not None
        if self.vpm.particles.number_of_particles == 0:
            self._last_omega_donor = np.zeros((n, 3), dtype=np.float64)
            return np.tile(self.u_inf, (n, 1)).astype(np.float64)

        exterior_mask = self._outside_box_mask()
        logger.info(
            "     [Donor] exterior particles=%d/%d + live FVM-interior BS per "
            "sub-step (donor_interior_source=fvm)",
            int(exterior_mask.sum()),
            self.vpm.particles.number_of_particles,
        )
        u_ext = self.vpm.compute_target_velocities(
            face_centers,
            include_freestream=True,
            zone_mask=exterior_mask,
        )
        if getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed":
            # VPM vorticity for the Robin BC Neumann target.  See comment in
            # _donor_velocity for why particle vorticity is used instead of FVM
            # vorticity (FVM vorticity at inlet causes exponential instability).
            self._last_omega_donor = np.asarray(
                self.vpm.compute_target_vorticities(face_centers), dtype=np.float64
            )
        else:
            self._last_omega_donor = None
        return np.asarray(u_ext, dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _project_to_solenoidal(
        u: np.ndarray, face_normals: np.ndarray, face_areas: np.ndarray
    ) -> np.ndarray:
        """Minimal-L² uniform-normal-shift projection so ∮u·n dA = 0.

        The same Gresho–Sani compatibility correction the donor BC applies, but
        reusable: the FVM sub-cycler interpolates two (already-projected) donor
        states and must re-project the interpolant so *every* sub-step sees a
        discretely solenoidal Dirichlet field, not only the interpolation
        endpoints (linear-combination flux residual is tiny but not exactly 0).
        """
        normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
        areas = np.asarray(face_areas, dtype=np.float64).ravel()
        if areas.size == 0:
            return u
        total_area = float(np.sum(areas))
        if total_area <= 0.0:
            return u
        eps = float(np.dot(np.einsum("ij,ij->i", u, normals), areas))
        return u - (eps / total_area) * normals

    def _outside_box_mask(self) -> np.ndarray:
        """Boolean mask of VPM particles strictly OUTSIDE the FVM box.

        Used by the Weymouth–Lauber coupled donor BC to take the *exterior*
        wake from particles and the *interior* from the FVM field — a clean
        overset split with no double-counting and, crucially, no vorticity
        dropped (the entire exterior wake is kept).
        """
        assert self.vpm is not None
        n = self.vpm.particles.number_of_particles
        pos = np.asarray(self.vpm.particles_positions, dtype=np.float64).reshape(-1, 3)[:n]
        x0, x1, y0, y1, z0, z1 = self.config.fvm_box
        inside = (
            (pos[:, 0] >= x0) & (pos[:, 0] <= x1)
            & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
            & (pos[:, 2] >= z0) & (pos[:, 2] <= z1)
        )
        return ~inside

    def _get_vorticity_field_buffer(self) -> np.ndarray:
        assert self.ofw is not None
        if self._omega_global_buffer is None:
            self._omega_global_buffer = np.ascontiguousarray(
                self.ofw.get_vorticity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.ofw.get_vorticity_field_into(self._omega_global_buffer)
        return self._omega_global_buffer

    def _fvm_interior_induced_velocity(
        self, targets: np.ndarray, omega: np.ndarray | None = None
    ) -> np.ndarray:
        """Biot–Savart velocity at ``targets`` induced by the FVM interior vorticity.

        Closes the Weymouth–Lauber BC↔pressure coupling: the box-boundary
        velocity must include the velocity induced by the vorticity *inside* the
        box (body boundary layer + near wake), re-evaluated from the freshly
        solved FVM field each sub-step / Picard iteration.

        Singular kernel regularised with an h-sized core to bound the near-face
        contribution; source selection (noise floor + hard cap) and the
        Numba-parallel direct sum live in ``helpers/interior_bs.py``.

        MPI: cell geometry comes from the injector's one-time collective gather;
        ``omega`` must be passed in when the caller already fetched it on all
        ranks (fetching here from a rank-0-only section would deadlock the
        collective getter).
        """
        from source.coupler.core.helpers import interior_bs

        assert self.ofw is not None
        targets = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1, 3)
        if omega is None:
            omega = self._get_vorticity_field_buffer()

        assert self.injector is not None
        centers = self.injector._cell_centers
        vols = self.injector._cell_volumes
        if centers is None or centers.shape[0] == 0:
            return np.zeros_like(targets)

        # Weak-remainder pooling at 2h: conserves the diffuse wake circulation
        # whose induction carves the outflow deficit into the trace (dropping
        # it pins the boundary near freestream → spurious fast wake recovery).
        src, gamma, info = interior_bs.select_sources(
            omega, centers, vols, pool_h=2.0 * float(self.config.h)
        )
        if src.shape[0] == 0:
            return np.zeros_like(targets)
        logger.info(
            "     [Donor] FVM-interior BS sources: %d exact + %d pooled bins "
            "(%d above the %.0e·peak floor; %.0f%% of Σ|Γ| represented)",
            src.shape[0] - info["n_pooled_bins"], info["n_pooled_bins"],
            info["n_above"], interior_bs.FLOOR_FRACTION,
            100.0 * info["kept_fraction"],
        )
        self._validate_bs_sign_once()
        return interior_bs.bs_velocity(targets, src, gamma, core=float(self.config.h))

    def _validate_bs_sign_once(self) -> None:
        """One-time analytic check of the FVM-interior BS kernel sign.

        A z-aligned vortex (Γ=+ẑ) at the origin must induce +y velocity at
        (+x,0,0) (counter-clockwise swirl).  Guards against the sign-convention
        error that previously inverted a Biot–Savart reconstruction.
        """
        if getattr(self, "_bs_sign_ok", False):
            return
        src = np.array([[0.0, 0.0, 0.0]])
        G = np.array([[0.0, 0.0, 1.0]])
        tgt = np.array([[1.0, 0.0, 0.0]])
        r = tgt[:, None, :] - src[None, :, :]
        r2 = np.einsum("mnk,mnk->mn", r, r) + 1e-6
        cross = np.cross(G[None, :, :], r)
        u = (1.0 / (4.0 * np.pi)) * np.einsum("mn,mnk->mk", r2 ** -1.5, cross)
        assert u[0, 1] > 0.0 and abs(u[0, 0]) < 1e-9, (
            f"FVM-interior BS sign check failed: u={u[0]} (expected +y swirl)"
        )
        self._bs_sign_ok = True

# =========================================================
    # STEP 2 — FVM advance with the donor BC pair
# =========================================================

    def _fvm_step(
        self,
        patch: str,
        u_target: np.ndarray,
        advance: bool = True,
        omega_target: np.ndarray | None = None,
    ) -> None:
        """Impose the donor BC and run the FVM PIMPLE solve.

        Velocity BC type is selected by ``config.donor_bc_mode``:

        * ``"dirichlet"`` — full-velocity Dirichlet. ``omega_target`` is ignored.

        * ``"mixed"`` — Robin / mixed (Billuart 2023 Eq. 11–14): normal
          Dirichlet ``u·n̂ = u_VPM·n̂`` + tangential Neumann
          ``∂u_t/∂n = ω_VPM × n̂`` via ``set_robin_velocity_boundary_condition``
          (requires the 0/U patch type to be ``directionMixed``).  Needs
          ``omega_target`` (VPM ω at the face centres).

        Pressure: fixedFluxPressure (0.orig/p) — the solver's constrainPressure
        (pEqn.H) computes ∂p/∂n with the exact discrete rAtU weights so the
        boundary pressure flux matches the imposed velocity flux identically.
        The pure-Neumann pressure problem is then compatible BY CONSTRUCTION
        and pRefCell only pins the level (no defect absorbed at the reference
        cell, no spurious corner vorticity).

        ``advance`` controls whether the time step is committed.  With the
        one-shot donor BC (``bc_coupling_iterations == 1``) it is True.  In the
        Weymouth–Lauber BC↔pressure Picard loop the BC *does* depend on the FVM
        answer (the pressure-generated interior vorticity feeds back into the
        boundary Biot–Savart velocity), so ``solve_pimple`` is called repeatedly
        with the BC re-imposed between solves and ``advance`` is True only on the
        final iteration.
        """
        assert self.ofw is not None
        u_inf_mag = float(np.linalg.norm(self.config.u_inf)) + 1e-30

        def _set_dirichlet_velocity(target: np.ndarray) -> None:
            self.ofw.set_dirichlet_velocity_boundary_condition_vec(
                np.ascontiguousarray(target, dtype=np.float64), patch
            )

        mode = getattr(self.config, "donor_bc_mode", "dirichlet")
        if mode == "mixed":
            if omega_target is None:
                logger.warning(
                    "     [FVM] donor_bc_mode='mixed' but omega_target is None — "
                    "using Dirichlet for this step."
                )
                _set_dirichlet_velocity(u_target)
                bc_label = "dirichlet"
            else:
                self.ofw.set_robin_velocity_boundary_condition(
                    np.ascontiguousarray(u_target, dtype=np.float64),
                    np.ascontiguousarray(omega_target, dtype=np.float64),
                    patch,
                )
                bc_label = "mixed (Robin: u_n Dirichlet + ω Neumann)"
        else:
            _set_dirichlet_velocity(u_target)
            bc_label = "dirichlet"

        with self.ofw_redirector:
            self.ofw.solve_pimple()
            if advance:
                self.ofw.advance_time()

        if u_target.shape[0] > 0:
            logger.info(
                "     [FVM] solved with donor BC [%s]  u_x/U∞ face[min=%.2f max=%.2f]%s",
                bc_label,
                u_target[:, 0].min() / u_inf_mag,
                u_target[:, 0].max() / u_inf_mag,
                "" if advance else "  (no advance — BC iteration)",
            )

    def _run_fvm_substeps(
        self,
        patch: str,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        u_prev: np.ndarray,
        u_next: np.ndarray,
        omega_prev: np.ndarray | None = None,
        omega_next: np.ndarray | None = None,
    ) -> None:
        """Advance the FVM ``period_multiplier`` sub-steps over one VPM cycle.

        Sub-step k (0..N−1) imposes the donor velocity linearly interpolated to
        α = (k+1)/N between the previous cycle's BC (``u_prev``, α→0) and this
        cycle's *future* BC (``u_next``, α=1) — i.e. the BC at each sub-step's
        **new** time level (backward-Euler-consistent) — re-projected solenoidal
        so every sub-step sees a discretely divergence-free Dirichlet field.

        When ``donor_bc_mode == "mixed"``, the donor vorticity is interpolated
        the same way (``omega_prev`` → ``omega_next``) and passed to
        ``_fvm_step`` as the Neumann target.

        The final sub-step (α=1) runs the Weymouth–Lauber donor↔pressure Picard
        when ``bc_coupling_iterations > 1``; otherwise each sub-step is a single
        ``solve_pimple`` + ``advance_time``.
        """
        N = max(1, int(self.period_multiplier))
        n_bc = max(1, int(self.config.bc_coupling_iterations))
        u_inf_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
        mixed = getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed"

        if getattr(self.config, "donor_interior_source", "particles") == "fvm":
            self._run_fvm_substeps_live_interior(
                patch, face_centers, face_normals, face_areas,
                u_prev, u_next, omega_prev, omega_next,
                N=N, n_bc=n_bc, mixed=mixed, u_inf_mag=u_inf_mag,
            )
            return

        if N > 1 and u_next.shape[0] > 0:
            # Physical guard: linear interpolation under-resolves the wake if the
            # donor BC changes a lot across one VPM cycle (Co_vpm too large).
            # Skipped on non-master MPI ranks (empty boundary arrays).
            dU = float(np.max(np.linalg.norm(u_next - u_prev, axis=1))) / u_inf_mag
            big = dU > 0.5
            logger.log(
                logging.WARNING if big else logging.INFO,
                "     [Sub-cycle] %d×dt_fvm=%.3e s  donor ΔBC max|Δu|/U∞=%.3f%s",
                N, self.dt_fvm, dU,
                "  (large — lower dt or period_multiplier)" if big else "",
            )

        for sub in range(N):
            alpha = (sub + 1) / N
            is_final = sub == N - 1
            if is_final and n_bc > 1:
                # Weymouth–Lauber donor↔pressure Picard at the full future BC.
                exterior_mask = self._outside_box_mask() if self._is_master else None
                for k in range(n_bc):
                    # Collective on all ranks.  Rank 0 assembles the full
                    # Weymouth–Lauber donor; non-master ranks pass zero arrays
                    # that the C++ pstreamScatter overwrites with rank-0's data.
                    # Both ranks MUST call the same setter (Robin or Dirichlet)
                    # to keep the MPI collectives in sync.
                    omega_interior = self._get_vorticity_field_buffer()
                    if self._is_master:
                        u_wl = self._donor_velocity(
                            face_centers, face_normals, face_areas,
                            exterior_mask=exterior_mask,
                            add_fvm_interior=True,
                            fvm_omega=omega_interior,
                        )
                        omega_wl = self._last_omega_donor
                    else:
                        u_wl = np.zeros_like(face_centers)
                        omega_wl = np.zeros_like(face_centers) if mixed else None
                    self._fvm_step(
                        patch, u_wl, advance=(k == n_bc - 1),
                        omega_target=omega_wl if mixed else None,
                    )
            else:
                u_bc = (1.0 - alpha) * u_prev + alpha * u_next
                u_bc = self._project_to_solenoidal(u_bc, face_normals, face_areas)
                omega_bc = None
                if mixed:
                    if omega_prev is not None and omega_next is not None:
                        omega_bc = (1.0 - alpha) * omega_prev + alpha * omega_next
                    elif omega_next is not None:
                        omega_bc = omega_next
                    elif omega_prev is not None:
                        omega_bc = omega_prev
                self._fvm_step(
                    patch, u_bc, advance=True,
                    omega_target=omega_bc if mixed else None,
                )

    def _run_fvm_substeps_live_interior(
        self,
        patch: str,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        u_ext_prev: np.ndarray,
        u_ext_next: np.ndarray,
        omega_prev: np.ndarray | None,
        omega_next: np.ndarray | None,
        *,
        N: int,
        n_bc: int,
        mixed: bool,
        u_inf_mag: float,
    ) -> None:
        """FVM sub-cycle with the Weymouth–Lauber-consistent split donor.

        Sub-step k imposes

            u_bc = interp_α(U∞ + BS(exterior particles))      [slow, interpolated]
                 + BS(FVM interior ω, LIVE)                     [fast, never stale]

        re-projected onto the discretely solenoidal subspace, where the
        interior term is re-evaluated from the current FVM vorticity before
        every solve.  With ``n_bc > 1`` the solve is Picard-iterated per
        sub-step: iteration k ≥ 1 re-fetches the post-solve vorticity, so the
        boundary trace becomes consistent with the interior at the sub-step's
        NEW time level (the BC↔pressure closure of Weymouth & Lauber); the
        time step is committed on the final iteration only.

        MPI: the vorticity gather is collective and runs on ALL ranks every
        iteration; the Biot–Savart and trace assembly are rank-0-only (the
        boundary arrays are empty elsewhere; the C++ BC scatter distributes).
        """
        if N > 1 and u_ext_next.shape[0] > 0:
            dU = float(np.max(np.linalg.norm(u_ext_next - u_ext_prev, axis=1))) / u_inf_mag
            big = dU > 0.5
            logger.log(
                logging.WARNING if big else logging.INFO,
                "     [Sub-cycle] %d×dt_fvm=%.3e s  exterior donor ΔBC "
                "max|Δu|/U∞=%.3f  (interior term is live)%s",
                N, self.dt_fvm, dU,
                "  (large — lower dt or period_multiplier)" if big else "",
            )

        for sub in range(N):
            alpha = (sub + 1) / N
            u_ext = (1.0 - alpha) * u_ext_prev + alpha * u_ext_next
            omega_bc = None
            if mixed:
                if omega_prev is not None and omega_next is not None:
                    omega_bc = (1.0 - alpha) * omega_prev + alpha * omega_next
                elif omega_next is not None:
                    omega_bc = omega_next
                elif omega_prev is not None:
                    omega_bc = omega_prev

            for k in range(n_bc):
                # Collective on ALL ranks; refreshed each iteration so k ≥ 1
                # sees the post-solve interior (new-time-level closure).
                t_w = time.time()
                omega_interior = self._get_vorticity_field_buffer()
                t_w = time.time() - t_w
                t_bs = time.time()
                if self._is_master and u_ext.shape[0] > 0:
                    u_int = self._fvm_interior_induced_velocity(
                        face_centers, omega=omega_interior
                    )
                    u_bc = self._project_to_solenoidal(
                        u_ext + u_int, face_normals, face_areas
                    )
                else:
                    u_bc = u_ext
                t_bs = time.time() - t_bs
                if self._is_master:
                    logger.info(
                        "     [Sub-cycle] sub=%d/%d it=%d/%d  ω-gather=%.2fs  BS+proj=%.2fs",
                        sub + 1, N, k + 1, n_bc, t_w, t_bs,
                    )
                    if sub == N - 1 and k == n_bc - 1 and u_bc.shape[0] > 0:
                        # Outflow deficit on the fully assembled (exterior +
                        # live interior) trace; direction-agnostic.
                        self._log_outflow_deficit(face_centers, u_bc)
                self._fvm_step(
                    patch, u_bc, advance=(k == n_bc - 1),
                    omega_target=omega_bc if mixed else None,
                )

# =========================================================
    # ETA (authority weight)
# =========================================================

    def _build_eta_fn(self):
        """Return a callable eta(x) for the continuous hand-off."""
        box = np.array(self.config.fvm_box, dtype=np.float64)
        ramp_width = max(self.config.buffer_thickness, 1e-12)
        dead_zone = self.config.dead_zone_h * self.config.h

        def eta_fn(points):
            return cosine_eta(points, box, ramp_width, dead_zone)

        return eta_fn

# =========================================================
    # Restart support
# =========================================================

    def load_vpm_from_backup(self, backup_h5_path: str) -> int:
        """Load VPM state from an H5 backup."""
        import h5py

        h5_path = backup_h5_path if backup_h5_path.endswith(".h5") else backup_h5_path + ".h5"
        with h5py.File(h5_path, "r") as f:
            flow_time = float(f["solver"].attrs["flow_time"])
            time_step = int(f["solver"].attrs["time_step"])
            p = f["particles"]
            pos = p["position"][:]
            circ = p["circulation"][:]
            vel = p["velocity"][:]
            rad = p["radius"][:]
            vol = p["volume"][:]
            visc = p["viscosity"][:]
            visc_t = p["viscosity_turbulent"][:]
            gid = p["group_id"][:]

        with self.vpm_redirector:
            self.vpm.remove_particles(remove_all=True)
            self.vpm.add_vortex_particles(
                position=pos,
                velocity=vel,
                circulation=circ,
                radius=rad,
                volume=vol,
                viscosity=visc,
                viscosity_turbulent=visc_t,
                group_id=gid,
            )
            self.vpm.flow_time = flow_time
            self.vpm.time_step = time_step

        n = len(pos)
        print(f"[Restart] Loaded {n} particles from backup (t={flow_time:.3f}s, step={time_step})")
        return time_step
