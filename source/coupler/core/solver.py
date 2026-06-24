"""
FVM-VPM Coupled Solver — overset/Chimera-style four-step loop.

0. VPM       : advance the particle cloud to t^{n+1} first (donor time level).
1. BC (donor): full-cloud Biot-Savart (all particles, no exterior masking)
   evaluated at the FVM boundary face centres, then projected onto the
   discretely solenoidal subspace (uniform normal shift ε/A_tot) so that
   Σ u·S = 0 to machine precision — the Gresho-Sani compatibility condition
   for the pure-Neumann pressure problem.  Direction-agnostic: no face is
   treated as inflow or outflow.
2. FVM       : impose Dirichlet-U on all faces; the pressure closure is
   fixedFluxPressure (0/p), whose gradient the solver's constrainPressure
   computes with exact discrete weights — compatibility by construction.
   The fringe relaxation (fvm_fringe.py) blends the FVM interior toward the
   VPM field in a buffer band so the interior match is smooth.
3. INJECT    : conservative continuous hand-off (η-blended M4' remesh) of the
   FVM near-field vorticity into the particle cloud (continuous_overlap.py).
"""

from __future__ import annotations

from dataclasses import replace
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

from source.coupler.config.types import CouplerConfig
from source.coupler.core.helpers.continuous_overlap import (
    ContinuousOverlapInjector,
    cosine_eta,
)
from source.coupler.core.helpers.fvm_velocity_blend import FVMVelocityBlend
from source.coupler.core.helpers.output_redirector import OutputRedirector
from source.coupler.core.helpers.setup import SetupHandler
from source.solvers.OFW.fvm_solver import fvm_solver
from source.solvers.VPM import Solver as VPM_Solver
from source.solvers.VPM import SolverConfig
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    TurbulenceConfig,
    VelocityConfig,
)
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

    def __init__(self, config: CouplerConfig):
        self.config = config
        self.case_dir = Path(".").absolute()

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
        self.body_panel = None
        self._u_bc_prev: np.ndarray | None = None  # donor BC carried between sub-cycles
        self._omega_bc_prev: np.ndarray | None = None  # donor ω BC carried between sub-cycles
        self._last_omega_donor: np.ndarray | None = None  # ω at faces from last _donor_velocity call

        # ── Multi-rate time stepping (FVM sub-cycling) ───────────────────────
        # The user configures ``config.dt`` = dt_fvm (the small, accurate FVM
        # step).  The VPM cloud and the whole coupling cadence (donor BC,
        # hand-off, samplers, backups) run on the LARGER dt_vpm = N · dt_fvm.
        # Internally ``self.dt`` IS the coupling step dt_vpm, so the VPM build,
        # loop cadence, injector buffer and fringe (which all read self.dt /
        # cfg.dt) size themselves on the inter-hand-off interval.  Only the FVM
        # deltaT and the per-sub-step BC interpolation use self.dt_fvm.
        self.period_multiplier = max(1, int(config.period_multiplier))
        self.dt_fvm = float(config.dt)
        self.dt_vpm = self.period_multiplier * self.dt_fvm
        self.dt = self.dt_vpm
        self.u_inf = np.array(config.u_inf, dtype=np.float64)

        if self._is_master:
            self._write_run_metadata()

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

    def initialize(self) -> None:
        """Build solvers, prepare directories, seed particles.

        The VPM is built FIRST: some viscous schemes (DVH) dictate their own
        time step Δt_d and override the configured dt.  Whatever dt the VPM
        settles on becomes THE coupled dt — it is propagated to the FVM
        (controlDict deltaT, write interval) and to the hand-off (buffer CFL
        sizing) before those components are built.
        """
        cfg = self.config

        # Radius consistency: DVH/GBD regen rebuilds every particle each step
        # with σ = regen_radius_ratio·h, silently overriding the hand-off radii.
        # The Beale correction deconvolves assuming σ = overlap_radius_ratio·h,
        # so the two MUST match (measured cost of the mismatch: ~4× in-box
        # velocity error at σ_regen=2.5h vs the corrected-for 1.5h).
        if self._is_master:
            if cfg.viscous_scheme is not None and hasattr(cfg.viscous_scheme, "regen_radius_ratio"):
                if cfg.viscous_scheme.regen_radius_ratio != cfg.overlap_radius_ratio:
                    logger.info(
                        "[Init] Syncing viscous regen radius to the hand-off: "
                        "regen_radius_ratio %.2f → %.2f (= overlap_radius_ratio).",
                        cfg.viscous_scheme.regen_radius_ratio,
                        cfg.overlap_radius_ratio,
                    )
                    cfg.viscous_scheme.regen_radius_ratio = float(cfg.overlap_radius_ratio)

        # VPM + GPU: rank 0 only.  Non-master ranks only participate in the
        # collective FVM solve; they never touch Taichi or particle data.
        if self._is_master:
            vpm_cfg = SolverConfig(
                time_step_size=self.dt,
                viscous=cfg.viscous_scheme,
                advection=AdvectionConfig(scheme=getattr(cfg, "advection_scheme", "RK2")),
                turbulence=TurbulenceConfig.les_smagorinsky(cs=cfg.les_smagorinsky_cs),
                stabilization=replace(
                    cfg.stabilization,
                    remove_particles_by_bounds=list(cfg.vpm_domain),
                ),
                particles_kernel=cfg.particles_kernel,
                background_velocity=list(cfg.u_inf),
                logging_frequency=cfg.log_period,
                backup_frequency=cfg.backup_period,
                backup_file_name="./solution/vpm_solution",
                samplers=cfg.samplers,
                velocity=VelocityConfig.treecode(theta=cfg.treecode_theta),
                vpm_domain_bounds=list(cfg.vpm_domain),
                max_particles=cfg.max_particles,
                precision=cfg.precision,
            )

            with self.vpm_redirector:
                self.vpm = VPM_Solver(config=vpm_cfg)

        # ── Time-step consistency: the VPM may override its OWN step (e.g. DVH
        # dictates a diffusion dt Δt_d).  Whatever the VPM settled on IS dt_vpm;
        # keep dt_fvm fixed (it is the FVM accuracy knob) and re-derive the
        # integer sub-cycle count period_multiplier = round(dt_vpm / dt_fvm). ──
        # NOTE: in parallel, period_multiplier is only updated on rank 0.  If a
        # viscous scheme overrides dt in a parallel run, all ranks must configure
        # period_multiplier explicitly in CouplerConfig to stay in sync.
        if self._is_master:
            vpm_dt = float(self.vpm.time_step_size)
            if abs(vpm_dt - self.dt_vpm) > 1e-12 * max(self.dt_vpm, vpm_dt):
                new_N = max(1, int(round(vpm_dt / self.dt_fvm)))
                logger.warning(
                    "[Init] VPM overrode the coupling step (viscous scheme %s): "
                    "dt_vpm %.4e → %.4e s.  Keeping dt_fvm=%.4e s and re-deriving "
                    "period_multiplier=%d FVM sub-steps per VPM step.",
                    getattr(cfg.viscous_scheme, "scheme", "?"),
                    self.dt_vpm,
                    vpm_dt,
                    self.dt_fvm,
                    new_N,
                )
                self.period_multiplier = new_N
                self.dt_vpm = vpm_dt
                self.dt = vpm_dt
                self._write_run_metadata()  # re-record with the final dt_vpm / N

        # Setup directories: FVM deltaT = dt_fvm, writeInterval at the VPM
        # (coupling) cadence = backup_period · period_multiplier FVM steps.
        # Rank 0 writes; non-master ranks read the result via shared filesystem.
        if self._is_master:
            setup_h = SetupHandler(cfg)
            setup_h.prepare_directories(self.period_multiplier, self.dt_fvm, restart=False)

        # Build the Eulerian backend — it marches on the small dt_fvm.  Default
        # is the OFW (OpenFOAM) wrapper; "FVM" selects the OpenONDA native solver,
        # which exposes the same OFW contract and builds from the case directory.
        with self.ofw_redirector:
            if str(getattr(cfg, "eulerian_backend", "OFW")).upper() == "FVM":
                from source.solvers.FVM import Solver as _FVMSolver

                self.ofw = _FVMSolver.from_case(str(self.case_dir))
            else:
                self.ofw = fvm_solver(str(self.case_dir))
        self.ofw.set_time_step(self.dt_fvm)
        self.ofw.set_kinematic_viscosity(cfg.nu)

        # Downstream coupling components (injector hand-off buffer, fringe) size
        # themselves on the inter-hand-off interval = dt_vpm; make cfg.dt report
        # the coupling step from here on.  The FVM alone uses dt_fvm (above).
        cfg.dt = self.dt_vpm

        # Build injector
        self.injector = ContinuousOverlapInjector(self)
        self.injector.setup(self.ofw)

        # Body panel: rank 0 only (uses VPM).
        if getattr(cfg, "body_panel_enabled", False) and self._is_master:
            from source.coupler.core.helpers.body_panel import BodyPanelModel

            with self.vpm_redirector:
                self.body_panel = BodyPanelModel(cfg)
            self.body_panel.solve(self.vpm, self.u_inf, 0.0)  # initial (zero-wake) solve
            self.vpm.set_body_induced_velocity(self.body_panel.induced)
            self.body_panel.log_diagnostics()
            logger.info("[BodyPanel] body panel model ENABLED (body_panel_enabled=True)")
        else:
            self.body_panel = None

        # Build velocity forcing.  FVMVelocityBlend is built on ALL ranks because
        # vel_blend.update() triggers collective OFW getters; only the VPM wiring
        # is rank-0-only.
        if cfg.overlap_velocity_forcing:
            self.vel_blend = FVMVelocityBlend(cfg, self.ofw)
            self.vel_blend.update()  # collective: snapshot initial 0/U field
            if self._is_master:
                # Body-panel advection term left OFF: the panel induction runs on
                # the numpy fallback (Vulkan rejects the f64 GPU kernel), too slow
                # per RK stage over the whole cloud.
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
            print("Initialization complete.\n")

    def run(self, start_step: int = 0) -> None:
        """Run the four-step coupling loop."""
        self.initialize()
        assert self._is_master == (self.vpm is not None)
        assert self.ofw is not None

        # MPI-safety gate.  Two coupling modes are NOT yet parallel-safe and would
        # deadlock or corrupt memory under mpirun -n>1 (see audit notes):
        #   * donor_bc_mode="mixed": set_robin_velocity_boundary_condition does not
        #     scatter and uses the GLOBAL face count to index a LOCAL-sized patch
        #     field (out-of-bounds write on master); the Python fallback also makes
        #     master/non-master call different collectives.
        #   * bc_coupling_iterations>1 (Weymouth–Lauber): _run_fvm_substeps calls
        #     _donor_velocity (VPM + interior-vorticity gather) on ALL ranks, but
        #     self.vpm is None on non-master and the gathers are master-only.
        # Fail loudly here rather than hang/segfault deep in the loop.
        n_procs = self.ofw.n_procs()
        if n_procs > 1:
            if getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed":
                raise NotImplementedError(
                    "donor_bc_mode='mixed' (Robin BC) is not parallel-safe "
                    f"(n_procs={n_procs}). Use donor_bc_mode='dirichlet' under MPI, "
                    "or run serially. See foamSolverCore.C "
                    "set_robin_velocity_boundary_condition (needs local face count "
                    "+ pstreamScatterDoubles)."
                )
            if int(self.config.bc_coupling_iterations) > 1:
                raise NotImplementedError(
                    "bc_coupling_iterations>1 (Weymouth–Lauber coupled BC) is not "
                    f"parallel-safe (n_procs={n_procs}). Set bc_coupling_iterations<=1 "
                    "under MPI, or run serially."
                )

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

                if self.body_panel is not None:
                    with self.vpm_redirector:
                        self.body_panel.solve(self.vpm, self.u_inf, t_end)
                    self.body_panel.log_diagnostics()
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
                u_bc_next = self._donor_velocity(face_centers, face_normals, face_areas)
                omega_bc_next = self._last_omega_donor
                if self._u_bc_prev is None:
                    self._u_bc_prev = u_bc_next.copy()
                if self._omega_bc_prev is None:
                    self._omega_bc_prev = (
                        omega_bc_next.copy() if omega_bc_next is not None else None
                    )
            else:
                # Non-master: provide zero-filled arrays of the right shape.
                # The C++ scatter in _fvm_step reads only from master’s buffer.
                u_bc_next = np.zeros_like(face_centers)
                omega_bc_next = None
                if self._u_bc_prev is None:
                    self._u_bc_prev = np.zeros_like(face_centers)
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

            # Refresh FVM velocity snapshot: ALL ranks (collective getter).
            if self.vel_blend is not None:
                self.vel_blend.update()
            t2 = time.time() - t2

            # ─────────────────────────────────────────────────────────────────
            # STEP 3 — PRE-FETCH vorticity (ALL ranks, collective getter),
            #          then INJECT (rank 0 only, uses pre-fetched omega)
            # ─────────────────────────────────────────────────────────────────
            t3 = time.time()
            # get_vorticity_field is collective; calling on all ranks avoids
            # deadlock when inject() is rank-0-gated below.
            omega_global = np.asarray(
                self.ofw.get_vorticity_field(), dtype=np.float64
            ).reshape(-1, 3)

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

    # =====================================================================
    # STEP 1 — Donor velocity BC (overset-style, full particle cloud)
    # =====================================================================

    def _donor_velocity(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        exterior_mask: np.ndarray | None = None,
        add_fvm_interior: bool = False,
    ) -> np.ndarray:
        """Donor velocity at boundary face centres (overset/Chimera-style).

        Legacy (default) path — ``exterior_mask=None, add_fvm_interior=False``:
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
                u_donor = u_donor + self._fvm_interior_induced_velocity(face_centers)
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
                zone_mask=exterior_mask,  # None → all particles (legacy)
            )
            # VPM vorticity at the boundary faces — the Neumann target for the
            # mixed/Robin donor BC (Billuart 2023 Eq. 12: ∂u_t/∂n = ω_VPM × n̂).
            # Computed on the full cloud (compute_target_vorticities has no
            # zone_mask); the in-box particles are hand-off-overwritten each
            # cycle so their boundary ω contribution is small.  The solenoidal
            # projection below is a gradient field (∇×∇φ=0) and does not alter ω.
            self._last_omega_donor = np.asarray(
                self.vpm.compute_target_vorticities(face_centers), dtype=np.float64
            )
            if add_fvm_interior:
                u_donor = u_donor + self._fvm_interior_induced_velocity(face_centers)

        # --- Discrete solenoidal projection -----------------------------------
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

        # Deficit probe on +x face
        u_inf_x = float(self.u_inf[0])
        x_max = self.config.fvm_box[1]
        plus_x = face_centers[:, 0] >= x_max - 1e-6
        if plus_x.any():
            ux_face = u_donor[plus_x, 0]
            logger.info(
                "     [Donor deficit +x] u_x/U∞ min=%.3f mean=%.3f max=%.3f  n_face=%d",
                ux_face.min() / u_inf_x,
                ux_face.mean() / u_inf_x,
                ux_face.max() / u_inf_x,
                int(plus_x.sum()),
            )

        return u_donor

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

    def _fvm_interior_induced_velocity(self, targets: np.ndarray) -> np.ndarray:
        """Biot–Savart velocity at ``targets`` induced by the FVM interior vorticity.

        Closes the Weymouth–Lauber BC↔pressure coupling: the box-boundary
        velocity must include the velocity induced by the vorticity *inside* the
        box (body boundary layer + near wake), re-evaluated from the freshly
        solved FVM field each Picard iteration.

        Singular kernel regularised with an h-sized core to bound the near-face
        contribution.  Sign convention matches the VPM solver: with r = target −
        source, u = (1/4π) Σ (Γ_c × r)/(|r|²+core²)^{3/2}  [≡ −(r×Γ)], validated
        once against an analytic single-element field.
        """
        assert self.ofw is not None
        targets = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1, 3)

        omega = np.asarray(self.ofw.get_vorticity_field(), dtype=np.float64).reshape(-1, 3)
        centers = np.asarray(
            self.ofw.get_cell_center_coordinates(), dtype=np.float64
        ).reshape(-1, 3)
        vols = np.asarray(self.ofw.get_cell_volumes(), dtype=np.float64).ravel()

        gamma = omega * vols[:, None]  # circulation per cell [m³/s]
        mag = np.linalg.norm(gamma, axis=1)
        if mag.size == 0:
            return np.zeros_like(targets)
        peak = float(mag.max())
        if peak <= 0.0:
            return np.zeros_like(targets)

        # Source selection.  Direct BS is O(n_face·n_src); the box is mostly
        # irrotational, so (1) drop cells below 0.1 % of the peak |Γ| (noise),
        # then (2) hard-cap to the N_MAX strongest cells.  The induced velocity
        # at the boundary is dominated by the strong near-cube + wake vorticity,
        # so the dropped tail is negligible while the cost stays bounded
        # (a too-loose 1e-6 floor previously kept ~all 4·10⁵ cells → minutes/eval).
        N_MAX = 20_000
        floor = 1e-3 * peak
        keep = np.flatnonzero(mag > floor)
        n_above = keep.size
        if keep.size > N_MAX:
            order = np.argpartition(mag[keep], keep.size - N_MAX)[keep.size - N_MAX:]
            keep = keep[order]
        if keep.size == 0:
            return np.zeros_like(targets)
        logger.info(
            "     [Donor] FVM-interior BS sources: %d (|Γ|>%.2e of peak %.2e; %d above floor)",
            keep.size, floor, peak, n_above,
        )

        self._validate_bs_sign_once()

        src = np.ascontiguousarray(centers[keep], dtype=np.float32)
        G = np.ascontiguousarray(gamma[keep], dtype=np.float32)
        tgt32 = targets.astype(np.float32)
        core2 = np.float32(float(self.config.h) ** 2)
        inv4pi = np.float32(1.0 / (4.0 * np.pi))
        out = np.zeros((len(targets), 3), dtype=np.float64)
        CH = 128  # chunk targets to bound (CH × n_src × 3) memory
        for i in range(0, len(tgt32), CH):
            tt = tgt32[i : i + CH]  # (m, 3)
            r = tt[:, None, :] - src[None, :, :]  # (m, n_src, 3)
            r2 = np.einsum("mnk,mnk->mn", r, r) + core2
            inv = r2 ** np.float32(-1.5)  # (m, n_src)
            cross = np.cross(G[None, :, :], r)  # Γ × r  (m, n_src, 3)
            out[i : i + CH] = (inv4pi * np.einsum("mn,mnk->mk", inv, cross)).astype(np.float64)
        return out

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

    # =====================================================================
    # STEP 2 — FVM advance with the donor BC pair
    # =====================================================================

    def _fvm_step(
        self,
        patch: str,
        u_target: np.ndarray,
        advance: bool = True,
        omega_target: np.ndarray | None = None,
    ) -> None:
        """Impose the donor BC and run the FVM PIMPLE solve.

        Velocity BC type is selected by ``config.donor_bc_mode``:

        * ``"dirichlet"`` (default, legacy) — full-velocity Dirichlet
          (``set_dirichlet_velocity_boundary_condition``).  ``omega_target`` is
          ignored.  Byte-identical to the pre-FIX-A path.

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

        ux = np.ascontiguousarray(u_target[:, 0], dtype=np.float64)
        uy = np.ascontiguousarray(u_target[:, 1], dtype=np.float64)
        uz = np.ascontiguousarray(u_target[:, 2], dtype=np.float64)

        mode = getattr(self.config, "donor_bc_mode", "dirichlet")
        if mode == "mixed":
            if omega_target is None:
                # Defensive: no ω available (e.g. empty cloud) → fall back to
                # Dirichlet so the step still runs (ω=0 Neumann = zero gradient
                # is NOT the same as full-Dirichlet, but safer than crashing).
                logger.warning(
                    "     [FVM] donor_bc_mode='mixed' but omega_target is None — "
                    "falling back to Dirichlet for this step."
                )
                self.ofw.set_dirichlet_velocity_boundary_condition(ux, uy, uz, patch)
                bc_label = "dirichlet (fallback)"
            else:
                self.ofw.set_robin_velocity_boundary_condition(
                    np.ascontiguousarray(u_target, dtype=np.float64),
                    np.ascontiguousarray(omega_target, dtype=np.float64),
                    patch,
                )
                bc_label = "mixed (Robin: u_n Dirichlet + ω Neumann)"
        else:
            self.ofw.set_dirichlet_velocity_boundary_condition(ux, uy, uz, patch)
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
        ``solve_pimple`` + ``advance_time``.  ``period_multiplier == 1`` reduces
        to exactly the legacy single-rate solve.
        """
        N = max(1, int(self.period_multiplier))
        n_bc = max(1, int(self.config.bc_coupling_iterations))
        u_inf_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
        mixed = getattr(self.config, "donor_bc_mode", "dirichlet") == "mixed"

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
                exterior_mask = self._outside_box_mask()
                for k in range(n_bc):
                    u_wl = self._donor_velocity(
                        face_centers, face_normals, face_areas,
                        exterior_mask=exterior_mask, add_fvm_interior=True,
                    )
                    omega_wl = self._last_omega_donor
                    self._fvm_step(
                        patch, u_wl, advance=(k == n_bc - 1),
                        omega_target=omega_wl if mixed else None,
                    )
            else:
                u_bc = (1.0 - alpha) * u_prev + alpha * u_next
                u_bc = self._project_to_solenoidal(u_bc, face_normals, face_areas)
                omega_bc = None
                if mixed and omega_prev is not None and omega_next is not None:
                    omega_bc = (1.0 - alpha) * omega_prev + alpha * omega_next
                self._fvm_step(
                    patch, u_bc, advance=True,
                    omega_target=omega_bc if mixed else None,
                )

    # =====================================================================
    # ETA (authority weight)
    # =====================================================================

    def _build_eta_fn(self):
        """Return a callable eta(x) for the continuous hand-off."""
        box = np.array(self.config.fvm_box, dtype=np.float64)
        ramp_width = max(self.config.buffer_thickness, 1e-12)
        dead_zone = self.config.dead_zone_h * self.config.h

        def eta_fn(points):
            return cosine_eta(points, box, ramp_width, dead_zone)

        return eta_fn

    # =====================================================================
    # Restart support
    # =====================================================================

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
