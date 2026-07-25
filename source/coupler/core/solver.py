"""FVM–VPM coupling driver with donor boundaries and conservative hand-off."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import TYPE_CHECKING

try:
    from mpi4py import MPI as _MPI

    _mpi4py_comm = _MPI.COMM_WORLD
except ImportError:
    _mpi4py_comm = None

import numpy as np

from source.coupler.config.types import CouplerSetup
from source.coupler.core.helpers.continuous_overlap import (
    ContinuousOverlapInjector,
    cosine_eta,
)
from source.coupler.core.helpers.fvm_velocity_blend import FVMVelocityBlend
from source.coupler.core.helpers.output_redirector import OutputRedirector
from source.coupler.core.helpers.setup import SetupHandler
from source.coupler.core.strategy import create_coupling_strategy

if TYPE_CHECKING:
    # OFW is a Linux/OpenFOAM extension and must remain a type-only import.
    from source.solvers.OFW.fvm_solver import fvm_solver
    from source.solvers.VPM import Solver as VPM_Solver

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


def flush_log():
    for handler in logger.handlers:
        handler.flush()


def _vpm_solver_info(vpm_solver) -> str:
    from source.solvers.VPM.io.logging import Logging

    return Logging.solver_info(vpm_solver)


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
        """Build a coupler from externally configured FVM and VPM solvers.

        The FVM solver is required on every rank. The VPM solver is required
        only on rank zero.
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
        self.coupling_strategy = create_coupling_strategy(coupler_setup.coupling_strategy)
        self._backend = coupler_setup.backend
        self.case_dir = Path(coupler_setup.case_dir).expanduser().absolute()

        # Injected sub-solvers.  The VPM may be None on non-master ranks.
        self._injected_fvm = fvm_solver
        self._injected_vpm = vpm_solver

        self._mpi_rank = _world_rank()
        self._is_master = self._mpi_rank == 0

        self.solution_dir = self.case_dir / "solution"
        if self._is_master:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            self._configure_logging()

        if self._is_master:
            self.vpm_redirector = OutputRedirector(
                logfile=str(self.solution_dir / "vpm.log"), append=True
            )
            if self._backend == "fvm":
                # The native solver owns solution/fvm.log through its rank-aware
                # logger. Redirecting stdout here would duplicate every line.
                self.ofw_redirector = OutputRedirector()
            else:
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
        self._last_omega_donor: np.ndarray | None = (
            None  # ω at faces from last _donor_velocity call
        )
        self._omega_global_buffer: np.ndarray | None = None
        self._bc_omega_tree = None  # lazy cKDTree on injector._cell_centers (mixed BC)
        self.picard_residual_history: list[dict[str, float | int | None]] = []
        self._last_donor_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self.coupling_diagnostics: list[dict] = []

        # ── Multi-rate time stepping (FVM sub-cycling) ───────────────────────
        # The FVM sub-step is OWNED by the injected FVM solver (native backend:
        # FVMSetup.time.delta_t); ``coupler_setup.dt`` is required only for
        # the case-writing OFW backend and is cross-checked otherwise.  The
        # authoritative values are resolved in initialize().
        self.dt_fvm = None if coupler_setup.dt is None else float(coupler_setup.dt)
        self.t_end = None if coupler_setup.t_end is None else float(coupler_setup.t_end)
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

            vpm = setup_vpm_solver(VPMSetup(...))
            FVMVPMCoupler.prepare_case(coupler_setup, vpm_solver=vpm)  # all ranks
            fvm = fvm_solver(case_dir)                                 # all ranks
            coupler = FVMVPMCoupler.from_solvers(
                coupler_setup, fvm_solver=fvm, vpm_solver=vpm)
            coupler.run()
        """
        return cls(vpm_solver, fvm_solver, coupler_setup)

    @staticmethod
    def _validate_injected_vpm(vpm, cfg: CouplerSetup, box, nu: float | None) -> None:
        """Fail-fast consistency checks between an injected VPM and the coupling
        config — the safety net that makes dependency injection safe without
        restricting which native VPM options the caller sets.

        Hard errors on show-stoppers: VPM removal domain must contain the FVM
        box (else injected near-body particles get culled every step),
        mismatched molecular viscosity, and a VPM background velocity that
        disagrees with the interface freestream (incompatible reference
        frames).  Warns on a mismatched hand-off radius.
        """
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
                float(regen),
                float(cfg.overlap_radius_ratio),
                float(cfg.overlap_radius_ratio),
            )
        # Regen threshold reference: the particle-population control must not
        # threshold the wake against a global |Γ| statistic.  A coupled field
        # spans ~4 decades — the maximum sits in the wall vortex sheet on the
        # body while the wake one body-length downstream is ~10⁻³ of it — so
        # every global-reference mode cuts the field along one iso-|Γ| surface
        # and deletes the far wake to keep the boundary layer.  Measured on the
        # cubeFlow hybrid case, one GBD regen at relative_max=5e-3 removed 100%
        # of the particles below |ω| = 0.243 s⁻¹ (67% of the cloud); 'budget'
        # at 1e-2 removed 100% below |ω| = 0.03 s⁻¹.  Because the reference
        # lives on the body, the cut level in the wake also jitters with the
        # near-wall solution and feeds back through the donor BC.
        scheme = str(getattr(vsc, "scheme", "") or "").upper()
        mode_attr = {"DVH": "dvh_threshold_mode", "GBD": "gbd_threshold_mode"}.get(scheme)
        if mode_attr is not None:
            mode = getattr(vsc, mode_attr, None)
            if mode in ("relative_max", "budget", "absolute"):
                logger.warning(
                    "[Init] Injected VPM uses %s='%s', which thresholds particle "
                    "regeneration against a GLOBAL |Γ| reference. In a coupled run "
                    "the global maximum is the body's wall vortex sheet, so the far "
                    "wake is pruned along an iso-|Γ| surface — real vortical "
                    "structures are cut into fragments and cannot be passed outward. "
                    "Use ViscousConfig.%s(threshold_mode='relative_local') so each "
                    "node is referenced to its own neighbourhood.",
                    mode_attr,
                    mode,
                    scheme.lower(),
                )
        dom = getattr(vpm, "vpm_domain_bounds", None)
        if dom is None:
            dom = getattr(getattr(vpm, "config", None), "vpm_domain_bounds", None)
        if box is not None and dom is not None and len(dom) == 6:
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
                    "Widen vpm_domain_bounds (or the VPM stabilization "
                    "remove_particles_by_bounds) to enclose fvm_box."
                )
        vpm_nu = getattr(getattr(getattr(vpm, "config", None), "viscous", None), "viscosity", None)
        if nu is not None and vpm_nu is not None and abs(float(vpm_nu) - float(nu)) > 1e-12:
            raise ValueError(
                f"Incompatible kinematic viscosity: VPM viscous.viscosity="
                f"{float(vpm_nu):g} but the Eulerian solver uses nu={float(nu):g}. "
                "The two solvers must model the same fluid."
            )
        bg = getattr(vpm, "background_velocity", None)
        if bg is not None:
            bg = np.asarray(bg, dtype=np.float64).reshape(-1)
            if bg.size == 3 and not np.allclose(bg, np.asarray(cfg.u_inf), atol=1e-9):
                raise ValueError(
                    f"Incompatible freestream: VPM background_velocity {tuple(bg)} "
                    f"!= coupling u_inf {tuple(cfg.u_inf)}. The donor far-field and "
                    "the VPM advection frame must agree."
                )

    @staticmethod
    def is_master_rank() -> bool:
        """True on the rank that owns the GPU VPM and all IO (rank 0).

        Lets an injection setup script build the VPM on the master only without
        hard-coding the ``OMPI_COMM_WORLD_RANK`` lookup."""
        return _world_rank() == 0

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
        both solvers are attached.

        With ``coupler_setup.backend == "fvm"`` (native Python FVM) there is
        no OpenFOAM case to prepare: the backend is configured
        programmatically (see ``helpers/fvm_backend.build_fvm_backend``), so
        this only creates the ``solution/`` directory."""
        if _world_rank() != 0:
            return
        if coupler_setup.backend == "fvm":
            solution = Path(coupler_setup.case_dir).absolute() / "solution"
            solution.mkdir(parents=True, exist_ok=True)
            if restart and not (solution / "coupled_checkpoint" / "manifest.json").is_file():
                raise FileNotFoundError(
                    "Native FVM restart requested, but no coupled checkpoint exists at "
                    f"{solution / 'coupled_checkpoint'}"
                )
            return
        coupler_setup.require_case_fields()
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
        """Send diagnostics to this case's log and, by default, the console."""
        log_path = (self.solution_dir / "coupler.log").resolve()
        for handler in list(logger.handlers):
            if getattr(handler, "_openonda_case_log", False):
                logger.removeHandler(handler)
                handler.close()

        has_external_handlers = bool(logger.handlers)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_h = logging.FileHandler(log_path, mode="w")
        file_h._openonda_case_log = True
        file_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(file_h)

        if not has_external_handlers:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console_h)

    def _write_run_metadata(self) -> None:
        metadata = {
            "generated_utc": datetime.now(UTC).isoformat(),
            "case_dir": str(self.case_dir),
            **self.config.to_dict(),
            "dt_fvm": float(self.dt_fvm),
            "dt_vpm": float(self.dt_vpm),
            "injection_period": int(self.period_multiplier),
            "backup_period": self.config.backup_period,
            "log_period": self.config.log_period,
            "period_multiplier": float(self.period_multiplier),
        }
        out_path = self.solution_dir / "run_metadata.json"
        out_path.write_text(json.dumps(metadata, indent=2))

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

    @staticmethod
    def _derive_coupling_step_count(t_end: float, dt_vpm: float) -> int:
        """Return an exact integer number of VPM/coupling intervals.

        ``int(t_end / dt)`` silently loses a final interval for ordinary
        decimal values such as ``0.15 / 0.05``.  The coupled FVM and VPM must
        finish at the same requested time, so accept roundoff around an
        integer and reject a genuinely incompatible horizon.
        """
        if dt_vpm <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {dt_vpm!r}.")
        if t_end < 0.0:
            raise ValueError(f"Coupling end time must be non-negative, got {t_end!r}.")
        ratio = t_end / dt_vpm
        n_steps = int(round(ratio))
        if not np.isclose(ratio, n_steps, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "The coupling end time must be an integer multiple of the VPM time "
                f"step. Got t_end={t_end:.12g}, dt_vpm={dt_vpm:.12g}, ratio={ratio:.12g}."
            )
        return n_steps

    def _derive_fvm_box(self) -> np.ndarray:
        """Bounds of the coupling patch, from the injected solver's geometry.

        The patch faces lie exactly on the six box planes, so the per-axis
        min/max of the face centroids reproduce the box bounds to round-off.
        Collective (all ranks) — the face-geometry getter gathers globally.
        """
        fc = np.asarray(
            self.ofw.get_boundary_face_center_coordinates(self.config.patch_name),
            dtype=np.float64,
        ).reshape(-1, 3)
        box = None
        error = None
        if self._is_master:
            if fc.shape[0] == 0:
                error = (
                    f"Coupling patch {self.config.patch_name!r} has no faces on the "
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
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            error, box = _mpi4py_comm.bcast((error, box) if self._is_master else None, root=0)
        if error is not None:
            raise ValueError(error)
        return np.asarray(box, dtype=np.float64)

    def _resolve_eulerian_ownership(self) -> None:
        """Resolve dt / t_end / nu / fvm_box between the coupling setup and
        the injected Eulerian solver.

        Native ``fvm`` backend: the injected solver's :class:`FVMSetup` OWNS
        these values.  CouplerSetup entries are optional cross-checks — a set
        value that contradicts the solver raises (nothing is silently
        overwritten), an unset one is filled from the solver so downstream
        coupling components keep a single consistent view.

        OFW backend: the OpenFOAM case was written from CouplerSetup
        (``prepare_case``), so the setup remains the source and the runtime
        setters stamp the wrapper exactly as the case dictionaries say.
        """
        cfg = self.config
        if self._backend == "fvm":
            fvm_cfg = self.ofw.config
            owned = {
                "dt": float(fvm_cfg.time.delta_t),
                "t_end": float(fvm_cfg.time.end_time),
                "nu": float(fvm_cfg.transport.nu),
            }
            for name, mine in (("dt", self.dt_fvm), ("t_end", self.t_end), ("nu", cfg.nu)):
                theirs = owned[name]
                if mine is not None and abs(float(mine) - theirs) > 1e-12 * max(abs(theirs), 1.0):
                    raise ValueError(
                        f"CouplerSetup.{name}={mine!r} contradicts the injected FVM "
                        f"solver's {name}={theirs!r}. The FVM solver owns this value; "
                        "leave the CouplerSetup field unset (None)."
                    )
            self.dt_fvm = owned["dt"]
            self.t_end = owned["t_end"]
            if cfg.nu is None:
                cfg.nu = owned["nu"]  # coupling components read cfg.nu
            box = self._derive_fvm_box()
            if cfg.fvm_box is not None and not np.allclose(
                np.asarray(cfg.fvm_box, dtype=np.float64), box, atol=1e-9
            ):
                raise ValueError(
                    f"CouplerSetup.fvm_box={tuple(cfg.fvm_box)} contradicts the "
                    f"injected solver's coupling-patch bounds {tuple(box)}. Leave "
                    "fvm_box unset (None); it is derived from the mesh."
                )
            if cfg.fvm_box is None:
                cfg.fvm_box = tuple(float(v) for v in box)
        else:
            cfg.require_case_fields(("nu", "dt", "t_end", "fvm_box"))
            self.ofw.set_time_step(self.dt_fvm)
            self.ofw.set_kinematic_viscosity(cfg.nu)

    def initialize(self) -> None:
        """Adopt the injected solvers, derive sub-cycling, and build coupling
        components.

        The injected FVM solver's configuration owns the FVM step (native
        backend); the injected VPM solver's ``time_step_size`` configures the
        coupling/VPM step.  After both are known, the coupler derives
        ``period_multiplier = round(dt_vpm / dt_fvm)`` internally.

        Idempotent: a second call is a no-op (the coupling components are built
        exactly once), so ``initialize`` then ``run``/``solve`` is safe.
        """
        if self.injector is not None:
            return  # already initialized
        cfg = self.config

        # Adopt the injected FVM (all ranks), built by the caller.  Ownership:
        # the FVM solver owns its physics/time configuration; the coupler
        # resolves the values it needs from it (native backend) or stamps the
        # wrapper from the case-writing setup (OFW backend).
        self.ofw = self._injected_fvm

        # Serial-backend guard: a serial Eulerian backend under mpirun would
        # leave every non-master rank solving its own detached copy (or hang
        # at the first collective) — fail loudly instead.
        world_size = 1
        if _mpi4py_comm is not None:
            world_size = int(_mpi4py_comm.Get_size())
        else:
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size > 1 and int(self.ofw.n_procs()) == 1:
            raise RuntimeError(
                f"Launched under MPI (world size {world_size}) but the injected "
                "Eulerian backend is serial (n_procs() == 1). Configure a parallel "
                "backend or launch one process."
            )

        self._resolve_eulerian_ownership()

        # Adopt the injected VPM (None on non-master).  Validation checks it is
        # mutually consistent with the resolved Eulerian values (domain ⊇ box,
        # viscosity, freestream frame, hand-off radius) — the VPM is already
        # built, so the coupler never silently fixes a mismatch; it raises.
        self.vpm = self._injected_vpm if self._is_master else None
        if self._is_master:
            if self.vpm is None:
                raise ValueError(
                    "from_solvers: vpm_solver is None on the master rank. "
                    "Build the VPM on the master (FVMVPMCoupler.is_master_rank())."
                )
            self._validate_injected_vpm(self.vpm, cfg, cfg.fvm_box, cfg.nu)
            logger.info("[Init] Using injected VPM solver.")

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
                "[Init] Time steps: dt_fvm=%.4e s, dt_vpm=%.4e s, period_multiplier=%d.",
                self.dt_fvm,
                self.dt_vpm,
                self.period_multiplier,
            )
            if self._backend == "ofw":
                # The native FVM backend has no controlDict; its write cadence
                # is configured programmatically (helpers/fvm_backend.py).
                SetupHandler(cfg).update_controldict(
                    self.period_multiplier, self.dt_fvm, restart=False
                )

        # Build injector
        self.injector = ContinuousOverlapInjector(self)
        self.injector.setup(self.ofw)
        if (
            self._is_master
            and self.injector._body_bounds is not None
            and hasattr(self.vpm.physics, "configure_body_box")
        ):
            self.vpm.physics.configure_body_box(self.injector._body_bounds)
            logger.info(
                "[Init] VPM grid diffusion body mask enabled for box %s.",
                self.injector._body_bounds.tolist(),
            )
        if (
            self._is_master
            and self.injector._cell_centers is not None
            and len(self.injector._cell_centers) > 0
            and hasattr(self.vpm.physics, "configure_grid_lattice_anchor")
        ):
            self.vpm.physics.configure_grid_lattice_anchor(
                self.injector._cell_centers[0], self.config.h
            )
            logger.info("[Init] VPM diffusion lattice phase aligned with FVM cell centres.")

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

        self.fringe = FringeFields(cfg, self.vpm, self.ofw, coupling_dt=self.dt_vpm)

        if self._is_master:
            logger.info("[Init] Impulsive start: zero VPM particles.")
            with self.vpm_redirector:
                print(_vpm_solver_info(self.vpm))
                sys.stdout.flush()
            self._write_run_metadata()
            print("Initialization complete.\n")

        self.coupling_strategy.initialize(self)

    def run(self, start_step: int = 0, restart_from=None) -> None:
        """Initialize (if not already) and run the coupling loop.

        Convenience wrapper = :meth:`initialize` + :meth:`solve`.  Callers who
        want the explicit build→initialize→solve flow can call those two
        directly; ``initialize`` is idempotent so ``run`` after a manual
        ``initialize`` will not rebuild.
        """
        if self.injector is None:  # not yet initialized
            self.initialize()
        if restart_from is not None:
            if start_step:
                raise ValueError("start_step and restart_from are mutually exclusive")
            start_step = self.load_state(restart_from)
        self.solve(start_step=start_step)

    # =========================================================
    # Coupling strategy hooks
    # =========================================================

    def solve(self, start_step: int = 0) -> None:
        """Run the selected coupling strategy after solver initialization."""
        self.coupling_strategy.solve(self, start_step=start_step)

    def _strategy_initialize(self) -> None:
        """Initialize the current strategy without changing solver state."""
        self._strategy_step_transfer_stats: dict[str, float | int] | None = None
        self.coupling_diagnostics = []

    def _strategy_prepare_run(self, start_step: int):
        """Validate a run and collect the immutable interface geometry."""
        if self.injector is None:
            raise RuntimeError(
                "solve() called before initialize(); call coupler.initialize() "
                "first, or use coupler.run() which does both."
            )
        assert self._is_master == (self.vpm is not None)
        assert self.ofw is not None

        n_steps = self._derive_coupling_step_count(self.t_end, self.dt)
        patch = self.config.patch_name
        if self._is_master:
            logger.info("=" * 60)
            logger.info("FVM-VPM COUPLED SOLVER")
            logger.info("=" * 60)

        face_centers = np.asarray(
            self.ofw.get_boundary_face_center_coordinates(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_normals = np.asarray(
            self.ofw.get_boundary_face_normals(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_areas = np.asarray(self.ofw.get_boundary_face_areas(patch), dtype=np.float64).ravel()
        self._strategy_face_centers = face_centers
        self._strategy_n_steps = n_steps
        return (face_centers, face_normals, face_areas), n_steps

    def _strategy_advance_vpm(self, step: int, time_end: float) -> float:
        """Advance VPM exactly as the pre-strategy loop did."""
        t0 = time.time()
        if self._is_master:
            with self.vpm_redirector:
                self.vpm.set_background_velocity(self.config.u_inf)
            print()
            print("─" * 60)
            print(f"STEP {step}/{self._strategy_n_steps}  (t={time_end:.3f}s)")

            if self.vel_blend is not None:
                self.vel_blend.begin_diagnostics()
            with _DisableSIGFPE(), self.vpm_redirector:
                self.vpm.update_state()
            import taichi as ti

            ti.sync()
            if self.vel_blend is not None and self.vpm.particles.number_of_particles > 0:
                pos_np = np.asarray(self.vpm.particles_positions, dtype=np.float64).reshape(-1, 3)
                vel_np = np.asarray(self.vpm.particles_velocities, dtype=np.float64).reshape(-1, 3)
                self.vel_blend.log_diagnostics(pos_np, vel_np, float(np.linalg.norm(self.u_inf)))
        return time.time() - t0

    def _strategy_transfer_vpm_to_fvm(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
    ):
        """Update fringe data and construct the next donor boundary trace."""
        t_fringe = time.time()
        self.fringe.update_target()
        t_fringe = time.time() - t_fringe

        t_donor = time.time()
        if self._is_master:
            if self.config.donor_interior_source == "fvm":
                u_bc_next = self._donor_exterior_velocity(face_centers)
            else:
                u_bc_next = self._donor_velocity(face_centers, face_normals, face_areas)
            omega_bc_next = self._last_omega_donor
            if self._u_bc_prev is None:
                self._u_bc_prev = u_bc_next.copy()
            if self._omega_bc_prev is None:
                self._omega_bc_prev = omega_bc_next.copy() if omega_bc_next is not None else None
        else:
            u_bc_next = np.zeros_like(face_centers)
            mixed = self.config.donor_bc_mode == "mixed"
            omega_bc_next = np.zeros_like(face_centers) if mixed else None
            if self._u_bc_prev is None:
                self._u_bc_prev = np.zeros_like(face_centers)
            if self._omega_bc_prev is None and mixed:
                self._omega_bc_prev = np.zeros_like(face_centers)
        t_donor = time.time() - t_donor
        return (
            self._u_bc_prev,
            u_bc_next,
            self._omega_bc_prev,
            omega_bc_next,
            t_fringe,
            t_donor,
        )

    def _strategy_advance_fvm(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        u_bc_prev: np.ndarray,
        u_bc_next: np.ndarray,
        omega_bc_prev: np.ndarray | None,
        omega_bc_next: np.ndarray | None,
        _fringe_time: float,
        _donor_time: float,
    ) -> float:
        """Run FVM sub-cycles and refresh its velocity snapshot."""
        t_fvm = time.time()
        self._run_fvm_substeps(
            self.config.patch_name,
            face_centers,
            face_normals,
            face_areas,
            u_bc_prev,
            u_bc_next,
            omega_bc_prev,
            omega_bc_next,
        )
        if self._is_master or self.config.donor_bc_mode == "mixed":
            self._u_bc_prev = u_bc_next
            self._omega_bc_prev = omega_bc_next
        if self.vel_blend is not None:
            self.vel_blend.update()
        return time.time() - t_fvm

    def _strategy_transfer_fvm_to_vpm(
        self,
        face_centers: np.ndarray,
        _face_normals: np.ndarray,
        _face_areas: np.ndarray,
    ):
        """Fetch FVM vorticity collectively, then perform the conservative hand-off."""
        t_handoff = time.time()
        omega_global = self._get_vorticity_field_buffer()
        handoff_result = None
        if self._is_master:
            eta_fn = self._build_eta_fn()
            n_before = self.vpm.particles.number_of_particles
            sum_before = (
                float(np.sum(np.linalg.norm(np.asarray(self.vpm.particles_circulation), axis=1)))
                if n_before > 0
                else 0.0
            )
            handoff_result = self.injector.inject(
                self.ofw, self.vpm, eta_fn=eta_fn, omega=omega_global
            )
            n_after = self.vpm.particles.number_of_particles
            sum_after = (
                float(np.sum(np.linalg.norm(np.asarray(self.vpm.particles_circulation), axis=1)))
                if n_after > 0
                else 0.0
            )
            self._strategy_step_transfer_stats = {
                "n_before": n_before,
                "n_after": n_after,
                "sum_before": sum_before,
                "sum_after": sum_after,
                "face_count": len(face_centers),
            }
        return handoff_result, time.time() - t_handoff

    def _strategy_apply_conservation_correction(self, handoff_result) -> None:
        """The existing injector already applies its conservative correction."""
        self._last_handoff_result = handoff_result

    def _strategy_compute_diagnostics(
        self,
        step: int,
        time_end: float,
        timing: tuple[float, float, float, float, float],
        _handoff_result,
    ) -> None:
        """Emit the legacy diagnostics and synchronize a completed coupling step."""
        t_vpm, t_fringe, t_donor, t_fvm, t_handoff = timing
        diagnostics = self.compute_diagnostics(_handoff_result)
        if self._is_master:
            diagnostics.update({"step": int(step), "time": float(time_end)})
            self.coupling_diagnostics.append(diagnostics)
        if self._is_master:
            stats = self._strategy_step_transfer_stats or {}
            logger.info(
                "     [Inject] N_before=%d  N_after=%d  |Γ|_before=%.4e  |Γ|_after=%.4e",
                int(stats.get("n_before", 0)),
                int(stats.get("n_after", 0)),
                float(stats.get("sum_before", 0.0)),
                float(stats.get("sum_after", 0.0)),
            )
            if self.config.donor_interior_source == "fvm":
                self._u_bc_prev = self._donor_exterior_velocity(self._strategy_face_centers)
                if self.config.donor_bc_mode == "mixed":
                    self._omega_bc_prev = self._last_omega_donor
            print()
            print(f"[Step {step:4d}] t={time_end:.3f}s | Particles: {int(stats.get('n_after', 0))}")
            print(
                f"     Timing: VPM={t_vpm:.2f}s | BC={t_donor:.2f}s | Fringe={t_fringe:.2f}s | "
                f"FVM={t_fvm:.2f}s | Inject={t_handoff:.2f}s"
            )
            sys.stdout.flush()
            flush_log()

        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            _mpi4py_comm.Barrier()
        if self.config.backup_period > 0 and step % self.config.backup_period == 0:
            self.save_state(self.solution_dir / "coupled_checkpoint", coupling_step=step)

    def _strategy_finalize_run(self) -> None:
        if self._is_master:
            flush_log()

    # =========================================================
    # Neumann vorticity helper (mixed / Robin BC)
    # =========================================================

    def _fvm_face_vorticity(self, face_centers: np.ndarray) -> np.ndarray | None:
        """Nearest-cell FVM vorticity for the mixed boundary condition.

        Returns ``None`` until an FVM vorticity buffer is available.
        """
        if self._omega_global_buffer is None:
            return None  # first step: no FVM solve has occurred yet

        assert self.injector is not None
        centers = self.injector._cell_centers
        if centers is None or centers.shape[0] == 0:
            return None

        if self._bc_omega_tree is None:
            from scipy.spatial import cKDTree

            self._bc_omega_tree = cKDTree(np.asarray(centers, dtype=np.float64))

        _, idx = self._bc_omega_tree.query(
            np.ascontiguousarray(face_centers, dtype=np.float64), k=1
        )
        omega = np.ascontiguousarray(self._omega_global_buffer[idx], dtype=np.float64)
        logger.info(
            "     [Donor] Neumann ω from FVM nearest-cell  n_face=%d  |ω|_max=%.3e s⁻¹",
            len(face_centers),
            float(np.max(np.linalg.norm(omega, axis=1))) if len(omega) else 0.0,
        )
        return omega

    def _donor_velocity(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        exterior_mask: np.ndarray | None = None,
        add_fvm_interior: bool = False,
        fvm_omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the donor velocity and remove its discrete boundary flux.

        ``exterior_mask`` selects particle sources. ``add_fvm_interior`` adds
        induction from the current FVM vorticity field.
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
                self._last_donor_flux_diagnostics = {
                    "raw_mismatch": 0.0,
                    "applied_correction": 0.0,
                    "corrected_mismatch": 0.0,
                }
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
                    "     [Donor] exterior particles=%d/%d + FVM-interior BS (coupled BC)",
                    int(exterior_mask.sum()),
                    n_particles,
                )

            u_donor = self.vpm.compute_target_velocities(
                face_centers,
                include_freestream=True,
                zone_mask=exterior_mask,  # None means all particles.
            )
            # Prefer the FVM vorticity for the mixed condition after startup.
            _omega_fvm = self._fvm_face_vorticity(face_centers)
            self._last_omega_donor = (
                _omega_fvm
                if _omega_fvm is not None
                else np.asarray(self.vpm.compute_target_vorticities(face_centers), dtype=np.float64)
            )
            if add_fvm_interior:
                u_donor = u_donor + self._fvm_interior_induced_velocity(
                    face_centers, omega=fvm_omega
                )

        # A uniform normal correction removes the quadrature flux residual.
        normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
        areas = np.asarray(face_areas, dtype=np.float64).ravel()
        flux_residual_raw = 0.0
        if len(areas) > 0:
            u_normal = np.einsum("ij,ij->i", u_donor, normals)
            flux_residual_raw = float(np.dot(u_normal, areas))
            total_area = float(np.sum(areas))

            if total_area > 0.0:
                delta_u_n = flux_residual_raw / total_area  # scalar [m/s]
                u_donor = u_donor - delta_u_n * normals

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
            self._last_donor_flux_diagnostics = {
                "raw_mismatch": float(abs(flux_residual_raw)),
                "applied_correction": float(abs(delta_u_n if total_area > 0.0 else 0.0)),
                "corrected_mismatch": float(abs(flux_residual_post)),
            }

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
        u_stream = u_field[mask] @ (np.asarray(self.u_inf) / u_mag)
        logger.info(
            "     [Donor deficit outflow axis=%d sign=%+d] u_s/U∞ min=%.3f "
            "mean=%.3f max=%.3f  n_face=%d",
            axis,
            int(sign),
            u_stream.min() / u_mag,
            u_stream.mean() / u_mag,
            u_stream.max() / u_mag,
            int(mask.sum()),
        )

    def _donor_exterior_velocity(self, face_centers: np.ndarray) -> np.ndarray:
        """Return freestream plus induction from particles outside the FVM box.

        Live FVM-interior induction and flux projection are added per substep.
        Mixed mode obtains its Neumann target from the particle field.
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
        if self.config.donor_bc_mode == "mixed":
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

    def compute_diagnostics(self, handoff_result=None) -> dict:
        """Return finite, structured transfer/correction diagnostics.

        The method is intentionally side-effect free.  Strategies decide when
        to persist its result, which keeps diagnostics available to tests and
        alternative coupling algorithms without tying them to log parsing.
        """
        result = (
            handoff_result
            if handoff_result is not None
            else getattr(self, "_last_handoff_result", None)
        )
        zero = {
            "circulation": 0.0,
            "linear_impulse": 0.0,
            "angular_impulse": 0.0,
        }

        def _finite(values):
            out = dict(zero)
            if values:
                out.update({key: float(value) for key, value in values.items()})
            if not all(np.isfinite(value) for value in out.values()):
                raise FloatingPointError("non-finite conservation diagnostic")
            return out

        conservation = {
            "raw_mismatch": _finite(getattr(result, "conservation_raw_mismatch", None)),
            "applied_correction": _finite(getattr(result, "conservation_applied_correction", None)),
            "corrected_mismatch": _finite(getattr(result, "conservation_corrected_mismatch", None)),
        }
        donor = {
            key: float(getattr(self, "_last_donor_flux_diagnostics", {}).get(key, 0.0))
            for key in ("raw_mismatch", "applied_correction", "corrected_mismatch")
        }
        if not all(np.isfinite(value) for value in donor.values()):
            raise FloatingPointError("non-finite donor-flux diagnostic")
        return {
            "conservation": conservation,
            "donor_flux": donor,
            "period_multiplier": int(self.period_multiplier),
            "handoff_particle_count": int(getattr(result, "n_total", 0)),
        }

    def _outside_box_mask(self) -> np.ndarray:
        """Return a mask of VPM particles outside the FVM box."""
        assert self.vpm is not None
        n = self.vpm.particles.number_of_particles
        pos = np.asarray(self.vpm.particles_positions, dtype=np.float64).reshape(-1, 3)[:n]
        x0, x1, y0, y1, z0, z1 = self.config.fvm_box
        inside = (
            (pos[:, 0] >= x0)
            & (pos[:, 0] <= x1)
            & (pos[:, 1] >= y0)
            & (pos[:, 1] <= y1)
            & (pos[:, 2] >= z0)
            & (pos[:, 2] <= z1)
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
        """Evaluate velocity induced by the current FVM vorticity field."""
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

        src, gamma, info = interior_bs.select_sources(
            omega,
            centers,
            vols,
            pool_h=2.0 * float(self.config.h),
            budget_fraction=0.999,
        )
        if src.shape[0] == 0:
            return np.zeros_like(targets)
        logger.info(
            "     [Donor] FVM-interior BS sources: %d exact + %d pooled bins "
            "(%d selected by %s; %.1f%% of Σ|Γ| represented)",
            src.shape[0] - info["n_pooled_bins"],
            info["n_pooled_bins"],
            info["n_above"],
            info["selection_mode"],
            100.0 * info["kept_fraction"],
        )
        self._validate_bs_sign_once()
        # Use the same Gaussian blob kernel on both sides of the hand-off.
        # The former Rosenhead cutoff changed the velocity operator abruptly
        # when a circulation element crossed the interface, creating a
        # non-physical step in the donor trace even when Gamma was conserved.
        sigma = float(self.config.overlap_radius_ratio) * float(self.config.h)
        theta = float(getattr(self.config, "donor_interior_treecode_theta", 0.0) or 0.0)
        if theta > 0.0:
            # Barnes-Hut with monopole+dipole: the direct sum is
            # O(n_faces * n_sources) and dominates the step once the wake fills
            # the box (measured 19s -> 334s per coupling step on the cube case).
            return interior_bs.gaussian_bs_velocity_treecode(
                targets, src, gamma, radii=sigma, theta=theta
            )
        return interior_bs.gaussian_bs_velocity(targets, src, gamma, radii=sigma)

    def _validate_bs_sign_once(self) -> None:
        """Check the Biot–Savart sign convention once per run."""
        if getattr(self, "_bs_sign_ok", False):
            return
        src = np.array([[0.0, 0.0, 0.0]])
        G = np.array([[0.0, 0.0, 1.0]])
        tgt = np.array([[1.0, 0.0, 0.0]])
        r = tgt[:, None, :] - src[None, :, :]
        r2 = np.einsum("mnk,mnk->mn", r, r) + 1e-6
        cross = np.cross(G[None, :, :], r)
        u = (1.0 / (4.0 * np.pi)) * np.einsum("mn,mnk->mk", r2**-1.5, cross)
        assert u[0, 1] > 0.0 and abs(u[0, 0]) < 1e-9, (
            f"FVM-interior BS sign check failed: u={u[0]} (expected +y swirl)"
        )
        self._bs_sign_ok = True

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

        mode = self.config.donor_bc_mode
        if mode == "mixed":
            if omega_target is None:
                raise RuntimeError(
                    "donor_bc_mode='mixed' requires a vorticity target; refusing to "
                    "replace the configured Robin condition with Dirichlet"
                )
            self.ofw.set_robin_velocity_boundary_condition(
                np.ascontiguousarray(u_target, dtype=np.float64),
                np.ascontiguousarray(omega_target, dtype=np.float64),
                patch,
            )
            bc_label = "mixed (Robin: u_n Dirichlet + ω Neumann)"
        elif mode == "characteristic":
            # Donor Dirichlet only where flow ENTERS the box; convective
            # (owner-extrapolated, upwinded) treatment where it exits.  This
            # breaks the deficit→ring-vorticity→Biot–Savart→deeper-deficit
            # feedback loop of the all-face Dirichlet cut: the outflow state
            # is set by the FVM's own transport, not by the donor's smeared
            # self-image (measured: all-Dirichlet drives the face deficit to
            # min u_x < 0 and blow-up by t≈2 while the monolith stays ≈0.89).
            setter = getattr(self.ofw, "set_freestream_velocity_boundary_condition_vec", None)
            if setter is None:
                raise RuntimeError(
                    "donor_bc_mode='characteristic' requires the Eulerian backend "
                    "to provide set_freestream_velocity_boundary_condition_vec "
                    "(native FVM backend only)."
                )
            setter(np.ascontiguousarray(u_target, dtype=np.float64), patch)
            bc_label = "characteristic (donor on inflow, convective outflow)"
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
        # Mirror the FVM y+ min/max to the coupler diagnostics as a compact
        # cross-solver health signal. The native FVM logger retains the full
        # wall-resolution section in solution/fvm.log.
        if advance:
            yplus = getattr(self.ofw, "last_yplus", None)
            if yplus:
                parts = [
                    f"{name}: y+ min={s['min']:.2f} max={s['max']:.2f} avg={s['avg']:.2f}"
                    for name, s in yplus.items()
                ]
                logger.info("     [FVM wall] %s", " | ".join(parts))

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
        """Advance FVM substeps with interpolated donor boundary data."""
        N = max(1, int(self.period_multiplier))
        n_bc = max(1, int(self.config.bc_coupling_iterations))
        u_inf_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
        mixed = self.config.donor_bc_mode == "mixed"

        if self.config.donor_interior_source == "fvm":
            self._run_fvm_substeps_live_interior(
                patch,
                face_centers,
                face_normals,
                face_areas,
                u_prev,
                u_next,
                omega_prev,
                omega_next,
                N=N,
                n_bc=n_bc,
                mixed=mixed,
                u_inf_mag=u_inf_mag,
            )
            return

        if N > 1 and u_next.shape[0] > 0:
            dU = float(np.max(np.linalg.norm(u_next - u_prev, axis=1))) / u_inf_mag
            big = dU > 0.5
            logger.log(
                logging.WARNING if big else logging.INFO,
                "     [Sub-cycle] %d×dt_fvm=%.3e s  donor ΔBC max|Δu|/U∞=%.3f%s",
                N,
                self.dt_fvm,
                dU,
                "  (large — lower dt or period_multiplier)" if big else "",
            )

        for sub in range(N):
            alpha = (sub + 1) / N
            is_final = sub == N - 1
            if is_final and n_bc > 1:
                # Only reachable with donor_interior_source="particles" (the
                # "fvm" path returns above), so the interior induction must come
                # from the particle cloud — the same whole-cloud donor that
                # produced u_prev/u_next.  Re-deriving this trace as "exterior
                # particles + FVM-interior vorticity" mixed two different models
                # of the same interior field, so the boundary condition jumped
                # between them once every N sub-steps and drove a Cd modulation
                # locked to the coupling cadence (measured 14.4% peak-to-peak,
                # 19x the monolithic reference's step-to-step jitter).
                previous_trace = None
                for k in range(n_bc):
                    if self._is_master:
                        u_wl = self._donor_velocity(face_centers, face_normals, face_areas)
                        omega_wl = self._last_omega_donor
                    else:
                        u_wl = np.zeros_like(face_centers)
                        omega_wl = np.zeros_like(face_centers) if mixed else None
                    residual = (
                        None
                        if previous_trace is None
                        else float(
                            np.linalg.norm(u_wl - previous_trace)
                            / max(np.linalg.norm(u_wl), np.linalg.norm(previous_trace), 1e-30)
                        )
                    )
                    self.picard_residual_history.append(
                        {"substep": sub + 1, "iteration": k + 1, "residual": residual}
                    )
                    previous_trace = u_wl.copy()
                    converged = (
                        residual is not None
                        and getattr(self.config, "bc_coupling_tolerance", None) is not None
                        and residual <= self.config.bc_coupling_tolerance
                    )
                    self._fvm_step(
                        patch,
                        u_wl,
                        advance=(k == n_bc - 1 or converged),
                        omega_target=omega_wl if mixed else None,
                    )
                    if converged:
                        break
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
                    patch,
                    u_bc,
                    advance=True,
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
        """Advance substeps while recomputing FVM-interior induction."""
        if N > 1 and u_ext_next.shape[0] > 0:
            dU = float(np.max(np.linalg.norm(u_ext_next - u_ext_prev, axis=1))) / u_inf_mag
            big = dU > 0.5
            logger.log(
                logging.WARNING if big else logging.INFO,
                "     [Sub-cycle] %d×dt_fvm=%.3e s  exterior donor ΔBC "
                "max|Δu|/U∞=%.3f  (interior term is live)%s",
                N,
                self.dt_fvm,
                dU,
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

            # Donor warmup: while t < donor_interior_warmup_time the interior
            # Biot–Savart term is left out (donor = exterior particles + U∞,
            # which is also what the monolith trace looks like at early times:
            # face min u_x stays ≈ 0.89–0.97 through t=2 in the matched
            # oracle).  The impulsively started wall sheet on a refined mesh
            # otherwise produces a wild trace (±1.2 U∞) whose Picard sequence
            # runs away within the first window.
            warmup = float(getattr(self.config, "donor_interior_warmup_time", 0.0) or 0.0)
            interior_on = self.ofw.flow_time + 1e-12 >= warmup
            # Picard under-relaxation of the imposed trace (standard fixed-
            # point damping; 1.0 recovers the undamped iteration).
            relax = float(getattr(self.config, "donor_bc_relax", 1.0) or 1.0)

            previous_trace = None
            for k in range(n_bc):
                t_w = time.time()
                omega_interior = self._get_vorticity_field_buffer()
                t_w = time.time() - t_w
                t_bs = time.time()
                if self._is_master and u_ext.shape[0] > 0 and interior_on:
                    u_int = self._fvm_interior_induced_velocity(face_centers, omega=omega_interior)
                    u_bc = self._project_to_solenoidal(u_ext + u_int, face_normals, face_areas)
                else:
                    u_bc = u_ext
                if previous_trace is not None and relax < 1.0:
                    u_bc = previous_trace + relax * (u_bc - previous_trace)
                    u_bc = self._project_to_solenoidal(u_bc, face_normals, face_areas)
                t_bs = time.time() - t_bs
                residual = (
                    None
                    if previous_trace is None
                    else float(
                        np.linalg.norm(u_bc - previous_trace)
                        / max(np.linalg.norm(u_bc), np.linalg.norm(previous_trace), 1e-30)
                    )
                )
                self.picard_residual_history.append(
                    {"substep": sub + 1, "iteration": k + 1, "residual": residual}
                )
                previous_trace = u_bc.copy()
                converged = (
                    residual is not None
                    and getattr(self.config, "bc_coupling_tolerance", None) is not None
                    and residual <= self.config.bc_coupling_tolerance
                )
                if self._is_master:
                    logger.info(
                        "     [Sub-cycle] sub=%d/%d it=%d/%d  ω-gather=%.2fs  BS+proj=%.2fs",
                        sub + 1,
                        N,
                        k + 1,
                        n_bc,
                        t_w,
                        t_bs,
                    )
                    if sub == N - 1 and k == n_bc - 1 and u_bc.shape[0] > 0:
                        # Outflow deficit on the fully assembled (exterior +
                        # live interior) trace; direction-agnostic.
                        self._log_outflow_deficit(face_centers, u_bc)
                self._fvm_step(
                    patch,
                    u_bc,
                    advance=(k == n_bc - 1 or converged),
                    omega_target=omega_bc if mixed else None,
                )
                if converged:
                    break

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

    def _config_digest(self) -> str:
        payload = json.dumps(self.config.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save_state(self, directory, *, coupling_step: int | None = None) -> Path:
        """Write a complete native FVM--VPM checkpoint, committing its manifest last."""
        if self._backend != "fvm":
            raise NotImplementedError("Coupled checkpoints currently require backend='fvm'")
        if self.ofw is None:
            raise RuntimeError("Initialize the coupler before saving a checkpoint")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        # The FVM checkpoint is COLLECTIVE under a partitioned backend: every
        # rank writes its own partition piece and joins a closing barrier.  It
        # must run on all ranks, so it precedes the master-only remainder --
        # returning early on the workers here deadlocks rank 0 in that barrier
        # while the workers advance to the next step's barrier.
        self.ofw.save_state(target / "fvm.npz")

        if not self._is_master:
            return target
        if self.vpm is None:
            raise RuntimeError("Initialize the coupler before saving a checkpoint")

        from source.solvers.VPM.io.backup import BackupSystem

        BackupSystem.backup_solver(
            self.vpm,
            str(target / "vpm_latest"),
            time_float32=float(self.vpm.flow_time),
            verbose=False,
        )

        donor_tmp = target / "donor_state.npz.tmp"
        with open(donor_tmp, "wb") as stream:
            np.savez_compressed(
                stream,
                u_present=np.asarray(self._u_bc_prev is not None),
                omega_present=np.asarray(self._omega_bc_prev is not None),
                u=np.empty((0, 3)) if self._u_bc_prev is None else self._u_bc_prev,
                omega=np.empty((0, 3)) if self._omega_bc_prev is None else self._omega_bc_prev,
            )
        os.replace(donor_tmp, target / "donor_state.npz")

        inferred_step = int(self.vpm.time_step)
        step = inferred_step if coupling_step is None else int(coupling_step)
        manifest = {
            "format_version": 1,
            "backend": "fvm",
            "config_sha256": self._config_digest(),
            "coupling_step": step,
            "flow_time": float(self.vpm.flow_time),
            "fvm_time_step": int(self.ofw.time_step),
            "vpm_time_step": int(self.vpm.time_step),
            "period_multiplier": int(self.period_multiplier),
            "files": ["fvm.npz", "vpm_latest.h5", "donor_state.npz"],
        }
        manifest_tmp = target / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, target / "manifest.json")
        return target

    def load_state(self, directory) -> int:
        """Restore both solvers and donor history from a compatible checkpoint."""
        if self._backend != "fvm":
            raise NotImplementedError("Coupled checkpoints currently require backend='fvm'")
        if not self._is_master:
            raise NotImplementedError("Partitioned coupled restart is not qualified")
        if self.ofw is None or self.vpm is None:
            raise RuntimeError("Initialize the coupler before loading a checkpoint")

        target = Path(directory)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("backend") != "fvm":
            raise ValueError("Unsupported coupled checkpoint format or backend")
        if manifest.get("config_sha256") != self._config_digest():
            raise ValueError("Coupled checkpoint configuration does not match this run")
        missing = [name for name in manifest["files"] if not (target / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete coupled checkpoint; missing: {', '.join(missing)}")

        self.ofw.load_state(target / "fvm.npz")
        self.load_vpm_from_backup(str(target / "vpm_latest.h5"))
        # The standalone VPM visualization backup stores time as float32; the
        # coupled manifest retains the authoritative float64 synchronization time.
        self.vpm.flow_time = float(manifest["flow_time"])
        with np.load(target / "donor_state.npz", allow_pickle=False) as donor:
            self._u_bc_prev = donor["u"].copy() if bool(donor["u_present"]) else None
            self._omega_bc_prev = donor["omega"].copy() if bool(donor["omega_present"]) else None

        expected_fvm_step = int(manifest["vpm_time_step"]) * self.period_multiplier
        if self.ofw.time_step != expected_fvm_step:
            raise ValueError(
                f"Coupled checkpoint time-step mismatch: FVM={self.ofw.time_step}, "
                f"expected {expected_fvm_step} from VPM={self.vpm.time_step}"
            )
        if not np.isclose(self.ofw.flow_time, self.vpm.flow_time, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Coupled checkpoint time mismatch: FVM={self.ofw.flow_time}, "
                f"VPM={self.vpm.flow_time}"
            )
        if self.vel_blend is not None:
            self.vel_blend.update()
        return int(manifest["coupling_step"])

    def load_vpm_from_backup(self, backup_h5_path: str) -> int:
        """Load VPM state from an H5 backup."""
        import h5py

        h5_path = backup_h5_path if backup_h5_path.endswith(".h5") else backup_h5_path + ".h5"
        with h5py.File(h5_path, "r") as f:
            flow_time = float(f["solver"].attrs["flow_time"])
            time_step = int(f["solver"].attrs["time_step"])
            p = f["particles"]
            count = int(f["solver"].attrs["number_of_particles"])
            if count:
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
            if count:
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

        n = count
        print(f"[Restart] Loaded {n} particles from backup (t={flow_time:.3f}s, step={time_step})")
        return time_step
