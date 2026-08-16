"""FVM–VPM coupling driver with donor boundaries and conservative hand-off."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
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
from source.coupler.core.helpers.continuous_overlap import ContinuousOverlapInjector
from source.coupler.core.helpers.fvm_fringe import FringeFields
from source.coupler.core.helpers.output_redirector import OutputRedirector

if TYPE_CHECKING:
    from source.solvers.FVM import Solver as FVM_Solver
    from source.solvers.VPM import Solver as VPM_Solver

logger = logging.getLogger("coupler")

CHECKPOINT_DIRECTORY = "checkpoint"
CHECKPOINT_FORMAT_VERSION = 3


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
        else:
            self.vpm_redirector = OutputRedirector()  # no-op

        # Solvers (built in initialize())
        self.vpm: VPM_Solver | None = None
        self.fvm: FVM_Solver | None = None
        self.injector: ContinuousOverlapInjector | None = None
        self.fringe = None
        self._u_bc_prev: np.ndarray | None = None
        self._pressure_gradient_bc_prev: np.ndarray | None = None
        self._pressure_gradient_bc_next: np.ndarray | None = None
        self._pressure_velocity_snapshot: np.ndarray | None = None
        self._velocity_global_buffer: np.ndarray | None = None
        self._velocity_gradient_global_buffer: np.ndarray | None = None
        self._last_donor_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self.coupling_diagnostics: list[dict] = []

        # ── Multi-rate time stepping (FVM sub-cycling) ───────────────────────
        # The FVM sub-step is owned by the injected FVM solver
        # (FVMSetup.time.delta_t); optional CouplerSetup values are
        # cross-checks. The authoritative values are resolved in initialize().
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
        """Validate the injected VPM against the coupling discretization."""
        vsc = getattr(getattr(vpm, "config", None), "viscous", None)
        scheme = str(getattr(vsc, "scheme", "") or "").upper()
        regen = getattr(vsc, "regen_radius_ratio", None)
        if (
            scheme in {"DVH", "GBD"}
            and regen is not None
            and abs(float(regen) - float(cfg.overlap_radius_ratio)) > 1e-9
        ):
            raise ValueError("VPM regen_radius_ratio must match the coupler overlap_radius_ratio")
        mode_attr = {"DVH": "dvh_threshold_mode", "GBD": "gbd_threshold_mode"}.get(scheme)
        if mode_attr is not None:
            mode = getattr(vsc, mode_attr, None)
            if mode in ("relative_max", "budget", "absolute"):
                raise ValueError(f"{scheme} coupling requires threshold_mode='relative_local'")
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
        """Prepare the native coupled solution directory and restart guard.

        The native solver is configured programmatically before it is injected;
        this helper remains as the common all-ranks pre-construction hook used
        by tutorials and applications.
        """
        if _world_rank() != 0:
            return
        solution = Path(coupler_setup.case_dir).absolute() / "solution"
        solution.mkdir(parents=True, exist_ok=True)
        checkpoint = solution / CHECKPOINT_DIRECTORY
        if restart and not (checkpoint / "manifest.json").is_file():
            raise FileNotFoundError(
                "Native FVM restart requested, but no coupled checkpoint exists at "
                f"{solution / CHECKPOINT_DIRECTORY}"
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
        """Return the number of VPM/coupling intervals for a given end time.

        The end time need not be an exact multiple of the VPM step size; the
        count is rounded to the nearest integer, landing on the closest
        coupling-step boundary.

        Args:
            t_end:  Requested simulation end time.
            dt_vpm: VPM (coupling) time-step size.

        Returns:
            Integer number of coupling steps.
        """
        if dt_vpm <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {dt_vpm!r}.")
        if t_end < 0.0:
            raise ValueError(f"Coupling end time must be non-negative, got {t_end!r}.")
        return max(0, int(round(t_end / dt_vpm)))

    def _derive_fvm_box(self) -> np.ndarray:
        """Bounds of the coupling patch, from the injected solver's geometry.

        The patch faces lie exactly on the six box planes, so the per-axis
        min/max of the face centroids reproduce the box bounds to round-off.
        Collective (all ranks) — the face-geometry getter gathers globally.
        """
        assert self.fvm is not None
        fc = np.asarray(
            self.fvm.get_boundary_face_center_coordinates(self.config.patch_name),
            dtype=np.float64,
        ).reshape(-1, 3)
        box = None
        error = None
        collective = _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1
        if self._is_master or not collective:
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
        if collective:
            error, box = _mpi4py_comm.bcast((error, box) if self._is_master else None, root=0)
        if error is not None:
            raise ValueError(error)
        return np.asarray(box, dtype=np.float64)

    def _resolve_eulerian_ownership(self) -> None:
        """Resolve dt / t_end / nu / fvm_box between the coupling setup and
        the injected Eulerian solver.

        The injected solver's :class:`FVMSetup` owns these values.
        CouplerSetup entries are optional cross-checks — a set
        value that contradicts the solver raises (nothing is silently
        overwritten), an unset one is filled from the solver so downstream
        coupling components keep a single consistent view.
        """
        cfg = self.config
        assert self.fvm is not None
        fvm_cfg = self.fvm.config
        owned = {
            "dt": float(fvm_cfg.time.delta_t),
            "t_end": float(fvm_cfg.time.end_time),
            "nu": float(fvm_cfg.transport.nu),
            "rho": float(fvm_cfg.transport.density),
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
        cfg.dt = owned["dt"]
        cfg.t_end = owned["t_end"]
        cfg.nu = owned["nu"]
        cfg.rho = owned["rho"]
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
        cfg.validate_handoff_box()

    def initialize(self) -> None:
        """Adopt the injected solvers, derive sub-cycling, and build coupling
        components.

        The injected FVM solver's configuration owns the FVM step; the
        injected VPM solver's ``time_step_size`` configures the
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
        # resolves the values it needs from that authoritative configuration.
        self.fvm = self._injected_fvm

        # Serial-backend guard: a serial Eulerian backend under mpirun would
        # leave every non-master rank solving its own detached copy (or hang
        # at the first collective) — fail loudly instead.
        world_size = 1
        if _mpi4py_comm is not None:
            world_size = int(_mpi4py_comm.Get_size())
        else:
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size > 1 and int(self.fvm.n_procs()) == 1:
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

        # Build injector
        self.injector = ContinuousOverlapInjector(self)
        self.injector.setup(self.fvm)
        physics = getattr(self.vpm, "physics", None)
        if (
            self._is_master
            and self.injector._body_bounds is not None
            and physics is not None
            and hasattr(physics, "configure_body_box")
        ):
            physics.configure_body_box(self.injector._body_bounds)
            logger.info(
                "[Init] VPM grid diffusion body mask enabled for box %s.",
                self.injector._body_bounds.tolist(),
            )
        if (
            self._is_master
            and physics is not None
            and hasattr(physics, "configure_grid_lattice_anchor")
        ):
            anchor = self.injector._lattice_anchor
            if anchor is None and self.injector._cell_centers is not None:
                anchor = self.injector._cell_centers[0]
            if anchor is not None:
                physics.configure_grid_lattice_anchor(anchor, self.config.h)
                logger.info("[Init] VPM diffusion lattice aligned with the handoff lattice.")

        self.fringe = FringeFields(cfg, self.vpm, self.fvm, coupling_dt=self.dt_vpm)

        if self._is_master:
            logger.info("[Init] Impulsive start: zero VPM particles.")
            with self.vpm_redirector:
                print(_vpm_solver_info(self.vpm))
                sys.stdout.flush()
            self._write_run_metadata()
            print("Initialization complete.\n")

        self._initialize_run_state()

    def run(
        self,
        start_step: int = 0,
        restart_from=None,
    ) -> None:
        """Initialize and run the coupling loop."""
        if self.injector is None:
            self.initialize()
        if restart_from is not None:
            if start_step:
                raise ValueError("start_step and restart_from are mutually exclusive")
            start_step = self.load_state(restart_from)
        self.solve(start_step=start_step)

    # =========================================================
    # Coupling loop
    # =========================================================

    def solve(self, start_step: int = 0) -> None:
        """Run the FVM--VPM coupling loop."""
        face_geometry, n_steps = self._prepare_run(start_step)
        for step in range(1 + start_step, n_steps + 1):
            time_end = step * self.dt
            vpm_time = self._advance_vpm(step, time_end)
            particle_guard = self._vpm_particle_fingerprint(validate=True)
            donor_state = self._transfer_vpm_to_fvm(*face_geometry)
            self._assert_vpm_particle_fingerprint(particle_guard, "target evaluation")
            fvm_time = self._advance_fvm(*face_geometry, *donor_state)
            self._assert_vpm_particle_fingerprint(particle_guard, "FVM subcycling")
            handoff_result, handoff_time = self._transfer_fvm_to_vpm(*face_geometry)
            self._last_handoff_result = handoff_result
            self._record_step(
                step,
                time_end,
                (vpm_time, donor_state[-2], donor_state[-1], fvm_time, handoff_time),
                handoff_result,
            )
        self._finalize_run()

    def _initialize_run_state(self) -> None:
        self._step_transfer_stats: dict[str, float | int] | None = None
        self.coupling_diagnostics = []

    def _vpm_particle_fingerprint(self, *, validate: bool = False) -> tuple[int, str, float]:
        """Return an exact fingerprint of particle state owned by the VPM.

        Target evaluation and FVM subcycling are read-only with respect to the
        particle cloud.  Hashing the two authoritative fields at those phase
        boundaries catches GPU/host-transfer corruption before a damaged cloud
        can be fed into the conservative handoff and contaminate later steps.
        """
        if not self._is_master:
            return (0, "", 0.0)
        assert self.vpm is not None
        count = int(self.vpm.particles.number_of_particles)
        if count == 0:
            return (0, hashlib.sha256(b"").hexdigest(), 0.0)
        positions = np.ascontiguousarray(self.vpm.particles_positions)
        circulation = np.ascontiguousarray(self.vpm.particles_circulation)
        if positions.shape != (count, 3) or circulation.shape != (count, 3):
            raise RuntimeError(
                "VPM particle readback shape does not match the active particle count: "
                f"count={count}, positions={positions.shape}, circulation={circulation.shape}"
            )
        if validate:
            radii = np.asarray(self.vpm.particles_radii)
            volumes = np.asarray(self.vpm.particles_volumes)
            if radii.shape != (count,) or volumes.shape != (count,):
                raise RuntimeError(
                    "VPM radius/volume readback shape does not match the active particle count: "
                    f"count={count}, radii={radii.shape}, volumes={volumes.shape}"
                )
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(circulation)):
                raise RuntimeError(
                    "VPM particle positions or circulations contain non-finite values"
                )
            if (
                not np.all(np.isfinite(radii))
                or not np.all(np.isfinite(volumes))
                or np.any(radii <= 0.0)
                or np.any(volumes <= 0.0)
            ):
                raise RuntimeError("VPM particle radii and volumes must be finite and positive")
            if not np.any(circulation != 0.0):
                raise RuntimeError(
                    "The active VPM cloud has identically zero circulation; "
                    "a GPU backend reset may have corrupted its fields"
                )
        digest = hashlib.sha256()
        digest.update(positions.tobytes(order="C"))
        digest.update(circulation.tobytes(order="C"))
        circulation_l1 = float(np.sum(np.linalg.norm(circulation, axis=1), dtype=np.float64))
        return (count, digest.hexdigest(), circulation_l1)

    def _assert_vpm_particle_fingerprint(
        self,
        expected: tuple[int, str, float],
        phase: str,
    ) -> None:
        """Fail if a nominally read-only coupling phase changed particles."""
        if not self._is_master:
            return
        actual = self._vpm_particle_fingerprint()
        if actual[:2] != expected[:2]:
            raise RuntimeError(
                "VPM particle state changed during read-only "
                f"{phase}: count {expected[0]} -> {actual[0]}, "
                f"sum|Gamma| {expected[2]:.6e} -> {actual[2]:.6e}. "
                "Aborting before the corrupted state reaches the FVM-to-VPM handoff."
            )

    def _prepare_run(self, start_step: int):
        """Validate a run and collect the immutable interface geometry."""
        if self.injector is None:
            raise RuntimeError(
                "solve() called before initialize(); call coupler.initialize() "
                "first, or use coupler.run() which does both."
            )
        assert self._is_master == (self.vpm is not None)
        assert self.fvm is not None

        n_steps = self._derive_coupling_step_count(self.t_end, self.dt)
        patch = self.config.patch_name
        if self._is_master:
            logger.info("=" * 60)
            logger.info("FVM-VPM COUPLED SOLVER")
            logger.info("=" * 60)

        face_centers = np.asarray(
            self.fvm.get_boundary_face_center_coordinates(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_normals = np.asarray(
            self.fvm.get_boundary_face_normals(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_areas = np.asarray(self.fvm.get_boundary_face_areas(patch), dtype=np.float64).ravel()
        self._face_centers = face_centers
        self._n_steps = n_steps
        return (face_centers, face_normals, face_areas), n_steps

    def _advance_vpm(self, step: int, time_end: float) -> float:
        t0 = time.perf_counter()
        if self._is_master:
            with self.vpm_redirector:
                self.vpm.set_background_velocity(self.config.u_inf)
            print()
            print("─" * 60)
            print(f"STEP {step}/{self._n_steps}  (t={time_end:.3f}s)")

            with self.vpm_redirector:
                self.vpm.update_state()
            import taichi as ti

            ti.sync()
        return time.perf_counter() - t0

    def _transfer_vpm_to_fvm(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
    ):
        """Update fringe data and construct the next donor boundary trace."""
        assert self.fringe is not None
        t_fringe = time.perf_counter()
        donor_velocity = None
        if self._is_master:
            assert self.vpm is not None
            fringe_points = self.fringe.active_cell_centres
            n_fringe = len(fringe_points)
            target_points = np.concatenate((fringe_points, face_centers), axis=0)
            target_velocity = np.asarray(
                self.vpm.compute_target_velocities(
                    target_points,
                    include_freestream=True,
                    zone_mask=None,
                    include_body=True,
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            expected_targets = n_fringe + len(face_centers)
            if target_velocity.shape != (expected_targets, 3):
                raise RuntimeError(
                    "VPM target evaluation returned an invalid shape: "
                    f"expected {(expected_targets, 3)}, got {target_velocity.shape}"
                )
            if not np.all(np.isfinite(target_velocity)):
                raise RuntimeError("VPM target evaluation returned non-finite velocities")
            freestream_speed = float(np.linalg.norm(self.u_inf))
            if (
                expected_targets > 0
                and freestream_speed > 0.0
                and float(np.max(np.linalg.norm(target_velocity, axis=1)))
                <= 1.0e-6 * freestream_speed
            ):
                raise RuntimeError(
                    "VPM target evaluation returned an identically zero field despite a "
                    "nonzero freestream; aborting before the corrupted donor data reaches the FVM"
                )
            self.fringe.update_target(target_velocity[:n_fringe])
            donor_velocity = target_velocity[n_fringe:]
        else:
            # The non-master rank has empty gathered cell geometry, but still
            # participates in the collective native-FVM boundary update.
            self.fringe.update_target()
        t_fringe = time.perf_counter() - t_fringe

        t_donor = time.perf_counter()
        if self._is_master:
            u_bc_next = self._donor_velocity(
                face_centers,
                face_normals,
                face_areas,
                evaluated_velocity=donor_velocity,
            )
            if self._u_bc_prev is None:
                self._u_bc_prev = u_bc_next.copy()
            if self.config.donor_boundary_mode == "pressure_gradient":
                assert self.vpm is not None
                assert self.config.rho is not None
                assert self.config.nu is not None
                assert self.dt is not None
                pressure_result, pressure_velocity = self.vpm.compute_target_pressure_gradients(
                    face_centers,
                    density=float(self.config.rho),
                    nu=float(self.config.nu),
                    include_viscous=False,
                    include_temporal=self._pressure_velocity_snapshot is not None,
                    include_freestream=True,
                    include_body=False,
                    h=self.config.h,
                    temporal_method="eulerian",
                    velocity_previous=self._pressure_velocity_snapshot,
                    dt=self.dt,
                    return_velocity=True,
                    treecode_theta=0.3,
                )
                pressure_gradient = np.asarray(pressure_result["grad_p"], dtype=np.float64).reshape(
                    -1, 3
                ) / float(self.config.rho)
                if pressure_gradient.shape != face_centers.shape or not np.all(
                    np.isfinite(pressure_gradient)
                ):
                    raise RuntimeError("VPM pressure-gradient donor returned invalid data")
                pressure_norm = np.linalg.norm(pressure_gradient, axis=1)
                logger.info(
                    "     [Donor pressure] |∇(p/ρ)| rms=%.3e max=%.3e m/s²  temporal=%s",
                    float(np.sqrt(np.mean(pressure_norm**2))) if len(pressure_norm) else 0.0,
                    float(np.max(pressure_norm)) if len(pressure_norm) else 0.0,
                    self._pressure_velocity_snapshot is not None,
                )
                self._pressure_velocity_snapshot = np.asarray(
                    pressure_velocity, dtype=np.float64
                ).reshape(-1, 3)
                self._pressure_gradient_bc_next = pressure_gradient
                if self._pressure_gradient_bc_prev is None:
                    self._pressure_gradient_bc_prev = pressure_gradient.copy()
        else:
            u_bc_next = np.zeros_like(face_centers)
            if self._u_bc_prev is None:
                self._u_bc_prev = np.zeros_like(face_centers)
            if self.config.donor_boundary_mode == "pressure_gradient":
                self._pressure_gradient_bc_next = np.zeros_like(face_centers)
                if self._pressure_gradient_bc_prev is None:
                    self._pressure_gradient_bc_prev = np.zeros_like(face_centers)
        t_donor = time.perf_counter() - t_donor
        return (
            self._u_bc_prev,
            u_bc_next,
            t_fringe,
            t_donor,
        )

    def _advance_fvm(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        u_bc_prev: np.ndarray,
        u_bc_next: np.ndarray,
        _fringe_time: float,
        _donor_time: float,
    ) -> float:
        """Run FVM sub-cycles and refresh its velocity snapshot."""
        t_fvm = time.perf_counter()
        self._run_fvm_substeps(
            self.config.patch_name,
            face_centers,
            face_normals,
            face_areas,
            u_bc_prev,
            u_bc_next,
            self._pressure_gradient_bc_prev,
            self._pressure_gradient_bc_next,
        )
        if self._is_master:
            self._u_bc_prev = u_bc_next
            if self._pressure_gradient_bc_next is not None:
                self._pressure_gradient_bc_prev = self._pressure_gradient_bc_next
        return time.perf_counter() - t_fvm

    def _transfer_fvm_to_vpm(
        self,
        face_centers: np.ndarray,
        _face_normals: np.ndarray,
        _face_areas: np.ndarray,
    ):
        """Fetch the FVM velocity trace and transfer it to the particle lattice."""
        t_handoff = time.perf_counter()
        if not callable(getattr(self.fvm, "get_velocity_gradient_field", None)):
            raise RuntimeError("FVM handoff requires the velocity-gradient API")
        velocity_global = self._get_velocity_field_buffer()
        gradient_global = self._get_velocity_gradient_field_buffer()
        handoff_result = None
        if self._is_master:
            assert self.vpm is not None
            assert self.injector is not None
            vpm = self.vpm
            injector = self.injector
            n_before = vpm.particles.number_of_particles
            sum_before = (
                float(np.sum(np.linalg.norm(np.asarray(vpm.particles_circulation), axis=1)))
                if n_before > 0
                else 0.0
            )
            handoff_result = injector.inject(
                vpm,
                velocity=velocity_global,
                velocity_gradient=gradient_global,
            )
            n_after = vpm.particles.number_of_particles
            sum_after = (
                float(np.sum(np.linalg.norm(np.asarray(vpm.particles_circulation), axis=1)))
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
        return handoff_result, time.perf_counter() - t_handoff

    def _record_step(
        self,
        step: int,
        time_end: float,
        timing: tuple[float, float, float, float, float],
        _handoff_result,
    ) -> None:
        """Persist diagnostics and synchronize a completed coupling step."""
        t_vpm, t_fringe, t_donor, t_fvm, t_handoff = timing
        diagnostics = self.compute_diagnostics(_handoff_result)
        timing_data = {
            "vpm": float(t_vpm),
            "donor": float(t_donor),
            "fringe": float(t_fringe),
            "fvm": float(t_fvm),
            "handoff": float(t_handoff),
            "total": float(sum(timing)),
        }
        if self._is_master:
            diagnostics.update(
                {"step": int(step), "time": float(time_end), "timing_seconds": timing_data}
            )
            self.coupling_diagnostics.append(diagnostics)
            with (self.solution_dir / "coupler_diagnostics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(diagnostics, separators=(",", ":")) + "\n")
        if self._is_master:
            stats = self._step_transfer_stats or {}
            logger.info(
                "     [Inject] N_before=%d  N_after=%d  |Γ|_before=%.4e  |Γ|_after=%.4e",
                int(stats.get("n_before", 0)),
                int(stats.get("n_after", 0)),
                float(stats.get("sum_before", 0.0)),
                float(stats.get("sum_after", 0.0)),
            )
            logger.info(
                "[Timing step=%d] VPM=%.3fs donor=%.3fs fringe=%.3fs "
                "FVM=%.3fs handoff=%.3fs total=%.3fs",
                step,
                timing_data["vpm"],
                timing_data["donor"],
                timing_data["fringe"],
                timing_data["fvm"],
                timing_data["handoff"],
                timing_data["total"],
            )
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
            self.save_state(self.solution_dir / CHECKPOINT_DIRECTORY, coupling_step=step)

    def _finalize_run(self) -> None:
        if self._is_master:
            flush_log()

    # =========================================================
    # Donor boundary
    # =========================================================

    def _donor_velocity(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        evaluated_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the complete VPM field and enforce zero net boundary flux."""
        assert self.vpm is not None
        normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
        logger.info("     [Donor] particles=%d", self.vpm.particles.number_of_particles)
        if evaluated_velocity is None:
            evaluated_velocity = self.vpm.compute_target_velocities(
                face_centers,
                include_freestream=True,
                zone_mask=None,
                include_body=True,
            )
        u_donor = np.asarray(evaluated_velocity, dtype=np.float64).reshape(-1, 3)
        if len(u_donor) != len(face_centers):
            raise ValueError("evaluated donor velocity count does not match boundary faces")

        # A uniform normal correction removes the quadrature flux residual.
        areas = np.asarray(face_areas, dtype=np.float64).ravel()
        flux_residual_raw = 0.0
        delta_u_n = 0.0
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
        """Return finite transfer and conservation diagnostics."""
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
        handoff = {
            "cfl": float(getattr(result, "cfl", 0.0)),
            "n_remesh_in": int(getattr(result, "n_remesh_in", 0)),
            "n_remesh_out": int(getattr(result, "n_remesh_out", 0)),
            "n_free": int(getattr(result, "n_free", 0)),
            "n_excluded": int(getattr(result, "n_excluded", 0)),
            "n_pruned": int(getattr(result, "n_pruned", 0)),
            "n_overlap_shell_pruned": int(getattr(result, "n_overlap_shell_pruned", 0)),
            "pruned_circulation_fraction": float(
                getattr(result, "pruned_circulation_fraction", 0.0)
            ),
            "overlap_shell_pruned_circulation_fraction": float(
                getattr(result, "overlap_shell_pruned_circulation_fraction", 0.0)
            ),
            "n_population_pruned": int(getattr(result, "n_population_pruned", 0)),
            "population_pruned_circulation_fraction": float(
                getattr(result, "population_pruned_circulation_fraction", 0.0)
            ),
            "population_pruned_velocity_bound": float(
                getattr(result, "population_pruned_velocity_bound", 0.0)
            ),
            "flux_ratio": float(getattr(result, "flux_ratio", 0.0)),
            "strength_correction_residual_pre": float(
                getattr(result, "strength_corr_residual_pre", 0.0)
            ),
            "strength_correction_residual_post": float(
                getattr(result, "strength_corr_residual_post", 0.0)
            ),
        }
        if not all(np.isfinite(value) for value in handoff.values()):
            raise FloatingPointError("non-finite handoff diagnostic")
        return {
            "conservation": conservation,
            "donor_flux": donor,
            "handoff": handoff,
            "period_multiplier": int(self.period_multiplier),
            "handoff_particle_count": int(getattr(result, "n_total", 0)),
        }

    def _get_velocity_field_buffer(self) -> np.ndarray:
        assert self.fvm is not None
        if self._velocity_global_buffer is None:
            self._velocity_global_buffer = np.ascontiguousarray(
                self.fvm.get_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm.get_velocity_field_into(self._velocity_global_buffer)
        return self._velocity_global_buffer

    def _get_velocity_gradient_field_buffer(self) -> np.ndarray:
        assert self.fvm is not None
        if self._velocity_gradient_global_buffer is None:
            self._velocity_gradient_global_buffer = np.ascontiguousarray(
                self.fvm.get_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm.get_velocity_gradient_field_into(self._velocity_gradient_global_buffer)
        return self._velocity_gradient_global_buffer

    def _fvm_step(
        self,
        patch: str,
        u_target: np.ndarray,
        pressure_gradient: np.ndarray | None = None,
    ) -> None:
        """Apply the configured donor boundary trace and advance one FVM step."""
        assert self.fvm is not None
        u_inf_mag = float(np.linalg.norm(self.config.u_inf)) + 1e-30
        boundary_mode = self.config.donor_boundary_mode
        u_target = np.ascontiguousarray(u_target, dtype=np.float64)
        if boundary_mode == "characteristic":
            self.fvm.set_freestream_velocity_boundary_condition_vec(u_target, patch)
            self.fvm.set_freestream_pressure_boundary_condition(patch, value=0.0)
            boundary_description = "characteristic U/p"
        elif boundary_mode == "directional_outflow":
            self.fvm.set_directional_freestream_velocity_boundary_condition_vec(
                u_target, patch, self.config.u_inf
            )
            self.fvm.set_directional_freestream_pressure_boundary_condition(patch, value=0.0)
            boundary_description = "directional-outflow mixed U/p"
        elif boundary_mode == "pressure_gradient":
            if pressure_gradient is None:
                raise RuntimeError("pressure_gradient donor mode requires pressure-gradient data")
            self.fvm.set_dirichlet_velocity_boundary_condition_vec(u_target, patch)
            self.fvm.set_neumann_pressure_boundary_condition(pressure_gradient, patch)
            boundary_description = "Dirichlet U / VPM pressure gradient"
        else:
            self.fvm.set_dirichlet_velocity_boundary_condition_vec(u_target, patch)
            boundary_description = "Dirichlet U / fixedFluxPressure"

        self.fvm.solve_pimple()
        self.fvm.advance_time()

        if u_target.shape[0] > 0:
            logger.info(
                "     [FVM] solved with %s  u_x/U∞ face[min=%.2f max=%.2f]",
                boundary_description,
                u_target[:, 0].min() / u_inf_mag,
                u_target[:, 0].max() / u_inf_mag,
            )
        yplus = getattr(self.fvm, "last_yplus", None)
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
        pressure_gradient_prev: np.ndarray | None = None,
        pressure_gradient_next: np.ndarray | None = None,
    ) -> None:
        """Advance FVM substeps with interpolated donor boundary data."""
        n_substeps = max(1, int(self.period_multiplier))
        u_inf_mag = float(np.linalg.norm(self.u_inf)) + 1e-30
        if n_substeps > 1 and u_next.shape[0] > 0:
            dU = float(np.max(np.linalg.norm(u_next - u_prev, axis=1))) / u_inf_mag
            big = dU > 0.5
            logger.log(
                logging.WARNING if big else logging.INFO,
                "     [Sub-cycle] %d×dt_fvm=%.3e s  donor ΔBC max|Δu|/U∞=%.3f%s",
                n_substeps,
                self.dt_fvm,
                dU,
                "  (large — lower dt or period_multiplier)" if big else "",
            )

        assert self.fringe is not None
        for substep in range(n_substeps):
            alpha = (substep + 1) / n_substeps
            self.fringe.push_target(alpha)
            u_bc = (1.0 - alpha) * u_prev + alpha * u_next
            u_bc = self._project_to_solenoidal(u_bc, face_normals, face_areas)
            pressure_gradient = None
            if pressure_gradient_prev is not None and pressure_gradient_next is not None:
                pressure_gradient = (
                    1.0 - alpha
                ) * pressure_gradient_prev + alpha * pressure_gradient_next
            self._fvm_step(patch, u_bc, pressure_gradient)

    # =========================================================
    # Restart support
    # =========================================================

    def _config_digest(self) -> str:
        payload = json.dumps(self.config.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save_state(self, directory, *, coupling_step: int | None = None) -> Path:
        """Write a complete native FVM--VPM checkpoint, committing its manifest last."""
        if self.fvm is None:
            raise RuntimeError("Initialize the coupler before saving a checkpoint")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        step = (
            int(coupling_step)
            if coupling_step is not None
            else int(self.fvm.time_step // self.period_multiplier)
        )
        suffix = f"{step:06d}"
        partitioned = bool(getattr(getattr(self.fvm, "parallel", None), "is_partitioned", False))
        fvm_artifact = f"fvm_{suffix}" if partitioned else f"fvm_{suffix}.npz"

        # The FVM checkpoint is COLLECTIVE under a partitioned backend: every
        # rank writes its own partition piece and joins a closing barrier.  It
        # must run on all ranks, so it precedes the master-only remainder --
        # returning early on the workers here deadlocks rank 0 in that barrier
        # while the workers advance to the next step's barrier.
        self.fvm.save_state(target / fvm_artifact)

        if not self._is_master:
            return target
        if self.vpm is None:
            raise RuntimeError("Initialize the coupler before saving a checkpoint")

        from source.solvers.VPM.io.backup import BackupSystem

        BackupSystem.backup_solver(
            self.vpm,
            str(target / f"vpm_{suffix}"),
            flow_time=float(self.vpm.flow_time),
            append_step=False,
            verbose=False,
        )

        donor_artifact = f"donor_{suffix}.npz"
        donor_tmp = target / f".{donor_artifact}.tmp"
        with open(donor_tmp, "wb") as stream:
            np.savez_compressed(
                stream,
                u_present=np.asarray(self._u_bc_prev is not None),
                u=np.empty((0, 3)) if self._u_bc_prev is None else self._u_bc_prev,
            )
        os.replace(donor_tmp, target / donor_artifact)

        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "openonda.coupled_checkpoint",
            "created_utc": datetime.now(UTC).isoformat(),
            "backend": "fvm",
            "config_sha256": self._config_digest(),
            "config": self.config.to_dict(),
            "coupling_step": step,
            "flow_time": float(self.vpm.flow_time),
            "fvm_time_step": int(self.fvm.time_step),
            "vpm_time_step": int(self.vpm.time_step),
            "period_multiplier": int(self.period_multiplier),
            "artifacts": {
                "fvm": fvm_artifact,
                "vpm": f"vpm_{suffix}.h5",
                "donor": donor_artifact,
            },
        }
        manifest_tmp = target / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, target / "manifest.json")
        keep = {"manifest.json", *manifest["artifacts"].values(), f"vpm_{suffix}.xdmf"}
        stale = {
            *target.glob("fvm_*"),
            *target.glob("vpm_*"),
            *target.glob("donor_*"),
        }
        for artifact in stale:
            if artifact.name in keep or not artifact.exists():
                continue
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        return target

    @staticmethod
    def _config_diff(stored: dict | None, current: dict) -> list[str]:
        """Return ``section.key: old -> new`` lines for a two-level config dict."""
        if not stored:
            return []
        lines: list[str] = []
        for section in sorted(set(stored) | set(current)):
            old_s, new_s = stored.get(section), current.get(section)
            if isinstance(old_s, dict) and isinstance(new_s, dict):
                for key in sorted(set(old_s) | set(new_s)):
                    if old_s.get(key) != new_s.get(key):
                        lines.append(f"{section}.{key}: {old_s.get(key)!r} -> {new_s.get(key)!r}")
            elif old_s != new_s:
                lines.append(f"{section}: {old_s!r} -> {new_s!r}")
        return lines

    def load_state(self, directory) -> int:
        """Restore both solvers and donor history from a checkpoint."""
        if self.fvm is None:
            raise RuntimeError("Initialize the coupler before loading a checkpoint")
        if self._is_master and self.vpm is None:
            raise RuntimeError("Initialize the coupler before loading a checkpoint")

        target = Path(directory)

        # ── Validate on the master, broadcast the verdict ────────────────────
        error: str | None = None
        manifest: dict | None = None
        artifacts: dict[str, str] = {}
        if self._is_master:
            try:
                manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            except OSError as exc:
                error = f"Cannot read coupled checkpoint manifest at {target}: {exc}"
            if error is None:
                assert manifest is not None
                version = manifest.get("format_version")
                if version != CHECKPOINT_FORMAT_VERSION or manifest.get("backend") != "fvm":
                    error = "Unsupported coupled checkpoint format or backend"
                else:
                    artifacts = manifest.get("artifacts", {})
                if error is None and (
                    missing := [
                        name
                        for name in ("fvm", "vpm", "donor")
                        if not artifacts.get(name) or not (target / artifacts[name]).exists()
                    ]
                ):
                    error = f"Incomplete coupled checkpoint; missing: {', '.join(missing)}"
                elif error is None and manifest.get("config_sha256") != self._config_digest():
                    changes = self._config_diff(manifest.get("config"), self.config.to_dict())
                    detail = "\n  ".join(changes) if changes else "(checkpoint stored no config)"
                    error = f"Coupled checkpoint configuration differs:\n  {detail}"
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            error, manifest = _mpi4py_comm.bcast(
                (error, manifest) if self._is_master else None, root=0
            )
        if error is not None:
            raise ValueError(error)
        assert manifest is not None
        artifacts = manifest["artifacts"]

        # ── Collective: each rank restores its own FVM partition ─────────────
        self.fvm.load_state(target / artifacts["fvm"])

        expected_fvm_step = int(manifest["vpm_time_step"]) * self.period_multiplier
        if self.fvm.time_step != expected_fvm_step:
            raise ValueError(
                f"Coupled checkpoint time-step mismatch: FVM={self.fvm.time_step}, "
                f"expected {expected_fvm_step} from VPM={manifest['vpm_time_step']}"
            )

        # ── Master-only: VPM particles and the donor trace ───────────────────
        if self._is_master:
            assert self.vpm is not None  # guarded above; narrows for the checker
            self.load_vpm_from_backup(str(target / artifacts["vpm"]))
            self.vpm.flow_time = float(manifest["flow_time"])
            with np.load(target / artifacts["donor"], allow_pickle=False) as donor:
                self._u_bc_prev = donor["u"].copy() if bool(donor["u_present"]) else None
            if not np.isclose(self.fvm.flow_time, self.vpm.flow_time, rtol=0.0, atol=1e-12):
                error = (
                    f"Coupled checkpoint time mismatch: FVM={self.fvm.flow_time}, "
                    f"VPM={self.vpm.flow_time}"
                )
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            error = _mpi4py_comm.bcast(error if self._is_master else None, root=0)
        if error is not None:
            raise ValueError(error)

        return int(manifest["coupling_step"])

    def load_vpm_from_backup(self, backup_h5_path: str) -> int:
        """Load VPM state from an H5 backup."""
        import h5py

        if self.vpm is None:
            raise RuntimeError("Initialize the VPM solver before loading a backup")
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
            else:
                pos = np.empty((0, 3))
                circ = np.empty((0, 3))
                vel = np.empty((0, 3))
                rad = np.empty(0)
                vol = np.empty(0)
                visc = np.empty(0)
                visc_t = np.empty(0)
                gid = np.empty(0, dtype=np.int32)

        with self.vpm_redirector:
            self.vpm.replace_vortex_particles(
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
