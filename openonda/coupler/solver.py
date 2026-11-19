"""Coupled FVM--VPM solver: particle boundary data drive the FVM domains, FVM vorticity is transferred to the particles."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
import logging
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.clock import time_tolerance
from openonda.core.metadata import write_run_metadata
from openonda.core.parallel import collective_phase
from openonda.core.restart import reset_run_outputs, select_backup
from openonda.core.schedules import EverySteps
from openonda.core.solution_layout import coupled_backup_directories
from openonda.core.status import RunStatus
from openonda.fvm.core.time_step import event_aligned_time_step_size
from openonda.runtime import detected_world_size

from .backup import coupled_backup_path, load_coupled_backup, save_coupled_backup
from .boundary import initialize_vpm_boundary_history, report_coupling_patch_quality
from .config import CouplerConfig
from .config.types import differing_shared_settings
from .domain import (
    CoupledDomain,
    check_interval_regions_apart,
    check_remeshing_regions_apart,
    lattice_phase,
)
from .reporting import (
    begin_coupling_step,
    close_coupler_logging,
    configure_coupler_logging,
    domain_records,
    flush_log,
    format_coupler_log,
    record_step,
)
from .vorticity_transfer import ParticleSetBalance, VorticityTransfer, remeshing_extension_length

if TYPE_CHECKING:
    from openonda.fvm import FVMCase, FVMSolver
    from openonda.vpm import VPMCase, VPMSolver

logger = logging.getLogger("openonda.coupler")


PARTICLE_STRAIN_CFL = 0.9
"""Largest particle strain over one coupling interval, ``|∇u|_max Δt``."""
GBD_MOMENT_RESIDUAL_LIMIT = 1.0e-5
"""Largest normalized moment residual of the GBD weak-strength removal, single precision."""


def _validate_gbd_moment_correction(
    correction: Mapping[str, bool | int | float] | None,
    correction_limit: float,
) -> None:
    """Stop when the GBD weak-strength removal of the last VPM step failed its moment checks."""
    if correction is None:
        return
    numeric_names = (
        "nonzero_node_count",
        "retained_node_count",
        "removed_node_count",
        "correction_fraction",
        "normalized_vortex_strength_residual",
        "normalized_linear_impulse_residual",
        "normalized_angular_impulse_residual",
    )
    try:
        values = {name: float(correction[name]) for name in numeric_names}
        values["support_augmented_node_count"] = float(
            correction.get("support_augmented_node_count", 0)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("GBD moment-correction diagnostics are incomplete or invalid") from error
    for name, value in values.items():
        if not np.isfinite(value):
            raise RuntimeError(f"GBD moment-correction diagnostic {name!r} is non-finite")
    for name in (
        "nonzero_node_count",
        "retained_node_count",
        "removed_node_count",
        "support_augmented_node_count",
    ):
        if values[name] < 0.0:
            raise RuntimeError(f"GBD moment-correction diagnostic {name!r} is negative")

    if values["removed_node_count"] > 0.0 and not bool(correction.get("applied", False)):
        raise RuntimeError("GBD removed weak strengths without the moment correction")
    maximum_residual = max(
        values["normalized_vortex_strength_residual"],
        values["normalized_linear_impulse_residual"],
        values["normalized_angular_impulse_residual"],
    )
    if maximum_residual > GBD_MOMENT_RESIDUAL_LIMIT:
        raise RuntimeError(
            f"the GBD moment correction leaves a normalized residual of {maximum_residual:.3e}, "
            f"above its storage-precision tolerance {GBD_MOMENT_RESIDUAL_LIMIT:.0e}"
        )
    if values["correction_fraction"] > float(correction_limit):
        raise RuntimeError(
            "the GBD weak-strength removal needed a moment correction above its limit: "
            f"fraction={values['correction_fraction']:.6e}, "
            f"limit={float(correction_limit):.6e}"
        )


def nominal_fvm_substeps_per_vpm_step(vpm_time_step_size: float, fvm_time_step_size: float) -> int:
    """Return the FVM substeps of one VPM step at the initial step sizes, at least one.

    A ratio within roundoff of an integer counts as that integer.
    """
    ratio = float(vpm_time_step_size) / float(fvm_time_step_size)
    return max(1, math.ceil(ratio - 1.0e-12 * max(1.0, ratio)))


def _as_tuple(value, kind: type, name: str) -> tuple:
    """One case or setting, or a sequence of them, as a tuple."""
    items = (value,) if isinstance(value, kind) else tuple(value)
    if not items or any(not isinstance(item, kind) for item in items):
        raise TypeError(f"CoupledSolver requires one {name} or a sequence of them")
    return items


def _validate_solution_layout(fvm_cases: tuple[FVMCase, ...], vpm_case: VPMCase) -> None:
    """Check that the domains write below the coupled solution directory, each to its own."""
    vpm_solution = Path(vpm_case.directory) / "solution" / (vpm_case.variant or "")
    if len(fvm_cases) == 1:
        fvm_case = fvm_cases[0]
        if fvm_case.solution_directory.resolve() != vpm_solution.resolve():
            raise ValueError(
                "The FVM and VPM cases must share their directory and variant; the FVM writes to "
                f"{fvm_case.solution_directory} and the VPM to {vpm_solution}"
            )
        return
    names = [fvm_case.name for fvm_case in fvm_cases]
    if len(set(names)) != len(names):
        raise ValueError(f"The FVM domains need distinct case names (got {names})")
    for fvm_case in fvm_cases:
        if (
            Path(fvm_case.directory).resolve() != Path(vpm_case.directory).resolve()
            or fvm_case.solution_directory.parent.resolve() != vpm_solution.resolve()
        ):
            raise ValueError(
                f"Each FVM domain writes to a sub-directory of the coupled solution {vpm_solution}: "
                f"give it the VPM directory and the variant '<VPM variant>/<domain>'; "
                f"{fvm_case.name!r} writes to {fvm_case.solution_directory}"
            )
    folders = [fvm_case.solution_directory.resolve() for fvm_case in fvm_cases]
    if len(set(folders)) != len(folders):
        raise ValueError("The FVM domains must write to distinct solution directories")


def _validate_cases(
    fvm_cases: tuple[FVMCase, ...], vpm_case: VPMCase, configs: tuple[CouplerConfig, ...]
) -> None:
    """Check the case settings the solvers and the coupler must share."""
    from openonda.fvm import FVMCase
    from openonda.vpm import VPMCase

    if any(not isinstance(fvm_case, FVMCase) for fvm_case in fvm_cases):
        raise TypeError("CoupledSolver requires FVMCase domains")
    if not isinstance(vpm_case, VPMCase):
        raise TypeError("CoupledSolver requires a VPMCase")
    if len(configs) != len(fvm_cases):
        raise ValueError(
            f"CoupledSolver requires one CouplerConfig per FVM domain "
            f"({len(fvm_cases)} domains, {len(configs)} settings)"
        )
    if differing := differing_shared_settings(configs):
        raise ValueError(
            "The FVM domains share one particle set and one clock; these coupling settings "
            f"must be equal in every domain: {', '.join(differing)}"
        )
    _validate_solution_layout(fvm_cases, vpm_case)
    config = configs[0]
    if vpm_case.backup.schedule is not None or any(
        fvm_case.backup.schedule is not None for fvm_case in fvm_cases
    ):
        raise ValueError(
            "Coupled backups are set by CouplerConfig.backup; "
            "the FVM and VPM cases take no backup schedule"
        )
    for fvm_case in fvm_cases:
        if fvm_case.run.start_time != 0.0:
            raise ValueError(
                "Coupled runs start at time zero "
                f"(fvm_case.run.start_time of {fvm_case.name!r} is {fvm_case.run.start_time})"
            )
        if not math.isclose(fvm_case.run.end_time, vpm_case.run.end_time, rel_tol=1.0e-12):
            raise ValueError(
                f"The FVM end time {fvm_case.run.end_time!r} s of {fvm_case.name!r} differs "
                f"from the VPM end time {vpm_case.run.end_time!r} s"
            )
    numerics = vpm_case.numerics
    if numerics.viscous.particle_spacing is None:
        raise ValueError(
            "Coupled VPM runs require viscous.particle_spacing so the transferred FVM "
            "vorticity and the boundary derivatives use the VPM discretization"
        )
    viscous = numerics.viscous
    if viscous.scheme != "gbd":
        raise ValueError(
            f"The vorticity transfer requires the GBD viscous scheme (got {viscous.scheme!r})"
        )
    if viscous.gbd_threshold_mode != "absolute":
        raise ValueError(
            "The vorticity transfer requires ViscousConfig.gbd_threshold_mode='absolute' "
            f"(got {viscous.gbd_threshold_mode!r}), so the removal threshold at the transfer "
            "surface is a resolved physical value"
        )
    spacing = float(viscous.particle_spacing)
    for domain_config in configs:
        transfer_threshold = domain_config.transfer_removal_threshold * spacing**3
        if float(viscous.gbd_threshold) > transfer_threshold:
            raise ValueError(
                f"The strength threshold h³ transfer_removal_threshold ({transfer_threshold:.6g} "
                f"m³/s) must be at least the GBD removal threshold of the VPM "
                f"({viscous.gbd_threshold:.6g} m³/s)"
            )
    has_walls = any(
        boundary.mesh_type == "wall" for fvm_case in fvm_cases for boundary in fvm_case.boundaries
    )
    if has_walls and viscous.gbd_remeshing_kernel != "m4_prime":
        raise ValueError(
            "GBD next to the FVM walls needs gbd_remeshing_kernel='m4_prime' "
            f"(got {viscous.gbd_remeshing_kernel!r})"
        )
    viscosity = numerics.viscous.kinematic_viscosity
    for fvm_case in fvm_cases:
        fvm_viscosity = fvm_case.numerics.transport.kinematic_viscosity
        if viscosity is not None and abs(float(viscosity) - fvm_viscosity) > 1.0e-12:
            raise ValueError(
                f"The VPM kinematic viscosity {float(viscosity):g} m^2/s differs from the FVM "
                f"kinematic viscosity {fvm_viscosity:g} m^2/s of {fvm_case.name!r}"
            )
        if abs(fvm_viscosity - fvm_cases[0].numerics.transport.kinematic_viscosity) > 1.0e-12:
            raise ValueError("The FVM domains must share one kinematic viscosity")
    if not np.allclose(numerics.freestream_velocity, config.freestream_velocity, atol=1.0e-9):
        raise ValueError(
            f"The VPM freestream velocity {tuple(numerics.freestream_velocity)} differs from "
            f"the coupling freestream velocity {tuple(config.freestream_velocity)}"
        )

    for fvm_case in fvm_cases:
        nominal_fvm_substeps = nominal_fvm_substeps_per_vpm_step(
            numerics.time_step_size, fvm_case.numerics.time_step_size
        )
        if config.max_coupling_iterations > 1 and (
            nominal_fvm_substeps > 1 or fvm_case.numerics.time_step_control is not None
        ):
            fvm_step_schedules = [fvm_case.output.schedule]
            fvm_step_schedules.extend(sampler.schedule for sampler in fvm_case.samplers.samples)
            if any(isinstance(schedule, EverySteps) for schedule in fvm_step_schedules):
                raise ValueError(
                    "FVM output and samplers that use EverySteps cannot be combined with "
                    "multiple FVM substeps and coupling iterations; use EveryTime schedules"
                )


def _validate_vpm_domain(vpm_case: VPMCase, config: CouplerConfig, fvm_bounds: np.ndarray) -> None:
    """Check the particle domain, slip planes and fringes against the FVM domain bounds."""
    numerics = vpm_case.numerics
    induction = numerics.induction
    if getattr(induction, "method", None) == "slip_slab":
        # The FVM extrusion, transfer region and VPM slab share one span.
        planes = np.array([induction.z_min, induction.z_max], dtype=np.float64)
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(planes))))
        transfer = config.transfer_region(
            fvm_bounds, numerics.viscous.particle_spacing, slip_planes=True
        )
        for name, bounds in (("FVM domain", fvm_bounds), ("transfer_region_bounds", transfer)):
            z_bounds = np.asarray(bounds, dtype=np.float64)[4:6]
            if np.any(np.abs(z_bounds - planes) > tolerance):
                raise ValueError(
                    f"{name} z {tuple(z_bounds.tolist())} must equal the VPM slip "
                    f"planes {tuple(planes.tolist())}"
                )
    domain = numerics.domain_bounds
    if domain is not None:
        # Projected wall vertices can move an outer face by a few picometres;
        # the VPM domain is specified at the intended plane.
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(fvm_bounds))))
        lower = np.asarray(domain[::2]) <= fvm_bounds[::2] + tolerance
        upper = np.asarray(domain[1::2]) >= fvm_bounds[1::2] - tolerance
        if not (np.all(lower) and np.all(upper)):
            raise ValueError(
                f"The VPM domain {tuple(domain)} does not contain the FVM domain {tuple(fvm_bounds)}: "
                "the transfer-lattice particles would be removed every step"
            )
    stabilization = numerics.stabilization
    fringe = getattr(stabilization, "outflow_fringe_depth", None)
    if fringe is not None:
        depth = np.asarray(fringe, dtype=np.float64)
        keep = np.asarray(stabilization.remove_particles_by_bounds, dtype=np.float64)
        inner = keep + depth * np.array([1.0, -1.0] * 3)
        lower_reach = (depth[::2] > 0.0) & (inner[::2] > fvm_bounds[::2])
        upper_reach = (depth[1::2] > 0.0) & (inner[1::2] < fvm_bounds[1::2])
        if np.any(lower_reach) or np.any(upper_reach):
            raise ValueError(
                f"The VPM outflow fringes {tuple(fringe)} reach the FVM domain {tuple(fvm_bounds)}; "
                "place each fringe outside the FVM domain"
            )


class CoupledSolver:
    """Inner FVM solutions coupled to one outer VPM wake.

    Each FVM domain has its own mesh, mesh motion, outer boundary and transfer
    region; all share one particle set and one coupling clock. The FVM CFL
    controller of each domain selects its substeps. The VPM interval is at
    most the nominal substeps of every domain times its CFL step, and the
    coupler shortens the interval when the particle strain limit requires it.
    Physical-time samples, frames, backups and the run horizon remain exact
    event times.

    One coupling interval advances the particles once to the predictor and
    evaluates the boundary data of every domain from the whole particle
    field. Every FVM domain then advances over the interval with its own
    substeps; the vorticity of every domain replaces the particles of its
    transfer region, on its own transfer lattice in the coordinates of its
    body; and the boundary data of all domains are evaluated again from the
    particles after all transfers. Coupling iterations repeat the FVM
    intervals and the transfers until the boundary data of every domain
    converge, and restart all domains together.

    With one FVM domain on a moving mesh the VPM runs in the frame
    translating with the rotation centre of the mesh, with the freestream
    relative to it. With several domains the VPM runs in the inertial frame.
    The coupling geometry of every domain stays in its body coordinates
    (:class:`~openonda.coupler.frame.MeshFrame`). Before the first step the
    remeshing regions of every two domains are checked to stay apart over the
    run (:func:`~openonda.coupler.domain.check_remeshing_regions_apart`).

    Parameters
    ----------
    fvm_case : FVMCase or sequence of FVMCase
        Inner flow, mesh and FVM output of each domain. Under MPI every rank
        constructs the coupled solver; the FVM domains run on all ranks.
        With several domains each FVM case has the VPM case's directory and
        the variant ``"<VPM variant>/<domain>"``, so that it writes below the
        coupled solution directory, and the case names differ.
    vpm_case : VPMCase
        Particle wake and its output; the VPM runs on the root rank. Its
        configured time-step size is the maximum coupling interval. With one
        domain its directory and variant are those of the FVM case.
    config : CouplerConfig or sequence of CouplerConfig
        Vorticity transfer, coupling iterations and coupled backups, one per
        FVM domain. The settings of
        :data:`~openonda.coupler.config.DOMAIN_SETTINGS` belong to their
        domain; all others must be equal.

    Attributes
    ----------
    config : CouplerConfig
        Coupling settings of the first domain; their shared settings hold for all.
    domains : tuple[CoupledDomain, ...]
        The FVM domains in the order given.
    vpm_case : VPMCase
        Case of the particle solver.
    fvm_solver, fvm_case, frame, vorticity_transfer
        Those of the single domain of a one-domain run.
    vpm_solver : VPMSolver or None
        VPM solver on the root rank, ``None`` elsewhere.
    solution_dir : pathlib.Path
        ``solution/`` of the case, below the VPM variant.
    vpm_time_step_size : float
        Current VPM coupling interval in s.
    freestream_velocity : numpy.ndarray
        VPM freestream in m/s, relative to the particle frame.
    absolute_freestream_velocity : numpy.ndarray
        Inertial-frame freestream in m/s.
    coupling_diagnostics : list[dict]
        Diagnostics of the coupling steps of this process, also appended to
        ``solution/coupler_diagnostics.jsonl``.

    Notes
    -----
    Coupled backups (``solution/coupled/coupled_{step:06d}/``) replace the
    backups of the solvers: the cases take no backup schedule and the solvers
    write no backup at their end. Every coupled backup also writes the FVM
    and particle frames of its time. Whenever the VPM writes a particle frame
    (``VPMCase.output``), every FVM domain writes a frame at the same time.
    The records of one domain in ``coupler_diagnostics.jsonl`` and in
    ``coupler_metadata.json`` are at the top level of a one-domain run and
    below ``domains.<name>`` with several domains. The solver is a context
    manager.

    Examples
    --------
    >>> with CoupledSolver(fvm_case, vpm_case, CouplerConfig()) as solver:
    ...     solver.run(start_from="latest")
    >>> with CoupledSolver((upstream, downstream), vpm_case, (config_a, config_b)) as solver:
    ...     solver.run(start_from="latest")
    """

    def __init__(
        self,
        fvm_case: FVMCase | Sequence[FVMCase],
        vpm_case: VPMCase,
        config: CouplerConfig | Sequence[CouplerConfig],
    ) -> None:
        """Construct the solvers, validate them together and build the transfer geometry."""
        from openonda.fvm import Backup as FVMBackup
        from openonda.fvm import FVMCase, FVMSolver
        from openonda.vpm import Backup as VPMBackup
        from openonda.vpm import VPMSolver

        fvm_cases = _as_tuple(fvm_case, FVMCase, "FVMCase")
        configs = _as_tuple(config, CouplerConfig, "CouplerConfig")
        _validate_cases(fvm_cases, vpm_case, configs)
        self.config = configs[0]
        self.vpm_case = vpm_case
        self.case_dir = Path(vpm_case.directory).resolve()
        self.solution_dir = (
            Path(vpm_case.directory) / "solution" / (vpm_case.variant or "")
        ).resolve()
        self.run_status = RunStatus.not_started
        self.run_failure: BaseException | None = None
        self.coupling_diagnostics: list[dict] = []
        self._closed = False
        self._run_started = False
        self._started = False
        self._restart_loaded = False
        self._frame_step: int | None = None
        self._backup_step: int | None = None
        self._coupling_step = 0
        # Accepted coupling time in s; the FVM solvers pass it while they advance an interval.
        self._accepted_time: float | None = None
        self._resources = ExitStack()
        self.vpm_solver: VPMSolver | None = None
        self._is_root = True
        self._comm = None
        self.domains: tuple[CoupledDomain, ...] = ()
        try:
            domains = []
            for domain_case, domain_config in zip(fvm_cases, configs, strict=True):
                fvm_solver: FVMSolver = self._resources.enter_context(
                    FVMSolver(replace(domain_case, backup=FVMBackup(write_at_end=False)))
                )
                # With one domain the VPM runs in the frame translating with the
                # rotation centre of the FVM mesh; with several, in the inertial frame.
                domains.append(
                    CoupledDomain(
                        self,
                        domain_case,
                        domain_config,
                        fvm_solver,
                        follows_mesh=len(fvm_cases) == 1,
                    )
                )
            self.domains = tuple(domains)
            self._comm = self.domains[0].fvm_solver.parallel.comm
            self._is_root = bool(self.domains[0].fvm_solver.parallel.is_root)
            if self._is_root:
                configure_coupler_logging(self.solution_dir / "coupler.log", console=True)
            with collective_phase(self._comm, "VPM construction"):
                if self._is_root:
                    self.vpm_solver = self._resources.enter_context(
                        VPMSolver(
                            replace(vpm_case, backup=VPMBackup(write_at_end=False)),
                            external_clock=True,
                        )
                    )
            self._initialize()
            self.run_status = RunStatus.created
            self.write_run_metadata(status=self.run_status)
        except BaseException as error:
            self._resources.__exit__(type(error), error, error.__traceback__)
            if self._is_root:
                close_coupler_logging()
            raise

    # The single domain of a one-domain run

    def _single_domain(self) -> CoupledDomain:
        if len(self.domains) != 1:
            raise AttributeError(
                f"The coupled run has {len(self.domains)} FVM domains; use CoupledSolver.domains"
            )
        return self.domains[0]

    @property
    def fvm_solver(self) -> FVMSolver:
        """The FVM solver of a one-domain run."""
        return self._single_domain().fvm_solver

    @property
    def fvm_case(self) -> FVMCase:
        """The FVM case of a one-domain run."""
        return self._single_domain().fvm_case

    @property
    def frame(self):
        """The mesh frame of a one-domain run."""
        return self._single_domain().frame

    @property
    def vorticity_transfer(self) -> VorticityTransfer:
        """The vorticity transfer of a one-domain run."""
        return self._single_domain().vorticity_transfer

    def _initialize(self) -> None:
        """Read the shared discretization and build the transfer geometry (collective)."""
        world_size = int(self._comm.Get_size()) if self._comm is not None else detected_world_size()
        for domain in self.domains:
            if world_size > 1 and domain.fvm_solver.n_ranks == 1:
                raise RuntimeError(
                    f"Launched under MPI with {world_size} processes but the FVM solver of "
                    f"{domain.name!r} is serial; set FVMCase.cores or launch one process"
                )
        numerics = self.vpm_case.numerics
        viscous = numerics.viscous
        self.vpm_time_step_size = float(numerics.time_step_size)
        for domain in self.domains:
            domain.nominal_fvm_substeps_per_vpm_step = nominal_fvm_substeps_per_vpm_step(
                self.vpm_time_step_size, domain.fvm_time_step_size
            )
        self.maximum_coupling_time_step_size = self.vpm_time_step_size
        # (start time, exact end time) of the selected interval when it lands on a deadline
        self._selected_interval_end: tuple[float, float] | None = None
        self.end_time = float(self.vpm_case.run.end_time)
        first_case = self.domains[0].fvm_case
        self.kinematic_viscosity = float(first_case.numerics.transport.kinematic_viscosity)
        self.density = float(first_case.numerics.transport.density)
        self.vpm_particle_spacing = float(viscous.particle_spacing)
        self.vpm_core_radius_ratio = float(viscous.core_radius_ratio)
        grid_spacing = {"gbd": viscous.gbd_grid_spacing, "dvh": viscous.dvh_grid_spacing}.get(
            viscous.scheme
        )
        self.vpm_diffusion_grid_spacing = float(
            self.vpm_particle_spacing if grid_spacing is None else grid_spacing
        )
        planar = getattr(numerics.induction, "method", None) == "slip_slab"
        start_time = self.domains[0].fvm_solver.time
        for domain in self.domains:
            domain.frame.validate(
                start_time, self.end_time, domain.fvm_time_step_size, planar=planar
            )
        self.freestream_velocity = self.config.freestream_velocity_vector
        self.absolute_freestream_velocity = self.freestream_velocity.copy()
        self._particle_set_balance = ParticleSetBalance(
            self.config, self.config.transfer_diagnostic_interval_steps
        )

        for domain in self.domains:
            domain.fvm_domain_bounds = self._gather_fvm_domain_bounds(domain)
            domain.config.validate_transfer_region(
                domain.fvm_domain_bounds, self.vpm_particle_spacing, planar
            )
            # A moving FVM domain sweeps a larger region of the particle domain.
            _validate_vpm_domain(
                self.vpm_case,
                domain.config,
                domain.frame.swept_bounds(
                    domain.fvm_domain_bounds, start_time, self.end_time, domain.fvm_time_step_size
                ),
            )
        if self._is_root:
            backup = self.config.backup.schedule
            for domain in self.domains:
                logger.info(
                    format_coupler_log(
                        "time stepping"
                        if len(self.domains) == 1
                        else f"time stepping, {domain.name}",
                        ("initial fvm time step", f"{domain.fvm_time_step_size:.4e}", "s"),
                        (
                            "maximum fvm time step",
                            "none"
                            if domain.maximum_fvm_time_step_size is None
                            else f"{domain.maximum_fvm_time_step_size:.4e}",
                            "s",
                        ),
                        (
                            "maximum vpm interval",
                            f"{self.maximum_coupling_time_step_size:.4e}",
                            "s",
                        ),
                        (
                            "nominal fvm substeps per vpm step",
                            domain.nominal_fvm_substeps_per_vpm_step,
                        ),
                        ("end time", f"{self.end_time:.6g}", "s"),
                        ("coupled backups", "none" if backup is None else backup.describe()),
                    )
                )

        for domain in self.domains:
            domain.vorticity_transfer = VorticityTransfer(domain)
            if domain.frame.moving:
                # The remeshing region must hold the maximum shared-step drift relative to the mesh.
                relative_speed = domain.frame.maximum_relative_speed(
                    self._absolute_freestream,
                    start_time,
                    self.end_time,
                    self.maximum_coupling_time_step_size,
                    domain.fvm_domain_bounds,
                )
                domain.vorticity_transfer.remeshing_extension_length = remeshing_extension_length(
                    [relative_speed, 0.0, 0.0],
                    self.maximum_coupling_time_step_size,
                    self.vpm_particle_spacing,
                )
            domain.vorticity_transfer.read_mesh(domain.fvm_solver)
        # Every domain's transfer lattice shares the phase of the particle lattice.
        anchors = [domain.vorticity_transfer.natural_lattice_anchor() for domain in self.domains]
        phase = None if any(anchor is None for anchor in anchors) else lattice_phase(anchors)
        spacing = self.vpm_particle_spacing
        for domain, anchor in zip(self.domains, anchors, strict=True):
            if phase is not None and len(self.domains) > 1:
                anchor = phase + np.round((anchor - phase) / spacing) * spacing
            domain.vorticity_transfer.build(domain.fvm_solver, anchor)
        self._lattice_phase = phase
        if len(self.domains) > 1:
            with collective_phase(self._comm, "remeshing regions of the FVM domains"):
                if self._is_root:
                    check_remeshing_regions_apart(
                        self.domains,
                        start_time,
                        self.end_time,
                        min(domain.fvm_time_step_size for domain in self.domains),
                    )
        with collective_phase(self._comm, "VPM grid configuration"):
            if self._is_root:
                self._configure_vpm_geometry()

    def _gather_fvm_domain_bounds(self, domain: CoupledDomain) -> np.ndarray:
        """Return the outer FVM bounds of a domain in body coordinates from its non-wall patches (collective)."""
        fvm = domain.fvm_solver
        patch = domain.config.outer_boundary_patch
        names = [patch] + [
            boundary.name
            for boundary in domain.fvm_case.boundaries
            if boundary.name != patch and boundary.mesh_type != "wall"
        ]
        faces = [
            domain.frame.to_body(
                domain.frame.to_frame(
                    np.asarray(fvm.gather_boundary_face_centre_coordinates(name)).reshape(-1, 3)
                )
            )
            for name in names
        ]
        bounds = None
        with collective_phase(self._comm, "FVM domain bounds"):
            if self._is_root:
                if len(faces[0]) == 0:
                    raise ValueError(
                        f"The coupling patch {patch!r} of {domain.name!r} has no faces"
                    )
                points = np.concatenate(faces)
                bounds = np.ravel(np.column_stack((points.min(axis=0), points.max(axis=0))))
        if self._comm is not None and self._comm.Get_size() > 1:
            bounds = self._comm.bcast(bounds, root=0)
        return np.asarray(bounds, dtype=np.float64)

    def _configure_vpm_geometry(self) -> None:
        """Give the VPM the FVM wall geometry, the strain-excluded regions and the lattice phase."""
        vpm = self.vpm_solver
        with_walls = [domain for domain in self.domains if domain.vorticity_transfer.solid_boundary]
        if any(domain.frame.moves for domain in with_walls):
            self._place_vpm_walls(None, self.time)
        elif with_walls:
            from .geometry import SolidBoundary
            from .solid import WallProjection

            boundary = (
                with_walls[0].vorticity_transfer.solid_boundary
                if len(with_walls) == 1
                else SolidBoundary(
                    body
                    for domain in with_walls
                    for body in domain.vorticity_transfer.solid_boundary.bodies
                )
            )
            vpm.physics.lattice.wall.configure_body_classifier(
                boundary.contains,
                revision=boundary.revision,
                query_bounds=boundary.bounds,
                segments_cross_wall=boundary.segments_cross_wall,
                geometry_queries=boundary.grid_geometry_queries(),
            )
            projection = WallProjection(boundary, vpm.physics, self.vpm_particle_spacing, logger)
            vpm.stage_rhs.position_guard = projection.stage
            vpm.stage_rhs.accepted_position_projector = projection.accepted
        if with_walls:
            revisions = [
                repr(domain.vorticity_transfer.solid_boundary.revision) for domain in with_walls
            ]
            logger.info(
                format_coupler_log(
                    "vpm diffusion grid",
                    ("solid mask", "FVM wall geometry"),
                    ("geometry revision", ", ".join(revisions)),
                )
            )
        # The FVM vorticity replaces these particles after every interval, so their
        # strain does not limit the interval; the FVM step follows its own Courant number.
        self._set_strain_excluded_regions(None)
        anchor = self.domains[0].vorticity_transfer._lattice_anchor
        if anchor is not None:
            vpm.physics.lattice.configure_grid_lattice_anchor(
                anchor, self.vpm_diffusion_grid_spacing
            )
            anchor_text = ", ".join(f"{value:.6g}" for value in np.asarray(anchor))
            logger.info(
                format_coupler_log(
                    "vpm diffusion lattice",
                    ("anchor", f"[{anchor_text}]", "m"),
                    ("spacing", f"{self.vpm_diffusion_grid_spacing:.6g}", "m"),
                )
            )

    def _set_strain_excluded_regions(self, interval: bool | None) -> None:
        """Exclude the remeshing regions of the domains, at the interval-end poses, from the strain limit."""
        regions = [
            region
            for domain in self.domains
            if (
                region := domain.remeshing_region_in_frame(
                    domain._interval_pose if interval else None
                )
            )
            is not None
        ]
        self.vpm_solver.strain_excluded_bounds = tuple(regions) if regions else None

    def _place_vpm_walls(self, interval: bool | None, time: float) -> None:
        """Give the VPM the walls of every domain, those of the moving meshes at their poses.

        ``interval`` takes the poses at the end of the coming interval, the
        current poses otherwise; ``time`` labels the geometry revision.
        """
        from .solid import PlacedSolidBoundary, SolidBoundaryUnion, WallProjection

        vpm = self.vpm_solver
        placed = []
        for domain in self.domains:
            boundary = domain.vorticity_transfer.solid_boundary
            if boundary is None:
                continue
            if domain.frame.moves:
                pose = domain._interval_pose if interval else domain.frame.pose_after(np.array([]))
                boundary = PlacedSolidBoundary(boundary, domain.frame, pose, time)
            placed.append(boundary)
        boundary = placed[0] if len(placed) == 1 else SolidBoundaryUnion(placed)
        vpm.physics.lattice.wall.configure_body_classifier(
            boundary.contains,
            revision=boundary.revision,
            query_bounds=boundary.bounds,
            segments_cross_wall=boundary.segments_cross_wall,
        )
        projection = WallProjection(boundary, vpm.physics, self.vpm_particle_spacing, logger)
        vpm.stage_rhs.position_guard = projection.stage
        vpm.stage_rhs.accepted_position_projector = projection.accepted

    # Clock

    @property
    def step(self) -> int:
        """Number of completed coupling steps."""
        return self._coupling_step

    @property
    def time(self) -> float:
        """Accepted physical time in s, that of every solver between two coupling steps."""
        if self._accepted_time is None:
            return float(self.domains[0].fvm_solver.time)
        return self._accepted_time

    def coupling_time(self, step: int) -> float:
        """Return the current or next shared physical time, in s."""
        if step == self.step:
            return float(self.time)
        if step == self.step + 1:
            # An interval that reaches an event or the end time ends exactly on it.
            landing = self._selected_interval_end
            if landing is not None and landing[0] == self.time:
                return landing[1]
            return float(self.time + self.vpm_time_step_size)
        raise ValueError("The coupled clock exposes only the current and next accepted times")

    # Lifecycle

    def __enter__(self) -> CoupledSolver:
        return self

    def __exit__(self, _exc_type, exc_value, _traceback) -> None:
        self.close(failure=exc_value)

    def start_from(self, selection: str | Path = "latest") -> bool:
        """Select the initial state or a coupled backup before the first coupling step.

        Parameters
        ----------
        selection : {'latest', 'initial'} or path, default='latest'
            ``latest`` loads the coupled backup with the highest step and
            starts from the initial state when there is none; ``initial``
            deletes the earlier output series and starts from the initial
            state; a path loads that coupled backup.

        Returns
        -------
        bool
            Whether a backup was loaded.
        """
        if selection == "initial":
            if self.step or self._restart_loaded or self._started:
                raise RuntimeError("start_from('initial') requires a new coupled solver")
            with collective_phase(self._comm, "coupled output reset"):
                if self._is_root:
                    reset_run_outputs(self.solution_dir, kind="coupled")
            for domain in self.domains:
                domain.fvm_solver.start_from("initial")
            self._accepted_time = float(self.domains[0].fvm_solver.time)
            with collective_phase(self._comm, "VPM output reset"):
                if self._is_root:
                    self.vpm_solver.start_from("initial")
            return False
        selected = None
        with collective_phase(self._comm, "coupled backup selection"):
            if self._is_root:
                selected = select_backup(selection, coupled_backup_directories(self.solution_dir))
        if self._comm is not None and self._comm.Get_size() > 1:
            selected = self._comm.bcast(selected, root=0)
        if selected is None:
            return self.start_from("initial")
        self.load_backup(selected)
        if self._is_root:
            logger.info(
                f"Continuing from {selected}: coupling step {self.step}, time {self.time:.17g}"
            )
        return True

    def run(self, *, start_from: str | Path | None = "latest") -> None:
        """Run the coupled case to ``RunPlan.end_time`` and close the solver.

        Parameters
        ----------
        start_from : {'latest', 'initial'}, path or None, default='latest'
            Passed to :meth:`start_from` before the run; ``None`` continues
            from the state in memory.

        Notes
        -----
        A fresh run writes the initial frames and samples of the solvers
        with their ``write_initial_output``; a restarted run does not repeat
        them. The run then advances the coupling steps, writes the scheduled output
        and backups, a final coupled backup with ``Backup.write_at_end``, and
        the final frames and final-only samples with ``write_final_output``.
        The run status is recorded in ``solution/coupler_metadata.json``.
        Every MPI rank calls this method.
        """
        if self._run_started:
            raise RuntimeError("CoupledSolver.run() may be called only once")
        self._run_started = True
        self.run_status = RunStatus.running
        failure: BaseException | None = None
        try:
            if start_from is not None:
                self.start_from(start_from)
            restored = self._restart_loaded
            self._start()
            if not restored:
                self._write_initial_output()
                if start_from is not None:
                    self.save_backup()
            self.write_run_metadata(status=self.run_status)
            while self.time < self.end_time - time_tolerance(self.end_time):
                self.advance()
            self._write_final_output()
            self.run_status = RunStatus.completed
        except BaseException as error:
            failure = error
            self.run_status = RunStatus.failed
            self.run_failure = error
        try:
            self.write_run_metadata(status=self.run_status)
        except BaseException as error:
            if failure is None:
                failure = error
                self.run_status = RunStatus.failed
        self.close(failure=failure)
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)

    def advance(self) -> None:
        """Advance one shared FVM--VPM step and write its due output and backup.

        The VPM advances by one shared selected step, the boundary data of the
        VPM predictor drive every FVM domain, the FVM vorticity of every domain
        is transferred to the particles of its transfer region, the boundary
        data of all domains are evaluated again, and the due samples, frames
        and coupled backup are written. Every MPI rank calls this method.
        """
        if self._closed:
            raise RuntimeError("The coupled solver is closed")
        self._start()
        step = self.step + 1
        if self.time >= self.end_time - time_tolerance(self.end_time):
            raise RuntimeError("The coupled run has reached its end time")
        if not self._run_started:
            self.run_status = RunStatus.running
        interval_started = time.perf_counter()
        self._select_coupling_interval()
        time_end = self.coupling_time(step)
        for domain in self.domains:
            # The outer-boundary faces and walls seen by the particles at the interval end
            domain.place_for_interval(time_end)
        if any(domain.frame.flow_driven for domain in self.domains) and len(self.domains) > 1:
            with collective_phase(self._comm, "remeshing regions at the interval end"):
                if self._is_root:
                    check_interval_regions_apart(self.domains, time_end)
        with collective_phase(self._comm, "free stream at the interval end"):
            self._update_freestream(time_end, interval=True)
        for domain in self.domains:
            domain._interval_seconds = dict.fromkeys(domain._interval_seconds, 0.0)
        vpm_seconds = self._advance_vpm(step, time_end)
        boundary_seconds = 0.0
        next_velocities = []
        for domain in self.domains:
            next_velocity, seconds = domain.predictor_boundary_data()
            next_velocities.append(next_velocity)
            boundary_seconds += seconds
        if self.config.max_coupling_iterations > 1:
            from .coupling_iteration import advance_coupling_iterations

            transfer_results, fvm_seconds, transfer_seconds = advance_coupling_iterations(
                self, next_velocities
            )
            boundary_seconds += self._last_coupling_iteration_diagnostics["phase_seconds"][
                "boundary_refresh"
            ]
        else:
            fvm_seconds = sum(
                domain.advance_fvm(next_velocity)
                for domain, next_velocity in zip(self.domains, next_velocities, strict=True)
            )
            transfer_results, transfer_seconds = self.transfer_vorticity_to_vpm()
            boundary_seconds += sum(domain.refresh_boundary_data() for domain in self.domains)
        self._coupling_step = step
        self._accepted_time = float(time_end)
        output_started = time.perf_counter()
        vpm_phase_seconds: dict[str, float] = {}
        with collective_phase(self._comm, "VPM particle-state check and output"):
            if self._is_root:
                self.vpm_solver.write_step_output()
                # The interval holds one particle step; its phases include this output.
                vpm_phase_seconds = dict(self.vpm_solver.phase_seconds)
        frame = self.vpm_case.output.schedule
        if frame is not None and frame.is_due(step, time_end, self.vpm_time_step_size):
            # The VPM wrote a particle frame; every FVM domain writes one at the same time.
            for domain in self.domains:
                if not domain.fvm_frame_written(self._frame_step):
                    domain.fvm_solver.write_vtk()
            self._frame_step = step
        output_seconds = time.perf_counter() - output_started
        record_step(
            self,
            step,
            time_end,
            (vpm_seconds, boundary_seconds, fvm_seconds, transfer_seconds),
            transfer_results,
            logger=logger,
            interval_started=interval_started,
            output_seconds=output_seconds,
            vpm_phase_seconds=vpm_phase_seconds,
        )

    def transfer_vorticity_to_vpm(self):
        """Transfer the FVM vorticity of every domain, then measure the particle-set balance.

        The transfers act on distinct particles, so their order does not
        change the particle state. Returns the transfer results in the order
        of the domains and the transfer seconds.
        """
        results = []
        seconds = 0.0
        for domain in self.domains:
            result, domain_seconds = domain.transfer_vorticity_to_vpm()
            results.append(result)
            seconds += domain_seconds
        if self._is_root:
            # The last transfer's particle-set fields describe the particles after all of them.
            last = self.domains[-1]
            results[-1] = last._last_transfer_result = self._particle_set_balance.update(
                results[-1], last.vorticity_transfer.step
            )
        return results, seconds

    def _start(self) -> None:
        """Prepare the first coupling step of this process once (collective).

        A fresh state first receives the FVM vorticity at the initial time,
        so every VPM interval starts from the FVM state of the same time.
        """
        if self._started:
            return
        for domain in self.domains:
            domain.fvm_solver.apply_velocity_boundaries(domain.fvm_solver.time)
        with collective_phase(self._comm, "initial freestream"):
            self._update_freestream(self.time, interval=False)
        for domain in self.domains:
            domain.read_outer_boundary()
            report_coupling_patch_quality(
                domain,
                domain._face_geometry[1],
                np.asarray(
                    domain.fvm_solver.gather_boundary_face_owner_offset(
                        domain.config.outer_boundary_patch
                    )
                ).reshape(-1, 3),
            )
        if self.step == 0 and not self._restart_loaded:
            self.transfer_vorticity_to_vpm()
        for domain in self.domains:
            initialize_vpm_boundary_history(domain, *domain._face_geometry)
        self._started = True

    def _write_frames(self, *, fvm_frame: bool, vpm_frame: bool) -> None:
        """Write the frames of the current coupling step; a particle frame brings the FVM frames."""
        for domain in self.domains:
            if (fvm_frame or vpm_frame) and not domain.fvm_frame_written(self._frame_step):
                domain.fvm_solver.write_vtk()
            domain.fvm_solver.flush_output()
        if vpm_frame and self._frame_step != self.step:
            with collective_phase(self._comm, "VPM frame"):
                if self._is_root:
                    self.vpm_solver.write_vtk()
            self._frame_step = self.step

    def _write_initial_output(self) -> None:
        """Write the initial frames and samples of the solvers; a particle frame brings the FVM frames."""
        vpm_frame = self.vpm_case.run.initial_output
        fvm_frame = False
        for domain in self.domains:
            fvm = domain.fvm_solver
            fvm.write_initial_output()
            if vpm_frame and not domain.fvm_case.run.initial_output:
                fvm.write_vtk()
            fvm.flush_output()
            fvm_frame = fvm_frame or domain.fvm_case.run.initial_output
        with collective_phase(self._comm, "VPM initial output"):
            if self._is_root:
                self.vpm_solver.write_initial_output()
        if fvm_frame or vpm_frame:
            self._frame_step = self.step

    def _write_final_output(self) -> None:
        """Write the final backup, frames and samples of the solvers."""
        if self.config.backup.write_at_end and self._backup_step != self.step:
            self.save_backup()
        vpm_frame = self.vpm_case.run.final_output
        fvm_frame = False
        for domain in self.domains:
            fvm = domain.fvm_solver
            if (
                vpm_frame
                and not domain.fvm_case.run.final_output
                and not domain.fvm_frame_written(self._frame_step)
            ):
                fvm.write_vtk()
            fvm.write_final_output()
            fvm.flush_output()
            fvm_frame = fvm_frame or domain.fvm_case.run.final_output
        with collective_phase(self._comm, "VPM final output"):
            if self._is_root:
                self.vpm_solver.write_final_output()
        if fvm_frame or vpm_frame:
            self._frame_step = self.step

    def _backup_due(self, step: int) -> bool:
        schedule = self.config.backup.schedule
        return schedule is not None and schedule.is_due(
            step,
            self.time,
            min(domain.fvm_solver._accepted_time_step_size for domain in self.domains),
        )

    def save_backup(self, path: str | Path | None = None) -> Path:
        """Save a coupled backup of the current coupling step with all frames.

        Parameters
        ----------
        path : str, pathlib.Path or None, default=None
            Backup directory; ``None`` writes
            ``solution/coupled/coupled_{step:06d}``.

        Returns
        -------
        pathlib.Path
            The backup directory, on every rank.

        Notes
        -----
        The backup holds the FVM backup and the boundary data of every
        domain and the VPM backup. The FVM and particle frames of the same
        time are written too. Collective under MPI.
        """
        self._start()
        destination = (
            coupled_backup_path(self.solution_dir, self.step) if path is None else Path(path)
        )
        save_coupled_backup(self, destination)
        self._backup_step = self.step
        self._write_frames(fvm_frame=True, vpm_frame=True)
        return destination

    def load_backup(
        self,
        path: str | Path,
        *,
        allowed_config_differences: Collection[str] = (),
        expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
    ) -> None:
        """Restore a coupled backup into all solvers.

        Parameters
        ----------
        path : str or pathlib.Path
            Coupled backup directory written by :meth:`save_backup`.
        allowed_config_differences : collection[str], default=()
            Exact configuration paths allowed to differ from the backup;
            ``vpm.``-prefixed paths are passed to the VPM backup reader.
        expected_config_differences : mapping or None, default=None
            ``path: (stored_value, current_value)`` expectations for the
            allowed differences.

        Notes
        -----
        The output series of all solvers and the coupler diagnostics are
        rewound to the restored time. Collective under MPI.
        """
        for domain in self.domains:
            domain.boundary_data_correction = None
            domain.previous_boundary_data_correction = None
        step = load_coupled_backup(
            self,
            Path(path),
            allowed_config_differences=allowed_config_differences,
            expected_config_differences=expected_config_differences,
        )
        for domain in self.domains:
            # A fresh run transfers once at the start and once per coupling step.
            domain.vorticity_transfer.step = step + 1
        self._restart_loaded = True
        self._started = False
        self._frame_step = self._backup_step = step

    def write_run_metadata(self, *, status: RunStatus | str) -> Path:
        """Write ``solution/coupler_metadata.json`` with the configuration and state.

        Returns
        -------
        pathlib.Path
            The metadata file, on every rank.
        """
        destination = self.solution_dir / "coupler_metadata.json"
        with collective_phase(self._comm, "coupler metadata"):
            if self._is_root:
                names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
                domain_sections = {
                    domain.name: {
                        "fvm_time_step_size": domain.fvm_time_step_size,
                        "fvm_substeps": domain.n_fvm_substeps,
                        "maximum_fvm_time_step_size": domain.maximum_fvm_time_step_size,
                        "nominal_fvm_substeps_per_vpm_step": (
                            domain.nominal_fvm_substeps_per_vpm_step
                        ),
                        "fvm_domain_bounds": dict(
                            zip(names, map(float, domain.fvm_domain_bounds), strict=True)
                        ),
                    }
                    for domain in self.domains
                }
                coupling = {
                    "vpm_time_step_size": self.vpm_time_step_size,
                    "maximum_time_step_size": self.maximum_coupling_time_step_size,
                    "end_time": self.end_time,
                    "kinematic_viscosity": self.kinematic_viscosity,
                    "density": self.density,
                    "particle_spacing": self.vpm_particle_spacing,
                    "core_radius_ratio": self.vpm_core_radius_ratio,
                    "diffusion_grid_spacing": self.vpm_diffusion_grid_spacing,
                    "viscous_scheme": self.vpm_case.numerics.viscous.scheme,
                }
                sections = {
                    "coupling": {**coupling, **domain_records(self, domain_sections)},
                    "state": {"step": self.step, "time": float(self.time)},
                }
                if len(self.domains) > 1:
                    sections["domain_settings"] = {
                        domain.name: domain.config for domain in self.domains
                    }
                write_run_metadata(
                    destination,
                    solver="coupler",
                    case_name=(
                        self.domains[0].fvm_case.name
                        if len(self.domains) == 1
                        else self.vpm_case.name
                    ),
                    status=status,
                    configuration=self.config,
                    sections=sections,
                )
        return destination

    def flush_output(self) -> None:
        """Finish the pending file output of all solvers and the coupler log."""
        for domain in self.domains:
            domain.fvm_solver.flush_output()
        if self._is_root:
            self.vpm_solver.flush_output()
            flush_log(logger)

    def close(self, *, failure: BaseException | None = None) -> None:
        """Close all solvers and the coupler log; safe to repeat.

        A coupled solver advanced without :meth:`run` records ``completed``
        when it reached the end time and ``partial`` otherwise. While an
        exception ``failure`` unwinds, cleanup errors are attached to it as
        notes. Every MPI rank closes its solvers.
        """
        if self._closed:
            return
        cleanup_error: BaseException | None = None
        if not self._run_started and self._started:
            self.run_status = (
                RunStatus.failed
                if failure is not None
                else RunStatus.completed
                if self.time >= self.end_time - time_tolerance(self.end_time)
                else RunStatus.partial
            )
            try:
                self.write_run_metadata(status=self.run_status)
            except BaseException as error:
                cleanup_error = error
        try:
            self._resources.__exit__(
                None if failure is None else type(failure),
                failure,
                None if failure is None else failure.__traceback__,
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
        finally:
            if self._is_root:
                close_coupler_logging()
            self._closed = True
        if cleanup_error is not None:
            if failure is not None:
                failure.add_note(f"Coupled solver cleanup also failed: {cleanup_error!r}")
            else:
                raise cleanup_error

    # Coupling-interval operations

    def _absolute_freestream(self, time: float) -> np.ndarray:
        """Inertial-frame freestream velocity in m/s at ``time``."""
        if self.config.freestream is not None:
            return np.asarray(self.config.freestream.at(time), dtype=np.float64)
        return self.config.freestream_velocity_vector

    def _coupling_event_schedules(self) -> tuple:
        """Return schedules that must land on VPM coupling times.

        The FVM schedules of every domain are included as well: an FVM-only
        output time inside a coupling interval would otherwise split it into a
        short FVM substep.
        """
        schedules = [
            schedule
            for domain in self.domains
            for schedule in domain.fvm_solver._run_event_schedules()
        ]
        schedules.append(self.vpm_case.logging.schedule)
        if self.vpm_case.output.schedule is not None:
            schedules.append(self.vpm_case.output.schedule)
        if self.config.backup.schedule is not None:
            schedules.append(self.config.backup.schedule)
        schedules.extend(
            sampler.schedule
            for sampler in self.vpm_case.samplers.samples
            if sampler.schedule is not None
        )
        return tuple(schedules)

    def _vpm_strain_time_step_limit(self) -> float:
        """Return a conservative step ceiling from the last accepted VPM strain."""
        vpm = self.vpm_solver
        metrics = vpm._accepted_state_metrics
        maximum = vpm.state_limits.lagrangian_cfl.maximum
        if maximum is None:
            return self.maximum_coupling_time_step_size
        if metrics is None:
            strain_rate = vpm.maximum_lagrangian_strain_rate()
        elif metrics.strain_increment_infinity > 0.0:
            strain_rate = metrics.strain_increment_infinity / vpm.time_step_size
        else:
            strain_rate = 0.0
        if strain_rate <= 0.0:
            return self.maximum_coupling_time_step_size
        return PARTICLE_STRAIN_CFL * maximum / strain_rate

    def _select_coupling_interval(self) -> float:
        """Select an event-aligned VPM interval from the FVM CFL limits and the VPM strain limit."""
        schedules = self._coupling_event_schedules()
        fvm_limits = []
        for domain in self.domains:
            fvm_limit = domain.fvm_solver.courant_time_step_limit()
            if domain.maximum_fvm_time_step_size is not None:
                fvm_limit = min(fvm_limit, domain.maximum_fvm_time_step_size)
            fvm_limits.append(fvm_limit)
        strain_limit = None
        if self._is_root:
            strain_limit = self._vpm_strain_time_step_limit()
        if self._comm is not None and self._comm.Get_size() > 1:
            strain_limit = self._comm.bcast(strain_limit, root=0)

        maximum_interval = self.maximum_coupling_time_step_size
        for domain, fvm_limit in zip(self.domains, fvm_limits, strict=True):
            maximum_interval = min(
                maximum_interval,
                domain.nominal_fvm_substeps_per_vpm_step * fvm_limit,
            )
        if strain_limit is not None:
            maximum_interval = min(maximum_interval, float(strain_limit))
        remaining = self.end_time - self.time
        maximum_interval = min(maximum_interval, remaining)
        next_event = min(
            (
                event_time
                for schedule in schedules
                if (event_time := schedule.next_time_after(self.time)) is not None
            ),
            default=self.end_time,
        )
        deadline = self.end_time if remaining <= next_event - self.time else float(next_event)
        event_distance = min(remaining, next_event - self.time)
        selected_interval = event_aligned_time_step_size(maximum_interval, event_distance)
        lands = selected_interval >= event_distance - time_tolerance(deadline)
        self._selected_interval_end = (float(self.time), deadline) if lands else None
        for domain, fvm_limit in zip(self.domains, fvm_limits, strict=True):
            domain.fvm_time_step_size = fvm_limit
            domain.vorticity_transfer.coupling_time_step_size = selected_interval
        self.vpm_time_step_size = selected_interval
        if self._is_root:
            self.vpm_solver.time_step_size = selected_interval
        return selected_interval

    def _update_freestream(self, time: float, *, interval: bool) -> None:
        """Set the freestream at ``time``, relative to the particle frame and to every mesh.

        ``interval`` takes the mean mesh velocities over the coming interval
        ending at ``time``; otherwise the mesh velocities at ``time``.
        """
        self.absolute_freestream_velocity = self._absolute_freestream(time)
        variable = self.config.freestream is not None
        mesh_velocities = [
            domain.frame.mesh_step_velocity(np.asarray([time]))
            if interval
            else domain.frame.mesh_velocity(time)
            for domain in self.domains
        ]
        for domain, mesh_velocity in zip(self.domains, mesh_velocities, strict=True):
            if variable or domain.frame.moving:
                domain.freestream_velocity = self.absolute_freestream_velocity - mesh_velocity
                domain.vorticity_transfer.config = replace(
                    domain.config, freestream_velocity=tuple(domain.freestream_velocity.tolist())
                )
        particle_frame = self.domains[0].frame
        if variable or (particle_frame.follows_mesh and particle_frame.moving):
            frame_velocity = mesh_velocities[0] if particle_frame.follows_mesh else np.zeros(3)
            self.freestream_velocity = self.absolute_freestream_velocity - frame_velocity
            if self._is_root:
                self.vpm_solver.set_freestream_velocity(self.freestream_velocity)

    def inertial_boundary_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Convert a VPM velocity, with the VPM freestream, to the inertial frame."""
        return velocity + (self.absolute_freestream_velocity - self.freestream_velocity)

    def _advance_vpm(self, step: int, time_end: float) -> float:
        started = time.perf_counter()
        with collective_phase(self._comm, "VPM advance"):
            if self._is_root:
                vpm = self.vpm_solver
                vpm.physics.last_solid_projection = {
                    "stage_count": 0,
                    "accepted_count": 0,
                    "stage_displacement_l1": 0.0,
                    "accepted_displacement_l1": 0.0,
                    "stage_impulse_change": np.zeros(3),
                    "accepted_impulse_change": np.zeros(3),
                }
                vpm.set_freestream_velocity(self.freestream_velocity)
                if any(
                    domain.frame.moves and domain.vorticity_transfer.solid_boundary is not None
                    for domain in self.domains
                ):
                    self._place_vpm_walls(True, time_end)
                if any(domain.frame.translates for domain in self.domains):
                    self._set_strain_excluded_regions(True)
                begin_coupling_step(logger, step, None, time_end)
                vpm.advance(defer_output=True, time=time_end)
                vpm.synchronize()
                if vpm.viscous_scheme == "gbd":
                    _validate_gbd_moment_correction(
                        getattr(vpm.physics.lattice, "last_gbd_moment_correction", None),
                        self.config.max_moment_correction_fraction,
                    )
        return time.perf_counter() - started
