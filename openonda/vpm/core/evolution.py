"""Time-evolution execution for the VPM solver.

:class:`EvolutionStepper` owns the per-step evolution algorithm: velocity and
gradient preparation, advection, stretching, coupled inviscid integration,
viscous diffusion, operator splitting, and the stabilization phases that
happen inside a step.  :class:`~openonda.vpm.core.solver.VPMSolver` is the
facade that composes subsystems; it delegates its step to the stepper and
keeps the diagnostics, backups, and I/O bookkeeping around the step.

The stepper holds a back-reference to the solver (``self.solver``) for the
state the solver changes between and during steps: the accepted clock, the
step size, the run state and the diffusion counter. The subsystems it drives
are plain attributes set when the solver builds the stepper. No physics is
implemented here: every numerical update is performed on the subsystems the
stepper calls. Diffusing and advecting kernels live in ``physics``; the
stabilization workers live in
``stabilization`` and are dispatched here only at their scheduled phases.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from ..io.logging import step_timing
from ..physics.induction.base import StageRates, StageState
from ..runtime.timing import remainder_section
from ..wall_errors import WallCorrectionTooLargeError

if TYPE_CHECKING:
    from openonda.core.log_style import StepLog
    from openonda.core.profiling import Profiler

    from ..config.case import Numerics
    from ..config.stabilization import StabilizationConfig
    from ..config.viscous import ViscousConfig
    from ..numerics.runge_kutta import RungeKutta
    from ..particles.container import Particles
    from ..physics.engine import PhysicsEngine
    from ..physics.stage_rhs import StageRHS
    from ..stabilization import StabilizationManager
    from ..turbulence.les import ParticleLES
    from ..vlm.solver import VLMSolver
    from .solver import VPMSolver


class EvolutionStepper:
    """Advance the VPM particle field by one time step.

    Constructed by the solver after the subsystems it drives are initialized.
    """

    def __init__(self, solver: VPMSolver) -> None:
        """Attach the accepted-step controller to its solver.

        Parameters
        ----------
        solver : VPMSolver
            Solver whose subsystems are driven. The stepper keeps the solver
            for its clock, time-step size, run state and diffusion counter,
            and binds the subsystems the solver builds once and never
            replaces: particles, physics, Runge--Kutta integrator, stage
            right-hand side, turbulence model, VLM solver, profiler and
            stabilization.

        Notes
        -----
        The owning solver remains responsible for accepted-step mutation,
        backups, diagnostics, and rollback. Construct this class through
        ``VPMSolver`` rather than instantiating it as an independent solver.
        """
        self.solver = solver
        self.particles: Particles = solver.particles
        self.physics: PhysicsEngine = solver.physics
        self.integrator: RungeKutta = solver.integrator
        self.stage_rhs: StageRHS = solver.stage_rhs
        self.turbulence_model: ParticleLES | None = solver.turbulence_model
        self.vlm_solver: VLMSolver | None = solver.vlm_solver
        self.profiler: Profiler = solver.profiler
        self.stabilization: StabilizationManager = solver.stabilization
        self.logger: StepLog = solver.logger
        self.numerics: Numerics = solver.numerics
        self.np_dtype: type[np.floating] = solver.np_dtype
        self.flow_model: str = solver.flow_model
        self.viscous_scheme: str = solver.viscous_scheme
        self._viscous_config: ViscousConfig = solver._viscous_config
        self.stabilization_config: StabilizationConfig = solver.stabilization_config
        self._staged_step: int | None = None
        self._staged_time: float | None = None

    @property
    def step(self) -> int:
        """Return the accepted or currently staged integer step index."""
        return self.solver.step if self._staged_step is None else self._staged_step

    @property
    def time(self) -> float:
        """Return the accepted or currently staged physical time in seconds."""
        return self.solver.time if self._staged_time is None else self._staged_time

    @property
    def time_step_size(self) -> float:
        """Return the active physical step size in seconds.

        The solver owns it: a restart with a changed step size sets it again.
        """
        return self.solver.time_step_size

    def advance(self, *, defer_output: bool = False, time: float | None = None) -> None:
        """Advance the VPM solution by one time step.

        The inviscid update advances particle motion and, when enabled, vortex
        stretching. Viscous diffusion is then applied by operator splitting. Core
        spreading uses symmetric Strang splitting in the coupled integrator.
        ``defer_output`` lets an external coupler synchronize the particle state
        before scheduled samples and backups are written. ``time`` is the
        accepted time imposed by a driver that owns the clock.
        """

        target_step = self.solver.step + 1
        target_time = self.solver._clock_time(target_step) if time is None else float(time)
        self.logger.begin_step(target_step, target_time, self.time_step_size)
        self.profiler.begin_step(self.solver.step, self.solver.time, self.time_step_size)

        self.stabilization.begin_step(
            step=self.solver.step,
            time=self.solver.time,
            time_step_size=self.time_step_size,
        )
        self.stabilization.run_phase("pre_evolution")

        # Stage the new clock for kernels and scheduled workers.  The standard
        # solver clock is committed only after all physical phases succeeded.
        self._staged_step = target_step
        self._staged_time = target_time
        self.stabilization.stage_clock(step=self._staged_step, time=self._staged_time)

        _gradients_required = (
            self.flow_model == "les"
            or (self.stabilization_config.selective_eddy_viscosity_coefficient or 0.0) > 0.0
            or self.stabilization_config.pedrizzetti_relaxation_enabled
        )
        if _gradients_required:
            self._update_velocity_and_gradients()

        with remainder_section(self.profiler, "LES update"):
            self._update_les_state()
            self.stabilization.update_selective_eddy_viscosity()

        # Relax against the same t_n gradient used by the strength update.
        self.stabilization.run_phase("pre_strength", profiler=self.profiler)

        # The inviscid particle state always advances through one coupled
        # position/strength RK call.  Diffusion remains a split operator.
        self._apply_coupled_update(
            self.time_step_size,
        )

        self.stabilization.run_phase("post_evolution", profiler=self.profiler)

        self.stabilization.run_phase("post_step", profiler=self.profiler)

        if self.vlm_solver is not None:
            # Transport the existing wake with the previous bound solution,
            # then solve at the new body/wake clock and insert the completed
            # near-wake row. Newborn elements must not receive a second full
            # step of advection or stretching during their birth interval.
            self.particles.touch_state()
            with remainder_section(self.profiler, "VLM coupling"):
                self._advance_vlm()

        # The evolution kernels mutate particle source fields directly on the
        # device.  Publish one new source revision after the complete physical
        # state (including any topology-changing stabilization) is committed so
        # post-step boundary queries cannot reuse the previous tree.
        self.particles.touch_state()
        self.profiler.end_step()
        self._commit_accepted_step()
        step_timing(
            self.solver,
            self.profiler.step_seconds,
            self.profiler.step_sections if self.profiler.enabled else None,
        )
        self.solver.wall_time = self.profiler.total_seconds
        total_steps = None if self.solver._external_clock else self.solver._run_final_step
        log = self.logger
        if log.is_due(self.solver.step, self.solver.time, self.time_step_size, total_steps):
            log.publish(
                self.solver.step,
                self.solver.time,
                self.solver.elapsed_wall_time,
                total_steps=total_steps,
                sections=(
                    ("particles", (("active particles", self.particles.n_particles_total),)),
                ),
            )

    def _advance_vlm(self) -> None:
        """Advance the VLM coupling by one step and append the wake particles it sheds."""
        particles_before = self.particles.n_particles_total
        self.vlm_solver.advance_coupled(
            particles=self.particles,
            physics=self.physics,
            freestream_velocity=self.numerics.freestream_velocity,
            time_step_size=self.time_step_size,
            time=self.time,
        )

        particles_added = self.particles.n_particles_total - particles_before
        if particles_added > 0 and self.stabilization.reference_vortex_strength is not None:
            # VLM inserts on the device; read only its shed batch for refinement reference.
            lattice = self.vlm_solver.lattice
            strength = lattice.wake_vortex_strength.to_numpy()[:particles_added]
            volume = lattice.wake_volume.to_numpy()[:particles_added]
            self.stabilization.on_add(np.linalg.norm(strength, axis=1), volume, particles_before)

    def _commit_accepted_step(self) -> None:
        """Publish a fully accepted numerical step to the standard clock."""
        assert self._staged_step is not None and self._staged_time is not None
        self.solver.step = self._staged_step
        self.solver.time = self._staged_time
        self._staged_step = None
        self._staged_time = None

    def _update_velocity_and_gradients(self) -> None:
        """Refresh derived particle velocity and gradient through the stage interface."""
        count = len(self.particles)
        if count == 0:
            return
        self.stage_rhs.evaluate(
            StageState(
                position=self.particles.position,
                vortex_strength=self.particles.vortex_strength,
                core_radius=self.particles.core_radius,
                count=count,
                time=self.solver.time,
                stage_index=None,
                accepted_revision=self.particles.state_revision,
            ),
            self.solver.time,
            StageRates(
                velocity=self.particles.velocity,
                vortex_strength_rate=self.physics.dstr_dt_temp,
                velocity_gradient=self.particles.velocity_gradient,
                strength_rate_enabled=False,
            ),
        )
        # External stage providers may add a non-uniform velocity gradient.
        # Rebuild the LES/stabilization strain tensor from the final combined
        # Jacobian rather than trusting a backend-private scratch field.
        with self.profiler.section("Strain-rate tensor"):
            self.particles._compute_strain_rate_tensor(
                self.particles.velocity_gradient,
                self.particles.strain_rate,
                count,
            )

    def _update_les_state(self) -> None:
        """Update LES viscosity from the current strain-rate field."""
        if self.flow_model == "les":
            self.turbulence_model.compute(self.particles, time_step_size=self.time_step_size)

    def _advance_particles(self, time_step_size: float) -> None:
        """Advance position and vortex strength at common Runge--Kutta stages."""
        accepted_projector = self.stage_rhs.accepted_position_projector
        guard = getattr(accepted_projector, "__self__", None)
        can_retry_wall = bool(
            guard is not None and hasattr(guard, "begin_trial") and hasattr(guard, "end_trial")
        )
        count = len(self.particles)
        accepted_logs = []
        wall_retries = 0
        accepted_intervals = 0

        def snapshot() -> tuple[np.ndarray, np.ndarray, object]:
            with self.profiler.section("Wall retry snapshot"):
                return (
                    self.physics.download_vector_field(self.particles.position, count),
                    self.physics.download_vector_field(self.particles.vortex_strength, count),
                    deepcopy(self.physics.last_solid_projection),
                )

        def project_accepted(position: object, vortex_strength: object, n: int) -> None:
            with self.profiler.section("Wall projection"):
                accepted_projector(position, vortex_strength, n)

        def restore(saved: tuple[np.ndarray, np.ndarray, object]) -> None:
            position, strength, budget = saved
            self.physics.upload_vector_array(position, self.particles.position, count)
            self.physics.upload_vector_array(strength, self.particles.vortex_strength, count)
            self.physics.last_solid_projection = deepcopy(budget)
            self.particles.touch_state()

        macro_snapshot = snapshot() if can_retry_wall and count else None

        def integrate(start_time: float, dt: float, depth: int) -> None:
            nonlocal wall_retries, accepted_intervals
            trial_snapshot = (
                (macro_snapshot if depth == 0 else snapshot())
                if macro_snapshot is not None
                else None
            )
            if trial_snapshot is not None and guard is not None:
                guard.begin_trial()
            try:
                try:
                    self.integrator.advance(
                        position=self.particles.position,
                        vortex_strength=self.particles.vortex_strength,
                        core_radius=self.particles.core_radius,
                        count=count,
                        time=start_time,
                        time_step_size=dt,
                        right_hand_side=self.stage_rhs,
                        accepted_position_projector=(
                            None if accepted_projector is None else project_accepted
                        ),
                        accepted_revision=self.particles.state_revision,
                    )
                finally:
                    # The RK combination writes the accepted source fields.
                    self.particles.touch_state()
            except BaseException as error:
                if trial_snapshot is not None and guard is not None:
                    restore(trial_snapshot)
                    guard.end_trial(accepted=False)
                if (
                    isinstance(error, WallCorrectionTooLargeError)
                    and can_retry_wall
                    and self.vlm_solver is None
                    and depth < 10
                ):
                    wall_retries += 1
                    half = 0.5 * dt
                    integrate(start_time, half, depth + 1)
                    integrate(start_time + half, dt - half, depth + 1)
                    return
                raise
            else:
                if trial_snapshot is not None and guard is not None:
                    accepted_logs.extend(guard.end_trial(accepted=True))
                accepted_intervals += 1

        try:
            integrate(self.solver.time, time_step_size, 0)
        except BaseException:
            if macro_snapshot is not None:
                restore(macro_snapshot)
            raise
        if guard is not None:
            for args in accepted_logs:
                guard.logger.warning(*args)
        if wall_retries and guard is not None:
            guard.logger.info(
                "VPM wall correction accepted after %d retries in %d RK intervals",
                wall_retries,
                accepted_intervals,
            )
        ti.sync()

    def _apply_coupled_update(
        self,
        time_step_size: float,
    ) -> None:
        """Advance position and strength together, with symmetric core spreading."""
        if self.viscous_scheme == "cs":
            with remainder_section(self.profiler, "Viscous diffusion"):
                self._apply_core_spreading_diffusion(0.5 * time_step_size)

        with remainder_section(self.profiler, "Runge-Kutta update"):
            self._advance_particles(time_step_size)
        if self.viscous_scheme == "cs":
            with remainder_section(self.profiler, "Viscous diffusion"):
                self._apply_core_spreading_diffusion(0.5 * time_step_size)

        if self.viscous_scheme in {"rwm", "dvh", "gbd"}:
            with remainder_section(self.profiler, "Viscous diffusion"):
                self._apply_viscous_diffusion(time_step_size)

        if self.turbulence_model is not None:
            with self.profiler.section("LES strength exchange"):
                self.turbulence_model.apply_strength_exchange(self.particles, time_step_size)

    def _apply_core_spreading_diffusion(self, time_step_size: float) -> None:
        """Advance the split Gaussian core-spreading operator."""
        if time_step_size <= 0.0 or len(self.particles) == 0:
            return
        self.physics.core_spreading_diffusion(self.particles, time_step_size)

    def _apply_viscous_diffusion(self, time_step_size: float) -> None:
        """Dispatch viscous diffusion by configured scheme."""
        if self.viscous_scheme == "none":
            return

        if self.viscous_scheme == "cs":
            self._apply_core_spreading_diffusion(time_step_size)
        elif self.viscous_scheme == "rwm":
            self.physics.random_walk_method_diffusion(
                self.particles,
                time_step_size=time_step_size,
                random_seed=self.numerics.random_seed,
                accepted_step=self.step,
            )
        elif self.viscous_scheme in ("dvh", "gbd"):
            # DVH fires only when its fixed diffusion increment has accumulated.
            # This is an operator-splitting approximation: intermediate accepted
            # states lag in diffusion time, and reducing RK dt at fixed h and
            # resolved interval does not remove that lag by itself.
            diffusion_time_step_size = time_step_size
            if self.viscous_scheme == "dvh" and self.solver._n_steps_per_dvh_diffusion > 1:
                pending_steps = self.solver._n_steps_since_dvh_diffusion + 1
                if pending_steps < self.solver._n_steps_per_dvh_diffusion:
                    self.solver._n_steps_since_dvh_diffusion = pending_steps
                    return
                # The heat-kernel width is 4*kinematic_viscosity*dt_d. Passing one macro-step
                # here after waiting several steps under-diffuses by exactly
                # _n_steps_per_dvh_diffusion. Apply the full accumulated interval.
                diffusion_time_step_size = time_step_size * pending_steps
            new_p = self._apply_grid_diffusion(self._viscous_config, diffusion_time_step_size)
            if new_p is not None:
                M = len(new_p["position"])
                retention_bounds = self.stabilization_config.remove_particles_by_bounds
                if retention_bounds is not None and M > 0:
                    position = np.asarray(new_p["position"])
                    xmin, xmax, ymin, ymax, zmin, zmax = retention_bounds
                    inside = (
                        (xmin <= position[:, 0])
                        & (position[:, 0] <= xmax)
                        & (ymin <= position[:, 1])
                        & (position[:, 1] <= ymax)
                        & (zmin <= position[:, 2])
                        & (position[:, 2] <= zmax)
                    )
                    if not np.any(inside):
                        lo = position.min(axis=0)
                        hi = position.max(axis=0)
                        raise RuntimeError(
                            "Grid diffusion generated no particles inside the configured "
                            "retention domain; refusing to replace the wake. "
                            f"Generated extent: x=[{lo[0]:.6g}, {hi[0]:.6g}], "
                            f"y=[{lo[1]:.6g}, {hi[1]:.6g}], "
                            f"z=[{lo[2]:.6g}, {hi[2]:.6g}]."
                        )
                    # Regeneration occurred on the configured fixed lattice and
                    # already left every new particle inside the same retention
                    # domain.  The post-step O(N) retention scan would be a no-op.
                    self.stabilization.ctx.state.domain_bounds_enforced = bool(np.all(inside))
                with self.profiler.section("Particle replacement"):
                    self.solver.replace_vortex_particles(
                        position=new_p["position"],
                        velocity=new_p.get("velocity", np.zeros((M, 3), dtype=self.np_dtype)),
                        vortex_strength=new_p["vortex_strength"],
                        core_radius=new_p["core_radius"],
                        particle_volume=new_p["particle_volume"],
                        kinematic_viscosity=new_p.get("kinematic_viscosity"),
                        eddy_viscosity=new_p.get("eddy_viscosity"),
                        zone_id=new_p.get("zone_id", np.zeros(M, dtype=np.int32)),
                        group_id=new_p.get("group_id", np.zeros(M, dtype=np.int32)),
                        report_removal=False,
                    )
                # Velocity is left stale; it is recomputed before the next consumer.
            if self.viscous_scheme == "dvh":
                # Commit the accepted-step counter only after validation and
                # regeneration succeed; a failed transfer must not discard
                # the accumulated physical interval.
                self.solver._n_steps_since_dvh_diffusion = 0
        ti.sync()

    def _apply_grid_diffusion(
        self, vc: ViscousConfig, time_step_size: float
    ) -> dict[str, np.ndarray] | None:
        """Run DVH or GBD grid-based diffusion; return new particle dict."""
        # Fall back to particle viscosity when the scheme has no scalar ν.
        kinematic_viscosity = vc.kinematic_viscosity
        if kinematic_viscosity is None or kinematic_viscosity <= 0.0:
            n_part = self.particles.n_particles_total
            kinematic_viscosity = (
                float(self.particles.kinematic_viscosity_cpu()[:n_part].mean())
                if n_part > 0
                else 0.0
            )
        if self.viscous_scheme == "dvh":
            # DVH requires a scalar, source-independent heat kernel. Pass the actual effective-viscosity field so both
            # molecular and LES-induced spatial variation is validated rather
            # than silently replaced by its mean.
            N = self.particles.n_particles_total
            effective_viscosity = (
                self.particles.effective_viscosity_cpu()[:N].copy() if N > 0 else None
            )
            return self.physics.grid_based_diffusion(
                self.particles,
                time_step_size=time_step_size,
                particle_spacing=vc.dvh_grid_spacing,
                kinematic_viscosity=kinematic_viscosity,
                domain_padding=vc.dvh_domain_padding,
                regen_threshold=vc.dvh_threshold,
                regen_threshold_mode=vc.dvh_threshold_mode,
                rd_ratio=vc.dvh_support_radius_ratio,
                effective_viscosity=effective_viscosity,
            )
        else:
            # LES uses per-particle effective viscosity in the grid Laplacian.
            effective_viscosity = None
            if self.flow_model == "les":
                N = self.particles.n_particles_total
                if N > 0:
                    effective_viscosity = self.particles.effective_viscosity_cpu()
            return self.physics.gbd_diffusion(
                self.particles,
                time_step_size=time_step_size,
                particle_spacing=vc.gbd_grid_spacing,
                kinematic_viscosity=kinematic_viscosity,
                domain_padding=vc.gbd_domain_padding,
                regen_threshold=vc.gbd_threshold,
                regen_threshold_mode=vc.gbd_threshold_mode,
                effective_viscosity=effective_viscosity,
                remeshing_kernel=vc.gbd_remeshing_kernel,
                wall_fallback_limit=vc.gbd_wall_fallback_limit,
            )
