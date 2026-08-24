"""Time-evolution orchestration for the VPM solver.

:class:`EvolutionStepper` owns the per-step evolution algorithm: velocity and
gradient preparation, advection, stretching, coupled inviscid integration,
viscous diffusion, operator splitting, and the stabilization phases that
happen inside a step.  :class:`~source.solvers.vpm.core.solver.VPMSolver` is the
facade that composes subsystems; it delegates its step to the stepper and
keeps the diagnostics, checkpointing, and IO bookkeeping around the step.

The stepper holds a back-reference to the solver (``self.solver``) and
delegates attribute *reads* to it through :meth:`__getattr__`, so the solver
remains the single owner of solver state.  Attribute writes stay on the
stepper; the few mutations that must reach solver state (the step clock, the
one-shot warning flags, and the DVH fire counter) are written to
``self.solver`` explicitly.  No physics is implemented here: every numerical
update is performed on the subsystems the stepper calls.  Diffusing and
advecting kernels live in ``physics``; the stabilization workers live in
``stabilization`` and are dispatched here only at their scheduled phases.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from ..io.logging import Logging

if TYPE_CHECKING:
    from .solver import VPMSolver


class EvolutionStepper:
    """Advance the VPM particle field by one time step.

    Constructed by the solver after the subsystems it drives are initialized.
    """

    def __init__(self, solver: VPMSolver) -> None:
        self.solver = solver

    def __getattr__(self, name: str):
        return getattr(self.solver, name)

    def advance(self) -> None:
        """Advance the VPM solution by one time step.

        The inviscid update advances particle motion and, when enabled, vortex
        stretching. Viscous diffusion is then applied by operator splitting. Core
        spreading uses symmetric Strang splitting in the coupled integrator.
        """

        self.solver._domain_bounds_enforced_this_step = False
        self._apply_pending_particle_regeneration()
        self.stabilization.run_phase("pre_evolution")

        self._advance_time_step()

        self.particles.step = self.step
        self._debug_validate_particle_geometry("step entry")

        diagnostics_due = (
            self.logging_interval_steps > 0 and self.step % self.logging_interval_steps == 0
        )

        with self.profiler.step():
            if self.vlm_solver is not None:
                with self.profiler.section("VLM coupling"):
                    self.coupling.advance_vlm(self.time_step_size)

            if self.panel_solver is not None:
                with self.profiler.section("Panel coupling"):
                    self.coupling.advance_panel()

            _adv = (self.setup.advection.scheme if self.setup.advection else "RK3").upper()
            _gradients_required = (
                self.stretching_enabled
                or self.flow_model == "LES"
                or self.time_integration == "COUPLED"
                or self.stabilization_config.stretching_viscosity_coefficient > 0.0
                or (
                    self.stabilization_config.pedrizzetti_relaxation_enabled
                    and self.flow_model != "POTENTIAL"
                )
            )
            _defer_stationary_velocity = (
                _adv == "NONE"
                and not _gradients_required
                and self.vlm_solver is None
                and self.panel_solver is None
                and self.n_sources == 0
                and getattr(self.physics, "velocity_override", None) is None
            )
            panel_affects_particle_velocity = (
                self.panel_solver is not None
                and getattr(self.panel_solver, "coupling_scope", "full") == "full"
            )
            _fuse_vel_grad = (
                self.flow_model != "POTENTIAL"
                and _adv != "NONE"
                and _gradients_required
                and self.n_sources == 0
                and not panel_affects_particle_velocity
                and getattr(self.physics, "velocity_override", None) is None
            )
            if _fuse_vel_grad:
                with self.profiler.section("Velocity + gradients"):
                    self._update_velocity_and_gradients()
                velocity_k1_ready = True
            else:
                velocity_k1_ready = False
                if not _defer_stationary_velocity and (
                    _adv == "NONE"
                    or not _gradients_required
                    or self.n_sources > 0
                    or panel_affects_particle_velocity
                ):
                    with self.profiler.section("velocity"):
                        self._update_velocities()
                    velocity_k1_ready = True
                if _gradients_required:
                    with self.profiler.section("Velocity gradients"):
                        self._update_velocity_gradients()

            with self.profiler.section("LES update"):
                self._update_les_state()
                self.stabilization.update_residual_viscosity()

            # Relax against the same t_n gradient used by the strength update.
            self.stabilization.run_phase("pre_strength", profiler=self.profiler)

            coupled_update = self.time_integration == "COUPLED" and self.flow_model != "POTENTIAL"

            # Inviscid evolution, followed by viscous diffusion.
            if not coupled_update:
                with self.profiler.section("Advection"):
                    self._update_positions(precomputed_k1=velocity_k1_ready)

            if self.flow_model != "POTENTIAL":
                if self.stretching_enabled:
                    self._announce_strength_update()

                if coupled_update:
                    with self.profiler.section("Coupled advection + stretching"):
                        self._apply_coupled_update_with_subcycling(
                            self.time_step_size, precomputed_velocity_k1=_fuse_vel_grad
                        )
                else:
                    with self.profiler.section("Stretching"):
                        self._apply_stretching(self.time_step_size)
                    self._debug_validate_particle_geometry("stretching")
                    with self.profiler.section("Viscous diffusion"):
                        self._apply_viscous_diffusion(self.time_step_size)
                    self._debug_validate_particle_geometry("viscous diffusion")

                self.stabilization.run_phase("post_evolution", profiler=self.profiler)

            if diagnostics_due:
                with self.profiler.section("Flow integrals"):
                    if _defer_stationary_velocity:
                        # Evaluate deferred velocity on the end-of-step state.
                        self._update_velocities()
                    elif (
                        self.flow_model == "LES"
                        or self.stabilization_config.stretching_viscosity_coefficient > 0.0
                    ):
                        # Keep ν_eff consistent with the end-of-step diagnostics.
                        self._update_velocity_and_gradients(announce=False)
                        self._update_les_state()
                        self.stabilization.update_residual_viscosity()
                    self._update_all_flow_integrals()
            elif self.vlm_solver is not None:
                with self.profiler.section("VLM diagnostics"):
                    self._record_vlm_diagnostics()

            self.stabilization.run_phase("post_step", profiler=self.profiler)
            self._debug_validate_particle_geometry("particle retention")

            with self.profiler.section("Checkpoint / IO"):
                self._write_checkpoint()

        # The evolution kernels mutate particle source fields directly on the
        # device.  Publish one new source revision after the complete physical
        # state (including any topology-changing stabilization) is committed so
        # post-step boundary/panel queries cannot reuse the previous tree.
        self.particles.touch_state()
        self.solver._execute_scheduled_samplers()
        self.profiler.report_step()
        self.solver.wall_time = self.profiler.wall_time

        if self.timing_interval_steps > 0 and self.step % self.timing_interval_steps == 0:
            self.profiler.set_particle_count(self.particles.n_particles_total)
            self.profiler.report()

        if self.logging_interval_steps > 0 and self.step % self.logging_interval_steps == 0:
            self.log_diagnostics()

    def _apply_pending_particle_regeneration(self) -> None:
        """Regenerate externally modified GBD particles without advancing time."""
        if not self.solver._is_particle_regeneration_pending:
            return
        if self.viscous_scheme != "GBD":
            self.solver._is_particle_regeneration_pending = False
            return
        self._apply_viscous_diffusion(0.0)
        self.solver._is_particle_regeneration_pending = False

    def _debug_validate_particle_geometry(self, stage: str) -> None:
        """Validate active particle core_radius and particle_volume when stage tracing is enabled."""
        if os.environ.get("VPM_VALIDATE_STAGES", "0") != "1":
            return
        n = self.particles.n_particles_total
        if n == 0:
            return
        core_radius = self.particles.core_radius_cpu(use_cache=False)
        particle_volume = self.particles.particle_volume_cpu(use_cache=False)
        invalid_radii = ~np.isfinite(core_radius) | (core_radius <= 0.0)
        invalid_volumes = ~np.isfinite(particle_volume) | (particle_volume <= 0.0)
        n_bad_radii = int(np.count_nonzero(invalid_radii))
        n_bad_volumes = int(np.count_nonzero(invalid_volumes))
        Logging.message(
            f"[Integrity:{stage}] N={n} core_radius=[{np.nanmin(core_radius):.6e}, "
            f"{np.nanmax(core_radius):.6e}] bad={n_bad_radii}; "
            f"particle_volume=[{np.nanmin(particle_volume):.6e}, {np.nanmax(particle_volume):.6e}] "
            f"bad={n_bad_volumes}"
        )
        if n_bad_radii or n_bad_volumes:
            bad_radius_index = int(np.flatnonzero(invalid_radii)[0]) if n_bad_radii else -1
            bad_volume_index = int(np.flatnonzero(invalid_volumes)[0]) if n_bad_volumes else -1
            raise RuntimeError(
                f"Invalid VPM particle geometry after {stage}: "
                f"radius_bad={n_bad_radii} first={bad_radius_index}, "
                f"volume_bad={n_bad_volumes} first={bad_volume_index}"
            )

    def _advance_time_step(self) -> None:
        """Advance the step counter and physical time."""

        self.solver.step += 1

        self.solver.time = round(self.solver.time + self.time_step_size, 12)

        Logging.message(
            f"\n[VPM][Step] step={self.step:d} time_s={self.time:.6e}",
            flush=True,
        )

    def _update_velocities(self) -> None:
        """Evaluate self-induced particle velocity and optional body/source contributions."""
        Logging.message(f"[VPM][Velocity] method={self.physics.velocity_method.lower()}")
        self.physics.compute_self_induced_velocity(
            self.particles.position,
            self.particles.vortex_strength,
            self.particles.core_radius,
            self.particles.velocity,
            self.particles.velocity_background,
            self.particles.n_particles_total,
        )

        if (
            self.panel_solver is not None
            and getattr(self.panel_solver, "coupling_scope", "full") == "full"
        ):
            # The panel solver reads velocity written by an asynchronous Taichi kernel.
            ti.sync()
            self.panel_solver.compute_induced_velocity_direct(self.particles)

        if self.n_sources > 0:
            self.physics.kernels["compute_target_source_velocity_kernel"](
                self.particles.position,
                self.source_position,
                self.source_strength,
                self.source_core_radius,
                self.particles.velocity,
                self.particles.n_particles_total,
                self.n_sources,
            )

    def _update_velocity_gradients(self, announce: bool = True) -> None:
        """Evaluate particle velocity gradients with the configured direct or tree method."""
        use_treecode = bool(self.setup.velocity and self.setup.velocity.method == "TREECODE")
        theta = self.setup.velocity.theta if self.setup.velocity else 0.5

        if use_treecode:
            if announce:
                Logging.message(f"[VPM][velocity_gradient] method=treecode theta={theta:g}")
            self.physics.compute_velocity_gradients_hierarchical(self.particles, theta=theta)
        else:
            if announce:
                Logging.message("[VPM][velocity_gradient] method=direct")
            self.physics.compute_velocity_gradients(self.particles)

    def _update_velocity_and_gradients(self, announce: bool = True) -> None:
        """Evaluate particle velocity and ``∇u`` in one direct pass or tree traversal."""
        use_treecode = bool(self.setup.velocity and self.setup.velocity.method == "TREECODE")
        theta = self.setup.velocity.theta if self.setup.velocity else 0.5
        if use_treecode:
            if announce:
                Logging.message(
                    f"[VPM][Velocity] fields=velocity,gradient method=treecode theta={theta:g}"
                )
            self.physics.compute_velocity_and_gradient_hierarchical(self.particles, theta=theta)
        else:
            if announce:
                Logging.message("[VPM][Velocity] fields=velocity,gradient method=direct")
            self.physics.compute_velocity_and_gradient(self.particles)

    def _update_les_state(self, time_step_size: float | None = None) -> None:
        """Update LES viscosity from the current strain-rate field."""
        if self.flow_model == "LES":
            self.turbulence_model.compute(
                self.particles,
                time_step_size=self.time_step_size if time_step_size is None else time_step_size,
            )
            if self.axisymmetric_axis >= 0:
                self._validate_axisymmetric_orbits()
                self.physics.average_axisymmetric_scalar(
                    self.particles.eddy_viscosity,
                    self.particles.zone_id,
                    len(self.particles),
                )
                self.physics.average_axisymmetric_scalar(
                    self.particles.effective_viscosity,
                    self.particles.zone_id,
                    len(self.particles),
                )

    def _update_strength(self, time_step_size: float | None = None, announce: bool = True) -> None:
        """Advance vortex stretching, then viscous diffusion, over ``time_step_size``."""
        if self.flow_model == "POTENTIAL":
            return

        if announce:
            self._announce_strength_update()

        time_step_size = self.time_step_size if time_step_size is None else time_step_size
        self._apply_stretching(time_step_size)
        self._apply_viscous_diffusion(time_step_size)

    def _announce_strength_update(self) -> None:
        """Log the stretching formulation used for this strength update."""
        effective_mode = self._effective_stretching_mode()
        Logging.message(f"[VPM][Stretching] formulation={effective_mode.lower()}")

    def _effective_stretching_mode(self) -> str:
        """Return the user-selected stretching formulation."""
        return self.stretching_mode

    def _apply_stretching(self, time_step_size: float) -> None:
        """Advance the configured vortex-stretching equation once per ``time_step_size``."""
        if self.stretching_enabled:
            # Warn once when the explicit stretching step exceeds the strain-based target.
            if not self._stretch_time_step_size_warned:
                gradient = self.particle_velocity_gradient
                strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
                max_strain_rate = (
                    float(np.max(np.abs(np.linalg.eigvalsh(strain)))) if len(strain) else 0.0
                )
                if max_strain_rate > 0.0:
                    recommended_time_step_size = 0.2 / max_strain_rate
                    if time_step_size > recommended_time_step_size:
                        Logging.stretching_time_step_size_warning(
                            time_step_size,
                            recommended_time_step_size,
                            max_strain_rate,
                        )
                        self.solver._stretch_time_step_size_warned = True
            self.physics.vortex_stretching(
                self.particles,
                time_step_size=time_step_size,
                scheme=self.stretching_scheme,
                mode=self.stretching_mode,
                use_treecode=self.stretching_use_treecode,
                treecode_theta=self.stretching_treecode_theta,
            )
            ti.sync()

    def _validate_axisymmetric_orbits(self) -> None:
        """Reject malformed orbit IDs before applying rotational stage averages."""
        if self.axisymmetric_axis < 0 or self._axisymmetric_orbits_validated:
            return
        position = self.particle_position.astype(np.float64)
        orbit_id = self.particle_zone_id.astype(np.int64)
        group_id = self.particle_group_id.astype(np.int64)
        count = len(position)
        if count == 0:
            return
        if np.any(orbit_id < 0) or int(orbit_id.max()) >= count:
            raise ValueError(
                "axisymmetric particles require non-negative zone_id orbit labels below "
                "the particle count"
            )

        present = np.unique(orbit_id)
        if not np.array_equal(present, np.arange(len(present))):
            raise ValueError("axisymmetric zone_id orbit labels must be contiguous from zero")

        axis = self.axisymmetric_axis
        b = (axis + 1) % 3
        c = (axis + 2) % 3
        radial = np.hypot(position[:, b], position[:, c])
        scale = max(1.0, float(np.abs(position).max(initial=0.0)))
        geometry_tolerance = 256.0 * np.finfo(self.np_dtype).eps * scale
        angle_tolerance = 512.0 * np.finfo(self.np_dtype).eps
        for orbit in present:
            selected = orbit_id == orbit
            orbit_count = int(np.count_nonzero(selected))
            if orbit_count < 8:
                raise ValueError(
                    f"axisymmetric orbit {orbit} has {orbit_count} particles; at least 8 are required"
                )
            if np.ptp(group_id[selected]) != 0:
                raise ValueError(f"axisymmetric orbit {orbit} spans more than one particle group")
            if (
                np.ptp(position[selected, axis]) > geometry_tolerance
                or np.ptp(radial[selected]) > geometry_tolerance
            ):
                raise ValueError(
                    f"axisymmetric orbit {orbit} does not have one axial and radial coordinate"
                )
            angles = np.sort(
                np.mod(np.arctan2(position[selected, c], position[selected, b]), 2 * np.pi)
            )
            gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
            expected_gap = 2.0 * np.pi / orbit_count
            if np.max(np.abs(gaps - expected_gap)) > angle_tolerance:
                raise ValueError(f"axisymmetric orbit {orbit} is not uniformly periodic")

        self.solver._axisymmetric_orbits_validated = True

    def _apply_coupled_advection_stretching(
        self, time_step_size: float, *, precomputed_velocity_k1: bool = False
    ) -> None:
        """Advance position and vortex_strength at the same Runge--Kutta stages."""
        self._validate_axisymmetric_orbits()
        self.physics.update_positions_and_strengths(
            self.particles,
            time_step_size=time_step_size,
            scheme=self.stretching_scheme,
            mode=self.stretching_mode,
            use_treecode=self.stretching_use_treecode,
            treecode_theta=self.stretching_treecode_theta,
            conserve_moments=self.stretching_conserve_moments,
            conserve_energy=self.stretching_conserve_energy,
            axisymmetric_axis=self.axisymmetric_axis,
            precomputed_velocity_k1=precomputed_velocity_k1,
        )
        ti.sync()

    def _coupled_stable_time_step_size(self, remaining_time_step_size: float) -> float:
        """Return a strain- and displacement-limited coupled substep."""
        grad = self.particle_velocity_gradient
        stable_time_step_size = float(remaining_time_step_size)
        if len(grad) and self.coupled_max_strain_increment is not None:
            strain = 0.5 * (grad + np.swapaxes(grad, 1, 2))
            max_strain = float(np.max(np.abs(np.linalg.eigvalsh(strain))))
            if np.isfinite(max_strain) and max_strain > 0.0:
                stable_time_step_size = min(
                    stable_time_step_size,
                    self.coupled_max_strain_increment / max_strain,
                )

        spacing = getattr(self._viscous_config, "particle_spacing", None)
        if (
            self.coupled_max_advection_fraction is not None
            and spacing is not None
            and spacing > 0.0
        ):
            velocity = self.particle_velocity
            max_speed = float(np.linalg.norm(velocity, axis=1).max()) if len(velocity) else 0.0
            if np.isfinite(max_speed) and max_speed > 0.0:
                stable_time_step_size = min(
                    stable_time_step_size,
                    self.coupled_max_advection_fraction * float(spacing) / max_speed,
                )
        return max(stable_time_step_size, np.finfo(float).eps)

    def _apply_coupled_update_with_subcycling(
        self, time_step_size: float, *, precomputed_velocity_k1: bool
    ) -> None:
        """Advance one macro step without clipping an inadmissible RK increment."""
        self.physics.rate_projection_max_correction_ratio = 0.0
        remaining = float(time_step_size)
        substeps = 0
        reuse_velocity = bool(precomputed_velocity_k1)
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(time_step_size))

        while remaining > tolerance:
            substep_size = min(remaining, self._coupled_stable_time_step_size(remaining))
            substeps += 1
            if substeps > self.coupled_max_substeps:
                raise RuntimeError(
                    "Coupled VPM step exceeded coupled_max_substeps. "
                    "The particle field is no longer temporally admissible at the "
                    "requested macro time_step_size; reduce time_step_size or refine the "
                    "particle spacing."
                )

            if self.viscous_scheme == "CS":
                # Symmetric core-spreading split around the coupled inviscid update.
                self._apply_core_spreading_diffusion(0.5 * substep_size)
                reuse_velocity = False

            target_moments = self._current_kernel_moments()
            self._apply_coupled_advection_stretching(
                substep_size, precomputed_velocity_k1=reuse_velocity
            )
            self._restore_coupled_step_moments(target_moments)

            if self.viscous_scheme == "CS":
                self._apply_core_spreading_diffusion(0.5 * substep_size)

            remaining -= substep_size
            if remaining <= tolerance:
                break

            # Refresh stability bounds and, for LES, eddy viscosity before the next substep.
            self._update_velocity_and_gradients()
            self._update_les_state()
            self.stabilization.update_residual_viscosity()
            reuse_velocity = self.viscous_scheme == "NONE"

        if substeps > 1:
            Logging.message(
                f"[VPM][TimeIntegration] substeps={substeps} time_step_size_s={time_step_size:.3e}"
            )

        if self.viscous_scheme in {"RWM", "DVH", "GBD"}:
            self._apply_viscous_diffusion(time_step_size)

    def _current_kernel_moments(self):
        """Return vortex_strength and both impulses for the active blob kernel."""
        if not self.stretching_conserve_moments or len(self.particles) == 0:
            return None
        from ..stabilization.filament_refinement import particle_moments

        return particle_moments(
            self.particles.position_cpu(use_cache=False).astype(np.float64),
            self.particles.vortex_strength_cpu(use_cache=False).astype(np.float64),
            self.particles.core_radius_cpu(use_cache=False).astype(np.float64),
            angular_core_coefficient=self.physics._angular_core_coefficient,
        )

    def _restore_coupled_step_moments(self, target_moments) -> None:
        """Correct finite-RK drift in the conserved coupled-step moments."""
        if target_moments is None or len(self.particles) == 0:
            return
        from ..stabilization.divergence_relaxation import (
            _MomentNullspace,
            invariant_rows,
        )
        from ..stabilization.filament_refinement import particle_moments

        position = self.particles.position_cpu(use_cache=False).astype(np.float64)
        vortex_strength = self.particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
        core_radius = self.particles.core_radius_cpu(use_cache=False).astype(np.float64)
        particle_volume = self.particles.particle_volume_cpu(use_cache=False).astype(np.float64)
        core_coefficient = self.physics._angular_core_coefficient
        current = particle_moments(
            position,
            vortex_strength,
            core_radius,
            angular_core_coefficient=core_coefficient,
        )
        moment_change = np.concatenate(
            (
                target_moments[0] - current[0],
                target_moments[2] - current[2],
                target_moments[3] - current[3],
            )
        )
        nullspace = _MomentNullspace(
            invariant_rows(
                position,
                core_radius,
                angular_core_coefficient=core_coefficient,
            ),
            particle_volume,
        )
        correction = nullspace.correction_for_moment_change(moment_change)
        correction_relative = float(
            np.linalg.norm(correction) / max(np.linalg.norm(vortex_strength), np.finfo(float).tiny)
        )
        self.update_particle_vortex_strength(
            np.ones(len(vortex_strength), dtype=bool),
            correction.astype(self.np_dtype),
        )
        self.physics.rate_projection_max_correction_ratio = max(
            self.physics.rate_projection_max_correction_ratio,
            correction_relative,
        )

        uploaded = self.particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
        restored = particle_moments(
            position,
            uploaded,
            core_radius,
            angular_core_coefficient=core_coefficient,
        )
        scale = max(target_moments[1], np.finfo(float).tiny)
        impulse_scale = max(
            0.5 * float(np.linalg.norm(np.cross(position, vortex_strength), axis=1).sum()),
            np.finfo(float).tiny,
        )
        angular_terms = (
            np.cross(position, np.cross(position, vortex_strength)) / 3.0
            - core_coefficient * core_radius[:, None] ** 2 * vortex_strength
        )
        angular_scale = max(
            float(np.linalg.norm(angular_terms, axis=1).sum()),
            np.finfo(float).tiny,
        )
        errors = (
            float(np.linalg.norm(restored[0] - target_moments[0])) / scale,
            float(np.linalg.norm(restored[2] - target_moments[2])) / impulse_scale,
            float(np.linalg.norm(restored[3] - target_moments[3])) / angular_scale,
        )
        if max(errors) > 4096.0 * np.finfo(self.np_dtype).eps:
            raise RuntimeError(
                "coupled-step moment projection exceeded its roundoff allowance: "
                f"vortex_strength={errors[0]:.3e}, linear_impulse={errors[1]:.3e}, "
                f"angular_impulse={errors[2]:.3e}"
            )

    def _apply_core_spreading_diffusion(self, time_step_size: float) -> None:
        """Advance Gaussian core spreading and optionally restore configured moments."""
        if time_step_size <= 0.0 or len(self.particles) == 0:
            return
        if not self.stretching_conserve_moments:
            self.physics.core_spreading_diffusion(self.particles, time_step_size)
            return

        from ..stabilization.divergence_relaxation import (
            _MomentNullspace,
            invariant_rows,
        )
        from ..stabilization.filament_refinement import particle_moments

        position = self.particles.position_cpu(use_cache=False).astype(np.float64)
        vortex_strength = self.particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
        core_radius = self.particles.core_radius_cpu(use_cache=False).astype(np.float64)
        particle_volume = self.particles.particle_volume_cpu(use_cache=False).astype(np.float64)
        core_coefficient = self.physics._angular_core_coefficient
        before = particle_moments(
            position,
            vortex_strength,
            core_radius,
            angular_core_coefficient=core_coefficient,
        )

        self.physics.core_spreading_diffusion(self.particles, time_step_size)
        new_core_radius = self.particles.core_radius_cpu(use_cache=False).astype(np.float64)
        uncorrected = particle_moments(
            position,
            vortex_strength,
            new_core_radius,
            angular_core_coefficient=core_coefficient,
        )
        moment_change = np.concatenate(
            (before[0] - uncorrected[0], before[2] - uncorrected[2], before[3] - uncorrected[3])
        )
        nullspace = _MomentNullspace(
            invariant_rows(
                position,
                new_core_radius,
                angular_core_coefficient=core_coefficient,
            ),
            particle_volume,
        )
        correction = nullspace.correction_for_moment_change(moment_change)
        self.solver.core_spreading_correction_relative = float(
            np.linalg.norm(correction) / max(np.linalg.norm(vortex_strength), np.finfo(float).tiny)
        )
        self.update_particle_vortex_strength(
            np.ones(len(vortex_strength), dtype=bool),
            correction.astype(self.np_dtype),
        )

        uploaded = self.particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
        after = particle_moments(
            position,
            uploaded,
            new_core_radius,
            angular_core_coefficient=core_coefficient,
        )
        impulse_scale = max(
            0.5 * float(np.linalg.norm(np.cross(position, vortex_strength), axis=1).sum()),
            np.finfo(float).tiny,
        )
        angular_terms = (
            np.cross(position, np.cross(position, vortex_strength)) / 3.0
            - core_coefficient * core_radius[:, None] ** 2 * vortex_strength
        )
        angular_scale = max(
            float(np.linalg.norm(angular_terms, axis=1).sum()),
            np.finfo(float).tiny,
        )
        errors = {
            "vortex_strength": float(np.linalg.norm(after[0] - before[0]))
            / max(before[1], np.finfo(float).tiny),
            "linear_impulse": float(np.linalg.norm(after[2] - before[2])) / impulse_scale,
            "angular_impulse": float(np.linalg.norm(after[3] - before[3])) / angular_scale,
        }
        roundoff_limit = 4096.0 * np.finfo(self.np_dtype).eps
        if max(errors.values()) > roundoff_limit:
            raise RuntimeError(
                "core-spreading moment projection exceeded its roundoff allowance: "
                + ", ".join(f"{name}={value:.3e}" for name, value in errors.items())
            )

    def _apply_viscous_diffusion(self, time_step_size: float) -> None:
        """Dispatch viscous diffusion by configured scheme."""
        if self.viscous_scheme == "NONE":
            return

        if self.viscous_scheme == "CS":
            Logging.message(f"[VPM][Diffusion] scheme=CS time_step_size_s={time_step_size:.3e}")
            self._apply_core_spreading_diffusion(time_step_size)
        elif self.viscous_scheme == "RWM":
            Logging.message(f"[VPM][Diffusion] scheme=RWM time_step_size_s={time_step_size:.3e}")
            self.physics.random_walk_method_diffusion(self.particles, time_step_size=time_step_size)
        elif self.viscous_scheme in ("DVH", "GBD"):
            # DVH fires only when its fixed diffusion increment has accumulated.
            diffusion_time_step_size = time_step_size
            if self.viscous_scheme == "DVH" and self._n_steps_per_dvh_diffusion > 1:
                self.solver._n_steps_since_dvh_diffusion += 1
                if self.solver._n_steps_since_dvh_diffusion < self._n_steps_per_dvh_diffusion:
                    return
                self.solver._n_steps_since_dvh_diffusion = 0
                # The heat-kernel width is 4*kinematic_viscosity*dt_d. Passing one macro-step
                # here after waiting several steps under-diffuses by exactly
                # _n_steps_per_dvh_diffusion. Apply the full accumulated interval.
                diffusion_time_step_size = time_step_size * self._n_steps_per_dvh_diffusion
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
                    self.solver._domain_bounds_enforced_this_step = bool(np.all(inside))
                self.replace_vortex_particles(
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
                # Velocity is intentionally left stale; it is recomputed before the next consumer.
        ti.sync()

    def _apply_grid_diffusion(self, vc, time_step_size: float):
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
        if self.viscous_scheme == "DVH":
            # LES uses per-particle effective viscosity for the heat-kernel width.
            effective_viscosity = None
            if self.flow_model == "LES":
                N = self.particles.n_particles_total
                if N > 0:
                    effective_viscosity = self.particles.effective_viscosity_cpu()
            fields = (
                f"[VPM][DVH] mode=diffusion time_step_size_s={time_step_size:.3e} "
                f"h_m={vc.dvh_grid_spacing:.3e} "
                f"kinematic_viscosity_m2_s={kinematic_viscosity:.3e} "
                f"threshold={vc.dvh_threshold:.2e}"
            )
            if effective_viscosity is not None and kinematic_viscosity > 0.0:
                fields += (
                    " max_effective_to_kinematic_viscosity_ratio="
                    f"{float(effective_viscosity.max()) / kinematic_viscosity:.2f}"
                )
            Logging.message(fields)
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
                max_nodes=getattr(vc, "dvh_max_nodes", None),
            )
        else:
            # LES uses per-particle effective viscosity in the grid Laplacian.
            effective_viscosity = None
            if self.flow_model == "LES":
                N = self.particles.n_particles_total
                if N > 0:
                    effective_viscosity = self.particles.effective_viscosity_cpu()
            mode = "regeneration" if time_step_size == 0.0 else "diffusion"
            fields = (
                f"[VPM][GBD] mode={mode} time_step_size_s={time_step_size:.3e} "
                f"h_m={vc.gbd_grid_spacing:.3e} "
                f"kinematic_viscosity_m2_s={kinematic_viscosity:.3e} "
                f"threshold={vc.gbd_threshold:.2e}"
            )
            if effective_viscosity is not None and kinematic_viscosity > 0.0:
                fields += (
                    " max_effective_to_kinematic_viscosity_ratio="
                    f"{float(effective_viscosity.max()) / kinematic_viscosity:.2f}"
                )
            Logging.message(fields)
            return self.physics.gbd_diffusion(
                self.particles,
                time_step_size=time_step_size,
                particle_spacing=vc.gbd_grid_spacing,
                kinematic_viscosity=kinematic_viscosity,
                domain_padding=vc.gbd_domain_padding,
                regen_threshold=vc.gbd_threshold,
                regen_threshold_mode=vc.gbd_threshold_mode,
                effective_viscosity=effective_viscosity,
                max_nodes=getattr(vc, "gbd_max_nodes", None),
            )

    def _update_positions(
        self, time_step_size: float | None = None, precomputed_k1: bool = False
    ) -> None:
        """Advect particles with the configured time integrator.

        A precomputed first-stage velocity may be reused when velocity and gradients
        were evaluated together at the beginning of the step.
        """
        if self.advection_scheme == "NONE":
            return
        self.physics.update_positions(
            self.particles,
            self.time_step_size if time_step_size is None else time_step_size,
            scheme=self.advection_scheme,
            precomputed_k1=precomputed_k1,
        )
