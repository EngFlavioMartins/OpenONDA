"""Master controller for the VPM stabilization mechanisms.

Every stabilization scheme in this package is a *worker*: it proposes a new
particle field and enforces its own internal admissibility rules, which are
part of the algorithm and stay with it.  :class:`StabilizationManager` is the
master.  It owns the schedule, dispatches the workers in a fixed order, and
judges the outcome of each event against one small set of criteria that is the
same for every mechanism.

The master's criteria are deliberately global, cheap, and physical.  They are
formed from :class:`StabilizationHealth`, an O(N) snapshot of the particle
cloud taken from arrays the solver already holds:

``vortex_strength error``   ``|sum vortex_strength_after - sum vortex_strength_before| / sum |vortex_strength|``
    Total vector strength is invariant under any admissible reassignment of
    a vortex field.  Only mechanisms that claim to preserve it are held to it;
    Pedrizzetti relaxation rotates vortex_strength and reports its transfer instead.

``strength growth``     ``(sum |vortex_strength|_after - sum |vortex_strength|_before) / sum |vortex_strength|_before``
``vorticity growth``    ``(max |omega|_after - max |omega|_before) / max |omega|_before``
    Both are one-sided.  A stabilization event may remove strength or peak
    vorticity — that is what most of these schemes are for, and each worker
    bounds its own dissipation — but no event may amplify the field.  Growth of
    the peak vorticity is the signature of the instability these mechanisms
    exist to suppress, so a scheme that produces it has failed.

    Both are also *discretization-dependent*: ``sum |vortex_strength|`` is a total
    variation over the particle set and ``max |omega|`` a per-particle maximum,
    so neither is comparable across an event that rebuilds the cloud on a new
    grid.  They are therefore measured and reported for every mechanism, but
    only enforced on the ones that keep the discretization they were given.
    A rebuilding worker is held to vortex_strength alone, and to the energy and
    enstrophy limits it enforces on itself.

Anything finer than this belongs inside the worker that can act on it, not in
a solver-level diagnostics table.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: August 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..io.logging import Logging
from .context import StabilizationContext
from .operators import StabilizationOperators

if TYPE_CHECKING:
    from ..config.types import StabilizationConfig


class StabilizationError(RuntimeError):
    """A stabilization event failed the master's global acceptance criteria."""


@dataclass(frozen=True)
class StabilizationHealth:
    """Global physical state of the particle cloud, measured in O(N)."""

    n_particles_total: int
    vortex_strength: np.ndarray
    vortex_strength_magnitude_sum: float
    max_vortex_strength_magnitude: float
    max_vorticity_magnitude: float

    @classmethod
    def measure(cls, particles) -> StabilizationHealth:
        """Snapshot the cloud from the vortex_strength and particle_volume already on hand."""
        count = particles.n_particles_total
        if count == 0:
            return cls(0, np.zeros(3), 0.0, 0.0, 0.0)
        vortex_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64)
        particle_volume = np.asarray(particles.particle_volume_cpu(), dtype=np.float64)
        magnitude = np.linalg.norm(vortex_strength, axis=1)
        vorticity = magnitude / np.maximum(particle_volume, np.finfo(float).tiny)
        return cls(
            n_particles_total=count,
            vortex_strength=vortex_strength.sum(axis=0),
            vortex_strength_magnitude_sum=float(magnitude.sum(dtype=np.float64)),
            max_vortex_strength_magnitude=float(magnitude.max(initial=0.0)),
            max_vorticity_magnitude=float(vorticity.max(initial=0.0)),
        )


# Lifecycle phases of one VPM time step, in execution order, with the worker
# methods that are allowed to act in each.  The schedule is owned here: a new
# stabilization worker registers in one of these tuples (and opens the manager
# call in ``run_phase``); the solver's step loop never grows another apply_*()
# call.  Phase names describe *where in the step* a worker may act:
#
# - ``pre_evolution``   at step entry, before the particle field changes.
# - ``pre_strength``    after velocity/gradients (and LES residual viscosity)
#                       are brought to the ``t_n`` state, before the strength
#                       update the relaxation must inform.
# - ``post_evolution``  after advection/stretching/diffusion have modified the
#                       field, while the updated gradients still describe it.
# - ``post_step``       end of the step, after diagnostics/IO.
PHASES: dict[str, tuple[str, ...]] = {
    "pre_evolution": ("capture_reference_state",),
    "pre_strength": ("apply_relaxation",),
    "post_evolution": (
        "apply_filament_refinement",
        "apply_divergence_relaxation",
        "apply_regularization",
    ),
    "post_step": ("apply_retention",),
}

# Profiler section labels for the phased workers, kept stable so the runtime
# timing report reads the same after a worker is re-registered under a phase.
_PHASE_SECTION_LABELS: dict[str, str] = {
    "apply_relaxation": "Pedrizzetti relaxation",
    "apply_filament_refinement": "Filament refinement",
    "apply_divergence_relaxation": "Divergence relaxation",
    "apply_regularization": "Conservative regularization",
    "apply_retention": "Particle retention",
}


class StabilizationManager:
    """Schedule the stabilization workers and audit what they did.

    The solver owns one instance and the step loop calls :meth:`run_phase` at
    each lifecycle phase declared in :data:`PHASES`; no stabilization state or
    bookkeeping lives on the solver itself.  The schedule — which worker runs
    in which phase — is owned here, so adding a stabilization mechanism means
    registering its worker in ``PHASES`` rather than growing the step loop.
    """

    def __init__(self, context: StabilizationContext) -> None:
        self.ctx = context
        self.config: StabilizationConfig = context.config
        # The stabilization subsystem owns its own kernels and fields; the
        # physics engine has no dependency on it.
        self.operators = StabilizationOperators(
            context.compute_dtype, int(context.particles._max_particles)
        )
        self.events = 0
        self.regularization_events = 0
        # A readable placeholder rather than "": the record goes to CSV, and an
        # empty field reads back as a missing value.
        self.last_mechanism = "none"
        self.last_vortex_strength_error = 0.0
        self.last_strength_growth = 0.0
        self.last_vorticity_growth = 0.0
        self.max_vorticity_growth = 0.0
        self.residual_viscosity_coefficient = self.config.stretching_viscosity_coefficient
        self._last_residual_feedback_step = -1
        # Lineage and reference state the workers need across events.  It is
        # part of the restart state, so the checkpoint reads and writes it.
        self.reference_vortex_strength: np.ndarray | None = None
        self.reference_lengths: np.ndarray | None = None
        self.reference_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- master criteria -------------------------------------------------------

    def measure(self) -> StabilizationHealth:
        """Return the current cloud health."""
        return StabilizationHealth.measure(self.ctx.particles)

    def accept(
        self,
        mechanism: str,
        before: StabilizationHealth,
        *,
        conserves_vortex_strength: bool = True,
        preserves_discretization: bool = True,
        detail: str = "",
    ) -> StabilizationHealth:
        """Judge one completed event and record it, or raise.

        The comparison is made on the uploaded field, so it also covers the
        cast to the solver's production precision.  Every number is recorded;
        the two flags say which of them this mechanism can be held to.
        """
        cfg = self.config
        after = self.measure()
        scale = max(before.vortex_strength_magnitude_sum, np.finfo(float).tiny)
        vortex_strength_error = (
            float(np.linalg.norm(after.vortex_strength - before.vortex_strength)) / scale
        )
        strength_growth = (
            after.vortex_strength_magnitude_sum - before.vortex_strength_magnitude_sum
        ) / scale
        vorticity_growth = (after.max_vorticity_magnitude - before.max_vorticity_magnitude) / max(
            before.max_vorticity_magnitude, np.finfo(float).tiny
        )

        self.events += 1
        self.last_mechanism = mechanism
        # Recorded for every mechanism.  A rotation carries vortex_strength with it
        # by construction, so for those this number is the reported transfer
        # rather than an error, and only the gate below is skipped.
        self.last_vortex_strength_error = vortex_strength_error
        self.last_strength_growth = strength_growth
        self.last_vorticity_growth = vorticity_growth
        self.max_vorticity_growth = max(self.max_vorticity_growth, vorticity_growth)

        rows = [
            ("mechanism", mechanism),
            ("particles, before", f"{before.n_particles_total:,}"),
            ("particles, after", f"{after.n_particles_total:,}"),
            ("vortex strength closure, relative", f"{vortex_strength_error:.2e}"),
            ("vortex strength sum, relative change", f"{strength_growth:+.2e}"),
            ("peak vorticity, relative change", f"{vorticity_growth:+.2e}"),
        ]
        if detail:
            rows.append(("detail", detail))
        Logging.record("stabilization", *rows)

        checks = []
        if conserves_vortex_strength:
            checks.append(
                ("vortex_strength error", vortex_strength_error, cfg.max_vortex_strength_error)
            )
        if preserves_discretization:
            checks += [
                ("strength growth", strength_growth, cfg.max_vortex_strength_growth),
                ("peak-vorticity growth", vorticity_growth, cfg.max_vorticity_growth),
            ]
        for name, value, limit in checks:
            if not np.isfinite(value) or value > limit:
                raise StabilizationError(
                    f"{mechanism} produced a {name} of {value:.3e}, beyond the admissible "
                    f"{limit:.3e}"
                )
        return after

    @property
    def diagnostics(self) -> dict[str, float | int | str]:
        """The compact per-step record exported with the flow integrals."""
        return {
            "n_stabilization_events": self.events,
            "n_regularization_events": self.regularization_events,
            "last_stabilization_mechanism": self.last_mechanism,
            "stabilization_vortex_strength_error": self.last_vortex_strength_error,
            "stabilization_vortex_strength_growth": self.last_strength_growth,
            "stabilization_vorticity_growth": self.last_vorticity_growth,
            "max_stabilization_vorticity_growth": self.max_vorticity_growth,
            "stretching_viscosity_feedback_coefficient": (
                self.residual_viscosity_coefficient
            ),
        }

    def restore_diagnostics(self, values: dict) -> None:
        """Reload the master's record from a checkpoint."""
        self.events = int(values.get("n_stabilization_events", self.events))
        self.regularization_events = int(
            values.get("n_regularization_events", self.regularization_events)
        )
        self.last_mechanism = str(values.get("last_stabilization_mechanism", self.last_mechanism))
        self.residual_viscosity_coefficient = float(
            values.get(
                "stretching_viscosity_feedback_coefficient",
                self.residual_viscosity_coefficient,
            )
        )
        for key, attribute in (
            ("stabilization_vortex_strength_error", "last_vortex_strength_error"),
            ("stabilization_vortex_strength_growth", "last_strength_growth"),
            ("stabilization_vorticity_growth", "last_vorticity_growth"),
            ("max_stabilization_vorticity_growth", "max_vorticity_growth"),
        ):
            if key in values:
                setattr(self, attribute, float(values[key]))

    def active_mechanisms(self) -> tuple[str, ...]:
        """Names of the mechanisms this configuration switches on."""
        cfg = self.config
        active = []
        if cfg.stretching_viscosity_coefficient > 0.0:
            active.append("residual stretching viscosity")
        if cfg.pedrizzetti_relaxation_enabled:
            active.append("Pedrizzetti relaxation")
        if cfg.filament_refinement.enabled:
            active.append("filament refinement")
        if cfg.divergence_relaxation.enabled:
            active.append("divergence relaxation")
        if cfg.regularization_interval_steps > 0:
            active.append("conservative regularization")
        if cfg.remove_particles_by_bounds is not None:
            active.append("bounded-domain retention")
        return tuple(active)

    def _due(self, interval_steps: int, start_step: int) -> bool:
        step = self.ctx.step()
        return (
            interval_steps > 0 and step >= start_step and (step - start_step) % interval_steps == 0
        )

    # -- lifecycle phases -------------------------------------------------------

    def run_phase(self, phase: str, profiler=None) -> None:
        """Run every stabilization worker scheduled in ``phase``.

        The phase schedule is the package-level :data:`PHASES` table, so the
        solver's step loop only ever calls this method — it never grows its own
        ``apply_*()`` sequence.  Each worker keeps its own admissibility rules
        and gating; this method only dispatches them in the declared order.

        When a profiler is supplied, each worker is timed under its stable
        :data:`_PHASE_SECTION_LABELS` section name so the timing report reads
        exactly as before the workers were folded into phases.
        """
        for worker_name in PHASES[phase]:
            worker = getattr(self, worker_name)
            if profiler is not None and worker_name in _PHASE_SECTION_LABELS:
                with profiler.section(_PHASE_SECTION_LABELS[worker_name]):
                    worker()
            else:
                worker()

    def phase_workers(self, phase: str) -> tuple[str, ...]:
        """Return the ordered worker method names scheduled in ``phase``."""
        return PHASES[phase]

    # -- mechanisms ------------------------------------------------------------

    def capture_reference_state(self) -> None:
        """Capture the lineage and moment references the workers relax toward."""
        particles = self.ctx.particles
        if self.reference_vortex_strength is not None or particles.n_particles_total == 0:
            return
        if not (
            self.config.filament_refinement.enabled or self.config.divergence_relaxation.enabled
        ):
            return

        from .filament_refinement import gaussian_particle_moments

        vortex_strength = particles.vortex_strength_cpu()
        particle_volume = particles.particle_volume_cpu()
        magnitude = np.linalg.norm(vortex_strength, axis=1)
        floor = max(float(magnitude.max(initial=0.0)) * 1e-12, np.finfo(np.float64).tiny)
        self.reference_vortex_strength = np.maximum(magnitude, floor)
        self.reference_lengths = np.cbrt(particle_volume)
        moments = gaussian_particle_moments(
            particles.position_cpu(),
            vortex_strength,
            particles.core_radius_cpu(),
        )
        self.reference_moments = tuple(
            np.asarray(moments[index], dtype=np.float64).copy() for index in (0, 2, 3)
        )

    def update_residual_viscosity(self) -> None:
        """Add the configured stretching-aware residual viscosity to ``effective_viscosity``."""
        cfg = self.config
        step = self.ctx.step()
        if step < cfg.stretching_viscosity_start_step:
            return
        if (
            cfg.stretching_viscosity_feedback_gain > 0.0
            and step != self._last_residual_feedback_step
            and self._due(
                cfg.stretching_viscosity_feedback_interval_steps,
                cfg.stretching_viscosity_start_step,
            )
        ):
            energy_rate = float(self.ctx.kinetic_energy_rate())
            viscous_rate = float(self.ctx.viscous_kinetic_energy_rate())
            if np.isfinite(energy_rate) and np.isfinite(viscous_rate):
                scale = max(abs(viscous_rate), np.finfo(float).eps)
                adjustment = np.clip(
                    1.0 + cfg.stretching_viscosity_feedback_gain * energy_rate / scale,
                    0.80,
                    1.0 + cfg.stretching_viscosity_feedback_growth_limit,
                )
                upper = (
                    cfg.stretching_viscosity_max_coefficient
                    if cfg.stretching_viscosity_max_coefficient is not None
                    else np.inf
                )
                self.residual_viscosity_coefficient = float(
                    np.clip(self.residual_viscosity_coefficient * adjustment, 0.0, upper)
                )
            self._last_residual_feedback_step = step
        coefficient = self.residual_viscosity_coefficient
        if coefficient <= 0.0:
            return
        self.operators.apply_stretching_viscosity(self.ctx.particles, coefficient)

    def apply_relaxation(self) -> None:
        """Rotate the scheduled fraction of the vortex_strength-omega misalignment away.

        The particle field is a vorticity field only while ``alpha_p`` stays
        parallel to the vorticity it induces, and the divergence of the
        discrete field grows exactly where it does not. Optional moment
        restoration removes the global impulse introduced by that rotation.
        """
        cfg = self.config
        if (
            not cfg.pedrizzetti_relaxation_enabled
            or self.ctx.flow_model == "POTENTIAL"
            or (
                cfg.pedrizzetti_relaxation_end_step is not None
                and self.ctx.step() >= cfg.pedrizzetti_relaxation_end_step
            )
            or not self._due(
                cfg.pedrizzetti_relaxation_interval_steps, cfg.pedrizzetti_relaxation_start_step
            )
        ):
            return

        before = self.measure()
        reference_vortex_strength = None
        if cfg.pedrizzetti_relaxation_preserve_moments:
            particles = self.ctx.particles
            position = particles.position_cpu(use_cache=False).astype(np.float64)
            reference_vortex_strength = particles.vortex_strength_cpu(use_cache=False).astype(
                np.float64
            )
            core_radius = particles.core_radius_cpu(use_cache=False).astype(np.float64)
            particle_volume = particles.particle_volume_cpu(use_cache=False).astype(np.float64)
        statistics = self.operators.apply_pedrizzetti_relaxation(
            self.ctx.particles,
            cfg.pedrizzetti_relaxation_factor,
            preserve_vortex_strength_magnitude=(
                cfg.pedrizzetti_relaxation_preserve_vortex_strength
            ),
        )
        correction_relative = 0.0
        if reference_vortex_strength is not None:
            from .divergence_relaxation import restore_particle_moments

            relaxed = self.ctx.particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
            corrected, correction_relative = restore_particle_moments(
                position,
                relaxed,
                core_radius,
                particle_volume,
                reference_vortex_strength,
                angular_core_coefficient=self.ctx.physics._angular_core_coefficient,
            )
            self.ctx.set_particles_properties(
                vortex_strength=corrected.astype(self.ctx.np_dtype)
            )
        self.accept(
            "Pedrizzetti relaxation",
            before,
            conserves_vortex_strength=reference_vortex_strength is not None,
            detail=(
                f"f={cfg.pedrizzetti_relaxation_factor:.3f}, "
                f"misalignment={statistics['pedrizzetti_misalignment_deg']:.2f} deg, "
                f"moment correction={correction_relative:.2e}"
            ),
        )

    def apply_filament_refinement(self) -> None:
        """Bisect over-stretched Lagrangian elements at the configured cadence."""
        ctx = self.ctx
        cfg = self.config.filament_refinement
        interval_steps = cfg.interval_steps
        late_stage = cfg.late_start_step is not None and ctx.step() >= cfg.late_start_step
        if late_stage:
            interval_steps = int(cfg.late_interval_steps)
        if (
            not cfg.enabled
            or (cfg.end_step is not None and ctx.step() >= cfg.end_step)
            or ctx.step() % interval_steps != 0
        ):
            return

        from .filament_refinement import FilamentRefinementError, split_stretched_filaments

        if self.reference_vortex_strength is None or self.reference_lengths is None:
            raise FilamentRefinementError(
                "filament-refinement lineage references were not captured before time integration"
            )
        particles = ctx.particles
        position = particles.position_cpu()
        if len(self.reference_vortex_strength) != len(position) or len(
            self.reference_lengths
        ) != len(position):
            raise FilamentRefinementError(
                "filament-refinement lineage state no longer matches the particle cloud"
            )
        capacity = int(particles._max_particles)
        if cfg.max_n_particles is not None:
            capacity = min(capacity, int(cfg.max_n_particles))

        before = self.measure()
        result = split_stretched_filaments(
            position,
            particles.vortex_strength_cpu(),
            particles.core_radius_cpu(),
            particles.particle_volume_cpu(),
            reference_vortex_strength=self.reference_vortex_strength,
            reference_length=self.reference_lengths,
            max_stretch_factor=(
                np.inf if late_stage and cfg.late_absolute_only else cfg.max_vortex_strength_factor
            ),
            offset_fraction=cfg.offset_fraction,
            max_n_particles=capacity,
            max_absolute_vortex_strength=cfg.max_absolute_vortex_strength,
        )
        if result.refined_particles == 0:
            return

        source = result.source_index
        ctx.replace_vortex_particles(
            position=result.position.astype(ctx.np_dtype),
            velocity=particles.velocity_cpu()[source],
            vortex_strength=result.vortex_strength.astype(ctx.np_dtype),
            core_radius=result.core_radius.astype(ctx.np_dtype),
            particle_volume=result.particle_volume.astype(ctx.np_dtype),
            kinematic_viscosity=particles.kinematic_viscosity_cpu()[source],
            eddy_viscosity=particles.eddy_viscosity_cpu()[source],
            group_id=particles.group_id_cpu()[source],
            zone_id=particles.zone_id_cpu()[source],
            report_removal=False,
        )
        # Refinement replaces particles without representing physical removal.
        self.reference_vortex_strength = result.reference_vortex_strength
        self.reference_lengths = result.reference_length
        self.accept(
            "filament refinement",
            before,
            detail=f"split={result.refined_particles}, stretch={result.max_stretch_ratio:.2f}",
        )

    def apply_divergence_relaxation(self) -> None:
        """Reassign vortex_strength onto the solenoidal subspace of the blob field."""
        ctx = self.ctx
        cfg = self.config.divergence_relaxation
        if not self._due(cfg.interval_steps, cfg.start_step):
            return

        from .divergence_relaxation import (
            DivergenceRelaxationError,
            constrained_divergence_relaxation,
        )

        if self.reference_moments is None:
            raise DivergenceRelaxationError(
                "divergence relaxation requires captured reference moments"
            )
        particles = ctx.particles
        position = particles.position_cpu().astype(np.float64)
        vortex_strength = particles.vortex_strength_cpu().astype(np.float64)
        core_radius = particles.core_radius_cpu().astype(np.float64)
        particle_volume = particles.particle_volume_cpu().astype(np.float64)
        reference_scales = (
            cfg.vortex_strength_reference_scale,
            cfg.linear_impulse_reference_scale,
            cfg.angular_impulse_reference_scale,
        )

        before = self.measure()
        result = constrained_divergence_relaxation(
            position,
            vortex_strength,
            core_radius,
            particle_volume,
            grid_spacing=cfg.grid_spacing,
            regularization=cfg.regularization,
            solver_relative_tolerance=cfg.solver_relative_tolerance,
            max_iterations=cfg.max_iterations,
            max_projection_sweeps=cfg.max_projection_sweeps,
            max_grid_nodes=cfg.max_grid_nodes,
            max_correction_norm=cfg.max_correction_norm,
            max_residual_ratio=cfg.max_residual_ratio,
            total_kinetic_energy_tolerance=cfg.total_kinetic_energy_tolerance,
            total_enstrophy_tolerance=cfg.total_enstrophy_tolerance,
            total_helicity_tolerance=cfg.total_helicity_tolerance,
            variation_tolerance=cfg.variation_tolerance,
            spectral_convergence_fraction=cfg.spectral_convergence_fraction,
            reference_scales=(
                reference_scales if all(value is not None for value in reference_scales) else None
            ),
            reference_tolerances=(
                cfg.vortex_strength_reference_tolerance,
                cfg.linear_impulse_reference_tolerance,
                cfg.angular_impulse_reference_tolerance,
            ),
            target_moments=self.reference_moments,
        )

        uploaded_vortex_strength = result.vortex_strength.astype(ctx.np_dtype)
        ctx.set_particles_properties(vortex_strength=uploaded_vortex_strength)
        self._rescale_lineage_reference(
            vortex_strength, uploaded_vortex_strength.astype(np.float64)
        )
        self.accept(
            "divergence relaxation",
            before,
            detail=(
                f"projection_sweeps={result.projection_sweeps},"
                f"divergence_residual_ratio={result.final_residual_ratio:.2e},"
                f"vortex_strength_correction_l2_rel={result.correction_norm_relative:.2e}"
            ),
        )

    def apply_regularization(self) -> None:
        """Redistribute a distorted cloud when its discretization health demands it."""
        cfg = self.config
        if (
            (
                cfg.regularization_max_events is not None
                and self.regularization_events >= cfg.regularization_max_events
            )
            or not self._due(cfg.regularization_interval_steps, cfg.regularization_start_step)
        ):
            return

        from .regularization import regularize

        before = self.measure()
        outcome = regularize(self.ctx, cfg)
        if outcome is None:
            return
        # Conservative regularization rebuilds a cloud on its own lattice, so
        # it invalidates any prior grid-regeneration bounds guarantee.
        self.ctx.set_domain_bounds_enforced(False)
        # This worker rebuilds the cloud on its own grid, so total variation and
        # peak vorticity are measured against a different discretization; its
        # energy and enstrophy limits are the physics gate, enforced inside it.
        self.accept(
            "conservative regularization",
            before,
            preserves_discretization=False,
            detail=outcome.detail,
        )
        self.regularization_events += 1

    def apply_retention(self) -> None:
        """Remove particles that have left the configured VPM domain."""
        ctx = self.ctx
        if ctx.flow_model == "POTENTIAL":
            return
        bounds = self.config.remove_particles_by_bounds
        if bounds is not None:
            if self.ctx.domain_bounds_enforced():
                return
            # Removal compacts the stored vorticity field; no O(N²) rebuild is needed.
            ctx.remove_particles_by_bounds(bounds, invert_selection=True)

    # -- lineage bookkeeping ---------------------------------------------------

    def _rescale_lineage_reference(self, vortex_strength: np.ndarray, relaxed: np.ndarray) -> None:
        """Keep the refinement lineage consistent with reassigned vortex_strength."""
        reference = self.reference_vortex_strength
        if reference is None or len(reference) != len(vortex_strength):
            return
        old_magnitude = np.linalg.norm(vortex_strength, axis=1)
        new_magnitude = np.linalg.norm(relaxed, axis=1)
        floor = max(float(old_magnitude.max(initial=0.0)) * 1e-14, np.finfo(float).tiny)
        updated = np.asarray(reference, dtype=np.float64).copy()
        scalable = old_magnitude > floor
        updated[scalable] *= new_magnitude[scalable] / old_magnitude[scalable]
        updated[~scalable] = np.maximum(updated[~scalable], new_magnitude[~scalable])
        self.reference_vortex_strength = np.maximum(updated, floor)

    def resize_lineage_reference(self, source_index: np.ndarray | None = None) -> None:
        """Re-map the lineage state after the particle set changed elsewhere."""
        if self.reference_vortex_strength is None:
            return
        if source_index is None:
            self.reference_vortex_strength = None
            self.reference_lengths = None
            self.reference_moments = None
            return
        self.reference_vortex_strength = self.reference_vortex_strength[source_index]
        self.reference_lengths = self.reference_lengths[source_index]

    def on_removal(self, *, indices=None, keep_mask=None, remove_all: bool = False) -> None:
        """Trim the lineage references to match a particle removal.

        Called by the solver's particle-mutation entry points so the refinement
        references never drift from the live cloud.  A no-op when no lineage has
        been captured yet (``reference_strengths is None``).
        """
        if self.reference_vortex_strength is None or self.reference_lengths is None:
            return
        if remove_all:
            self.reference_vortex_strength = np.empty(0, dtype=np.float64)
            self.reference_lengths = np.empty(0, dtype=np.float64)
            return
        if keep_mask is not None:
            keep = np.asarray(keep_mask, dtype=bool)
        elif indices is not None and len(indices) > 0:
            keep = np.ones(len(self.reference_vortex_strength), dtype=bool)
            keep[np.asarray(indices, dtype=np.int64)] = False
        else:
            return
        self.reference_vortex_strength = np.asarray(self.reference_vortex_strength)[keep]
        self.reference_lengths = np.asarray(self.reference_lengths)[keep]

    def on_replacement(self, magnitude: np.ndarray, particle_volume: np.ndarray) -> None:
        """Reset the lineage references to the new cloud's own magnitudes."""
        if self.reference_vortex_strength is None:
            return
        floor = max(float(magnitude.max(initial=0.0)) * 1e-12, np.finfo(np.float64).tiny)
        self.reference_vortex_strength = np.maximum(magnitude, floor)
        self.reference_lengths = np.cbrt(np.asarray(particle_volume, dtype=np.float64))

    def on_add(
        self, magnitude: np.ndarray, particle_volume: np.ndarray, start: int, loading: bool = False
    ) -> None:
        """Extend the lineage references for an appended batch of particles."""
        if self.reference_vortex_strength is None or self.reference_lengths is None:
            return
        if loading:
            return
        if len(self.reference_vortex_strength) != start:
            raise RuntimeError(
                "filament-refinement lineage state did not match the cloud before insertion"
            )
        floor = max(float(magnitude.max(initial=0.0)) * 1e-12, np.finfo(np.float64).tiny)
        self.reference_vortex_strength = np.concatenate(
            (self.reference_vortex_strength, np.maximum(magnitude, floor))
        )
        self.reference_lengths = np.concatenate(
            (self.reference_lengths, np.cbrt(np.asarray(particle_volume, dtype=np.float64)))
        )
