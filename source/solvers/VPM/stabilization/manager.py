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

``circulation error``   ``|sum Gamma_after - sum Gamma_before| / sum |Gamma|``
    Total vector circulation is invariant under any admissible reassignment of
    a vortex field.  Only mechanisms that claim to preserve it are held to it;
    Pedrizzetti relaxation rotates strengths and reports its transfer instead.

``strength growth``     ``(sum |Gamma|_after - sum |Gamma|_before) / sum |Gamma|_before``
``vorticity growth``    ``(max |omega|_after - max |omega|_before) / max |omega|_before``
    Both are one-sided.  A stabilization event may remove strength or peak
    vorticity — that is what most of these schemes are for, and each worker
    bounds its own dissipation — but no event may amplify the field.  Growth of
    the peak vorticity is the signature of the instability these mechanisms
    exist to suppress, so a scheme that produces it has failed.

    Both are also *discretization-dependent*: ``sum |Gamma|`` is a total
    variation over the particle set and ``max |omega|`` a per-particle maximum,
    so neither is comparable across an event that rebuilds the cloud on a new
    grid.  They are therefore measured and reported for every mechanism, but
    only enforced on the ones that keep the discretization they were given.
    A rebuilding worker is held to circulation alone, and to the energy and
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
from .operators import StabilizationOperators

if TYPE_CHECKING:
    from ..config.types import StabilizationConfig
    from ..core.solver import Solver


class StabilizationError(RuntimeError):
    """A stabilization event failed the master's global acceptance criteria."""


@dataclass(frozen=True)
class StabilizationHealth:
    """Global physical state of the particle cloud, measured in O(N)."""

    particles: int
    circulation: np.ndarray
    strength_magnitude: float
    peak_strength: float
    peak_vorticity: float

    @classmethod
    def measure(cls, particles) -> StabilizationHealth:
        """Snapshot the cloud from the strengths and volumes already on hand."""
        count = particles.number_of_particles
        if count == 0:
            return cls(0, np.zeros(3), 0.0, 0.0, 0.0)
        circulation = np.asarray(particles.circulation_cpu(), dtype=np.float64)
        volume = np.asarray(particles.volume_cpu(), dtype=np.float64)
        magnitude = np.linalg.norm(circulation, axis=1)
        vorticity = magnitude / np.maximum(volume, np.finfo(float).tiny)
        return cls(
            particles=count,
            circulation=circulation.sum(axis=0),
            strength_magnitude=float(magnitude.sum(dtype=np.float64)),
            peak_strength=float(magnitude.max(initial=0.0)),
            peak_vorticity=float(vorticity.max(initial=0.0)),
        )


class StabilizationManager:
    """Schedule the stabilization workers and audit what they did.

    The solver owns one instance and calls it at the fixed points of a time
    step; no stabilization state or bookkeeping lives on the solver itself.
    """

    def __init__(self, solver: Solver) -> None:
        self.solver = solver
        self.config: StabilizationConfig = solver.config.stabilization
        # The stabilization subsystem owns its own kernels and fields; the
        # physics engine has no dependency on it.
        self.operators = StabilizationOperators(
            solver.compute_dtype, int(solver.particles._max_particles)
        )
        self.events = 0
        # A readable placeholder rather than "": the record goes to CSV, and an
        # empty field reads back as a missing value.
        self.last_mechanism = "none"
        self.last_circulation_error = 0.0
        self.last_strength_growth = 0.0
        self.last_vorticity_growth = 0.0
        self.max_vorticity_growth = 0.0
        # Lineage and reference state the workers need across events.  It is
        # part of the restart state, so the checkpoint reads and writes it.
        self.reference_strengths: np.ndarray | None = None
        self.reference_lengths: np.ndarray | None = None
        self.reference_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- master criteria -------------------------------------------------------

    def measure(self) -> StabilizationHealth:
        """Return the current cloud health."""
        return StabilizationHealth.measure(self.solver.particles)

    def accept(
        self,
        mechanism: str,
        before: StabilizationHealth,
        *,
        conserves_circulation: bool = True,
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
        scale = max(before.strength_magnitude, np.finfo(float).tiny)
        circulation_error = float(np.linalg.norm(after.circulation - before.circulation)) / scale
        strength_growth = (after.strength_magnitude - before.strength_magnitude) / scale
        vorticity_growth = (after.peak_vorticity - before.peak_vorticity) / max(
            before.peak_vorticity, np.finfo(float).tiny
        )

        self.events += 1
        self.last_mechanism = mechanism
        # Recorded for every mechanism.  A rotation carries circulation with it
        # by construction, so for those this number is the reported transfer
        # rather than an error, and only the gate below is skipped.
        self.last_circulation_error = circulation_error
        self.last_strength_growth = strength_growth
        self.last_vorticity_growth = vorticity_growth
        self.max_vorticity_growth = max(self.max_vorticity_growth, vorticity_growth)

        Logging.message(
            f"[Stabilization] {mechanism}: {before.particles} -> {after.particles} particles, "
            f"dSumGamma/S={circulation_error:.2e}, dS/S={strength_growth:+.2e}, "
            f"dOmegaMax/OmegaMax={vorticity_growth:+.2e}" + (f", {detail}" if detail else "")
        )

        checks = []
        if conserves_circulation:
            checks.append(("circulation error", circulation_error, cfg.max_circulation_error))
        if preserves_discretization:
            checks += [
                ("strength growth", strength_growth, cfg.max_strength_growth),
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
            "stabilization_events": self.events,
            "stabilization_last_mechanism": self.last_mechanism,
            "stabilization_circulation_error": self.last_circulation_error,
            "stabilization_strength_growth": self.last_strength_growth,
            "stabilization_vorticity_growth": self.last_vorticity_growth,
            "stabilization_max_vorticity_growth": self.max_vorticity_growth,
        }

    def restore_diagnostics(self, values: dict) -> None:
        """Reload the master's record from a checkpoint."""
        self.events = int(values.get("stabilization_events", self.events))
        self.last_mechanism = str(values.get("stabilization_last_mechanism", self.last_mechanism))
        for key, attribute in (
            ("stabilization_circulation_error", "last_circulation_error"),
            ("stabilization_strength_growth", "last_strength_growth"),
            ("stabilization_vorticity_growth", "last_vorticity_growth"),
            ("stabilization_max_vorticity_growth", "max_vorticity_growth"),
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
        if cfg.regularization_frequency > 0:
            active.append("conservative regularization")
        if cfg.remove_particles_by_bounds is not None:
            active.append("bounded-domain retention")
        return tuple(active)

    def _due(self, frequency: int, start_step: int) -> bool:
        step = self.solver.time_step
        return frequency > 0 and step >= start_step and (step - start_step) % frequency == 0

    # -- mechanisms ------------------------------------------------------------

    def capture_reference_state(self) -> None:
        """Capture the lineage and moment references the workers relax toward."""
        solver = self.solver
        if self.reference_strengths is not None or solver.particles.number_of_particles == 0:
            return
        if not (
            self.config.filament_refinement.enabled or self.config.divergence_relaxation.enabled
        ):
            return

        from .filament_refinement import gaussian_particle_moments

        circulation = solver.particles.circulation_cpu()
        volume = solver.particles.volume_cpu()
        magnitude = np.linalg.norm(circulation, axis=1)
        floor = max(float(magnitude.max(initial=0.0)) * 1e-12, np.finfo(np.float64).tiny)
        self.reference_strengths = np.maximum(magnitude, floor)
        self.reference_lengths = np.cbrt(volume)
        moments = gaussian_particle_moments(
            solver.particles.position_cpu(),
            circulation,
            solver.particles.radius_cpu(),
        )
        self.reference_moments = tuple(
            np.asarray(moments[index], dtype=np.float64).copy() for index in (0, 2, 3)
        )

    def update_residual_viscosity(self) -> None:
        """Add the configured stretching-aware residual viscosity to ``nu_eff``."""
        coefficient = self.config.stretching_viscosity_coefficient
        if coefficient <= 0.0:
            return
        self.operators.apply_stretching_viscosity(self.solver.particles, coefficient)

    def apply_relaxation(self) -> None:
        """Rotate the scheduled fraction of the Gamma-omega misalignment away.

        The particle field is a vorticity field only while ``Gamma_p`` stays
        parallel to the vorticity it induces, and the divergence of the
        discrete field grows exactly where it does not.  The rotation carries
        vector circulation with it, so the master reports that transfer instead
        of gating it.
        """
        solver = self.solver
        cfg = self.config
        if (
            not cfg.pedrizzetti_relaxation_enabled
            or solver.flow_model == "POTENTIAL"
            or not self._due(
                cfg.pedrizzetti_relaxation_frequency, cfg.pedrizzetti_relaxation_start_step
            )
        ):
            return

        before = self.measure()
        statistics = self.operators.apply_pedrizzetti_relaxation(
            solver.particles,
            cfg.pedrizzetti_relaxation_factor,
            conserve_strength=cfg.pedrizzetti_relaxation_conserve_strength,
        )
        self.accept(
            "Pedrizzetti relaxation",
            before,
            conserves_circulation=False,
            detail=(
                f"f={cfg.pedrizzetti_relaxation_factor:.3f}, "
                f"misalignment={statistics['pedrizzetti_misalignment_deg']:.2f} deg"
            ),
        )

    def apply_filament_refinement(self) -> None:
        """Bisect over-stretched Lagrangian elements at the configured cadence."""
        solver = self.solver
        cfg = self.config.filament_refinement
        if not cfg.enabled or solver.time_step % cfg.frequency != 0:
            return

        from .filament_refinement import FilamentRefinementError, split_stretched_filaments

        if self.reference_strengths is None or self.reference_lengths is None:
            raise FilamentRefinementError(
                "filament-refinement lineage references were not captured before time integration"
            )
        position = solver.particles.position_cpu()
        if len(self.reference_strengths) != len(position) or len(self.reference_lengths) != len(
            position
        ):
            raise FilamentRefinementError(
                "filament-refinement lineage state no longer matches the particle cloud"
            )
        capacity = int(solver.particles._max_particles)
        if cfg.max_particles is not None:
            capacity = min(capacity, int(cfg.max_particles))

        before = self.measure()
        result = split_stretched_filaments(
            position,
            solver.particles.circulation_cpu(),
            solver.particles.radius_cpu(),
            solver.particles.volume_cpu(),
            reference_strength=self.reference_strengths,
            reference_length=self.reference_lengths,
            max_stretch_factor=cfg.max_strength_factor,
            offset_fraction=cfg.offset_fraction,
            max_particles=capacity,
        )
        if result.refined_particles == 0:
            return

        source = result.source_index
        solver.replace_vortex_particles(
            position=result.position.astype(solver.np_dtype),
            velocity=solver.particles.velocity_cpu()[source],
            circulation=result.circulation.astype(solver.np_dtype),
            radius=result.radius.astype(solver.np_dtype),
            volume=result.volume.astype(solver.np_dtype),
            viscosity=solver.particles.viscosity_cpu()[source],
            viscosity_turbulent=solver.particles.viscosity_turbulent_cpu()[source],
            group_id=solver.particles.group_id_cpu()[source],
            zone_id=solver.particles.zone_id_cpu()[source],
        )
        # Refinement replaces particles without representing physical removal.
        solver._particles_removed_this_step = 0
        solver._circulation_removed_this_step = np.zeros(3, dtype=solver.np_dtype)
        self.reference_strengths = result.reference_strength
        self.reference_lengths = result.reference_length
        self.accept(
            "filament refinement",
            before,
            detail=f"split={result.refined_particles}, stretch={result.maximum_stretch_ratio:.2f}",
        )

    def apply_divergence_relaxation(self) -> None:
        """Reassign strengths onto the solenoidal subspace of the blob field."""
        solver = self.solver
        cfg = self.config.divergence_relaxation
        if not self._due(cfg.frequency, cfg.start_step):
            return

        from .divergence_relaxation import (
            DivergenceRelaxationError,
            constrained_divergence_relaxation,
        )

        if self.reference_moments is None:
            raise DivergenceRelaxationError(
                "divergence relaxation requires captured reference moments"
            )
        position = solver.particles.position_cpu().astype(np.float64)
        circulation = solver.particles.circulation_cpu().astype(np.float64)
        radius = solver.particles.radius_cpu().astype(np.float64)
        volume = solver.particles.volume_cpu().astype(np.float64)
        reference_scales = (
            cfg.circulation_reference_scale,
            cfg.linear_impulse_reference_scale,
            cfg.angular_impulse_reference_scale,
        )

        before = self.measure()
        result = constrained_divergence_relaxation(
            position,
            circulation,
            radius,
            volume,
            grid_spacing=cfg.grid_spacing,
            regularization=cfg.regularization,
            solver_rtol=cfg.solver_rtol,
            max_iterations=cfg.max_iterations,
            max_projection_sweeps=cfg.max_projection_sweeps,
            max_grid_nodes=cfg.max_grid_nodes,
            max_correction_norm=cfg.max_correction_norm,
            max_residual_ratio=cfg.max_residual_ratio,
            energy_tolerance=cfg.energy_tolerance,
            enstrophy_tolerance=cfg.enstrophy_tolerance,
            helicity_tolerance=cfg.helicity_tolerance,
            variation_tolerance=cfg.variation_tolerance,
            spectral_convergence_fraction=cfg.spectral_convergence_fraction,
            reference_scales=(
                reference_scales if all(value is not None for value in reference_scales) else None
            ),
            reference_tolerances=(
                cfg.circulation_reference_tolerance,
                cfg.linear_impulse_reference_tolerance,
                cfg.angular_impulse_reference_tolerance,
            ),
            target_moments=self.reference_moments,
        )

        uploaded_circulation = result.circulation.astype(solver.np_dtype)
        solver.set_particles_properties(strengths=uploaded_circulation)
        self._rescale_lineage_reference(circulation, uploaded_circulation.astype(np.float64))
        self.accept(
            "divergence relaxation",
            before,
            detail=(
                f"sweeps={result.projection_sweeps}, residual={result.final_residual_ratio:.2e}, "
                f"|dGamma|/|Gamma|={result.correction_norm_relative:.2e}"
            ),
        )

    def apply_regularization(self) -> None:
        """Redistribute a distorted cloud when its discretization health demands it."""
        cfg = self.config
        if not self._due(cfg.regularization_frequency, cfg.regularization_start_step):
            return

        from .regularization import regularize

        before = self.measure()
        outcome = regularize(self.solver, cfg)
        if outcome is None:
            return
        # This worker rebuilds the cloud on its own grid, so total variation and
        # peak vorticity are measured against a different discretization; its
        # energy and enstrophy limits are the physics gate, enforced inside it.
        self.accept(
            "conservative regularization",
            before,
            preserves_discretization=False,
            detail=outcome.detail,
        )

    def apply_retention(self) -> None:
        """Remove particles that have left the configured VPM domain."""
        solver = self.solver
        if solver.flow_model == "POTENTIAL":
            return
        bounds = self.config.remove_particles_by_bounds
        if bounds is not None:
            # Removal compacts the stored vorticity field; no O(N²) rebuild is needed.
            solver.remove_particles_by_bounds(bounds, invert_selection=True)

    # -- lineage bookkeeping ---------------------------------------------------

    def _rescale_lineage_reference(self, circulation: np.ndarray, relaxed: np.ndarray) -> None:
        """Keep the refinement lineage consistent with reassigned strengths."""
        reference = self.reference_strengths
        if reference is None or len(reference) != len(circulation):
            return
        old_magnitude = np.linalg.norm(circulation, axis=1)
        new_magnitude = np.linalg.norm(relaxed, axis=1)
        floor = max(float(old_magnitude.max(initial=0.0)) * 1e-14, np.finfo(float).tiny)
        updated = np.asarray(reference, dtype=np.float64).copy()
        scalable = old_magnitude > floor
        updated[scalable] *= new_magnitude[scalable] / old_magnitude[scalable]
        updated[~scalable] = np.maximum(updated[~scalable], new_magnitude[~scalable])
        self.reference_strengths = np.maximum(updated, floor)

    def resize_lineage_reference(self, source_index: np.ndarray | None = None) -> None:
        """Re-map the lineage state after the particle set changed elsewhere."""
        if self.reference_strengths is None:
            return
        if source_index is None:
            self.reference_strengths = None
            self.reference_lengths = None
            self.reference_moments = None
            return
        self.reference_strengths = self.reference_strengths[source_index]
        self.reference_lengths = self.reference_lengths[source_index]
