"""Conservative regularization of a distorted vortex-particle cloud.

This is the most invasive stabilization worker: it rebuilds the cloud on a
Gaussian redistribution grid, which restores overlap but throws away the
Lagrangian history of the discretization.  It therefore runs only when the
discretization-health diagnostics say the current cloud has stopped being a
faithful vorticity field, and it enforces its own admissibility rules before it
lets the new field stand:

* vector strength, linear impulse, and finite-core angular impulse are
  restored by a minimum-norm correction and re-checked after the upload;
* a candidate that dissipates both energy and enstrophy is taken directly,
  otherwise enstrophy is restored exactly along a moment-preserving direction;
* every accepted event stays inside the declared dissipation limits, and a
  rejected one restores the original field before raising.

The global judgement of whether the event helped is left to
:class:`~source.solvers.vpm.stabilization.manager.StabilizationManager`.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: August 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..diagnostics.resolution import discretization_health
from ..io.logging import Logging

if TYPE_CHECKING:
    from ..config.types import StabilizationConfig
    from .context import StabilizationContext

ENSTROPHY_RESTORATION_TOLERANCE = 1.0e-4  # float32 reduction roundoff scale


@dataclass(frozen=True)
class RegularizationOutcome:
    """What one accepted regularization event did, in one line of numbers."""

    particles_before: int
    particles_after: int
    total_kinetic_energy_change_relative: float
    total_enstrophy_change_relative: float
    projected: bool

    @property
    def detail(self) -> str:
        return (
            f"dE/E={self.total_kinetic_energy_change_relative:.2e}, dZ/Z={self.total_enstrophy_change_relative:.2e}"
            + (", solenoidal projection" if self.projected else "")
        )


def regularize(ctx: StabilizationContext, cfg: StabilizationConfig) -> RegularizationOutcome | None:
    """Redistribute the cloud if its health has fallen below the triggers.

    Returns ``None`` when the health triggers are not met and nothing was done.
    """

    from .divergence_relaxation import (
        _MomentNullspace,
        constrained_divergence_relaxation,
        gaussian_invariant_rows,
    )
    from .filament_refinement import gaussian_particle_moments

    particles = ctx.particles
    position = particles.position_cpu().astype(np.float64)
    vortex_strength = particles.vortex_strength_cpu().astype(np.float64)
    core_radius = particles.core_radius_cpu().astype(np.float64)
    particle_volume = particles.particle_volume_cpu().astype(np.float64)
    kinematic_viscosity = particles.kinematic_viscosity_cpu().astype(np.float64)
    if len(position) == 0:
        return

    before_health = discretization_health(position, vortex_strength, core_radius)
    capacity_count = None
    if cfg.regularization_max_particles is not None:
        capacity_count = max(
            1,
            int(np.ceil(cfg.regularization_capacity_fraction * cfg.regularization_max_particles)),
        )
    at_capacity = capacity_count is not None and len(position) >= capacity_count
    max_particles = (
        cfg.regularization_capacity_max_particles
        if at_capacity and cfg.regularization_capacity_max_particles is not None
        else cfg.regularization_max_particles
    )
    spacing = float(
        cfg.regularization_capacity_grid_spacing
        if at_capacity and cfg.regularization_capacity_grid_spacing is not None
        else cfg.regularization_grid_spacing
    )
    divergence_trigger = (
        cfg.regularization_capacity_divergence_trigger
        if at_capacity and cfg.regularization_capacity_divergence_trigger is not None
        else cfg.regularization_divergence_trigger
    )
    misalignment_trigger = (
        cfg.regularization_capacity_misalignment_trigger
        if at_capacity and cfg.regularization_capacity_misalignment_trigger is not None
        else cfg.regularization_misalignment_trigger
    )
    energy_rate_trigger = (
        cfg.regularization_capacity_energy_rate_trigger if at_capacity else None
    )
    energy_growth = (
        energy_rate_trigger is not None
        and float(ctx.kinetic_energy_rate()) > energy_rate_trigger
    )
    if (
        before_health["vorticity_divergence_error"] <= divergence_trigger
        and before_health["vortex_strength_misalignment_degrees"] <= misalignment_trigger
        and not energy_growth
    ):
        return

    before_moments = gaussian_particle_moments(position, vortex_strength, core_radius)
    before_integrals = ctx.field_diagnostics.compute_flow_integrals(
        particles,
        ctx.time(),
        record_history=False,
    )
    old_state = {
        "position": position.astype(ctx.np_dtype),
        "velocity": particles.velocity_cpu().astype(ctx.np_dtype),
        "vortex_strength": vortex_strength.astype(ctx.np_dtype),
        "core_radius": core_radius.astype(ctx.np_dtype),
        "particle_volume": particle_volume.astype(ctx.np_dtype),
        "kinematic_viscosity": kinematic_viscosity.astype(ctx.np_dtype),
        "eddy_viscosity": particles.eddy_viscosity_cpu().astype(ctx.np_dtype),
        "zone_id": particles.zone_id_cpu().astype(np.int32),
        "group_id": particles.group_id_cpu().astype(np.int32),
    }
    removed_before = ctx.particles_removed()
    vortex_strength_removed_before = ctx.vortex_strength_removed().copy()
    mean_kinematic_viscosity = float(kinematic_viscosity.mean())
    projection_only = (
        max_particles is not None and len(position) > max_particles
    )
    if projection_only:
        proposal = old_state.copy()
    else:
        proposal = ctx.physics.grid_based_diffusion(
            particles,
            time_step_size=ctx.time_step_size(),
            particle_spacing=spacing,
            kinematic_viscosity=mean_kinematic_viscosity,
            domain_padding=4.0,
            regen_threshold=cfg.regularization_tail_budget,
            regen_threshold_mode="budget",
            rd_ratio=4.0,
            effective_viscosity=None,
            max_nodes=max_particles,
        )
    if proposal is None:
        raise RuntimeError("conservative regularization produced no particle field")
    configured_core_radius = (
        cfg.regularization_capacity_core_radius
        if at_capacity and cfg.regularization_capacity_core_radius is not None
        else cfg.regularization_core_radius
    )
    if configured_core_radius is not None:
        proposal["core_radius"] = np.full(
            len(proposal["position"]),
            configured_core_radius,
            dtype=ctx.np_dtype,
        )

    new_position = np.asarray(proposal["position"], dtype=np.float64)
    proposed_vortex_strength = np.asarray(proposal["vortex_strength"], dtype=np.float64)
    new_core_radius = np.asarray(proposal["core_radius"], dtype=np.float64)
    new_particle_volume = np.asarray(proposal["particle_volume"], dtype=np.float64)
    count = len(new_position)
    new_velocity = np.asarray(proposal.get("velocity", np.zeros((count, 3))), dtype=ctx.np_dtype)
    new_kinematic_viscosity = np.asarray(
        proposal.get("kinematic_viscosity", np.full(count, mean_kinematic_viscosity)),
        dtype=ctx.np_dtype,
    )
    new_eddy_viscosity = np.asarray(
        proposal.get("eddy_viscosity", np.zeros(count)), dtype=ctx.np_dtype
    )
    new_zone_id = np.asarray(
        proposal.get("zone_id", np.zeros(count, dtype=np.int32)), dtype=np.int32
    )
    new_group_id = np.asarray(
        proposal.get("group_id", np.zeros(count, dtype=np.int32)), dtype=np.int32
    )

    def upload_and_integrate(vortex_strength: np.ndarray) -> tuple[np.ndarray, dict]:
        uploaded_vortex_strength = np.asarray(vortex_strength, dtype=ctx.np_dtype)
        ctx.replace_vortex_particles(
            position=np.asarray(proposal["position"], dtype=ctx.np_dtype),
            velocity=new_velocity,
            vortex_strength=uploaded_vortex_strength,
            core_radius=np.asarray(proposal["core_radius"], dtype=ctx.np_dtype),
            particle_volume=np.asarray(proposal["particle_volume"], dtype=ctx.np_dtype),
            kinematic_viscosity=new_kinematic_viscosity,
            eddy_viscosity=new_eddy_viscosity,
            zone_id=new_zone_id,
            group_id=new_group_id,
        )
        integrals = ctx.field_diagnostics.compute_flow_integrals(
            particles,
            ctx.time(),
            record_history=False,
        )
        return uploaded_vortex_strength, integrals

    def restore_old_field() -> None:
        ctx.replace_vortex_particles(**old_state)
        ctx.set_particles_removed(removed_before)
        ctx.set_vortex_strength_removed(vortex_strength_removed_before)

    def evaluate_moment_corrected_candidate():
        candidate_core_radius = np.asarray(proposal["core_radius"], dtype=np.float64)
        candidate_nullspace = _MomentNullspace(
            gaussian_invariant_rows(new_position, candidate_core_radius),
            new_particle_volume,
        )
        proposed_moments = gaussian_particle_moments(
            new_position,
            proposed_vortex_strength,
            candidate_core_radius,
        )
        moment_change = np.concatenate(
            (
                before_moments[0] - proposed_moments[0],
                before_moments[2] - proposed_moments[2],
                before_moments[3] - proposed_moments[3],
            )
        )
        corrected = proposed_vortex_strength + candidate_nullspace.correction_for_moment_change(
            moment_change
        )
        candidate, integrals = upload_and_integrate(corrected)
        total_kinetic_energy_change_relative = (
            float(integrals["total_kinetic_energy"])
            - float(before_integrals["total_kinetic_energy"])
        ) / max(abs(float(before_integrals["total_kinetic_energy"])), np.finfo(float).tiny)
        total_enstrophy_change_relative = (
            float(integrals["total_enstrophy"]) - float(before_integrals["total_enstrophy"])
        ) / max(abs(float(before_integrals["total_enstrophy"])), np.finfo(float).tiny)
        correction_relative = float(
            np.linalg.norm(candidate.astype(np.float64) - proposed_vortex_strength)
            / max(np.linalg.norm(proposed_vortex_strength), np.finfo(float).tiny)
        )
        return (
            candidate_core_radius,
            candidate_nullspace,
            corrected,
            candidate,
            integrals,
            total_kinetic_energy_change_relative,
            total_enstrophy_change_relative,
            correction_relative,
        )

    projection_result = None
    adaptive_core_used = False
    try:
        (
            new_core_radius,
            nullspace,
            moment_corrected,
            candidate,
            candidate_integrals,
            candidate_energy_change,
            candidate_enstrophy_change,
            moment_correction_relative,
        ) = evaluate_moment_corrected_candidate()

        # Broaden a fixed regenerated core only when it would inject energy or enstrophy.
        if configured_core_radius is not None and not projection_only:
            for retry in range(1, 9):
                if candidate_energy_change <= 1.0e-7 and candidate_enstrophy_change <= 1.0e-7:
                    break
                adaptive_core_used = True
                trial_core_radius = configured_core_radius * 1.05**retry
                proposal["core_radius"] = np.full(
                    count,
                    trial_core_radius,
                    dtype=ctx.np_dtype,
                )
                (
                    new_core_radius,
                    nullspace,
                    moment_corrected,
                    candidate,
                    candidate_integrals,
                    candidate_energy_change,
                    candidate_enstrophy_change,
                    moment_correction_relative,
                ) = evaluate_moment_corrected_candidate()
            if candidate_energy_change > 1.0e-7 or candidate_enstrophy_change > 1.0e-7:
                raise RuntimeError(
                    "regularization could not find a non-injecting Gaussian core: "
                    f"core_radius={float(new_core_radius.mean()):.3e}, "
                    f"dE/E={candidate_energy_change:.3e}, "
                    f"dZ/Z={candidate_enstrophy_change:.3e}"
                )

        if adaptive_core_used:
            Logging.record(
                "regularization, adaptive core",
                ("core radius, configured", f"{configured_core_radius:.3e}", "m"),
                ("core radius, selected", f"{float(new_core_radius.mean()):.3e}", "m"),
            )

        if projection_only or (
            -cfg.regularization_total_kinetic_energy_dissipation_limit
            <= candidate_energy_change
            <= 1.0e-7
            and -cfg.regularization_total_enstrophy_dissipation_limit
            <= candidate_enstrophy_change
            <= 1.0e-7
        ):
            uploaded = candidate
            after_integrals = candidate_integrals
            total_kinetic_energy_change_relative = candidate_energy_change
        else:
            direction = nullspace.to_correction(moment_corrected / nullspace.sqrt_volume[:, None])
            direction_norm = float(np.linalg.norm(direction))
            vortex_strength_norm = max(
                float(np.linalg.norm(moment_corrected)),
                np.finfo(float).tiny,
            )
            if direction_norm <= np.finfo(float).tiny:
                raise RuntimeError("regularization has no enstrophy-restoration direction")
            direction *= vortex_strength_norm / direction_norm
            _, plus_integrals = upload_and_integrate(moment_corrected + direction)
            _, minus_integrals = upload_and_integrate(moment_corrected - direction)
            candidate_enstrophy = float(candidate_integrals["total_enstrophy"])
            target_enstrophy = float(before_integrals["total_enstrophy"])
            linear = 0.5 * (
                float(plus_integrals["total_enstrophy"]) - float(minus_integrals["total_enstrophy"])
            )
            quadratic = (
                0.5
                * (
                    float(plus_integrals["total_enstrophy"])
                    + float(minus_integrals["total_enstrophy"])
                )
                - candidate_enstrophy
            )
            roots = np.roots((quadratic, linear, candidate_enstrophy - target_enstrophy))
            real_roots = sorted(
                (
                    float(value.real)
                    for value in roots
                    if np.isfinite(value) and abs(value.imag) <= 1.0e-9
                ),
                key=abs,
            )
            if not real_roots:
                raise RuntimeError("regularization could not restore enstrophy")

            admissible: list[tuple[float, np.ndarray, dict]] = []
            rejected: list[str] = []
            energy_before = float(before_integrals["total_kinetic_energy"])
            energy_scale = max(abs(energy_before), np.finfo(float).tiny)
            enstrophy_scale = max(abs(target_enstrophy), np.finfo(float).tiny)
            for multiplier in real_roots:
                uploaded, trial_integrals = upload_and_integrate(
                    candidate.astype(np.float64) + multiplier * direction
                )
                trial_energy_change = (
                    float(trial_integrals["total_kinetic_energy"]) - energy_before
                ) / energy_scale
                trial_enstrophy_change = (
                    float(trial_integrals["total_enstrophy"]) - target_enstrophy
                ) / enstrophy_scale
                if (
                    -cfg.regularization_total_kinetic_energy_dissipation_limit
                    <= trial_energy_change
                    <= 1.0e-7
                    and abs(trial_enstrophy_change) <= ENSTROPHY_RESTORATION_TOLERANCE
                ):
                    admissible.append((trial_energy_change, uploaded.copy(), trial_integrals))
                else:
                    rejected.append(
                        f"lambda={multiplier:.3e}: dE/E={trial_energy_change:.3e}, "
                        f"dZ/Z={trial_enstrophy_change:.3e}"
                    )
            if not admissible:
                raise RuntimeError(
                    "regularization candidate changed "
                    f"dE/E={candidate_energy_change:.3e}, "
                    f"dZ/Z={candidate_enstrophy_change:.3e}; "
                    "no dissipative enstrophy-preserving root (" + "; ".join(rejected) + ")"
                )

            # Choose the admissible root with the least energy loss.
            total_kinetic_energy_change_relative, uploaded, after_integrals = max(
                admissible, key=lambda item: item[0]
            )
            uploaded, after_integrals = upload_and_integrate(uploaded)

        preliminary_health = discretization_health(
            new_position,
            uploaded.astype(np.float64),
            new_core_radius,
        )
        if (
            projection_only
            or preliminary_health["vorticity_divergence_error"]
            > cfg.regularization_projection_trigger
        ):
            projection_result = constrained_divergence_relaxation(
                new_position,
                uploaded.astype(np.float64),
                new_core_radius,
                new_particle_volume,
                grid_spacing=spacing,
                max_correction_norm=cfg.regularization_projection_max_correction,
                max_residual_ratio=0.9,
                total_kinetic_energy_tolerance=0.02,
                total_enstrophy_tolerance=cfg.regularization_total_enstrophy_dissipation_limit,
                total_helicity_tolerance=0.05,
                variation_tolerance=0.05,
                spectral_convergence_fraction=0.25,
                reference_tolerances=(1.0e-4, 1.0e-4, 1.0e-4),
                max_projection_sweeps=1,
                target_moments=(before_moments[0], before_moments[2], before_moments[3]),
            )
            uploaded, after_integrals = upload_and_integrate(projection_result.vortex_strength)

    except Exception:
        restore_old_field()
        raise

    audited = uploaded.astype(np.float64)
    after_moments = gaussian_particle_moments(new_position, audited, new_core_radius)
    energy_transfer = float(after_integrals["total_kinetic_energy"]) - float(
        before_integrals["total_kinetic_energy"]
    )
    total_kinetic_energy_change_relative = energy_transfer / max(
        abs(float(before_integrals["total_kinetic_energy"])), np.finfo(float).tiny
    )
    total_enstrophy_change_relative = (
        float(after_integrals["total_enstrophy"]) - float(before_integrals["total_enstrophy"])
    ) / max(abs(float(before_integrals["total_enstrophy"])), np.finfo(float).tiny)
    if (
        not -cfg.regularization_total_kinetic_energy_dissipation_limit
        <= total_kinetic_energy_change_relative
        <= 1.0e-7
    ):
        restore_old_field()
        raise RuntimeError(
            f"regularization changed energy by {total_kinetic_energy_change_relative:.3e}, outside its "
            "declared dissipative interval"
        )
    if not (
        -cfg.regularization_total_enstrophy_dissipation_limit
        <= total_enstrophy_change_relative
        <= ENSTROPHY_RESTORATION_TOLERANCE
    ):
        restore_old_field()
        raise RuntimeError(
            f"regularization changed enstrophy by {total_enstrophy_change_relative:.3e}, outside "
            "its declared non-injecting interval"
        )

    impulse_scale = max(
        0.5
        * float(np.linalg.norm(np.cross(position, vortex_strength), axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    angular_terms = (
        np.cross(position, np.cross(position, vortex_strength)) / 3.0
        - core_radius[:, None] ** 2 * vortex_strength / 3.0
    )
    angular_scale = max(
        float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    errors = {
        "vortex_strength": float(np.linalg.norm(after_moments[0] - before_moments[0]))
        / max(before_moments[1], np.finfo(float).tiny),
        "linear_impulse": float(np.linalg.norm(after_moments[2] - before_moments[2]))
        / impulse_scale,
        "angular_impulse": float(np.linalg.norm(after_moments[3] - before_moments[3]))
        / angular_scale,
    }
    roundoff_limit = 1024.0 * np.finfo(ctx.np_dtype).eps
    if max(errors.values()) > roundoff_limit:
        restore_old_field()
        raise RuntimeError(
            "uploaded regularization exceeded its moment roundoff allowance: "
            + ", ".join(f"{name}={value:.3e}" for name, value in errors.items())
        )

    ctx.set_particles_removed(0)
    ctx.set_vortex_strength_removed(np.zeros(3, dtype=ctx.np_dtype))

    return RegularizationOutcome(
        particles_before=len(position),
        particles_after=len(new_position),
        total_kinetic_energy_change_relative=total_kinetic_energy_change_relative,
        total_enstrophy_change_relative=total_enstrophy_change_relative,
        projected=projection_result is not None,
    )
