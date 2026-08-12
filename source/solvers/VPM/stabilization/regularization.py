"""Conservative regularization of a distorted vortex-particle cloud.

This is the most invasive stabilization worker: it rebuilds the cloud on a
Gaussian redistribution grid, which restores overlap but throws away the
Lagrangian history of the discretization.  It therefore runs only when the
discretization-health diagnostics say the current cloud has stopped being a
faithful vorticity field, and it enforces its own admissibility rules before it
lets the new field stand:

* vector circulation, linear impulse, and finite-core angular impulse are
  restored by a minimum-norm correction and re-checked after the upload;
* a candidate that dissipates both energy and enstrophy is taken directly,
  otherwise enstrophy is restored exactly along a moment-preserving direction;
* every accepted event stays inside the declared dissipation limits, and a
  rejected one restores the original field before raising.

The global judgement of whether the event helped is left to
:class:`~source.solvers.VPM.stabilization.manager.StabilizationManager`.

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
    from ..core.solver import Solver


@dataclass(frozen=True)
class RegularizationOutcome:
    """What one accepted regularization event did, in one line of numbers."""

    particles_before: int
    particles_after: int
    energy_change: float
    enstrophy_change: float
    projected: bool

    @property
    def detail(self) -> str:
        return f"dE/E={self.energy_change:.2e}, dZ/Z={self.enstrophy_change:.2e}" + (
            ", solenoidal projection" if self.projected else ""
        )


def regularize(solver: Solver, cfg: StabilizationConfig) -> RegularizationOutcome | None:
    """Redistribute the cloud if its health has fallen below the triggers.

    Returns ``None`` when the health triggers are not met and nothing was done.
    """

    from .divergence_relaxation import (
        _MomentNullspace,
        constrained_divergence_relaxation,
        gaussian_invariant_rows,
    )
    from .filament_refinement import gaussian_particle_moments

    position = solver.particles.position_cpu().astype(np.float64)
    circulation = solver.particles.circulation_cpu().astype(np.float64)
    radius = solver.particles.radius_cpu().astype(np.float64)
    volume = solver.particles.volume_cpu().astype(np.float64)
    viscosity = solver.particles.viscosity_cpu().astype(np.float64)
    if len(position) == 0:
        return

    before_health = discretization_health(position, circulation, radius)
    capacity_count = None
    if cfg.regularization_max_particles is not None:
        capacity_count = max(
            1,
            int(np.ceil(cfg.regularization_capacity_fraction * cfg.regularization_max_particles)),
        )
    at_capacity = capacity_count is not None and len(position) >= capacity_count
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
    if (
        before_health["vorticity_divergence_error"] <= divergence_trigger
        and before_health["strength_misalignment_deg"] <= misalignment_trigger
    ):
        return

    before_moments = gaussian_particle_moments(position, circulation, radius)
    before_integrals = solver.field_diagnostics.compute_flow_integrals(
        solver.particles,
        solver.flow_time,
        record_history=False,
    )
    old_state = {
        "position": position.astype(solver.np_dtype),
        "velocity": solver.particles.velocity_cpu().astype(solver.np_dtype),
        "circulation": circulation.astype(solver.np_dtype),
        "radius": radius.astype(solver.np_dtype),
        "volume": volume.astype(solver.np_dtype),
        "viscosity": viscosity.astype(solver.np_dtype),
        "viscosity_turbulent": solver.particles.viscosity_turbulent_cpu().astype(solver.np_dtype),
        "zone_id": solver.particles.zone_id_cpu().astype(np.int32),
        "group_id": solver.particles.group_id_cpu().astype(np.int32),
    }
    removed_before = solver._particles_removed_this_step
    circulation_removed_before = solver._circulation_removed_this_step.copy()
    molecular_viscosity = float(viscosity.mean())
    projection_only = (
        cfg.regularization_max_particles is not None
        and len(position) > cfg.regularization_max_particles
    )
    if projection_only:
        proposal = old_state.copy()
    else:
        proposal = solver.physics.grid_based_diffusion(
            solver.particles,
            dt=solver.time_step_size,
            h=spacing,
            nu=molecular_viscosity,
            domain_padding=4.0,
            regen_threshold=cfg.regularization_tail_budget,
            regen_threshold_mode="budget",
            rd_ratio=4.0,
            nu_eff=None,
            max_nodes=cfg.regularization_max_particles,
            cap_abs_fraction=0.995,
        )
    if proposal is None:
        raise RuntimeError("conservative regularization produced no particle field")
    configured_core_radius = (
        cfg.regularization_capacity_core_radius
        if at_capacity and cfg.regularization_capacity_core_radius is not None
        else cfg.regularization_core_radius
    )
    if configured_core_radius is not None:
        proposal["radius"] = np.full(
            len(proposal["position"]),
            configured_core_radius,
            dtype=solver.np_dtype,
        )

    new_position = np.asarray(proposal["position"], dtype=np.float64)
    proposed_circulation = np.asarray(proposal["circulation"], dtype=np.float64)
    new_radius = np.asarray(proposal["radius"], dtype=np.float64)
    new_volume = np.asarray(proposal["volume"], dtype=np.float64)
    count = len(new_position)
    new_velocity = np.asarray(proposal.get("velocity", np.zeros((count, 3))), dtype=solver.np_dtype)
    new_viscosity = np.asarray(
        proposal.get("viscosity", np.full(count, molecular_viscosity)),
        dtype=solver.np_dtype,
    )
    new_viscosity_turbulent = np.asarray(
        proposal.get("viscosity_turbulent", np.zeros(count)), dtype=solver.np_dtype
    )
    new_zone_id = np.asarray(
        proposal.get("zone_id", np.zeros(count, dtype=np.int32)), dtype=np.int32
    )
    new_group_id = np.asarray(
        proposal.get("group_id", np.zeros(count, dtype=np.int32)), dtype=np.int32
    )

    def upload_and_integrate(strength: np.ndarray) -> tuple[np.ndarray, dict]:
        uploaded_strength = np.asarray(strength, dtype=solver.np_dtype)
        solver.replace_vortex_particles(
            position=np.asarray(proposal["position"], dtype=solver.np_dtype),
            velocity=new_velocity,
            circulation=uploaded_strength,
            radius=np.asarray(proposal["radius"], dtype=solver.np_dtype),
            volume=np.asarray(proposal["volume"], dtype=solver.np_dtype),
            viscosity=new_viscosity,
            viscosity_turbulent=new_viscosity_turbulent,
            zone_id=new_zone_id,
            group_id=new_group_id,
        )
        integrals = solver.field_diagnostics.compute_flow_integrals(
            solver.particles,
            solver.flow_time,
            record_history=False,
        )
        return uploaded_strength, integrals

    def restore_old_field() -> None:
        solver.replace_vortex_particles(**old_state)
        solver._particles_removed_this_step = removed_before
        solver._circulation_removed_this_step = circulation_removed_before

    def evaluate_moment_corrected_candidate():
        candidate_radius = np.asarray(proposal["radius"], dtype=np.float64)
        candidate_nullspace = _MomentNullspace(
            gaussian_invariant_rows(new_position, candidate_radius),
            new_volume,
        )
        proposed_moments = gaussian_particle_moments(
            new_position,
            proposed_circulation,
            candidate_radius,
        )
        moment_change = np.concatenate(
            (
                before_moments[0] - proposed_moments[0],
                before_moments[2] - proposed_moments[2],
                before_moments[3] - proposed_moments[3],
            )
        )
        corrected = proposed_circulation + candidate_nullspace.correction_for_moment_change(
            moment_change
        )
        candidate, integrals = upload_and_integrate(corrected)
        energy_change = (
            float(integrals["kinetic_energy"]) - float(before_integrals["kinetic_energy"])
        ) / max(abs(float(before_integrals["kinetic_energy"])), np.finfo(float).tiny)
        enstrophy_change = (
            float(integrals["enstrophy"]) - float(before_integrals["enstrophy"])
        ) / max(abs(float(before_integrals["enstrophy"])), np.finfo(float).tiny)
        correction_relative = float(
            np.linalg.norm(candidate.astype(np.float64) - proposed_circulation)
            / max(np.linalg.norm(proposed_circulation), np.finfo(float).tiny)
        )
        return (
            candidate_radius,
            candidate_nullspace,
            corrected,
            candidate,
            integrals,
            energy_change,
            enstrophy_change,
            correction_relative,
        )

    projection_result = None
    adaptive_core_used = False
    try:
        (
            new_radius,
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
                proposal["radius"] = np.full(
                    count,
                    trial_core_radius,
                    dtype=solver.np_dtype,
                )
                (
                    new_radius,
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
                    f"core={float(new_radius.mean()):.3e}, "
                    f"dE/E={candidate_energy_change:.3e}, "
                    f"dZ/Z={candidate_enstrophy_change:.3e}"
                )

        if adaptive_core_used:
            Logging.message(
                "[Conservative regularization] broadened regenerated core "
                f"{configured_core_radius:.3e}->{float(new_radius.mean()):.3e} m "
                "to prevent energy/enstrophy injection"
            )

        if projection_only or (
            -cfg.regularization_energy_dissipation_limit <= candidate_energy_change <= 1.0e-7
            and -cfg.regularization_enstrophy_dissipation_limit
            <= candidate_enstrophy_change
            <= 1.0e-7
        ):
            uploaded = candidate
            after_integrals = candidate_integrals
            energy_change = candidate_energy_change
        else:
            direction = nullspace.to_correction(moment_corrected / nullspace.sqrt_volume[:, None])
            direction_norm = float(np.linalg.norm(direction))
            circulation_norm = max(
                float(np.linalg.norm(moment_corrected)),
                np.finfo(float).tiny,
            )
            if direction_norm <= np.finfo(float).tiny:
                raise RuntimeError("regularization has no enstrophy-restoration direction")
            direction *= circulation_norm / direction_norm
            _, plus_integrals = upload_and_integrate(moment_corrected + direction)
            _, minus_integrals = upload_and_integrate(moment_corrected - direction)
            candidate_enstrophy = float(candidate_integrals["enstrophy"])
            target_enstrophy = float(before_integrals["enstrophy"])
            linear = 0.5 * (
                float(plus_integrals["enstrophy"]) - float(minus_integrals["enstrophy"])
            )
            quadratic = (
                0.5 * (float(plus_integrals["enstrophy"]) + float(minus_integrals["enstrophy"]))
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
            energy_before = float(before_integrals["kinetic_energy"])
            energy_scale = max(abs(energy_before), np.finfo(float).tiny)
            enstrophy_scale = max(abs(target_enstrophy), np.finfo(float).tiny)
            for multiplier in real_roots:
                uploaded, trial_integrals = upload_and_integrate(
                    candidate.astype(np.float64) + multiplier * direction
                )
                trial_energy_change = (
                    float(trial_integrals["kinetic_energy"]) - energy_before
                ) / energy_scale
                trial_enstrophy_change = (
                    float(trial_integrals["enstrophy"]) - target_enstrophy
                ) / enstrophy_scale
                if (
                    -cfg.regularization_energy_dissipation_limit <= trial_energy_change <= 1.0e-7
                    and abs(trial_enstrophy_change) <= 5.0e-6
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
            energy_change, uploaded, after_integrals = max(admissible, key=lambda item: item[0])
            uploaded, after_integrals = upload_and_integrate(uploaded)

        preliminary_health = discretization_health(
            new_position,
            uploaded.astype(np.float64),
            new_radius,
        )
        if (
            projection_only
            or preliminary_health["vorticity_divergence_error"]
            > cfg.regularization_projection_trigger
        ):
            projection_result = constrained_divergence_relaxation(
                new_position,
                uploaded.astype(np.float64),
                new_radius,
                new_volume,
                grid_spacing=spacing,
                max_correction_norm=cfg.regularization_projection_max_correction,
                max_residual_ratio=0.9,
                energy_tolerance=0.02,
                enstrophy_tolerance=cfg.regularization_enstrophy_dissipation_limit,
                helicity_tolerance=0.05,
                variation_tolerance=0.05,
                spectral_convergence_fraction=0.25,
                reference_tolerances=(1.0e-4, 1.0e-4, 1.0e-4),
                max_projection_sweeps=1,
                target_moments=(before_moments[0], before_moments[2], before_moments[3]),
            )
            uploaded, after_integrals = upload_and_integrate(projection_result.circulation)

    except Exception:
        restore_old_field()
        raise

    audited = uploaded.astype(np.float64)
    after_moments = gaussian_particle_moments(new_position, audited, new_radius)
    energy_transfer = float(after_integrals["kinetic_energy"]) - float(
        before_integrals["kinetic_energy"]
    )
    energy_change = energy_transfer / max(
        abs(float(before_integrals["kinetic_energy"])), np.finfo(float).tiny
    )
    enstrophy_change = (
        float(after_integrals["enstrophy"]) - float(before_integrals["enstrophy"])
    ) / max(abs(float(before_integrals["enstrophy"])), np.finfo(float).tiny)
    if not -cfg.regularization_energy_dissipation_limit <= energy_change <= 1.0e-7:
        restore_old_field()
        raise RuntimeError(
            f"regularization changed energy by {energy_change:.3e}, outside its "
            "declared dissipative interval"
        )
    if not (-cfg.regularization_enstrophy_dissipation_limit <= enstrophy_change <= 5.0e-6):
        restore_old_field()
        raise RuntimeError(
            f"regularization changed enstrophy by {enstrophy_change:.3e}, outside "
            "its declared non-injecting interval"
        )

    impulse_scale = max(
        0.5 * float(np.linalg.norm(np.cross(position, circulation), axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    angular_terms = (
        np.cross(position, np.cross(position, circulation)) / 3.0
        - radius[:, None] ** 2 * circulation / 3.0
    )
    angular_scale = max(
        float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    errors = {
        "circulation": float(np.linalg.norm(after_moments[0] - before_moments[0]))
        / max(before_moments[1], np.finfo(float).tiny),
        "linear_impulse": float(np.linalg.norm(after_moments[2] - before_moments[2]))
        / impulse_scale,
        "angular_impulse": float(np.linalg.norm(after_moments[3] - before_moments[3]))
        / angular_scale,
    }
    roundoff_limit = 1024.0 * np.finfo(solver.np_dtype).eps
    if max(errors.values()) > roundoff_limit:
        restore_old_field()
        raise RuntimeError(
            "uploaded regularization exceeded its moment roundoff allowance: "
            + ", ".join(f"{name}={value:.3e}" for name, value in errors.items())
        )

    solver._particles_removed_this_step = 0
    solver._circulation_removed_this_step = np.zeros(3, dtype=solver.np_dtype)

    return RegularizationOutcome(
        particles_before=len(position),
        particles_after=len(new_position),
        energy_change=energy_change,
        enstrophy_change=enstrophy_change,
        projected=projection_result is not None,
    )
