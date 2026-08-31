"""Typed accepted-step health limits for VPM particle states.

Corrective stabilization and health assessment deliberately have different
ownership.  Stabilization workers may modify the particle cloud and validate
their own correction; these limits are evaluated by :class:`VPMSolver` only
after an accepted physical step has refreshed its diagnostic dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _optional_positive(value: float | None, name: str) -> None:
    if value is not None and (not np.isfinite(value) or value <= 0.0):
        raise ValueError(f"{name} must be finite and positive or None")


def _optional_non_negative(value: float | None, name: str) -> None:
    if value is not None and (not np.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative or None")


@dataclass(frozen=True, slots=True)
class FiniteStateCheck:
    """Require finite particle fields and strictly positive radii and volumes."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("FiniteStateCheck.enabled must be a boolean")


@dataclass(frozen=True, slots=True)
class LagrangianCFLLimit:
    """Limit the accepted-step material deformation CFL number."""

    maximum: float | None = 1.0

    def __post_init__(self) -> None:
        _optional_positive(self.maximum, "LagrangianCFLLimit.maximum")


@dataclass(frozen=True, slots=True)
class ParticleStrengthLimit:
    """Limit the magnitude of every particle vortex-strength vector [m³/s]."""

    maximum: float | None = None

    def __post_init__(self) -> None:
        _optional_positive(self.maximum, "ParticleStrengthLimit.maximum")


@dataclass(frozen=True, slots=True)
class DivergenceLimit:
    """Limit weighted vorticity-divergence error from resolution diagnostics."""

    maximum: float | None = None

    def __post_init__(self) -> None:
        _optional_non_negative(self.maximum, "DivergenceLimit.maximum")


@dataclass(frozen=True, slots=True)
class MisalignmentLimit:
    """Limit mean vortex-strength/vorticity misalignment in degrees."""

    maximum_degrees: float | None = None

    def __post_init__(self) -> None:
        _optional_non_negative(self.maximum_degrees, "MisalignmentLimit.maximum_degrees")
        if self.maximum_degrees is not None and self.maximum_degrees > 180.0:
            raise ValueError("MisalignmentLimit.maximum_degrees must not exceed 180")


@dataclass(frozen=True, slots=True)
class GrowthLimit:
    """One-step relative-growth limits for particle strength and peak vorticity."""

    maximum_particle_strength_growth: float | None = None
    maximum_vorticity_growth: float | None = None

    def __post_init__(self) -> None:
        _optional_non_negative(
            self.maximum_particle_strength_growth,
            "GrowthLimit.maximum_particle_strength_growth",
        )
        _optional_non_negative(
            self.maximum_vorticity_growth,
            "GrowthLimit.maximum_vorticity_growth",
        )


@dataclass(frozen=True, slots=True)
class HealthLimits:
    """Complete accepted-step particle-health limits owned by ``VPMSolver``.

    The defaults enforce finite state and a conservative Lagrangian CFL bound.
    Strength, resolution, and growth limits are opt-in because their safe
    values are case-dependent physical choices.
    """

    finite_state: FiniteStateCheck = FiniteStateCheck()
    lagrangian_cfl: LagrangianCFLLimit = LagrangianCFLLimit()
    maximum_particle_strength: ParticleStrengthLimit = ParticleStrengthLimit()
    divergence: DivergenceLimit = DivergenceLimit()
    misalignment: MisalignmentLimit = MisalignmentLimit()
    growth: GrowthLimit = GrowthLimit()

    def __post_init__(self) -> None:
        expected = (
            ("finite_state", FiniteStateCheck),
            ("lagrangian_cfl", LagrangianCFLLimit),
            ("maximum_particle_strength", ParticleStrengthLimit),
            ("divergence", DivergenceLimit),
            ("misalignment", MisalignmentLimit),
            ("growth", GrowthLimit),
        )
        for name, limit_type in expected:
            if not isinstance(getattr(self, name), limit_type):
                raise TypeError(f"HealthLimits.{name} must be a {limit_type.__name__}")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Measured accepted-step health values used by :class:`HealthLimits`."""

    lagrangian_cfl: float
    maximum_particle_strength: float
    maximum_vorticity: float


class HealthError(RuntimeError):
    """An accepted VPM particle state violates its declared health limits."""


def accepted_step_health(
    *,
    limits: HealthLimits,
    step: int,
    time_step_size: float,
    position: np.ndarray,
    velocity: np.ndarray,
    velocity_gradient: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    resolution: dict[str, float],
    previous: HealthSnapshot | None,
) -> HealthSnapshot:
    """Measure and enforce ``limits`` for one fully refreshed accepted state."""
    arrays = {
        "position": np.asarray(position),
        "velocity": np.asarray(velocity),
        "velocity_gradient": np.asarray(velocity_gradient),
        "vortex_strength": np.asarray(vortex_strength),
        "core_radius": np.asarray(core_radius),
        "particle_volume": np.asarray(particle_volume),
    }
    count = len(arrays["position"])
    if limits.finite_state.enabled:
        invalid = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if np.any(arrays["core_radius"] <= 0.0):
            invalid.append("core_radius")
        if np.any(arrays["particle_volume"] <= 0.0):
            invalid.append("particle_volume")
        if invalid:
            raise HealthError(
                f"VPM accepted state at step {step} is invalid: " + ", ".join(sorted(set(invalid)))
            )

    if count:
        gradient = np.asarray(arrays["velocity_gradient"], dtype=np.float64)
        strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
        cfl = float(time_step_size * np.abs(strain).sum(axis=1).max(initial=0.0))
        strength = np.linalg.norm(arrays["vortex_strength"], axis=1)
        maximum_strength = float(strength.max(initial=0.0))
        maximum_vorticity = float(
            (strength / np.maximum(arrays["particle_volume"], np.finfo(float).tiny)).max(
                initial=0.0
            )
        )
    else:
        cfl = maximum_strength = maximum_vorticity = 0.0
    snapshot = HealthSnapshot(cfl, maximum_strength, maximum_vorticity)

    if limits.lagrangian_cfl.maximum is not None and cfl > limits.lagrangian_cfl.maximum:
        raise HealthError(
            f"VPM accepted state at step {step}: Lagrangian CFL number {cfl:.3g} exceeds "
            f"maximum={limits.lagrangian_cfl.maximum:.3g}; reduce time_step_size."
        )
    if (
        limits.maximum_particle_strength.maximum is not None
        and maximum_strength > limits.maximum_particle_strength.maximum
    ):
        raise HealthError(
            f"VPM accepted state at step {step}: maximum particle strength {maximum_strength:.3e} "
            f"exceeds maximum={limits.maximum_particle_strength.maximum:.3e}."
        )

    checks = (
        (
            "vorticity_divergence_error",
            limits.divergence.maximum,
            "vorticity divergence error",
        ),
        (
            "vortex_strength_misalignment_degrees",
            limits.misalignment.maximum_degrees,
            "vortex-strength misalignment (degrees)",
        ),
    )
    for key, maximum, label in checks:
        if maximum is None:
            continue
        value = float(resolution.get(key, float("nan")))
        if not np.isfinite(value) or value > maximum:
            raise HealthError(
                f"VPM accepted state at step {step}: {label} {value:.3e} exceeds "
                f"maximum={maximum:.3e}."
            )

    if previous is not None:
        growth_checks = (
            (
                "particle-strength growth",
                (snapshot.maximum_particle_strength - previous.maximum_particle_strength)
                / max(previous.maximum_particle_strength, np.finfo(float).tiny),
                limits.growth.maximum_particle_strength_growth,
            ),
            (
                "peak-vorticity growth",
                (snapshot.maximum_vorticity - previous.maximum_vorticity)
                / max(previous.maximum_vorticity, np.finfo(float).tiny),
                limits.growth.maximum_vorticity_growth,
            ),
        )
        for label, value, maximum in growth_checks:
            if maximum is not None and (not np.isfinite(value) or value > maximum):
                raise HealthError(
                    f"VPM accepted state at step {step}: {label} {value:.3e} exceeds "
                    f"maximum={maximum:.3e}."
                )
    return snapshot
