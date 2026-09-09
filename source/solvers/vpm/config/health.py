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
    """Configure the non-negotiable finite-state gate after accepted VPM steps.

    Parameters
    ----------
    enabled : bool, default=True
        When true, require finite position, velocity, velocity-gradient, and
        particle-strength fields plus strictly positive core radii and volumes.
        Disable only for specialized diagnostics because an invalid accepted
        state cannot be safely restarted.
    """

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("FiniteStateCheck.enabled must be a boolean")


@dataclass(frozen=True, slots=True)
class LagrangianCFLLimit:
    """Limit material deformation accumulated during one accepted VPM step.

    Parameters
    ----------
    maximum : float or None, default=1.0
        Positive dimensionless upper bound on ``dt * ||S||_infinity``, where
        ``S = 0.5 * (grad(u) + grad(u).T)``. ``None`` disables this gate.
    """

    maximum: float | None = 1.0

    def __post_init__(self) -> None:
        _optional_positive(self.maximum, "LagrangianCFLLimit.maximum")


@dataclass(frozen=True, slots=True)
class ParticleStrengthLimit:
    """Limit the largest particle strength (vector circulation) magnitude.

    Parameters
    ----------
    maximum : float or None, optional
        Positive bound on ``max_p |Gamma_p|`` in m³/s. ``None`` disables the
        case-dependent gate. Particle strength obeys ``Gamma = omega * V``.
    """

    maximum: float | None = None

    def __post_init__(self) -> None:
        _optional_positive(self.maximum, "ParticleStrengthLimit.maximum")


@dataclass(frozen=True, slots=True)
class DivergenceLimit:
    """Limit normalized vorticity-divergence error on an accepted particle cloud.

    Parameters
    ----------
    maximum : float or None, optional
        Non-negative dimensionless threshold for the weighted divergence metric
        reported by :func:`discretization_health`; ``None`` disables the gate.
    """

    maximum: float | None = None

    def __post_init__(self) -> None:
        _optional_non_negative(self.maximum, "DivergenceLimit.maximum")


@dataclass(frozen=True, slots=True)
class MisalignmentLimit:
    """Limit directional disagreement between represented and evaluated vorticity.

    Parameters
    ----------
    maximum_degrees : float or None, optional
        Mean angular threshold in degrees, in ``[0, 180]``. ``None`` disables
        the gate. This metric compares particle ``Gamma`` to sampled ``omega``.
    """

    maximum_degrees: float | None = None

    def __post_init__(self) -> None:
        _optional_non_negative(self.maximum_degrees, "MisalignmentLimit.maximum_degrees")
        if self.maximum_degrees is not None and self.maximum_degrees > 180.0:
            raise ValueError("MisalignmentLimit.maximum_degrees must not exceed 180")


@dataclass(frozen=True, slots=True)
class GrowthLimit:
    """Bound relative growth of accepted VPM extrema between consecutive steps.

    Parameters
    ----------
    maximum_particle_strength_growth : float or None, optional
        Non-negative fractional increase allowed for ``max |Gamma|`` per step.
    maximum_vorticity_growth : float or None, optional
        Non-negative fractional increase allowed for ``max |omega|`` per step.
        ``None`` disables the corresponding history-dependent check.
    """

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

    Parameters
    ----------
    finite_state : FiniteStateCheck
        Finite/positive field validation.
    lagrangian_cfl : LagrangianCFLLimit
        Dimensionless material-deformation bound.
    maximum_particle_strength : ParticleStrengthLimit
        Optional vector-circulation bound in m³/s.
    divergence : DivergenceLimit
        Optional normalized vorticity-divergence bound.
    misalignment : MisalignmentLimit
        Optional angular alignment bound in degrees.
    growth : GrowthLimit
        Optional accepted-step relative-growth bounds.

    Notes
    -----
    The solver evaluates these limits only after refreshing all derived fields
    for an accepted physical state. Candidate Runge--Kutta stages are excluded.
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

    strain_increment_infinity: float
    strain_increment_spectral: float
    maximum_particle_strength: float
    maximum_vorticity: float
    strain_increment_infinity_particle: int = -1
    strain_increment_spectral_particle: int = -1

    @property
    def lagrangian_cfl(self) -> float:
        """Compatibility alias for the legacy health-limit storage field."""
        return self.strain_increment_infinity


def strain_increments(
    velocity_gradient: np.ndarray, time_step_size: float
) -> tuple[float, float, int, int]:
    """Return infinity/spectral strain increments and their particle indices."""
    gradient = np.asarray(velocity_gradient, dtype=np.float64)
    if gradient.size == 0:
        return 0.0, 0.0, -1, -1
    strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
    infinity_values = float(time_step_size) * np.abs(strain).sum(axis=2).max(axis=1)
    spectral_values = float(time_step_size) * np.linalg.norm(strain, ord=2, axis=(1, 2))
    return (
        float(infinity_values.max(initial=0.0)),
        float(spectral_values.max(initial=0.0)),
        int(np.argmax(infinity_values)),
        int(np.argmax(spectral_values)),
    )


class HealthError(RuntimeError):
    """An accepted VPM particle state violates its declared health limits."""

    def __init__(self, message: str, *, restartable: bool = True) -> None:
        """Create an accepted-state health failure.

        Parameters
        ----------
        message : str
            Human-readable description of the violated health gate. It is
            passed unchanged to :class:`RuntimeError`.
        restartable : bool, default=True
            Whether the caller may safely write or resume from the state that
            triggered the error. Non-finite accepted states should pass
            ``False``.

        Notes
        -----
        The exception carries the advisory :attr:`restartable` attribute; it
        does not roll back or mutate particle state.
        """
        super().__init__(message)
        self.restartable = bool(restartable)


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
                f"VPM accepted state at step {step} is invalid: " + ", ".join(sorted(set(invalid))),
                restartable=False,
            )

    if count:
        (
            strain_increment_infinity,
            strain_increment_spectral,
            infinity_particle,
            spectral_particle,
        ) = strain_increments(arrays["velocity_gradient"], time_step_size)
        strength = np.linalg.norm(arrays["vortex_strength"], axis=1)
        maximum_strength = float(strength.max(initial=0.0))
        maximum_vorticity = float(
            (strength / np.maximum(arrays["particle_volume"], np.finfo(float).tiny)).max(
                initial=0.0
            )
        )
    else:
        strain_increment_infinity = strain_increment_spectral = 0.0
        infinity_particle = spectral_particle = -1
        maximum_strength = maximum_vorticity = 0.0
    snapshot = HealthSnapshot(
        strain_increment_infinity,
        strain_increment_spectral,
        maximum_strength,
        maximum_vorticity,
        infinity_particle,
        spectral_particle,
    )

    if (
        limits.lagrangian_cfl.maximum is not None
        and strain_increment_infinity > limits.lagrangian_cfl.maximum
    ):
        raise HealthError(
            f"VPM accepted state at step {step}: Lagrangian CFL number "
            f"{strain_increment_infinity:.3g} (strain increment infinity norm) exceeds "
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
