"""Accepted-step limits for VPM particle fields.

Corrective stabilization modifies solver; these checks measure and limit
accepted particle fields.  Stabilization workers may modify the particle cloud and validate
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
    """Configure the non-negotiable finite-state check after accepted VPM steps.

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
        ``S = 0.5 * (grad(u) + grad(u).T)``. ``None`` disables this check.
    """

    maximum: float | None = 1.0

    def __post_init__(self) -> None:
        _optional_positive(self.maximum, "LagrangianCFLLimit.maximum")


@dataclass(frozen=True, slots=True)
class ParticleStrengthLimit:
    """Limit the largest particle vortex-strength magnitude.

    Parameters
    ----------
    maximum : float or None, optional
        Positive bound on ``max_p |Gamma_p|`` in m³/s. ``None`` disables the
        case-dependent check. Particle strength obeys ``Gamma = omega * V``.
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
        reported by :func:`particle_resolution_metrics`; ``None`` disables the check.
        The solver confirms sampled crossings using all distinct blobs before
        enforcing this threshold.
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
        the check. This metric compares particle ``Gamma`` to sampled ``omega``.
        A sampled crossing is confirmed over all distinct blobs before stopping.
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
class ParticleStateLimits:
    """Limits checked after each accepted VPM time step.

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
        Optional vortex-strength bound in m³/s.
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
                raise TypeError(f"ParticleStateLimits.{name} must be a {limit_type.__name__}")
