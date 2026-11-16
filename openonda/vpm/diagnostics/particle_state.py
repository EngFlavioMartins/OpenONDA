"""Device reductions for the accepted-step particle state limits.

After every accepted step the solver measures, over all active particles,
whether the particle fields are finite, the largest Lagrangian strain
increment ``dt * ||S||`` in the infinity and spectral norms, and the largest
particle strength ``|Gamma|`` and vorticity ``|Gamma| / V``. The reductions
run where the particle fields live; only the extrema, the indices of the
particles that attain the strain extrema, and the position of the particle
limiting the Lagrangian CFL number reach the host. The strain extrema may
skip the particles inside one axis-aligned box: the coupler excludes the
particles that the FVM vorticity replaces after every interval.

All metric arithmetic is float64. The strain ``S = 0.5 * (J + J^T)`` and its
infinity norm (largest absolute row sum) are formed exactly as in a float64
host evaluation. The spectral norm of the symmetric strain is its largest
absolute eigenvalue, obtained with cyclic Jacobi rotations; it agrees with a
singular value decomposition to round-off, including for repeated eigenvalues.
A NaN value makes the corresponding maximum NaN and the first NaN particle its
argument, as for ``numpy.max`` and ``numpy.argmax``; ties report the lowest
particle index.
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti

from ..config.state_limits import ParticleStateLimits

__all__ = [
    "MeasuredParticleState",
    "ParticleStateError",
    "ParticleStateMetrics",
    "check_accepted_particle_state",
    "measure_particle_state",
]

# Bit order of the invalid-field mask returned by the reduction.
_CHECKED_FIELDS = (
    "position",
    "velocity",
    "velocity_gradient",
    "vortex_strength",
    "core_radius",
    "particle_volume",
)

# Cyclic Jacobi converges quadratically; a 3x3 symmetric matrix reaches round-off
# in about five sweeps. The cap only bounds non-finite input.
_MAXIMUM_JACOBI_SWEEPS = 16
# Stop once the off-diagonal energy is below (1e-18)^2 of the diagonal energy:
# further rotations no longer change any diagonal entry in float64.
_JACOBI_TOLERANCE = 1.0e-36

# Volume floor of the particle vorticity. Kernels must use it as ti.f64: with a
# float32 default float type the literal would round to zero.
_TINY_VOLUME = float(np.finfo(np.float64).tiny)

# Result layout: masks, four maxima, two particle indices, limiting position.
_RESULT_SIZE = 11


@dataclass(frozen=True, slots=True)
class ParticleStateMetrics:
    """Measured accepted-step field values used by :class:`ParticleStateLimits`."""

    strain_increment_infinity: float
    strain_increment_spectral: float
    maximum_particle_strength: float
    maximum_vorticity: float
    strain_increment_infinity_particle: int = -1
    strain_increment_spectral_particle: int = -1


@dataclass(frozen=True, slots=True)
class MeasuredParticleState:
    """Device-reduced accepted particle state used to enforce the limits.

    Parameters
    ----------
    invalid_fields : tuple of str
        Particle fields holding non-finite values, or non-positive core radii
        and volumes.
    metrics : ParticleStateMetrics
        Accepted-step extrema and the particles attaining the strain extrema.
    strain_limiting_position : tuple of float
        Position in m of the particle with the largest infinity-norm strain
        increment.
    """

    invalid_fields: tuple[str, ...]
    metrics: ParticleStateMetrics
    strain_limiting_position: tuple[float, float, float]


class ParticleStateError(RuntimeError):
    """An accepted VPM particle state violates its declared particle state limits."""

    def __init__(self, message: str, *, restartable: bool = True) -> None:
        """Report an accepted particle field outside its configured limits.

        Parameters
        ----------
        message : str
            Human-readable description of the violated particle state check. It is
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


def check_accepted_particle_state(
    *,
    limits: ParticleStateLimits,
    step: int,
    time_step_size: float,
    state: MeasuredParticleState,
    resolution: dict[str, float],
    previous: ParticleStateMetrics | None,
) -> ParticleStateMetrics:
    """Enforce ``limits`` on one fully refreshed and measured accepted state."""
    if limits.finite_state.enabled and state.invalid_fields:
        raise ParticleStateError(
            f"VPM accepted state at step {step} is invalid: "
            + ", ".join(sorted(set(state.invalid_fields))),
            restartable=False,
        )

    snapshot = state.metrics
    strain_increment_infinity = snapshot.strain_increment_infinity
    infinity_particle = snapshot.strain_increment_infinity_particle
    maximum_strength = snapshot.maximum_particle_strength

    if (
        limits.lagrangian_cfl.maximum is not None
        and strain_increment_infinity > limits.lagrangian_cfl.maximum
    ):
        maximum = limits.lagrangian_cfl.maximum
        strain_rate = strain_increment_infinity / float(time_step_size)
        position_text = ", ".join(
            f"{coordinate:.9g}" for coordinate in state.strain_limiting_position
        )
        raise ParticleStateError(
            f"VPM accepted state at step {step}: Lagrangian CFL number "
            f"{strain_increment_infinity:.9g} (strain increment infinity norm) exceeds "
            f"maximum={maximum:.9g}; particle={infinity_particle}, "
            f"position=({position_text}) m, strain_rate={strain_rate:.9g} 1/s, "
            f"time_step_size={time_step_size:.9g} s. "
            f"The frozen-state time-step ceiling is {maximum / strain_rate:.9g} s. "
            "Reduce time_step_size with a safety margin; if strain keeps growing, "
            "check spatial resolution and vorticity divergence/alignment as well."
        )
    if (
        limits.maximum_particle_strength.maximum is not None
        and maximum_strength > limits.maximum_particle_strength.maximum
    ):
        raise ParticleStateError(
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
            raise ParticleStateError(
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
                raise ParticleStateError(
                    f"VPM accepted state at step {step}: {label} {value:.3e} exceeds "
                    f"maximum={maximum:.3e}."
                )
    return snapshot


@ti.func
def _symmetric_spectral_norm(a):
    """Largest absolute eigenvalue of a symmetric float64 3x3 matrix."""
    for _ in range(_MAXIMUM_JACOBI_SWEEPS):
        off_diagonal = a[0, 1] ** 2 + a[0, 2] ** 2 + a[1, 2] ** 2
        diagonal = a[0, 0] ** 2 + a[1, 1] ** 2 + a[2, 2] ** 2
        if off_diagonal <= _JACOBI_TOLERANCE * diagonal:
            break
        for p, q, r in ti.static(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            apq = a[p, q]
            if apq != 0.0:
                # Rotation annihilating a[p, q] (Golub & Van Loan, Alg. 8.4.1).
                theta = (a[q, q] - a[p, p]) / (2.0 * apq)
                t = 1.0 / (ti.abs(theta) + ti.sqrt(1.0 + theta * theta))
                if theta < 0.0:
                    t = -t
                c = 1.0 / ti.sqrt(1.0 + t * t)
                s = t * c
                arp = a[r, p]
                arq = a[r, q]
                a[p, p] -= t * apq
                a[q, q] += t * apq
                a[p, q] = 0.0
                a[q, p] = 0.0
                a[r, p] = c * arp - s * arq
                a[p, r] = a[r, p]
                a[r, q] = s * arp + c * arq
                a[q, r] = a[r, q]
    return ti.max(ti.abs(a[0, 0]), ti.abs(a[1, 1]), ti.abs(a[2, 2]))


@ti.func
def _strain_increments(gradient, time_step_size):
    """Return ``dt * ||S||_infinity`` and ``dt * ||S||_2`` for one particle."""
    jacobian = ti.cast(gradient, ti.f64)
    strain = 0.5 * (jacobian + jacobian.transpose())
    infinity = ti.cast(0.0, ti.f64)
    for row in ti.static(range(3)):
        row_sum = ti.abs(strain[row, 0]) + ti.abs(strain[row, 1]) + ti.abs(strain[row, 2])
        infinity = ti.max(infinity, row_sum)
    nan = ti.cast(ti.math.nan, ti.f64)
    spectral = nan
    if ti.math.isnan(strain).any():
        infinity = nan
    elif not ti.math.isinf(strain).any():
        spectral = _symmetric_spectral_norm(strain)
    return time_step_size * infinity, time_step_size * spectral


@ti.func
def _particle_strength_and_vorticity(strength, volume):
    """Return ``|Gamma|`` and ``|Gamma| / max(V, tiny)`` in float64."""
    gamma = ti.cast(strength, ti.f64)
    magnitude = ti.sqrt(gamma[0] * gamma[0] + gamma[1] * gamma[1] + gamma[2] * gamma[2])
    vorticity = magnitude / ti.max(ti.cast(volume, ti.f64), ti.f64(_TINY_VOLUME))
    if ti.math.isnan(volume):
        vorticity = ti.cast(ti.math.nan, ti.f64)
    return magnitude, vorticity


@ti.func
def _accumulate_maximum(value, maximum: ti.template(), nan_mask: ti.template(), bit: ti.template()):
    if ti.math.isnan(value):
        ti.atomic_or(nan_mask, bit)
    else:
        ti.atomic_max(maximum, value)


@ti.func
def _attains_maximum(value, maximum, nan_mask, bit: ti.template()):
    return (ti.math.isnan(value) and (nan_mask & bit) != 0) or (
        (nan_mask & bit) == 0 and value == maximum
    )


@ti.func
def _strain_counted(position, excluded, use_exclusion):
    counted = True
    if use_exclusion != 0:
        inside = True
        for axis in ti.static(range(3)):
            coordinate = ti.cast(position[axis], ti.f64)
            if coordinate < excluded[2 * axis] or coordinate > excluded[2 * axis + 1]:
                inside = False
        counted = not inside
    return counted


@ti.kernel
def _reduce_particle_state(
    particles: ti.template(),
    count: ti.i32,
    time_step_size: ti.f64,
    excluded: ti.types.vector(6, ti.f64),
    use_exclusion: ti.i32,
) -> ti.types.vector(_RESULT_SIZE, ti.f64):
    invalid_mask = 0
    nan_mask = 0
    infinity_maximum = ti.cast(0.0, ti.f64)
    spectral_maximum = ti.cast(0.0, ti.f64)
    strength_maximum = ti.cast(0.0, ti.f64)
    vorticity_maximum = ti.cast(0.0, ti.f64)
    for i in range(count):
        for bit, field in ti.static(
            enumerate(
                (
                    particles.position,
                    particles.velocity,
                    particles.velocity_gradient,
                    particles.vortex_strength,
                )
            )
        ):
            if ti.math.isnan(field[i]).any() or ti.math.isinf(field[i]).any():
                ti.atomic_or(invalid_mask, 1 << bit)
        for bit, field in ti.static(((4, particles.core_radius), (5, particles.particle_volume))):
            value = field[i]
            if ti.math.isnan(value) or ti.math.isinf(value) or value <= 0.0:
                ti.atomic_or(invalid_mask, 1 << bit)

        if _strain_counted(particles.position[i], excluded, use_exclusion):
            infinity, spectral = _strain_increments(particles.velocity_gradient[i], time_step_size)
            _accumulate_maximum(infinity, infinity_maximum, nan_mask, 1)
            _accumulate_maximum(spectral, spectral_maximum, nan_mask, 2)
        strength, vorticity = _particle_strength_and_vorticity(
            particles.vortex_strength[i], particles.particle_volume[i]
        )
        _accumulate_maximum(strength, strength_maximum, nan_mask, 4)
        _accumulate_maximum(vorticity, vorticity_maximum, nan_mask, 8)

    infinity_particle = count
    spectral_particle = count
    for i in range(count):
        if _strain_counted(particles.position[i], excluded, use_exclusion):
            infinity, spectral = _strain_increments(particles.velocity_gradient[i], time_step_size)
            if _attains_maximum(infinity, infinity_maximum, nan_mask, 1):
                ti.atomic_min(infinity_particle, i)
            if _attains_maximum(spectral, spectral_maximum, nan_mask, 2):
                ti.atomic_min(spectral_particle, i)

    limiting_position = ti.cast(particles.position[ti.min(infinity_particle, count - 1)], ti.f64)
    return ti.Vector(
        [
            invalid_mask,
            nan_mask,
            infinity_maximum,
            spectral_maximum,
            strength_maximum,
            vorticity_maximum,
            infinity_particle,
            spectral_particle,
            limiting_position[0],
            limiting_position[1],
            limiting_position[2],
        ],
        dt=ti.f64,
    )


def measure_particle_state(
    particles,
    time_step_size: float,
    strain_excluded_bounds: tuple[float, ...] | None = None,
) -> MeasuredParticleState:
    """Reduce the active particle fields to the accepted-state limit metrics.

    Parameters
    ----------
    particles : Particles
        Particle container whose velocity and velocity gradient were refreshed
        for the accepted state.
    time_step_size : float
        Accepted step size in s that scales the strain rate to an increment.
    strain_excluded_bounds : tuple of float or None
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m. Particles inside this
        box do not enter the strain extrema; every other metric covers all
        particles.

    Returns
    -------
    MeasuredParticleState
        Invalid field names, the metric extrema with their particle indices,
        and the position of the particle with the largest infinity-norm strain
        increment. An empty cloud gives zero metrics and particle index ``-1``.
    """
    count = int(particles.n_particles_total)
    if count == 0:
        return MeasuredParticleState((), ParticleStateMetrics(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    use_exclusion = strain_excluded_bounds is not None
    excluded = np.asarray(
        strain_excluded_bounds if use_exclusion else np.zeros(6), dtype=np.float64
    ).reshape(6)
    result = _reduce_particle_state(
        particles, count, float(time_step_size), excluded, int(use_exclusion)
    ).to_numpy()
    invalid_mask = int(result[0])
    nan_mask = int(result[1])
    maxima = [float("nan") if nan_mask & (1 << bit) else float(result[2 + bit]) for bit in range(4)]
    return MeasuredParticleState(
        invalid_fields=tuple(
            name for bit, name in enumerate(_CHECKED_FIELDS) if invalid_mask & (1 << bit)
        ),
        metrics=ParticleStateMetrics(
            strain_increment_infinity=maxima[0],
            strain_increment_spectral=maxima[1],
            maximum_particle_strength=maxima[2],
            maximum_vorticity=maxima[3],
            strain_increment_infinity_particle=int(result[6]),
            strain_increment_spectral_particle=int(result[7]),
        ),
        strain_limiting_position=tuple(float(value) for value in result[8:11]),
    )
