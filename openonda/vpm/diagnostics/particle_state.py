"""Device reductions for the accepted-step particle state limits.

After every accepted step the solver measures, over all active particles,
whether the particle fields are finite, the largest Lagrangian strain
increment ``dt * ||S||`` in the infinity and spectral norms, and the largest
particle strength ``|Gamma|`` and vorticity ``|Gamma| / V``. The reductions
run where the particle fields live; only the extrema, the indices of the
particles that attain the strain extrema, and the position of the particle
limiting the Lagrangian CFL number reach the host.

All metric arithmetic is float64. The strain ``S = 0.5 * (J + J^T)`` and its
infinity norm (largest absolute row sum) are formed exactly as in a float64
host evaluation. The spectral norm of the symmetric strain is its largest
absolute eigenvalue, obtained with cyclic Jacobi rotations; it agrees with a
singular value decomposition to round-off, including for repeated eigenvalues.
A NaN value makes the corresponding maximum NaN and the first NaN particle its
argument, as for ``numpy.max`` and ``numpy.argmax``; ties report the lowest
particle index.
"""

import numpy as np
import taichi as ti

from ..config.state_limits import MeasuredParticleState, ParticleStateMetrics

__all__ = ["measure_particle_state"]

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


@ti.kernel
def _reduce_particle_state(
    particles: ti.template(), count: ti.i32, time_step_size: ti.f64
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

        infinity, spectral = _strain_increments(particles.velocity_gradient[i], time_step_size)
        strength, vorticity = _particle_strength_and_vorticity(
            particles.vortex_strength[i], particles.particle_volume[i]
        )
        _accumulate_maximum(infinity, infinity_maximum, nan_mask, 1)
        _accumulate_maximum(spectral, spectral_maximum, nan_mask, 2)
        _accumulate_maximum(strength, strength_maximum, nan_mask, 4)
        _accumulate_maximum(vorticity, vorticity_maximum, nan_mask, 8)

    infinity_particle = count
    spectral_particle = count
    for i in range(count):
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


def measure_particle_state(particles, time_step_size: float) -> MeasuredParticleState:
    """Reduce the active particle fields to the accepted-state limit metrics.

    Parameters
    ----------
    particles : Particles
        Particle container whose velocity and velocity gradient were refreshed
        for the accepted state.
    time_step_size : float
        Accepted step size in s that scales the strain rate to an increment.

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
    result = _reduce_particle_state(particles, count, float(time_step_size)).to_numpy()
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
