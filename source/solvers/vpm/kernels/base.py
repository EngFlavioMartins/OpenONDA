"""Shared radial vortex-blob kernel contract for all induction methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class RadialVortexKernel:
    """Numerical contract for an isotropic regularized Biot--Savart kernel.

    ``q`` includes the ``1/(4π)`` Biot--Savart constant and ``zeta`` is the
    normalized radial vorticity profile.  Particle-to-particle operators use
    the symmetric pair radius ``(σ_target + σ_source)/2``.  Arbitrary target
    field evaluation is a source-field operator and uses the source radius
    only; this distinction is intentional and is shared by direct, treecode,
    and FMM target paths.  All induction backends consume this object for
    host-side near/far decisions; the existing Taichi factories are still used
    for device kernels by the direct and Barnes--Hut paths.
    """

    name: str
    q_function: ArrayFunction
    zeta_function: ArrayFunction
    q_infinity: float = 1.0 / (4.0 * math.pi)
    angular_impulse_constant: float = 1.5

    def q(self, rho):
        """Return the dimensionless enclosed-circulation factor."""
        return self.q_function(np.asarray(rho, dtype=np.float64))

    def zeta(self, rho):
        """Return the dimensionless radial vorticity profile."""
        return self.zeta_function(np.asarray(rho, dtype=np.float64))

    def q_prime(self, rho):
        """Return ``dq/dρ = ρ²ζ(ρ)`` for normalized radial blobs."""
        rho = np.asarray(rho, dtype=np.float64)
        return rho * rho * self.zeta(rho)

    def pair_radius(self, target_core, source_core):
        """Return the symmetric radius used by the canonical particle pair."""
        return 0.5 * (np.asarray(target_core) + np.asarray(source_core))

    def velocity_pair(self, displacement, source_strength, target_core, source_core):
        """Evaluate source velocity at a particle target using pair radius."""
        displacement = np.asarray(displacement, dtype=np.float64)
        source_strength = np.asarray(source_strength, dtype=np.float64)
        radius = np.linalg.norm(displacement, axis=-1)
        core = self.pair_radius(target_core, source_core)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        rho = np.divide(radius, core, out=np.zeros_like(radius), where=core > 0.0)
        scale = np.divide(
            self.q(rho), safe_radius**3, out=np.zeros_like(radius), where=radius > 0.0
        )
        return scale[..., None] * np.cross(source_strength, displacement)

    def gradient_pair(self, displacement, source_strength, target_core, source_core):
        """Evaluate the particle-pair velocity Jacobian ``∂u/∂x_target``."""
        displacement = np.asarray(displacement, dtype=np.float64)
        source_strength = np.asarray(source_strength, dtype=np.float64)
        radius = np.linalg.norm(displacement, axis=-1)
        core = self.pair_radius(target_core, source_core)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        rho = np.divide(radius, core, out=np.zeros_like(radius), where=core > 0.0)
        q_value = self.q(rho)
        q_prime = self.q_prime(rho)
        scale = np.divide(q_value, safe_radius**3, out=np.zeros_like(radius), where=radius > 0.0)
        derivative = np.divide(
            q_prime / core,
            safe_radius**3,
            out=np.zeros_like(radius),
            where=(radius > 0.0) & (core > 0.0),
        ) - np.divide(3.0 * q_value, safe_radius**4, out=np.zeros_like(radius), where=radius > 0.0)
        cross_matrix = np.zeros(displacement.shape[:-1] + (3, 3), dtype=np.float64)
        cross_matrix[..., 0, 1] = -source_strength[..., 2]
        cross_matrix[..., 0, 2] = source_strength[..., 1]
        cross_matrix[..., 1, 0] = source_strength[..., 2]
        cross_matrix[..., 1, 2] = -source_strength[..., 0]
        cross_matrix[..., 2, 0] = -source_strength[..., 1]
        cross_matrix[..., 2, 1] = source_strength[..., 0]
        cross_matrix *= scale[..., None, None]
        cross = np.cross(source_strength, displacement)
        cross_matrix += (
            derivative[..., None, None]
            * cross[..., :, None]
            * displacement[..., None, :]
            / safe_radius[..., None, None]
        )
        return np.where((radius > 0.0)[..., None, None], cross_matrix, 0.0)

    def transposed_rate_pair(
        self, displacement, target_strength, source_strength, target_core, source_core
    ):
        """Evaluate the canonical conservative transposed pair contribution."""
        displacement = np.asarray(displacement, dtype=np.float64)
        target_strength = np.asarray(target_strength, dtype=np.float64)
        source_strength = np.asarray(source_strength, dtype=np.float64)
        radius = np.linalg.norm(displacement, axis=-1)
        core = self.pair_radius(target_core, source_core)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        rho = np.divide(radius, core, out=np.zeros_like(radius), where=core > 0.0)
        q_value = self.q(rho)
        zeta_value = self.zeta(rho)
        coefficient_a = np.divide(
            q_value, safe_radius**3, out=np.zeros_like(radius), where=radius > 0.0
        )
        coefficient_b = np.divide(
            3.0 * q_value - zeta_value * rho**3,
            core**5 * np.where(rho > 0.0, rho**5, 1.0),
            out=np.zeros_like(radius),
            where=(radius > 0.0) & (core > 0.0),
        )
        result = coefficient_a[..., None] * np.cross(target_strength, source_strength)
        result += (
            coefficient_b
            * np.sum(target_strength * np.cross(displacement, source_strength), axis=-1)
        )[..., None] * displacement
        return np.where((radius > 0.0)[..., None], result, 0.0)

    def far_field_error(self, rho):
        """Estimate regularization error relative to the singular far field."""
        return np.abs(self.q_infinity - self.q(rho))

    def gradient_far_field_error(self, rho):
        """Estimate the radial gradient-coefficient error in the far field."""
        rho = np.asarray(rho, dtype=np.float64)
        regularized = 3.0 * self.q(rho) - self.zeta(rho) * rho**3
        return np.abs(3.0 * self.q_infinity - regularized)

    def dimensionless_tail_cutoffs(
        self,
        velocity_relative_tolerance: float,
        gradient_relative_tolerance: float,
    ) -> tuple[float, float]:
        """Return cached velocity and gradient regularization cutoffs."""
        velocity = _cached_dimensionless_tail_cutoff(
            self,
            float(velocity_relative_tolerance),
            False,
        )
        gradient = _cached_dimensionless_tail_cutoff(
            self,
            float(gradient_relative_tolerance),
            True,
        )
        return velocity, gradient

    def near_field_cutoff(self, core_radius: float, tolerance: float) -> float:
        """Return a conservative physical near-field radius for ``tolerance``."""
        return _cached_near_field_cutoff(self, float(core_radius), float(tolerance))


@lru_cache(maxsize=512)
def _cached_near_field_cutoff(
    kernel: RadialVortexKernel, core_radius: float, tolerance: float
) -> float:
    """Cache scalar kernel cutoff solves without retaining mutable instances."""
    if core_radius <= 0.0 or not 0.0 < tolerance < 1.0:
        raise ValueError("core_radius must be positive and tolerance must lie in (0, 1)")
    low, high = 0.0, 1.0
    while float(np.max(kernel.far_field_error(high))) > tolerance and high < 1.0e6:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if float(np.max(kernel.far_field_error(middle))) > tolerance:
            low = middle
        else:
            high = middle
    return float(core_radius * high)


@lru_cache(maxsize=128)
def _cached_dimensionless_tail_cutoff(
    kernel: RadialVortexKernel,
    relative_tolerance: float,
    gradient: bool,
) -> float:
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    reference = 3.0 * kernel.q_infinity if gradient else kernel.q_infinity

    def relative_error(rho: float) -> float:
        error = kernel.gradient_far_field_error(rho) if gradient else kernel.far_field_error(rho)
        return float(np.max(error)) / reference

    low, high = 0.0, 1.0
    while relative_error(high) > relative_tolerance and high < 1.0e6:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if relative_error(middle) > relative_tolerance:
            low = middle
        else:
            high = middle
    return float(high)


def _erf(values: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[float])(values)


def _gaussian_q(rho):
    rho = np.asarray(rho, dtype=np.float64)
    result = (_erf(rho) - 2.0 / math.sqrt(math.pi) * rho * np.exp(-rho * rho)) / (4.0 * math.pi)
    small = rho < 0.2
    d2 = rho * rho
    series = (
        (4.0 / (3.0 * math.sqrt(math.pi)))
        * rho
        * d2
        * (1.0 - 0.6 * d2 + 3.0 / 14.0 * d2 * d2)
        / (4.0 * math.pi)
    )
    return np.where(small, series, result)


def _gaussian_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return np.exp(-rho * rho) / math.pi**1.5


def _winckelmans_q(rho):
    rho = np.asarray(rho, dtype=np.float64)
    d2 = rho * rho
    base = d2 + 1.0
    return rho * d2 * (d2 + 2.5) / (base * base * np.sqrt(base)) / (4.0 * math.pi)


def _winckelmans_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    base = rho * rho + 1.0
    return 7.5 / (base * base * base * np.sqrt(base)) / (4.0 * math.pi)


def _high_order_q(rho):
    rho = np.asarray(rho, dtype=np.float64)
    d2 = rho * rho
    closed = (_erf(rho) + 2.0 / math.sqrt(math.pi) * rho * (d2 - 1.0) * np.exp(-d2)) / (
        4.0 * math.pi
    )
    series = (
        2.0
        / math.sqrt(math.pi)
        * rho
        * d2
        * (
            5.0 / 3.0
            + d2
            * (
                -7.0 / 5.0
                + d2
                * (9.0 / 14.0 + d2 * (-11.0 / 54.0 + d2 * (13.0 / 264.0 + d2 * (-1.0 / 104.0))))
            )
        )
        / (4.0 * math.pi)
    )
    return np.where(rho < 0.5, series, closed)


def _high_order_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return (2.5 - rho * rho) * np.exp(-rho * rho) / math.pi**1.5


def _super_gaussian_q(rho):
    rho = np.asarray(rho, dtype=np.float64)
    d2 = rho * rho
    closed = (
        _erf(rho / math.sqrt(2.0))
        - math.sqrt(2.0 / math.pi) * rho * (1.0 - d2 / 2.0) * np.exp(-d2 / 2.0)
    ) / (4.0 * math.pi)
    series = (
        math.sqrt(2.0 / math.pi)
        * rho
        * d2
        * (
            5.0 / 6.0
            + d2
            * (
                -7.0 / 20.0
                + d2
                * (9.0 / 112.0 + d2 * (-11.0 / 864.0 + d2 * (13.0 / 8448.0 + d2 * (-1.0 / 6656.0))))
            )
        )
        / (4.0 * math.pi)
    )
    return np.where(rho < 0.5, series, closed)


def _super_gaussian_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return (
        math.sqrt(2.0 / math.pi)
        * (2.5 - rho * rho / 2.0)
        * np.exp(-rho * rho / 2.0)
        / (4.0 * math.pi)
    )


def _gaussian_device_factory(dtype):
    from .gaussian import create_gaussian_kernels

    return create_gaussian_kernels(dtype)


def _high_order_device_factory(dtype):
    from .high_order_gaussian import create_high_order_gaussian_kernels

    return create_high_order_gaussian_kernels(dtype)


def _super_gaussian_device_factory(dtype):
    from .super_gaussian import create_super_gaussian_kernels

    return create_super_gaussian_kernels(dtype)


def _winckelmans_device_factory(dtype):
    from .winckelmans import create_winckelmans_kernels

    return create_winckelmans_kernels(dtype)


_KERNEL_REGISTRY = {
    "GAUSSIAN": (_gaussian_q, _gaussian_zeta, 1.5, _gaussian_device_factory),
    "HIGH_ORDER_GAUSSIAN": (
        _high_order_q,
        _high_order_zeta,
        0.0,
        _high_order_device_factory,
    ),
    "SUPER_GAUSSIAN": (
        _super_gaussian_q,
        _super_gaussian_zeta,
        0.0,
        _super_gaussian_device_factory,
    ),
    "WINCKELMANS": (_winckelmans_q, _winckelmans_zeta, 1.5, _winckelmans_device_factory),
}


def make_vortex_kernel(name: str) -> RadialVortexKernel:
    """Construct one of the supported isotropic radial vortex kernels."""
    key = name.upper()
    try:
        q_function, zeta_function, angular_constant, _ = _KERNEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unsupported vortex kernel {name!r}") from exc
    return RadialVortexKernel(
        key, q_function, zeta_function, angular_impulse_constant=angular_constant
    )


def make_device_vortex_kernels(name: str, dtype):
    """Build the Taichi radial functions from the authoritative registry."""
    key = name.upper()
    try:
        device_factory = _KERNEL_REGISTRY[key][3]
    except KeyError as exc:
        raise ValueError(f"unsupported vortex kernel {name!r}") from exc
    return device_factory(dtype)


__all__ = ["RadialVortexKernel", "make_device_vortex_kernels", "make_vortex_kernel"]
