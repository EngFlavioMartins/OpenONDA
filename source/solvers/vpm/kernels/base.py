"""Shared radial vortex-blob kernel contract for all induction methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class RadialVortexKernel:
    """Numerical contract for an isotropic regularized Biot--Savart kernel.

    ``q`` includes the ``1/(4π)`` Biot--Savart constant and ``zeta`` is the
    normalized radial vorticity profile.  All induction backends consume this
    object for host-side near/far decisions; the existing Taichi factories are
    still used for device kernels by the direct and Barnes--Hut paths.
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

    def near_field_cutoff(self, core_radius: float, tolerance: float) -> float:
        """Return a conservative physical near-field radius for ``tolerance``."""
        if core_radius <= 0.0 or not 0.0 < tolerance < 1.0:
            raise ValueError("core_radius must be positive and tolerance must lie in (0, 1)")
        low, high = 0.0, 1.0
        while float(np.max(self.far_field_error(high))) > tolerance and high < 1.0e6:
            high *= 2.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if float(np.max(self.far_field_error(middle))) > tolerance:
                low = middle
            else:
                high = middle
        return float(core_radius * high)


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
    return (_erf(rho) + 2.0 / math.sqrt(math.pi) * rho * (rho * rho - 1.0) * np.exp(-rho * rho)) / (
        4.0 * math.pi
    )


def _high_order_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return (2.5 - rho * rho) * np.exp(-rho * rho) / math.pi**1.5


def _super_gaussian_q(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return (
        _erf(rho / math.sqrt(2.0))
        - math.sqrt(2.0 / math.pi) * rho * (1.0 - rho * rho / 2.0) * np.exp(-rho * rho / 2.0)
    ) / (4.0 * math.pi)


def _super_gaussian_zeta(rho):
    rho = np.asarray(rho, dtype=np.float64)
    return (
        math.sqrt(2.0 / math.pi)
        * (2.5 - rho * rho / 2.0)
        * np.exp(-rho * rho / 2.0)
        / (4.0 * math.pi)
    )


def make_vortex_kernel(name: str) -> RadialVortexKernel:
    """Construct one of the supported isotropic radial vortex kernels."""
    key = name.upper()
    factories = {
        "GAUSSIAN": (_gaussian_q, _gaussian_zeta, 1.5),
        "HIGH_ORDER_GAUSSIAN": (_high_order_q, _high_order_zeta, 0.0),
        "SUPER_GAUSSIAN": (_super_gaussian_q, _super_gaussian_zeta, 0.0),
        "WINCKELMANS": (_winckelmans_q, _winckelmans_zeta, 1.5),
    }
    try:
        q_function, zeta_function, angular_constant = factories[key]
    except KeyError as exc:
        raise ValueError(f"unsupported vortex kernel {name!r}") from exc
    return RadialVortexKernel(
        key, q_function, zeta_function, angular_impulse_constant=angular_constant
    )


__all__ = ["RadialVortexKernel", "make_vortex_kernel"]
