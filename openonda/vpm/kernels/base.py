"""Shared radial vortex-blob kernel conditions for all induction methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .registry import kernel_definition

ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class RadialVortexKernel:
    """Numerical conditions for an isotropic regularized Biot--Savart kernel.

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

    def q(self, rho: np.ndarray | float) -> np.ndarray:
        """Return the regularized Biot--Savart circulation factor.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Non-negative dimensionless radius ``r / sigma``. Any input shape
            is preserved.

        Returns
        -------
        numpy.ndarray
            Dimensionless factor including ``1/(4*pi)``. Its far-field limit
            is :attr:`q_infinity`.
        """
        return self.q_function(np.asarray(rho, dtype=np.float64))

    def zeta(self, rho: np.ndarray | float) -> np.ndarray:
        """Return the normalized dimensionless radial vorticity profile.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Non-negative radius normalized by the core radius ``sigma``.

        Returns
        -------
        numpy.ndarray
            Profile with the normalization used by ``q_prime = rho²*zeta``.
        """
        return self.zeta_function(np.asarray(rho, dtype=np.float64))

    def q_prime(self, rho: np.ndarray | float) -> np.ndarray:
        """Return the derivative ``dq/dρ = ρ² zeta(rho)``.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Dimensionless radius ``r / sigma``.

        Returns
        -------
        numpy.ndarray
            Derivative of :meth:`q` with respect to dimensionless radius.
        """
        rho = np.asarray(rho, dtype=np.float64)
        return rho * rho * self.zeta(rho)

    def pair_radius(
        self, target_core: np.ndarray | float, source_core: np.ndarray | float
    ) -> np.ndarray:
        """Return the symmetric core radius for a particle pair.

        Parameters
        ----------
        target_core, source_core : float or numpy.ndarray
            Target and source core radii in metres. Inputs broadcast under
            NumPy rules.

        Returns
        -------
        numpy.ndarray
            ``0.5 * (target_core + source_core)`` in metres.
        """
        return 0.5 * (np.asarray(target_core) + np.asarray(source_core))

    def velocity_pair(
        self,
        displacement: np.ndarray,
        source_strength: np.ndarray,
        target_core: np.ndarray | float,
        source_core: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate regularized source velocity at particle targets.

        Parameters
        ----------
        displacement : numpy.ndarray
            Target-minus-source displacement with shape ``(..., 3)`` in m.
        source_strength : numpy.ndarray
            Source circulation vector(s), shape ``(..., 3)`` in m³/s.
        target_core, source_core : float or numpy.ndarray
            Pair core radii in m, broadcastable to the displacement batch.

        Returns
        -------
        numpy.ndarray
            Induced velocity with shape ``(..., 3)`` in m/s. Coincident points
            return zero rather than a singular value.

        Notes
        -----
        The result is ``q(r/sigma) * (Gamma_source × r) / r³`` using the
        symmetric pair radius. This vector convention is shared by direct,
        treecode, FMM near-field, and host reference calculations.
        """
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

    def gradient_pair(
        self,
        displacement: np.ndarray,
        source_strength: np.ndarray,
        target_core: np.ndarray | float,
        source_core: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate the regularized particle-pair velocity Jacobian.

        Parameters
        ----------
        displacement : numpy.ndarray
            Target-minus-source displacement, shape ``(..., 3)``, in m.
        source_strength : numpy.ndarray
            Source circulation vector(s), shape ``(..., 3)``, in m³/s.
        target_core, source_core : float or numpy.ndarray
            Pair core radii in m.

        Returns
        -------
        numpy.ndarray
            Jacobian ``G[i, j] = d u[i] / d x_target[j]`` with shape
            ``(..., 3, 3)`` in 1/s. Coincident points use the finite core limit.
        """
        displacement = np.asarray(displacement, dtype=np.float64)
        source_strength = np.asarray(source_strength, dtype=np.float64)
        radius = np.linalg.norm(displacement, axis=-1)
        core = self.pair_radius(target_core, source_core)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        rho = np.divide(radius, core, out=np.zeros_like(radius), where=core > 0.0)
        q_value = self.q(rho)
        q_prime = self.q_prime(rho)
        scale = np.divide(q_value, safe_radius**3, out=np.zeros_like(radius), where=radius > 0.0)
        # q(rho) = zeta(0) rho³/3 + O(rho⁵). The velocity vanishes at
        # a source centre, but its Jacobian retains this finite skew part.
        origin_scale = np.divide(
            self.zeta(0.0),
            3.0 * core**3,
            out=np.zeros_like(core, dtype=np.float64),
            where=core > 0.0,
        )
        scale = np.where(radius > 0.0, scale, origin_scale)
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
        return cross_matrix

    def transposed_rate_pair(
        self,
        displacement: np.ndarray,
        target_strength: np.ndarray,
        source_strength: np.ndarray,
        target_core: np.ndarray | float,
        source_core: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate one conservative transposed stretching pair contribution.

        Parameters
        ----------
        displacement : numpy.ndarray
            Target-minus-source displacement, shape ``(..., 3)``, in m.
        target_strength, source_strength : numpy.ndarray
            Target/source circulation vectors, shape ``(..., 3)``, in m³/s.
        target_core, source_core : float or numpy.ndarray
            Pair core radii in m.

        Returns
        -------
        numpy.ndarray
            Contribution to ``dGamma_target/dt`` with shape ``(..., 3)`` in
            m³/s². Coincident points return zero.

        Notes
        -----
        Pair contributions are antisymmetric under target/source exchange in
        the discrete structure used by the solver, which is why the
        transposed formulation preserves total particle strength to round-off
        in the qualification tests.
        """
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

    def far_field_error(self, rho: np.ndarray | float) -> np.ndarray:
        """Estimate absolute velocity-factor error in the singular far field.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Dimensionless radius ``r / sigma``.

        Returns
        -------
        numpy.ndarray
            ``abs(q_infinity - q(rho))``; dimensionless and shape-preserving.
        """
        return np.abs(self.q_infinity - self.q(rho))

    def gradient_far_field_error(self, rho: np.ndarray | float) -> np.ndarray:
        """Estimate absolute gradient-coefficient error in the far field.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Dimensionless radius ``r / sigma``.

        Returns
        -------
        numpy.ndarray
            Error in ``3 q(rho) - rho³ zeta(rho)`` relative to its singular
            coefficient ``3*q_infinity``.
        """
        rho = np.asarray(rho, dtype=np.float64)
        regularized = 3.0 * self.q(rho) - self.zeta(rho) * rho**3
        return np.abs(3.0 * self.q_infinity - regularized)

    def dimensionless_tail_cutoffs(
        self,
        velocity_relative_tolerance: float,
        gradient_relative_tolerance: float,
    ) -> tuple[float, float]:
        """Return dimensionless velocity and gradient tail cutoffs.

        Parameters
        ----------
        velocity_relative_tolerance, gradient_relative_tolerance : float
            Strict tolerances in ``(0, 1)`` used to decide when regularized
            tails are close enough to the singular far field.

        Returns
        -------
        tuple[float, float]
            ``(rho_velocity, rho_gradient)`` dimensionless cutoffs. The
            values are cached per kernel and tolerance.

        Raises
        ------
        ValueError
            If either tolerance is not strictly between zero and one.
        """
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
        """Return a conservative physical near-field radius.

        Parameters
        ----------
        core_radius : float
            Source/core radius in metres.
        tolerance : float
            Relative velocity-factor tolerance in ``(0, 1)``.

        Returns
        -------
        float
            Physical cutoff radius in metres.

        Raises
        ------
        ValueError
            If ``core_radius`` is not positive or ``tolerance`` is outside
            ``(0, 1)``.
        """
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


def make_vortex_kernel(name: str) -> RadialVortexKernel:
    """Construct the host conditions of a registered radial vortex kernel.

    Parameters
    ----------
    name : str
        Case-insensitive name of a kernel in ``kernels.registry``.

    Returns
    -------
    RadialVortexKernel
        Immutable host-side kernel conditions with the NumPy functions of the
        kernel's definition. Equal names give equal objects.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    definition = kernel_definition(name)
    return RadialVortexKernel(
        definition.name,
        definition.q,
        definition.zeta,
    )


def make_device_vortex_kernels(name: str, dtype: object) -> dict[str, Callable]:
    """Build the Taichi radial functions of a registered kernel."""
    return kernel_definition(name).device_factory(dtype)


__all__ = ["RadialVortexKernel", "make_device_vortex_kernels", "make_vortex_kernel"]
