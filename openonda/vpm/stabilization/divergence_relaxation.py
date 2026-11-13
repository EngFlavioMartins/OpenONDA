"""Constrained divergence relaxation for Gaussian vortex particles.

Winckelmans' relaxation reassigns particle vortex_strength so the reconstructed blob
vorticity equals the curl of its induced velocity.  In a particle-mesh
discretization, that curl is the Helmholtz projection of the reconstructed
vorticity.  This module solves the corresponding correction equation

    K delta_vortex_strength = G[P(B vortex_strength) - B vortex_strength],

where ``B`` reconstructs the Gaussian blob field on the grid, ``G`` is its
exact-transpose gather, and ``K = G B`` is the symmetric particle-space
operator.

The inverse is regularized: strongly overlapping Gaussian blobs
make the unregularized interpolation matrix ill-conditioned, and fitting its
near-null modes creates large, non-physical strength oscillations.  The
correction keeps the vector vortex_strength, linear impulse, and
Gaussian-corrected angular impulse of the field, or restores given target
moments with a minimum-norm particular correction.  The divergence solve is
restricted to the exact null space of those nine moments, and one null-space
direction restores the kinetic energy exactly in the Fourier audit.  Enstrophy,
helicity, total variation, correction size, and residual reduction are then
hard acceptance checks.  If one admissible sweep does not reach the global
residual target, later sweeps are searched by amplitude and audited together
as one atomic original-to-final state update.

Between free-slip planes (``slip_slab``) the grid spans one image period in
z, the blob field includes every particle image, and the restored moments are
those of the spanwise strength, the invariants of slab flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

from openonda.core.slip_slab_images import AXIAL_PARITY, slab_period_images, slab_period_lattice

from ..config.divergence_relaxation import DivergenceRelaxationConfig
from ..numerics.fourier_integrals import gaussian_fourier_integrals
from ..numerics.m4_kernel import m4_prime
from .filament_refinement import gaussian_particle_moments, particle_moments


class DivergenceRelaxationError(RuntimeError):
    """A divergence-relaxation proposal failed a declared physics check."""

    def __init__(self, message: str, *, check: str | None = None) -> None:
        """Create a rejected divergence-relaxation proposal error.

        Parameters
        ----------
        message : str
            Human-readable explanation passed to :class:`RuntimeError`.
        check : str or None, optional
            Stable name of the violated acceptance check, such as
            ``"heterogeneous core width"`` or ``"moment restoration"``.

        Notes
        -----
        The exception reports why a proposal was rejected; it does not mutate
        the particle-strength field or perform rollback itself.
        """
        super().__init__(message)
        self.check = check


@dataclass(frozen=True)
class DivergenceRelaxationResult:
    """Accepted reassignment and the diagnostics the stabilization log reports."""

    vortex_strength: np.ndarray
    projection_sweeps: int
    final_residual_ratio: float
    correction_norm_relative: float


# Halvings of the moment restoration, the correction amplitude, and the
# sweep amplitude that a proposal may try before it is rejected.
_MAX_LINE_SEARCH_STEPS = 8


def _wave_numbers(
    shape: tuple[int, int, int],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = [
        2.0 * np.pi * fft.fftfreq(shape[0], d=spacing),
        2.0 * np.pi * fft.fftfreq(shape[1], d=spacing),
        2.0 * np.pi * fft.rfftfreq(shape[2], d=spacing),
    ]
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            frequencies[axis][size // 2] = 0.0
    return np.broadcast_arrays(
        frequencies[0][:, None, None],
        frequencies[1][None, :, None],
        frequencies[2][None, None, :],
    )


def projection_padding(
    shape: tuple[int, ...], *, periodic_z: bool = False
) -> list[tuple[int, int]]:
    """Padding that doubles each free-space axis of a periodic projection grid.

    Outside a compact field the gradient part of its projection decays like a
    dipole field, r^-3, so the periodic images of the doubled box perturb it
    eight times less than those of the bare box. A slab z axis is one image
    period and is not padded.
    """
    padding = [(size // 2, size - size // 2) for size in shape[:3]]
    if periodic_z:
        padding[2] = (0, 0)
    return padding


def helmholtz_project(field: np.ndarray, spacing: float) -> tuple[np.ndarray, float, float]:
    """Periodic Helmholtz projection of a Cartesian vector field onto its solenoidal part.

    Returns the projected field and the relative divergence before and after.
    Gaussian convolution commutes with this projection. Pad free-space axes by
    :func:`projection_padding` first.
    """
    transformed = [fft.rfftn(field[..., component], workers=-1) for component in range(3)]
    wave_numbers = _wave_numbers(field.shape[:3], spacing)
    norm_sq = sum(component * component for component in wave_numbers)
    dot = sum(wave_numbers[axis] * transformed[axis] for axis in range(3))
    nonzero = norm_sq > 0.0
    divergence_before = float(
        np.sqrt(
            np.sum(np.abs(dot) ** 2, dtype=np.float64)
            / max(
                sum(
                    np.sum(
                        np.abs(wave_numbers[direction] * transformed[component]) ** 2,
                        dtype=np.float64,
                    )
                    for direction in range(3)
                    for component in range(3)
                ),
                np.finfo(float).tiny,
            )
        )
    )
    for axis in range(3):
        transformed[axis][nonzero] -= wave_numbers[axis][nonzero] * dot[nonzero] / norm_sq[nonzero]
    dot_after = sum(wave_numbers[axis] * transformed[axis] for axis in range(3))
    divergence_after = float(
        np.sqrt(
            np.sum(np.abs(dot_after) ** 2, dtype=np.float64)
            / max(
                sum(
                    np.sum(
                        np.abs(wave_numbers[direction] * transformed[component]) ** 2,
                        dtype=np.float64,
                    )
                    for direction in range(3)
                    for component in range(3)
                ),
                np.finfo(float).tiny,
            )
        )
    )
    projected = np.stack(
        [
            fft.irfftn(transformed[component], s=field.shape[:3], workers=-1).real
            for component in range(3)
        ],
        axis=-1,
    )
    return projected, divergence_before, divergence_after


class GaussianParticleGridOperator:
    """Symmetric M4 scatter/Gaussian-convolution/M4 gather operator."""

    def __init__(
        self,
        position: np.ndarray,
        core_radius: np.ndarray,
        vortex_strength_weight: np.ndarray,
        *,
        spacing: float,
        slip_slab: tuple[float, float] | None = None,
    ) -> None:
        """Prepare a symmetric M4/Gaussian grid operator.

        Parameters
        ----------
        position : ndarray, shape (N, 3)
            Particle centres in metres. The values are copied into a float64
            host representation used to determine the grid bounds.
        core_radius : ndarray, shape (N,)
            Positive particle core radii in metres.
        vortex_strength_weight : ndarray, shape (N,)
            Non-negative scalar weights used to choose the reconstruction
            width; in the stabilization path these are typically
            ``|Gamma|`` with units m³/s.
        spacing : float
            Uniform Cartesian grid spacing in metres.
        slip_slab : tuple of float, optional
            Free-slip planes ``(z_min, z_max)``. The z axis then spans one
            image period and is periodic; every scatter adds the particles'
            reflections, and gathers read the particles only.

        Raises
        ------
        ValueError
            If shapes or positivity are invalid.
        DivergenceRelaxationError
            If the relative core-radius range ``ptp(core_radius) / mean(core_radius)``
            exceeds 0.25.

        Notes
        -----
        Construction computes grid geometry and allocates spectral work
        arrays; it does not alter the supplied particle strengths.
        """
        position = np.asarray(position, dtype=np.float64)
        core_radius = np.asarray(core_radius, dtype=np.float64)
        vortex_strength_weight = np.asarray(vortex_strength_weight, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or len(position) == 0:
            raise ValueError("position must have shape (N, 3) with N > 0")
        if core_radius.shape != (len(position),):
            raise ValueError("core_radius must have shape (N,)")
        if vortex_strength_weight.shape != (len(position),):
            raise ValueError("vortex_strength_weight must have shape (N,)")
        if spacing <= 0.0 or np.any(core_radius <= 0.0):
            raise ValueError("spacing and all core_radius must be positive")
        core_radius_scale = max(float(core_radius.mean()), np.finfo(float).tiny)
        self.core_radius_spread = float(np.ptp(core_radius)) / core_radius_scale
        if self.core_radius_spread > 0.25:
            raise DivergenceRelaxationError(
                "divergence relaxation uses one Gaussian reconstruction width, but the "
                f"active core-radius spread is {self.core_radius_spread:.3g} > "
                "the admissible 0.25",
                check="heterogeneous core width",
            )

        weight_sum = float(vortex_strength_weight.sum(dtype=np.float64))
        self.smoothing_radius = (
            float(
                np.sqrt(
                    np.sum(
                        vortex_strength_weight * core_radius * core_radius,
                        dtype=np.float64,
                    )
                    / weight_sum
                )
            )
            if weight_sum > 0.0
            else float(np.sqrt(np.mean(core_radius * core_radius)))
        )
        self.spacing = float(spacing)
        # Gaussian support radius divided by the reconstruction width.
        self.support_radius_multiplier = 4.0
        padding = int(np.ceil(4.0 * self.smoothing_radius / spacing)) + 3
        lower = np.floor(position.min(axis=0) / spacing).astype(np.int64) - padding
        upper = np.ceil(position.max(axis=0) / spacing).astype(np.int64) + padding
        shape = [int(value) for value in upper - lower + 1]
        origin = lower.astype(np.float64) * spacing
        self.count = len(position)
        self.slip_slab = None
        if slip_slab is not None:
            self.slip_slab = (float(slip_slab[0]), float(slip_slab[1]))
            origin[2], shape[2] = slab_period_lattice(spacing, self.slip_slab)
            position = slab_period_images(position, np.zeros_like(position), self.slip_slab)[0]
        self.shape = tuple(shape)
        self.nodes = prod(self.shape)
        coordinates = (position - origin) / spacing
        base = np.floor(coordinates).astype(np.int64)
        fractions = coordinates - base
        offsets = np.arange(-1, 3, dtype=np.int64)
        axis_weights = tuple(
            np.stack(
                [m4_prime(fractions[:, axis] - offset) for offset in offsets],
                axis=1,
            )
            for axis in range(3)
        )

        indices: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        _, ny, nz = self.shape
        for oi, di in enumerate(offsets):
            ix = base[:, 0] + di
            for oj, dj in enumerate(offsets):
                iy = base[:, 1] + dj
                weight_xy = axis_weights[0][:, oi] * axis_weights[1][:, oj]
                for ok, dk in enumerate(offsets):
                    iz = base[:, 2] + dk
                    if self.slip_slab is not None:
                        iz = np.mod(iz, nz)
                    indices.append((ix * ny + iy) * nz + iz)
                    weights.append(weight_xy * axis_weights[2][:, ok])
        self.indices = np.ascontiguousarray(np.stack(indices, axis=1), dtype=np.int32)
        self.weights = np.ascontiguousarray(np.stack(weights, axis=1), dtype=np.float64)
        partition_error = float(np.max(np.abs(self.weights.sum(axis=1) - 1.0)))
        if partition_error > 2e-13:
            raise DivergenceRelaxationError(
                f"M4 interpolation lost partition of unity by {partition_error:.3e}"
            )
        self._flat_indices = self.indices.ravel()
        self._filter_sigma = self.smoothing_radius / (np.sqrt(2.0) * spacing)

    def scatter(self, values: np.ndarray) -> np.ndarray:
        """M4-adjoint scatter of particle vectors to grid nodes."""

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.count, 3):
            raise ValueError("values must have shape (N, 3)")
        if self.slip_slab is not None:
            values = np.concatenate((values, values * AXIAL_PARITY))
        return np.stack(
            [
                np.bincount(
                    self._flat_indices,
                    weights=(self.weights * values[:, component, None]).ravel(),
                    minlength=self.nodes,
                ).reshape(self.shape)
                for component in range(3)
            ],
            axis=-1,
        )

    def gather(self, field: np.ndarray) -> np.ndarray:
        """M4 gather using the exact transpose weights of :meth:`scatter`."""

        field = np.asarray(field, dtype=np.float64)
        if field.shape != (*self.shape, 3):
            raise ValueError(f"field must have shape {(*self.shape, 3)}")
        indices, weights = self.indices[: self.count], self.weights[: self.count]
        return np.stack(
            [
                np.sum(field[..., component].ravel()[indices] * weights, axis=1)
                for component in range(3)
            ],
            axis=1,
        )

    def smooth(self, field: np.ndarray) -> np.ndarray:
        """Apply the normalized Gaussian blob kernel on the padded grid."""

        mode = ["constant", "constant", "constant" if self.slip_slab is None else "grid-wrap"]
        return np.stack(
            [
                gaussian_filter(
                    field[..., component],
                    self._filter_sigma,
                    mode=mode,
                    truncate=self.support_radius_multiplier,
                )
                / self.spacing**3
                for component in range(3)
            ],
            axis=-1,
        )

    def apply(self, vortex_strength: np.ndarray) -> np.ndarray:
        """Apply the symmetric particle-space interpolation operator."""

        return self.gather(self.smooth(self.scatter(vortex_strength)))

    def relaxation_residual(
        self,
        vortex_strength: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return ``P(K vortex_strength) - K vortex_strength`` at particle locations."""

        blob_grid = self.smooth(self.scatter(vortex_strength))
        padding = projection_padding(self.shape, periodic_z=self.slip_slab is not None)
        target_grid, before, after = helmholtz_project(
            np.pad(blob_grid, (*padding, (0, 0))), self.spacing
        )
        crop = tuple(
            slice(low, low + size) for (low, _), size in zip(padding, self.shape, strict=True)
        )
        return self.gather(target_grid[crop]) - self.gather(blob_grid), before, after


def invariant_rows(
    position: np.ndarray,
    core_radius: np.ndarray,
    *,
    angular_core_coefficient: float,
    slip_slab: bool = False,
) -> np.ndarray:
    """Rows for vortex_strength and both impulses for a supported blob kernel.

    With ``slip_slab`` the rows are those of the slab moments of
    :func:`particle_moments`: only spanwise strength enters, and the rows of
    the non-invariant components are zero.
    """

    position = np.asarray(position, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    n = len(position)
    rows = np.zeros((9, n, 3), dtype=np.float64)
    identity = np.eye(3)
    rows[0:3] = identity[:, None, :]

    skew = np.zeros((n, 3, 3), dtype=np.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -position[:, 2], position[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = position[:, 2], -position[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -position[:, 1], position[:, 0]
    rows[3:6] = 0.5 * np.transpose(skew, (1, 0, 2))

    skew_sq = (
        np.einsum("pa,pb->pab", position, position)
        - np.einsum("pa,pa->p", position, position)[:, None, None] * identity
    )
    angular = skew_sq / 3.0 - (
        angular_core_coefficient * core_radius[:, None, None] ** 2 * identity
    )
    rows[6:9] = np.transpose(angular, (1, 0, 2))
    if slip_slab:
        rows[..., :2] = 0.0
        rows[6:8] = 0.0
    return rows


def gaussian_invariant_rows(
    position: np.ndarray,
    core_radius: np.ndarray,
    *,
    slip_slab: bool = False,
) -> np.ndarray:
    """Rows for vector strength, linear impulse, and Gaussian angular impulse."""
    return invariant_rows(
        position,
        core_radius,
        angular_core_coefficient=1.0 / 3.0,
        slip_slab=slip_slab,
    )


class _MomentNullspace:
    def __init__(self, rows: np.ndarray, particle_volume: np.ndarray) -> None:
        self.sqrt_volume = np.sqrt(np.asarray(particle_volume, dtype=np.float64))
        self.rows = np.asarray(rows, dtype=np.float64)
        self.scaled_rows = self.rows * self.sqrt_volume[None, :, None]
        gram = np.einsum("mpa,npa->mn", self.scaled_rows, self.scaled_rows)
        self.gram_inverse = np.linalg.pinv(gram, rcond=1e-12)

    def project_transformed(self, value: np.ndarray) -> np.ndarray:
        multipliers = self.gram_inverse @ np.einsum("mpa,pa->m", self.scaled_rows, value)
        return value - np.einsum("m,mpa->pa", multipliers, self.scaled_rows)

    def correction_for_moment_change(self, change: np.ndarray) -> np.ndarray:
        """Return the minimum transformed-norm correction satisfying the constraint."""

        change = np.asarray(change, dtype=np.float64)
        if change.shape != (len(self.rows),):
            raise ValueError(f"moment change must have shape ({len(self.rows)},)")
        multipliers = self.gram_inverse @ change
        transformed = np.einsum("m,mpa->pa", multipliers, self.scaled_rows)
        return self.sqrt_volume[:, None] * transformed

    def to_correction(self, transformed: np.ndarray) -> np.ndarray:
        return self.sqrt_volume[:, None] * self.project_transformed(transformed)


def restore_particle_moments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    reference_vortex_strength: np.ndarray,
    *,
    angular_core_coefficient: float,
    slip_slab: bool = False,
) -> tuple[np.ndarray, float]:
    """Restore vector strength and both impulses with a minimum-norm correction.

    With ``slip_slab`` only the slab moments of the spanwise strength are restored.
    """

    target = particle_moments(
        position,
        reference_vortex_strength,
        core_radius,
        angular_core_coefficient=angular_core_coefficient,
        slip_slab=slip_slab,
    )
    current = particle_moments(
        position,
        vortex_strength,
        core_radius,
        angular_core_coefficient=angular_core_coefficient,
        slip_slab=slip_slab,
    )
    moment_change = np.concatenate(
        (target[0] - current[0], target[2] - current[2], target[3] - current[3])
    )
    correction = _MomentNullspace(
        invariant_rows(
            position,
            core_radius,
            angular_core_coefficient=angular_core_coefficient,
            slip_slab=slip_slab,
        ),
        particle_volume,
    ).correction_for_moment_change(moment_change)
    scale = max(np.linalg.norm(vortex_strength), np.finfo(float).tiny)
    return vortex_strength + correction, float(np.linalg.norm(correction) / scale)


@dataclass(frozen=True)
class _Rejection:
    """A proposal that failed an amplitude-dependent acceptance check."""

    check: str
    message: str


class _SweepState:
    """Quantities of one working strength that no proposal amplitude changes."""

    def __init__(
        self,
        position: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        *,
        grid_spacing: float,
        slip_slab: tuple[float, float] | None,
    ) -> None:
        self.position = position
        self.vortex_strength = vortex_strength
        self.core_radius = core_radius
        self.particle_volume = particle_volume
        self.grid_spacing = grid_spacing
        self.slip_slab = slip_slab
        self.moments = gaussian_particle_moments(
            position, vortex_strength, core_radius, slip_slab=slip_slab is not None
        )
        self.operator = GaussianParticleGridOperator(
            position,
            core_radius,
            np.linalg.norm(vortex_strength, axis=1),
            spacing=grid_spacing,
            slip_slab=slip_slab,
        )
        residual, _, _ = self.operator.relaxation_residual(vortex_strength)
        self.residual_norm = float(np.linalg.norm(residual))
        self.blob_norm = float(np.linalg.norm(self.operator.apply(vortex_strength)))
        self.rows = gaussian_invariant_rows(position, core_radius, slip_slab=slip_slab is not None)
        self.nullspace = _MomentNullspace(self.rows, particle_volume)
        self.norm = max(float(np.linalg.norm(vortex_strength)), np.finfo(float).tiny)
        self._integrals = None
        self._energy_direction = None
        self._energy_direction_computed = False
        self._divergence_corrections: dict[int, np.ndarray] = {}

    def integrals(self, vortex_strength: np.ndarray | None = None):
        """Fourier-audited quadratic integrals of this or another strength field."""
        if vortex_strength is None:
            if self._integrals is None:
                self._integrals = self.integrals(self.vortex_strength)
            return self._integrals
        return gaussian_fourier_integrals(
            self.position,
            vortex_strength,
            self.core_radius,
            self.particle_volume,
            spacing=self.grid_spacing,
            slip_slab=self.slip_slab,
        )

    def energy_direction(self) -> np.ndarray | None:
        """Moment-preserving part of the strength itself, scaled to its norm.

        Returns ``None`` when that part is round-off: the strength then lies in
        the span of its own moment rows (a uniform discrete ring lies in the
        span of the impulse rows), so no moment-preserving rescaling exists.
        """
        if not self._energy_direction_computed:
            self._energy_direction_computed = True
            sqrt_volume = self.nullspace.sqrt_volume[:, None]
            direction = self.nullspace.to_correction(self.vortex_strength / sqrt_volume)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= np.sqrt(np.finfo(float).eps) * self.norm:
                return None
            # Scaling up amplifies the round-off left in the moment rows by
            # norm / direction_norm; a second projection removes it again.
            direction *= self.norm / direction_norm
            self._energy_direction = self.nullspace.to_correction(direction / sqrt_volume)
        return self._energy_direction

    def divergence_correction(
        self,
        restoration_index: int,
        repaired: np.ndarray,
        config: DivergenceRelaxationConfig,
    ) -> np.ndarray:
        """Regularized moment-preserving solution of ``K delta = residual(repaired)``."""
        if restoration_index in self._divergence_corrections:
            return self._divergence_corrections[restoration_index]
        nullspace = self.nullspace
        sqrt_volume = nullspace.sqrt_volume
        repaired_residual, _, _ = self.operator.relaxation_residual(repaired)
        right_hand_side = nullspace.project_transformed(sqrt_volume[:, None] * repaired_residual)

        def matrix_vector(flat_value: np.ndarray) -> np.ndarray:
            transformed = nullspace.project_transformed(flat_value.reshape(-1, 3))
            physical = sqrt_volume[:, None] * transformed
            applied = sqrt_volume[:, None] * self.operator.apply(physical)
            return (
                nullspace.project_transformed(applied) + config.regularization * transformed
            ).ravel()

        dimension = 3 * len(self.position)
        solution, info = cg(
            LinearOperator((dimension, dimension), matvec=matrix_vector, dtype=np.float64),
            right_hand_side.ravel(),
            rtol=config.solver_relative_tolerance,
            atol=0.0,
            maxiter=config.max_iterations,
        )
        if info != 0:
            raise DivergenceRelaxationError(
                f"regularized divergence relaxation did not converge in {config.max_iterations} "
                f"iterations (CG info={info})"
            )
        correction = nullspace.to_correction(solution.reshape(-1, 3))
        self._divergence_corrections[restoration_index] = correction
        return correction


def _target_moment_change(
    state: _SweepState,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if target_moments is None:
        return np.zeros(9)
    target_total, target_impulse, target_angular = target_moments
    return np.concatenate(
        (
            target_total - state.moments[0],
            target_impulse - state.moments[2],
            target_angular - state.moments[3],
        )
    )


def _audit(
    before: _SweepState,
    relaxed: np.ndarray,
    config: DivergenceRelaxationConfig,
    *,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    restoration_scale: float,
    residual_limit: float,
) -> tuple[float, float] | _Rejection:
    """Check a proposed original-to-relaxed update against every physics limit.

    Returns the residual ratio and the relative correction norm, or the first
    violated amplitude-dependent check. A moment change beyond roundoff is a
    defect, not a rejection, and raises.
    """
    position = before.position
    vortex_strength = before.vortex_strength
    core_radius = before.core_radius
    slab = before.slip_slab is not None
    before_moments = before.moments
    after_moments = gaussian_particle_moments(position, relaxed, core_radius, slip_slab=slab)
    achieved = np.concatenate(
        (before_moments[0], before_moments[2], before_moments[3])
    ) + restoration_scale * _target_moment_change(before, target_moments)
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
    moment_tolerance = 4096.0 * np.finfo(float).eps
    for name, after, target, scale in (
        ("vector strength", after_moments[0], achieved[0:3], before_moments[1]),
        ("linear impulse", after_moments[2], achieved[3:6], impulse_scale),
        ("angular impulse", after_moments[3], achieved[6:9], angular_scale),
    ):
        error = float(np.linalg.norm(after - target))
        allowance = moment_tolerance * max(scale, np.finfo(float).tiny)
        if error > allowance:
            raise DivergenceRelaxationError(
                f"divergence relaxation changed {name} by {error:.3e}, beyond "
                f"its roundoff allowance {allowance:.3e}"
            )

    tiny = np.finfo(float).tiny
    before_integrals = before.integrals()
    after_integrals = before.integrals(relaxed)

    def relative_changes(energy: str, enstrophy: str, helicity: str) -> tuple[float, ...]:
        energy_before = getattr(before_integrals, energy)
        enstrophy_before = getattr(before_integrals, enstrophy)
        helicity_scale = np.sqrt(max(2.0 * abs(energy_before) * abs(enstrophy_before), tiny))
        return (
            (getattr(after_integrals, energy) - energy_before) / max(abs(energy_before), tiny),
            (getattr(after_integrals, enstrophy) - enstrophy_before)
            / max(abs(enstrophy_before), tiny),
            (getattr(after_integrals, helicity) - getattr(before_integrals, helicity))
            / helicity_scale,
        )

    changes = relative_changes("total_kinetic_energy", "total_enstrophy", "total_helicity")
    previous_order_changes = relative_changes(
        "previous_order_total_kinetic_energy",
        "previous_order_total_enstrophy",
        "previous_order_total_helicity",
    )
    spectral_errors = [
        abs(change - previous)
        for change, previous in zip(changes, previous_order_changes, strict=True)
    ]
    variation_change = (after_moments[1] - before_moments[1]) / max(before_moments[1], tiny)
    correction_norm_relative = float(np.linalg.norm(relaxed - vortex_strength) / before.norm)
    final_residual, _, _ = before.operator.relaxation_residual(relaxed)
    final_residual_ratio = (
        float(np.linalg.norm(final_residual) / before.residual_norm)
        if before.residual_norm > tiny
        else 0.0
    )
    fraction = config.spectral_convergence_fraction
    for name, value, limit in (
        ("correction norm", correction_norm_relative, config.max_correction_norm),
        ("residual ratio", final_residual_ratio, residual_limit),
        ("kinetic-energy transfer", abs(changes[0]), config.total_kinetic_energy_tolerance),
        ("enstrophy transfer", abs(changes[1]), config.total_enstrophy_tolerance),
        ("helicity transfer", abs(changes[2]), config.total_helicity_tolerance),
        ("total-variation transfer", abs(variation_change), config.variation_tolerance),
        (
            "kinetic-energy spectral convergence",
            spectral_errors[0],
            fraction * config.total_kinetic_energy_tolerance,
        ),
        (
            "enstrophy spectral convergence",
            spectral_errors[1],
            fraction * config.total_enstrophy_tolerance,
        ),
        (
            "helicity spectral convergence",
            spectral_errors[2],
            fraction * config.total_helicity_tolerance,
        ),
    ):
        if not np.isfinite(value) or value > limit:
            return _Rejection(
                name, f"divergence-relaxation {name} {value:.3e} exceeds the admissible {limit:.3e}"
            )
    return final_residual_ratio, correction_norm_relative


def _propose(
    state: _SweepState,
    config: DivergenceRelaxationConfig,
    *,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    restoration_index: int,
    correction_scale: float,
    residual_limit: float,
) -> DivergenceRelaxationResult | _Rejection:
    """One Winckelmans relaxation of ``state`` at the given amplitudes."""

    vortex_strength = state.vortex_strength
    restoration_scale = 0.5**restoration_index
    moment_change = restoration_scale * _target_moment_change(state, target_moments)
    moment_correction = state.nullspace.correction_for_moment_change(moment_change)
    restored_moment_change = np.einsum("mpa,pa->m", state.rows, moment_correction)
    targets = (
        target_moments
        if target_moments is not None
        else (state.moments[0], state.moments[2], state.moments[3])
    )
    moment_change_scale = max(
        float(np.linalg.norm(moment_change)),
        *(float(np.linalg.norm(value)) for value in targets),
        np.finfo(float).tiny,
    )
    if not np.allclose(
        restored_moment_change,
        moment_change,
        rtol=0.0,
        atol=4096.0 * np.finfo(float).eps * moment_change_scale,
    ):
        raise DivergenceRelaxationError(
            "reference moments are rank-deficient for the current particle cloud"
        )
    moment_correction_norm = float(np.linalg.norm(moment_correction))
    available_norm = 0.9 * config.max_correction_norm * state.norm - moment_correction_norm
    if available_norm < 0.0:
        return _Rejection(
            "moment restoration",
            "reference-moment restoration alone requires a correction norm of "
            f"{moment_correction_norm / state.norm:.3e}, beyond the admissible "
            f"{0.9 * config.max_correction_norm:.3e}",
        )

    roundoff = 128.0 * np.finfo(float).eps
    if (
        state.residual_norm <= roundoff * state.blob_norm
        and moment_correction_norm <= roundoff * state.norm
    ):
        return DivergenceRelaxationResult(
            vortex_strength=vortex_strength.copy(),
            projection_sweeps=1,
            final_residual_ratio=0.0,
            correction_norm_relative=0.0,
        )

    divergence_correction = state.divergence_correction(
        restoration_index, vortex_strength + moment_correction, config
    )
    trust_region_scale = (
        min(
            1.0,
            available_norm
            / max(float(np.linalg.norm(divergence_correction)), np.finfo(float).tiny),
        )
        * correction_scale
    )
    raw_correction = moment_correction + trust_region_scale * divergence_correction
    candidate = vortex_strength + raw_correction

    # Kinetic energy is a quadratic form of the strength, so along the
    # moment-preserving direction d, E(c + m d) = E(c) + b m + a m^2 exactly; the
    # central differences give b and a, and the smaller root restores E.
    # Without such a direction the audit judges the energy change of the
    # candidate itself against its tolerance.
    energy_direction = state.energy_direction()
    relaxed = candidate
    if energy_direction is not None:
        energy = state.integrals(candidate).total_kinetic_energy
        plus = state.integrals(candidate + energy_direction).total_kinetic_energy
        minus = state.integrals(candidate - energy_direction).total_kinetic_energy
        energy_roots = np.roots(
            (
                0.5 * (plus + minus) - energy,
                0.5 * (plus - minus),
                energy - state.integrals().total_kinetic_energy,
            )
        )
        real_energy_roots = [
            float(value.real)
            for value in energy_roots
            if np.isfinite(value) and abs(value.imag) <= 1e-10
        ]
        if not real_energy_roots:
            return _Rejection(
                "kinetic-energy restoration",
                "kinetic energy could not be restored in the moment null space",
            )
        energy_multiplier = min(real_energy_roots, key=abs)
        relaxed = vortex_strength + (raw_correction + energy_multiplier * energy_direction)

    audit = _audit(
        state,
        relaxed,
        config,
        target_moments=target_moments,
        restoration_scale=restoration_scale,
        residual_limit=residual_limit,
    )
    if isinstance(audit, _Rejection):
        return audit
    return DivergenceRelaxationResult(
        vortex_strength=relaxed,
        projection_sweeps=1,
        final_residual_ratio=audit[0],
        correction_norm_relative=audit[1],
    )


def _sweep(
    state: _SweepState,
    config: DivergenceRelaxationConfig,
    *,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    residual_limit: float,
    initial_correction_scale: float,
) -> tuple[DivergenceRelaxationResult, float] | _Rejection:
    """One projection sweep, halving its moment restoration and then its amplitude.

    Returns the accepted proposal with its moment-restoration fraction, or the
    last rejection once every amplitude failed.
    """
    last: _Rejection | None = None
    for restoration_index in range(_MAX_LINE_SEARCH_STEPS):
        for divergence_index in range(_MAX_LINE_SEARCH_STEPS):
            outcome = _propose(
                state,
                config,
                target_moments=target_moments,
                restoration_index=restoration_index,
                correction_scale=initial_correction_scale * 0.5**divergence_index,
                residual_limit=residual_limit,
            )
            if not isinstance(outcome, _Rejection):
                return outcome, 0.5**restoration_index
            last = outcome
            # Only a smaller moment restoration can help these two.
            if outcome.check in ("residual ratio", "moment restoration"):
                break
    assert last is not None
    return _Rejection(
        last.check,
        "divergence-relaxation line search exhausted its physics-admissible "
        f"amplitudes; last rejection: {last.message}",
    )


def constrained_divergence_relaxation(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    config: DivergenceRelaxationConfig,
    *,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    slip_slab: tuple[float, float] | None = None,
) -> DivergenceRelaxationResult:
    """Return an iterated Helmholtz projection with physical error checks.

    The relaxation keeps the vector strength and both impulses of the given
    field, or restores ``target_moments`` (vector strength, linear impulse,
    angular impulse) when they are given. ``config`` supplies the projection
    spacing and every limit; its schedule fields are not read.
    """

    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    particle_volume = np.asarray(particle_volume, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex_strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),) or particle_volume.shape != (len(position),):
        raise ValueError("core_radius and particle_volume must have shape (N,)")
    if np.any(particle_volume <= 0.0):
        raise ValueError("all particle volume must be positive")
    if config.grid_spacing is None or config.grid_spacing <= 0.0:
        raise ValueError("divergence relaxation requires a positive grid_spacing")
    if target_moments is not None:
        if len(target_moments) != 3:
            raise ValueError(
                "target_moments must contain vortex strength, linear impulse, and angular impulse"
            )
        target_moments = tuple(np.asarray(value, dtype=np.float64) for value in target_moments)
        if any(value.shape != (3,) for value in target_moments):
            raise ValueError("each target moment must have shape (3,)")
        if not all(np.isfinite(value).all() for value in target_moments):
            raise ValueError("target moments must be finite")

    def sweep_state(strength: np.ndarray) -> _SweepState:
        return _SweepState(
            position,
            strength,
            core_radius,
            particle_volume,
            grid_spacing=config.grid_spacing,
            slip_slab=slip_slab,
        )

    max_sweeps = config.max_projection_sweeps
    sweep_residual_limit = config.max_residual_ratio ** (1.0 / max_sweeps)
    monotone_residual_limit = np.nextafter(1.0, 0.0)
    original = sweep_state(vortex_strength)
    working = original
    unrestored_fraction = 1.0
    last: _Rejection | None = None
    for sweep_index in range(max_sweeps):
        full_sweep: DivergenceRelaxationResult | None = None
        full_restoration = 1.0
        for amplitude_index in range(_MAX_LINE_SEARCH_STEPS):
            outcome = _sweep(
                working,
                config,
                target_moments=target_moments,
                residual_limit=(
                    sweep_residual_limit if amplitude_index == 0 else monotone_residual_limit
                ),
                initial_correction_scale=0.5**amplitude_index,
            )
            if isinstance(outcome, _Rejection):
                last = outcome
                if outcome.check == "residual ratio":
                    break
                raise DivergenceRelaxationError(outcome.message, check=outcome.check)
            candidate, restoration_scale = outcome
            if full_sweep is None:
                full_sweep, full_restoration = candidate, restoration_scale
            if sweep_index == 0:
                if candidate.final_residual_ratio <= config.max_residual_ratio:
                    return candidate
                break
            # Several sweeps are accepted only as one original-to-final update.
            audit = _audit(
                original,
                candidate.vortex_strength,
                config,
                target_moments=target_moments,
                restoration_scale=1.0 - unrestored_fraction * (1.0 - restoration_scale),
                residual_limit=config.max_residual_ratio,
            )
            if not isinstance(audit, _Rejection):
                return DivergenceRelaxationResult(
                    vortex_strength=candidate.vortex_strength,
                    projection_sweeps=sweep_index + 1,
                    final_residual_ratio=audit[0],
                    correction_norm_relative=audit[1],
                )
            last = audit
            if audit.check == "residual ratio":
                break
        if full_sweep is None or sweep_index == max_sweeps - 1:
            assert last is not None
            raise DivergenceRelaxationError(last.message, check=last.check)
        unrestored_fraction *= 1.0 - full_restoration
        working = sweep_state(full_sweep.vortex_strength)
    raise AssertionError("projection sweep loop terminated without a result")
