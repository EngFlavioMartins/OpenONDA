"""Constrained divergence relaxation for Gaussian vortex particles.

Winckelmans' relaxation reassigns particle strengths so the reconstructed blob
vorticity equals the curl of its induced velocity.  In a particle-mesh
discretization, that curl is the Helmholtz projection of the reconstructed
vorticity.  This module solves the corresponding correction equation

    K delta_Gamma = G[P(B Gamma) - B Gamma],

where ``B`` reconstructs the Gaussian blob field on the grid, ``G`` is its
exact-transpose gather, and ``K = G B`` is the symmetric particle-space
operator.

The inverse is deliberately regularized: strongly overlapping Gaussian blobs
make the unregularized interpolation matrix ill-conditioned, and fitting its
near-null modes creates large, non-physical strength oscillations.  A
minimum-norm particular correction first restores the reference vector
circulation, linear impulse, and Gaussian-corrected angular impulse.  The
divergence solve is restricted to the exact null space of those nine moments,
and two independent null-space directions restore the quadratic kinetic energy
and enstrophy exactly in the Fourier audit.  Helicity, total variation,
correction size, and residual reduction are then hard acceptance gates.  If
one admissible sweep does not reach the global residual target, later sweeps
are searched by amplitude and audited together as one atomic original-to-final
transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, cg

from ..diagnostics.fourier_integrals import gaussian_fourier_integrals
from .filament_refinement import gaussian_particle_moments


class DivergenceRelaxationError(RuntimeError):
    """A divergence-relaxation proposal failed a declared physics gate."""

    def __init__(self, message: str, *, gate: str | None = None) -> None:
        super().__init__(message)
        self.gate = gate


@dataclass(frozen=True)
class DivergenceRelaxationResult:
    """Accepted reassignment and its independently measurable diagnostics."""

    circulation: np.ndarray
    correction: np.ndarray
    projection_sweeps: int
    iterations: int
    regularization: float
    trust_region_scale: float
    initial_residual_norm: float
    final_residual_ratio: float
    correction_norm_relative: float
    quadratic_restoration_fraction: float
    reference_restoration_scale: float
    circulation_restored: float
    linear_impulse_restored: float
    angular_impulse_restored: float
    circulation_reference_error: float
    linear_impulse_reference_error: float
    angular_impulse_reference_error: float
    circulation_error: float
    linear_impulse_error: float
    angular_impulse_error: float
    total_variation_change_relative: float
    energy_change_relative: float
    enstrophy_change_relative: float
    helicity_change_relative: float
    energy_spectral_error: float
    enstrophy_spectral_error: float
    helicity_spectral_error: float
    grid_divergence_before: float
    grid_divergence_after: float


def _m4_prime(distance: np.ndarray) -> np.ndarray:
    distance = np.abs(np.asarray(distance))
    weight = np.zeros_like(distance, dtype=np.result_type(distance, np.float64))
    inner = distance <= 1.0
    outer = (distance > 1.0) & (distance <= 2.0)
    weight[inner] = 1.0 - 2.5 * distance[inner] ** 2 + 1.5 * distance[inner] ** 3
    weight[outer] = 0.5 * (2.0 - distance[outer]) ** 2 * (1.0 - distance[outer])
    return weight


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


class GaussianParticleGridOperator:
    """Symmetric M4 scatter/Gaussian-convolution/M4 gather operator."""

    def __init__(
        self,
        position: np.ndarray,
        radius: np.ndarray,
        circulation_weight: np.ndarray,
        *,
        spacing: float,
        support_radii: float = 4.0,
        max_grid_nodes: int = 8_000_000,
    ) -> None:
        position = np.asarray(position, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)
        circulation_weight = np.asarray(circulation_weight, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or len(position) == 0:
            raise ValueError("position must have shape (N, 3) with N > 0")
        if radius.shape != (len(position),):
            raise ValueError("radius must have shape (N,)")
        if circulation_weight.shape != (len(position),):
            raise ValueError("circulation_weight must have shape (N,)")
        if spacing <= 0.0 or np.any(radius <= 0.0):
            raise ValueError("spacing and all radii must be positive")
        if support_radii < 3.0:
            raise ValueError("support_radii must be at least three")
        radius_scale = max(float(radius.mean()), np.finfo(float).tiny)
        self.radius_spread = float(np.ptp(radius)) / radius_scale

        weight_sum = float(circulation_weight.sum(dtype=np.float64))
        self.smoothing_radius = (
            float(
                np.sqrt(
                    np.sum(
                        circulation_weight * radius * radius,
                        dtype=np.float64,
                    )
                    / weight_sum
                )
            )
            if weight_sum > 0.0
            else float(np.sqrt(np.mean(radius * radius)))
        )
        self.spacing = float(spacing)
        self.support_radii = float(support_radii)
        padding = int(np.ceil(support_radii * self.smoothing_radius / spacing)) + 3
        lower = np.floor(position.min(axis=0) / spacing).astype(np.int64) - padding
        upper = np.ceil(position.max(axis=0) / spacing).astype(np.int64) + padding
        self.shape = tuple(int(value) for value in upper - lower + 1)
        self.nodes = int(np.prod(self.shape))
        if self.nodes > max_grid_nodes:
            raise DivergenceRelaxationError(
                f"divergence-relaxation grid requires {self.nodes:,} nodes, exceeding "
                f"the declared limit {max_grid_nodes:,}"
            )
        origin = lower.astype(np.float64) * spacing
        coordinates = (position - origin) / spacing
        base = np.floor(coordinates).astype(np.int64)
        fractions = coordinates - base
        offsets = np.arange(-1, 3, dtype=np.int64)
        axis_weights = tuple(
            np.stack(
                [_m4_prime(fractions[:, axis] - offset) for offset in offsets],
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
        if values.shape != (len(self.indices), 3):
            raise ValueError("values must have shape (N, 3)")
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
        return np.stack(
            [
                np.sum(
                    field[..., component].ravel()[self.indices] * self.weights,
                    axis=1,
                )
                for component in range(3)
            ],
            axis=1,
        )

    def smooth(self, field: np.ndarray) -> np.ndarray:
        """Apply the normalized Gaussian blob kernel on the padded grid."""

        return np.stack(
            [
                gaussian_filter(
                    field[..., component],
                    self._filter_sigma,
                    mode="constant",
                    truncate=self.support_radii,
                )
                / self.spacing**3
                for component in range(3)
            ],
            axis=-1,
        )

    def apply(self, circulation: np.ndarray) -> np.ndarray:
        """Apply the symmetric particle-space interpolation operator."""

        return self.gather(self.smooth(self.scatter(circulation)))

    def helmholtz_project(self, field: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Project a padded grid vector field onto its solenoidal subspace."""

        transformed = [fft.rfftn(field[..., component], workers=-1) for component in range(3)]
        wave_numbers = _wave_numbers(self.shape, self.spacing)
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
            transformed[axis][nonzero] -= (
                wave_numbers[axis][nonzero] * dot[nonzero] / norm_sq[nonzero]
            )
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
                fft.irfftn(transformed[component], s=self.shape, workers=-1).real
                for component in range(3)
            ],
            axis=-1,
        )
        return projected, divergence_before, divergence_after

    def relaxation_residual(
        self,
        circulation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return ``P(K Gamma) - K Gamma`` at particle locations."""

        blob_grid = self.smooth(self.scatter(circulation))
        target_grid, before, after = self.helmholtz_project(blob_grid)
        return self.gather(target_grid) - self.gather(blob_grid), before, after


def gaussian_invariant_rows(
    position: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Rows for vector circulation, linear impulse, and Gaussian angular impulse."""

    position = np.asarray(position, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
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
    angular = (skew_sq - radius[:, None, None] ** 2 * identity) / 3.0
    rows[6:9] = np.transpose(angular, (1, 0, 2))
    return rows


class _MomentNullspace:
    def __init__(self, rows: np.ndarray, volume: np.ndarray) -> None:
        self.sqrt_volume = np.sqrt(np.asarray(volume, dtype=np.float64))
        self.rows = np.asarray(rows, dtype=np.float64)
        self.scaled_rows = self.rows * self.sqrt_volume[None, :, None]
        gram = np.einsum("mpa,npa->mn", self.scaled_rows, self.scaled_rows)
        self.gram_inverse = np.linalg.pinv(gram, rcond=1e-12)

    def project_transformed(self, value: np.ndarray) -> np.ndarray:
        multipliers = self.gram_inverse @ np.einsum("mpa,pa->m", self.scaled_rows, value)
        return value - np.einsum("m,mpa->pa", multipliers, self.scaled_rows)

    def correction_for_moment_change(self, change: np.ndarray) -> np.ndarray:
        """Return the minimum transformed-norm correction satisfying ``C dGamma=change``."""

        change = np.asarray(change, dtype=np.float64)
        if change.shape != (len(self.rows),):
            raise ValueError(f"moment change must have shape ({len(self.rows)},)")
        multipliers = self.gram_inverse @ change
        transformed = np.einsum("m,mpa->pa", multipliers, self.scaled_rows)
        return self.sqrt_volume[:, None] * transformed

    def to_correction(self, transformed: np.ndarray) -> np.ndarray:
        return self.sqrt_volume[:, None] * self.project_transformed(transformed)


def _constrained_divergence_relaxation_once(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
    volume: np.ndarray,
    *,
    grid_spacing: float,
    regularization: float = 0.1,
    solver_rtol: float = 1e-5,
    max_iterations: int = 30,
    max_grid_nodes: int = 8_000_000,
    max_correction_norm: float = 2e-2,
    max_residual_ratio: float = 0.9,
    energy_tolerance: float = 1e-6,
    enstrophy_tolerance: float = 1e-4,
    helicity_tolerance: float = 1e-4,
    variation_tolerance: float = 1e-3,
    spectral_convergence_fraction: float = 0.1,
    reference_scales: tuple[float, float, float] | None = None,
    reference_tolerances: tuple[float, float, float] = (1e-3, 1e-2, 1e-2),
    correction_scale: float = 1.0,
    restoration_scale: float = 1.0,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> DivergenceRelaxationResult:
    """Return one accepted, reference-restoring Winckelmans relaxation."""

    position = np.asarray(position, dtype=np.float64)
    circulation = np.asarray(circulation, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    if position.shape != circulation.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and circulation must both have shape (N, 3)")
    if radius.shape != (len(position),) or volume.shape != (len(position),):
        raise ValueError("radius and volume must have shape (N,)")
    if np.any(volume <= 0.0):
        raise ValueError("all particle volumes must be positive")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if solver_rtol <= 0.0 or max_iterations < 1:
        raise ValueError("solver_rtol must be positive and max_iterations at least one")
    if not 0.0 < spectral_convergence_fraction <= 1.0:
        raise ValueError("spectral_convergence_fraction must be in (0, 1]")
    if not 0.0 < restoration_scale <= 1.0:
        raise ValueError("restoration_scale must be in (0, 1]")
    if len(reference_tolerances) != 3 or any(value < 0.0 for value in reference_tolerances):
        raise ValueError("reference_tolerances must contain three non-negative values")
    if reference_scales is not None and (
        len(reference_scales) != 3 or any(value <= 0.0 for value in reference_scales)
    ):
        raise ValueError("reference_scales must contain three positive values")

    before_moments = gaussian_particle_moments(position, circulation, radius)
    if target_moments is None:
        target_total = before_moments[0]
        target_impulse = before_moments[2]
        target_angular = before_moments[3]
    else:
        if len(target_moments) != 3:
            raise ValueError(
                "target_moments must contain circulation, linear impulse, and angular impulse"
            )
        target_total, target_impulse, target_angular = (
            np.asarray(value, dtype=np.float64) for value in target_moments
        )
        if any(value.shape != (3,) for value in (target_total, target_impulse, target_angular)):
            raise ValueError("each target moment must have shape (3,)")
        if not all(
            np.isfinite(value).all() for value in (target_total, target_impulse, target_angular)
        ):
            raise ValueError("target moments must be finite")

    strength = np.linalg.norm(circulation, axis=1)
    operator = GaussianParticleGridOperator(
        position,
        radius,
        strength,
        spacing=grid_spacing,
        max_grid_nodes=max_grid_nodes,
    )
    residual, grid_divergence_before, _ = operator.relaxation_residual(circulation)
    initial_residual_norm = float(np.linalg.norm(residual))
    blob_norm = float(np.linalg.norm(operator.apply(circulation)))
    rows = gaussian_invariant_rows(position, radius)
    nullspace = _MomentNullspace(rows, volume)
    full_moment_change = np.concatenate(
        (
            target_total - before_moments[0],
            target_impulse - before_moments[2],
            target_angular - before_moments[3],
        )
    )
    moment_change = restoration_scale * full_moment_change
    achieved_total = before_moments[0] + moment_change[0:3]
    achieved_impulse = before_moments[2] + moment_change[3:6]
    achieved_angular = before_moments[3] + moment_change[6:9]
    moment_correction = nullspace.correction_for_moment_change(moment_change)
    restored_moment_change = np.einsum("mpa,pa->m", rows, moment_correction)
    moment_change_scale = max(
        float(np.linalg.norm(moment_change)),
        float(np.linalg.norm(target_total)),
        float(np.linalg.norm(target_impulse)),
        float(np.linalg.norm(target_angular)),
        1.0,
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
    circulation_norm = max(
        float(np.linalg.norm(circulation)),
        np.finfo(float).tiny,
    )
    if moment_correction_norm > 0.9 * max_correction_norm * circulation_norm:
        raise DivergenceRelaxationError(
            "reference-moment restoration alone requires a correction norm of "
            f"{moment_correction_norm / circulation_norm:.3e}, beyond the admissible "
            f"{0.9 * max_correction_norm:.3e}",
            gate="correction norm",
        )

    repaired = circulation + moment_correction
    repaired_residual, _, _ = operator.relaxation_residual(repaired)
    negligible_residual = initial_residual_norm <= 128.0 * np.finfo(float).eps * max(blob_norm, 1.0)
    negligible_moment_change = moment_correction_norm <= 128.0 * np.finfo(float).eps * max(
        circulation_norm, 1.0
    )
    if negligible_residual and negligible_moment_change:
        return DivergenceRelaxationResult(
            circulation=circulation.copy(),
            correction=np.zeros_like(circulation),
            projection_sweeps=1,
            iterations=0,
            regularization=regularization,
            trust_region_scale=1.0,
            initial_residual_norm=initial_residual_norm,
            final_residual_ratio=0.0,
            correction_norm_relative=0.0,
            quadratic_restoration_fraction=0.0,
            reference_restoration_scale=restoration_scale,
            circulation_restored=0.0,
            linear_impulse_restored=0.0,
            angular_impulse_restored=0.0,
            circulation_reference_error=0.0,
            linear_impulse_reference_error=0.0,
            angular_impulse_reference_error=0.0,
            circulation_error=0.0,
            linear_impulse_error=0.0,
            angular_impulse_error=0.0,
            total_variation_change_relative=0.0,
            energy_change_relative=0.0,
            enstrophy_change_relative=0.0,
            helicity_change_relative=0.0,
            energy_spectral_error=0.0,
            enstrophy_spectral_error=0.0,
            helicity_spectral_error=0.0,
            grid_divergence_before=grid_divergence_before,
            grid_divergence_after=grid_divergence_before,
        )

    sqrt_volume = nullspace.sqrt_volume
    right_hand_side = nullspace.project_transformed(sqrt_volume[:, None] * repaired_residual)
    iterations = 0

    def matrix_vector(flat_value: np.ndarray) -> np.ndarray:
        transformed = nullspace.project_transformed(flat_value.reshape(-1, 3))
        physical = sqrt_volume[:, None] * transformed
        applied = sqrt_volume[:, None] * operator.apply(physical)
        return (nullspace.project_transformed(applied) + regularization * transformed).ravel()

    def note_iteration(_value: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    dimension = 3 * len(position)
    linear_operator = LinearOperator(
        (dimension, dimension),
        matvec=matrix_vector,
        dtype=np.float64,
    )
    solution, info = cg(
        linear_operator,
        right_hand_side.ravel(),
        rtol=solver_rtol,
        atol=0.0,
        maxiter=max_iterations,
        callback=note_iteration,
    )
    if info != 0:
        raise DivergenceRelaxationError(
            f"regularized divergence relaxation did not converge in {max_iterations} "
            f"iterations (CG info={info})"
        )
    divergence_correction = nullspace.to_correction(solution.reshape(-1, 3))
    divergence_correction_norm = float(np.linalg.norm(divergence_correction))
    available_norm = max(
        0.0,
        0.9 * max_correction_norm * circulation_norm - moment_correction_norm,
    )
    trust_region_scale = (
        min(
            1.0,
            available_norm / max(divergence_correction_norm, np.finfo(float).tiny),
        )
        * correction_scale
    )
    raw_correction = moment_correction + trust_region_scale * divergence_correction
    candidate = circulation + raw_correction

    energy_direction = nullspace.to_correction(circulation / sqrt_volume[:, None])
    energy_direction_norm = float(np.linalg.norm(energy_direction))
    if energy_direction_norm <= np.finfo(float).tiny:
        raise DivergenceRelaxationError(
            "no moment-preserving direction is available to restore quadratic invariants"
        )
    energy_direction *= circulation_norm / energy_direction_norm

    enstrophy_gradient = operator.apply(circulation)
    enstrophy_direction = nullspace.to_correction(enstrophy_gradient / sqrt_volume[:, None])
    enstrophy_direction -= (
        np.vdot(enstrophy_direction, energy_direction) / np.vdot(energy_direction, energy_direction)
    ) * energy_direction
    enstrophy_direction_norm = float(np.linalg.norm(enstrophy_direction))
    for axis in range(3):
        trial = nullspace.to_correction(
            position[:, axis, None] * circulation / sqrt_volume[:, None]
        )
        trial -= (
            np.vdot(trial, energy_direction) / np.vdot(energy_direction, energy_direction)
        ) * energy_direction
        trial_norm = float(np.linalg.norm(trial))
        if enstrophy_direction_norm <= 1e-10 * circulation_norm and trial_norm > 0.0:
            enstrophy_direction = trial
            enstrophy_direction_norm = trial_norm
    if enstrophy_direction_norm <= 1e-10 * circulation_norm:
        raise DivergenceRelaxationError(
            "no second moment-preserving direction is available to restore quadratic invariants"
        )
    enstrophy_direction *= circulation_norm / enstrophy_direction_norm

    before_integrals = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        volume,
        spacing=grid_spacing,
    )
    candidate_integrals = gaussian_fourier_integrals(
        position,
        candidate,
        radius,
        volume,
        spacing=grid_spacing,
    )
    plus_energy = gaussian_fourier_integrals(
        position,
        candidate + energy_direction,
        radius,
        volume,
        spacing=grid_spacing,
    )
    minus_energy = gaussian_fourier_integrals(
        position,
        candidate - energy_direction,
        radius,
        volume,
        spacing=grid_spacing,
    )
    plus_enstrophy = gaussian_fourier_integrals(
        position,
        candidate + enstrophy_direction,
        radius,
        volume,
        spacing=grid_spacing,
    )
    minus_enstrophy = gaussian_fourier_integrals(
        position,
        candidate - enstrophy_direction,
        radius,
        volume,
        spacing=grid_spacing,
    )
    plus_both = gaussian_fourier_integrals(
        position,
        candidate + energy_direction + enstrophy_direction,
        radius,
        volume,
        spacing=grid_spacing,
    )
    values = np.array(
        [
            [candidate_integrals.energy, candidate_integrals.enstrophy],
            [plus_energy.energy, plus_energy.enstrophy],
            [minus_energy.energy, minus_energy.enstrophy],
            [plus_enstrophy.energy, plus_enstrophy.enstrophy],
            [minus_enstrophy.energy, minus_enstrophy.enstrophy],
            [plus_both.energy, plus_both.enstrophy],
        ],
        dtype=np.float64,
    ).T
    linear = np.stack(
        (
            0.5 * (values[:, 1] - values[:, 2]),
            0.5 * (values[:, 3] - values[:, 4]),
        ),
        axis=1,
    )
    diagonal = np.stack(
        (
            0.5 * (values[:, 1] + values[:, 2]) - values[:, 0],
            0.5 * (values[:, 3] + values[:, 4]) - values[:, 0],
        ),
        axis=1,
    )
    cross = 0.5 * (
        values[:, 5] - values[:, 0] - linear[:, 0] - linear[:, 1] - diagonal[:, 0] - diagonal[:, 1]
    )
    energy_roots = np.roots(
        (
            diagonal[0, 0],
            linear[0, 0],
            values[0, 0] - before_integrals.energy,
        )
    )
    real_energy_roots = [
        float(value.real)
        for value in energy_roots
        if np.isfinite(value) and abs(value.imag) <= 1e-10
    ]
    if not real_energy_roots:
        raise DivergenceRelaxationError(
            "kinetic energy could not be restored in the moment null space",
            gate="quadratic restoration",
        )
    energy_multiplier = min(real_energy_roots, key=abs)
    scalar_restored_enstrophy = (
        values[1, 0] + linear[1, 0] * energy_multiplier + diagonal[1, 0] * energy_multiplier**2
    )
    targets = np.array(
        [
            before_integrals.energy,
            (
                before_integrals.enstrophy
                if reference_scales is not None
                else scalar_restored_enstrophy
            ),
        ],
        dtype=np.float64,
    )
    scales = np.maximum(np.abs(targets), np.finfo(float).tiny)

    def quadratic_errors(multipliers: np.ndarray) -> np.ndarray:
        first, second = multipliers
        restored = (
            values[:, 0]
            + linear @ multipliers
            + diagonal[:, 0] * first * first
            + 2.0 * cross * first * second
            + diagonal[:, 1] * second * second
        )
        return (restored - targets) / scales

    def quadratic_jacobian(multipliers: np.ndarray) -> np.ndarray:
        first, second = multipliers
        return (
            np.stack(
                (
                    linear[:, 0] + 2.0 * diagonal[:, 0] * first + 2.0 * cross * second,
                    linear[:, 1] + 2.0 * cross * first + 2.0 * diagonal[:, 1] * second,
                ),
                axis=1,
            )
            / scales[:, None]
        )

    initial_multipliers = (
        np.linalg.lstsq(
            linear / scales[:, None],
            (targets - values[:, 0]) / scales,
            rcond=1e-12,
        )[0]
        if reference_scales is not None
        else np.array([energy_multiplier, 0.0])
    )
    restoration = root(
        quadratic_errors,
        initial_multipliers,
        jac=quadratic_jacobian,
        method="hybr",
    )
    if not restoration.success or not np.isfinite(restoration.x).all():
        raise DivergenceRelaxationError(
            "kinetic energy and enstrophy could not be restored in the moment null space",
            gate="quadratic restoration",
        )
    invariant_correction = (
        restoration.x[0] * energy_direction + restoration.x[1] * enstrophy_direction
    )
    correction = raw_correction + invariant_correction
    relaxed = circulation + correction
    after_integrals = gaussian_fourier_integrals(
        position,
        relaxed,
        radius,
        volume,
        spacing=grid_spacing,
    )

    after_moments = gaussian_particle_moments(position, relaxed, radius)
    circulation_restored = float(np.linalg.norm(achieved_total - before_moments[0]))
    linear_impulse_restored = float(np.linalg.norm(achieved_impulse - before_moments[2]))
    angular_impulse_restored = float(np.linalg.norm(achieved_angular - before_moments[3]))
    circulation_error = float(np.linalg.norm(after_moments[0] - achieved_total))
    linear_impulse_error = float(np.linalg.norm(after_moments[2] - achieved_impulse))
    angular_impulse_error = float(np.linalg.norm(after_moments[3] - achieved_angular))
    circulation_reference_error = float(np.linalg.norm(after_moments[0] - target_total))
    linear_impulse_reference_error = float(np.linalg.norm(after_moments[2] - target_impulse))
    angular_impulse_reference_error = float(np.linalg.norm(after_moments[3] - target_angular))
    variation_change_relative = (after_moments[1] - before_moments[1]) / max(
        before_moments[1], np.finfo(float).tiny
    )
    energy_change_relative = (after_integrals.energy - before_integrals.energy) / max(
        abs(before_integrals.energy), np.finfo(float).tiny
    )
    enstrophy_change_relative = (after_integrals.enstrophy - before_integrals.enstrophy) / max(
        abs(before_integrals.enstrophy), np.finfo(float).tiny
    )
    helicity_scale = np.sqrt(
        max(
            2.0 * abs(before_integrals.energy) * abs(before_integrals.enstrophy),
            np.finfo(float).tiny,
        )
    )
    helicity_change_relative = (
        after_integrals.helicity - before_integrals.helicity
    ) / helicity_scale
    previous_energy_change = (
        after_integrals.previous_order_energy - before_integrals.previous_order_energy
    ) / max(
        abs(before_integrals.previous_order_energy),
        np.finfo(float).tiny,
    )
    previous_enstrophy_change = (
        after_integrals.previous_order_enstrophy - before_integrals.previous_order_enstrophy
    ) / max(
        abs(before_integrals.previous_order_enstrophy),
        np.finfo(float).tiny,
    )
    previous_helicity_scale = np.sqrt(
        max(
            2.0
            * abs(before_integrals.previous_order_energy)
            * abs(before_integrals.previous_order_enstrophy),
            np.finfo(float).tiny,
        )
    )
    previous_helicity_change = (
        after_integrals.previous_order_helicity - before_integrals.previous_order_helicity
    ) / previous_helicity_scale
    energy_spectral_error = abs(energy_change_relative - previous_energy_change)
    enstrophy_spectral_error = abs(enstrophy_change_relative - previous_enstrophy_change)
    helicity_spectral_error = abs(helicity_change_relative - previous_helicity_change)
    correction_norm_relative = float(np.linalg.norm(correction) / circulation_norm)
    relaxed_residual, relaxed_grid_divergence, _ = operator.relaxation_residual(relaxed)
    final_residual_ratio = float(np.linalg.norm(relaxed_residual) / initial_residual_norm)
    quadratic_restoration_fraction = float(
        np.linalg.norm(invariant_correction)
        / max(np.linalg.norm(raw_correction), np.finfo(float).tiny)
    )

    moment_scale = max(before_moments[1], np.finfo(float).tiny)
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
    reference_gates = (
        (
            (
                "circulation reference error",
                circulation_reference_error / reference_scales[0],
                reference_tolerances[0],
            ),
            (
                "linear-impulse reference error",
                linear_impulse_reference_error / reference_scales[1],
                reference_tolerances[1],
            ),
            (
                "angular-impulse reference error",
                angular_impulse_reference_error / reference_scales[2],
                reference_tolerances[2],
            ),
        )
        if reference_scales is not None
        else ()
    )
    moment_tolerance = 4096.0 * np.finfo(float).eps
    moment_checks = (
        ("vector circulation", circulation_error, moment_scale),
        ("linear impulse", linear_impulse_error, impulse_scale),
        ("angular impulse", angular_impulse_error, angular_scale),
    )
    for name, error, scale in moment_checks:
        if error > moment_tolerance * scale:
            raise DivergenceRelaxationError(
                f"divergence relaxation changed {name} by {error:.3e}, beyond "
                f"its roundoff allowance {moment_tolerance * scale:.3e}"
            )
    gates = reference_gates + (
        ("correction norm", correction_norm_relative, max_correction_norm),
        ("residual ratio", final_residual_ratio, max_residual_ratio),
        ("kinetic-energy transfer", abs(energy_change_relative), energy_tolerance),
        ("enstrophy transfer", abs(enstrophy_change_relative), enstrophy_tolerance),
        ("helicity transfer", abs(helicity_change_relative), helicity_tolerance),
        ("total-variation transfer", abs(variation_change_relative), variation_tolerance),
        (
            "kinetic-energy spectral convergence",
            energy_spectral_error,
            spectral_convergence_fraction * energy_tolerance,
        ),
        (
            "enstrophy spectral convergence",
            enstrophy_spectral_error,
            spectral_convergence_fraction * enstrophy_tolerance,
        ),
        (
            "helicity spectral convergence",
            helicity_spectral_error,
            spectral_convergence_fraction * helicity_tolerance,
        ),
    )
    for name, value, limit in gates:
        if not np.isfinite(value) or value > limit:
            raise DivergenceRelaxationError(
                f"divergence-relaxation {name} {value:.3e} exceeds the admissible {limit:.3e}",
                gate=name,
            )
    return DivergenceRelaxationResult(
        circulation=relaxed,
        correction=correction,
        projection_sweeps=1,
        iterations=iterations,
        regularization=regularization,
        trust_region_scale=trust_region_scale,
        initial_residual_norm=initial_residual_norm,
        final_residual_ratio=final_residual_ratio,
        correction_norm_relative=correction_norm_relative,
        quadratic_restoration_fraction=quadratic_restoration_fraction,
        reference_restoration_scale=restoration_scale,
        circulation_restored=circulation_restored,
        linear_impulse_restored=linear_impulse_restored,
        angular_impulse_restored=angular_impulse_restored,
        circulation_reference_error=circulation_reference_error,
        linear_impulse_reference_error=linear_impulse_reference_error,
        angular_impulse_reference_error=angular_impulse_reference_error,
        circulation_error=circulation_error,
        linear_impulse_error=linear_impulse_error,
        angular_impulse_error=angular_impulse_error,
        total_variation_change_relative=variation_change_relative,
        energy_change_relative=energy_change_relative,
        enstrophy_change_relative=enstrophy_change_relative,
        helicity_change_relative=helicity_change_relative,
        energy_spectral_error=energy_spectral_error,
        enstrophy_spectral_error=enstrophy_spectral_error,
        helicity_spectral_error=helicity_spectral_error,
        grid_divergence_before=grid_divergence_before,
        grid_divergence_after=relaxed_grid_divergence,
    )


def _constrained_divergence_relaxation_sweep(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
    volume: np.ndarray,
    *,
    grid_spacing: float,
    regularization: float = 0.1,
    solver_rtol: float = 1e-5,
    max_iterations: int = 30,
    max_grid_nodes: int = 8_000_000,
    max_correction_norm: float = 2e-2,
    max_residual_ratio: float = 0.9,
    energy_tolerance: float = 1e-6,
    enstrophy_tolerance: float = 1e-4,
    helicity_tolerance: float = 1e-4,
    variation_tolerance: float = 1e-3,
    spectral_convergence_fraction: float = 0.1,
    reference_scales: tuple[float, float, float] | None = None,
    reference_tolerances: tuple[float, float, float] = (1e-3, 1e-2, 1e-2),
    max_line_search_steps: int = 8,
    initial_correction_scale: float = 1.0,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> DivergenceRelaxationResult:
    """Return one monotone projection sweep, backtracking its amplitude if needed."""

    if max_line_search_steps < 1:
        raise ValueError("max_line_search_steps must be at least one")
    if not 0.0 < initial_correction_scale <= 1.0:
        raise ValueError("initial_correction_scale must lie in (0, 1]")
    scalable_gates = {
        "correction norm",
        "kinetic-energy transfer",
        "enstrophy transfer",
        "helicity transfer",
        "total-variation transfer",
        "kinetic-energy spectral convergence",
        "enstrophy spectral convergence",
        "helicity spectral convergence",
        "quadratic restoration",
    }
    last_error: DivergenceRelaxationError | None = None
    for restoration_attempt in range(max_line_search_steps):
        for divergence_attempt in range(max_line_search_steps):
            try:
                return _constrained_divergence_relaxation_once(
                    position,
                    circulation,
                    radius,
                    volume,
                    grid_spacing=grid_spacing,
                    regularization=regularization,
                    solver_rtol=solver_rtol,
                    max_iterations=max_iterations,
                    max_grid_nodes=max_grid_nodes,
                    max_correction_norm=max_correction_norm,
                    max_residual_ratio=max_residual_ratio,
                    energy_tolerance=energy_tolerance,
                    enstrophy_tolerance=enstrophy_tolerance,
                    helicity_tolerance=helicity_tolerance,
                    variation_tolerance=variation_tolerance,
                    spectral_convergence_fraction=spectral_convergence_fraction,
                    reference_scales=reference_scales,
                    reference_tolerances=reference_tolerances,
                    correction_scale=initial_correction_scale * 0.5**divergence_attempt,
                    restoration_scale=0.5**restoration_attempt,
                    target_moments=target_moments,
                )
            except DivergenceRelaxationError as error:
                if error.gate == "residual ratio":
                    last_error = error
                    break
                if error.gate not in scalable_gates:
                    raise
                last_error = error
                if str(error).startswith("reference-moment restoration alone"):
                    break
    assert last_error is not None
    raise DivergenceRelaxationError(
        "divergence-relaxation line search exhausted its physics-admissible "
        f"amplitudes; last rejection: {last_error}",
        gate=last_error.gate,
    )


def _combine_projection_sweeps(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
    volume: np.ndarray,
    sweeps: list[DivergenceRelaxationResult],
    *,
    grid_spacing: float,
    regularization: float,
    max_grid_nodes: int,
    max_correction_norm: float,
    max_residual_ratio: float,
    energy_tolerance: float,
    enstrophy_tolerance: float,
    helicity_tolerance: float,
    variation_tolerance: float,
    spectral_convergence_fraction: float,
    reference_scales: tuple[float, float, float] | None,
    reference_tolerances: tuple[float, float, float],
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> DivergenceRelaxationResult:
    """Audit several monotone sweeps as one original-to-final transaction."""

    relaxed = sweeps[-1].circulation
    correction = relaxed - circulation
    before_moments = gaussian_particle_moments(position, circulation, radius)
    after_moments = gaussian_particle_moments(position, relaxed, radius)
    if target_moments is None:
        target_total = before_moments[0]
        target_impulse = before_moments[2]
        target_angular = before_moments[3]
    else:
        target_total, target_impulse, target_angular = (
            np.asarray(value, dtype=np.float64) for value in target_moments
        )

    unrestored_fraction = float(
        np.prod([1.0 - sweep.reference_restoration_scale for sweep in sweeps])
    )
    restoration_scale = 1.0 - unrestored_fraction
    achieved_total = before_moments[0] + restoration_scale * (target_total - before_moments[0])
    achieved_impulse = before_moments[2] + restoration_scale * (target_impulse - before_moments[2])
    achieved_angular = before_moments[3] + restoration_scale * (target_angular - before_moments[3])
    circulation_restored = float(np.linalg.norm(achieved_total - before_moments[0]))
    linear_impulse_restored = float(np.linalg.norm(achieved_impulse - before_moments[2]))
    angular_impulse_restored = float(np.linalg.norm(achieved_angular - before_moments[3]))
    circulation_error = float(np.linalg.norm(after_moments[0] - achieved_total))
    linear_impulse_error = float(np.linalg.norm(after_moments[2] - achieved_impulse))
    angular_impulse_error = float(np.linalg.norm(after_moments[3] - achieved_angular))
    circulation_reference_error = float(np.linalg.norm(after_moments[0] - target_total))
    linear_impulse_reference_error = float(np.linalg.norm(after_moments[2] - target_impulse))
    angular_impulse_reference_error = float(np.linalg.norm(after_moments[3] - target_angular))
    variation_change_relative = (after_moments[1] - before_moments[1]) / max(
        before_moments[1], np.finfo(float).tiny
    )

    before_integrals = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        volume,
        spacing=grid_spacing,
    )
    after_integrals = gaussian_fourier_integrals(
        position,
        relaxed,
        radius,
        volume,
        spacing=grid_spacing,
    )
    energy_change_relative = (after_integrals.energy - before_integrals.energy) / max(
        abs(before_integrals.energy), np.finfo(float).tiny
    )
    enstrophy_change_relative = (after_integrals.enstrophy - before_integrals.enstrophy) / max(
        abs(before_integrals.enstrophy), np.finfo(float).tiny
    )
    helicity_scale = np.sqrt(
        max(
            2.0 * abs(before_integrals.energy) * abs(before_integrals.enstrophy),
            np.finfo(float).tiny,
        )
    )
    helicity_change_relative = (
        after_integrals.helicity - before_integrals.helicity
    ) / helicity_scale
    previous_energy_change = (
        after_integrals.previous_order_energy - before_integrals.previous_order_energy
    ) / max(abs(before_integrals.previous_order_energy), np.finfo(float).tiny)
    previous_enstrophy_change = (
        after_integrals.previous_order_enstrophy - before_integrals.previous_order_enstrophy
    ) / max(abs(before_integrals.previous_order_enstrophy), np.finfo(float).tiny)
    previous_helicity_scale = np.sqrt(
        max(
            2.0
            * abs(before_integrals.previous_order_energy)
            * abs(before_integrals.previous_order_enstrophy),
            np.finfo(float).tiny,
        )
    )
    previous_helicity_change = (
        after_integrals.previous_order_helicity - before_integrals.previous_order_helicity
    ) / previous_helicity_scale
    energy_spectral_error = abs(energy_change_relative - previous_energy_change)
    enstrophy_spectral_error = abs(enstrophy_change_relative - previous_enstrophy_change)
    helicity_spectral_error = abs(helicity_change_relative - previous_helicity_change)

    circulation_norm = max(float(np.linalg.norm(circulation)), np.finfo(float).tiny)
    correction_norm_relative = float(np.linalg.norm(correction) / circulation_norm)
    operator = GaussianParticleGridOperator(
        position,
        radius,
        np.linalg.norm(circulation, axis=1),
        spacing=grid_spacing,
        max_grid_nodes=max_grid_nodes,
    )
    initial_residual, grid_divergence_before, _ = operator.relaxation_residual(circulation)
    final_residual, grid_divergence_after, _ = operator.relaxation_residual(relaxed)
    initial_residual_norm = float(np.linalg.norm(initial_residual))
    final_residual_ratio = (
        float(np.linalg.norm(final_residual) / initial_residual_norm)
        if initial_residual_norm > np.finfo(float).tiny
        else 0.0
    )

    moment_scale = max(before_moments[1], np.finfo(float).tiny)
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
    moment_tolerance = 4096.0 * np.finfo(float).eps
    for name, error, scale in (
        ("vector circulation", circulation_error, moment_scale),
        ("linear impulse", linear_impulse_error, impulse_scale),
        ("angular impulse", angular_impulse_error, angular_scale),
    ):
        if error > moment_tolerance * scale:
            raise DivergenceRelaxationError(
                f"iterated divergence relaxation changed {name} by {error:.3e}, beyond "
                f"its roundoff allowance {moment_tolerance * scale:.3e}"
            )

    reference_gates = (
        (
            (
                "circulation reference error",
                circulation_reference_error / reference_scales[0],
                reference_tolerances[0],
            ),
            (
                "linear-impulse reference error",
                linear_impulse_reference_error / reference_scales[1],
                reference_tolerances[1],
            ),
            (
                "angular-impulse reference error",
                angular_impulse_reference_error / reference_scales[2],
                reference_tolerances[2],
            ),
        )
        if reference_scales is not None
        else ()
    )
    gates = reference_gates + (
        ("correction norm", correction_norm_relative, max_correction_norm),
        ("residual ratio", final_residual_ratio, max_residual_ratio),
        ("kinetic-energy transfer", abs(energy_change_relative), energy_tolerance),
        ("enstrophy transfer", abs(enstrophy_change_relative), enstrophy_tolerance),
        ("helicity transfer", abs(helicity_change_relative), helicity_tolerance),
        ("total-variation transfer", abs(variation_change_relative), variation_tolerance),
        (
            "kinetic-energy spectral convergence",
            energy_spectral_error,
            spectral_convergence_fraction * energy_tolerance,
        ),
        (
            "enstrophy spectral convergence",
            enstrophy_spectral_error,
            spectral_convergence_fraction * enstrophy_tolerance,
        ),
        (
            "helicity spectral convergence",
            helicity_spectral_error,
            spectral_convergence_fraction * helicity_tolerance,
        ),
    )
    for name, value, limit in gates:
        if not np.isfinite(value) or value > limit:
            raise DivergenceRelaxationError(
                f"iterated divergence-relaxation {name} {value:.3e} "
                f"exceeds the admissible {limit:.3e}",
                gate=name,
            )

    return DivergenceRelaxationResult(
        circulation=relaxed,
        correction=correction,
        projection_sweeps=len(sweeps),
        iterations=sum(sweep.iterations for sweep in sweeps),
        regularization=regularization,
        trust_region_scale=min(sweep.trust_region_scale for sweep in sweeps),
        initial_residual_norm=initial_residual_norm,
        final_residual_ratio=final_residual_ratio,
        correction_norm_relative=correction_norm_relative,
        quadratic_restoration_fraction=max(
            sweep.quadratic_restoration_fraction for sweep in sweeps
        ),
        reference_restoration_scale=restoration_scale,
        circulation_restored=circulation_restored,
        linear_impulse_restored=linear_impulse_restored,
        angular_impulse_restored=angular_impulse_restored,
        circulation_reference_error=circulation_reference_error,
        linear_impulse_reference_error=linear_impulse_reference_error,
        angular_impulse_reference_error=angular_impulse_reference_error,
        circulation_error=circulation_error,
        linear_impulse_error=linear_impulse_error,
        angular_impulse_error=angular_impulse_error,
        total_variation_change_relative=variation_change_relative,
        energy_change_relative=energy_change_relative,
        enstrophy_change_relative=enstrophy_change_relative,
        helicity_change_relative=helicity_change_relative,
        energy_spectral_error=energy_spectral_error,
        enstrophy_spectral_error=enstrophy_spectral_error,
        helicity_spectral_error=helicity_spectral_error,
        grid_divergence_before=grid_divergence_before,
        grid_divergence_after=grid_divergence_after,
    )


def constrained_divergence_relaxation(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
    volume: np.ndarray,
    *,
    grid_spacing: float,
    regularization: float = 0.1,
    solver_rtol: float = 1e-5,
    max_iterations: int = 30,
    max_grid_nodes: int = 8_000_000,
    max_correction_norm: float = 2e-2,
    max_residual_ratio: float = 0.9,
    energy_tolerance: float = 1e-6,
    enstrophy_tolerance: float = 1e-4,
    helicity_tolerance: float = 1e-4,
    variation_tolerance: float = 1e-3,
    spectral_convergence_fraction: float = 0.1,
    reference_scales: tuple[float, float, float] | None = None,
    reference_tolerances: tuple[float, float, float] = (1e-3, 1e-2, 1e-2),
    max_line_search_steps: int = 8,
    max_projection_sweeps: int = 3,
    target_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> DivergenceRelaxationResult:
    """Return an atomic, iterated, physics-gated Helmholtz projection."""

    if max_projection_sweeps < 1:
        raise ValueError("max_projection_sweeps must be at least one")
    sweep_residual_limit = max_residual_ratio ** (1.0 / max_projection_sweeps)
    monotone_residual_limit = np.nextafter(1.0, 0.0)
    amplitude_retryable_gates = {
        "correction norm",
        "kinetic-energy transfer",
        "enstrophy transfer",
        "helicity transfer",
        "total-variation transfer",
        "kinetic-energy spectral convergence",
        "enstrophy spectral convergence",
        "helicity spectral convergence",
    }
    working_circulation = np.asarray(circulation, dtype=np.float64)
    sweeps: list[DivergenceRelaxationResult] = []
    last_error: DivergenceRelaxationError | None = None
    for sweep_index in range(max_projection_sweeps):
        full_sweep: DivergenceRelaxationResult | None = None
        for amplitude_attempt in range(max_line_search_steps):
            try:
                candidate = _constrained_divergence_relaxation_sweep(
                    position,
                    working_circulation,
                    radius,
                    volume,
                    grid_spacing=grid_spacing,
                    regularization=regularization,
                    solver_rtol=solver_rtol,
                    max_iterations=max_iterations,
                    max_grid_nodes=max_grid_nodes,
                    max_correction_norm=max_correction_norm,
                    max_residual_ratio=(
                        sweep_residual_limit if amplitude_attempt == 0 else monotone_residual_limit
                    ),
                    energy_tolerance=energy_tolerance,
                    enstrophy_tolerance=enstrophy_tolerance,
                    helicity_tolerance=helicity_tolerance,
                    variation_tolerance=variation_tolerance,
                    spectral_convergence_fraction=spectral_convergence_fraction,
                    reference_scales=reference_scales,
                    reference_tolerances=reference_tolerances,
                    max_line_search_steps=max_line_search_steps,
                    initial_correction_scale=0.5**amplitude_attempt,
                    target_moments=target_moments,
                )
            except DivergenceRelaxationError as error:
                last_error = error
                if error.gate == "residual ratio":
                    break
                raise
            if full_sweep is None:
                full_sweep = candidate
            candidate_sweeps = [*sweeps, candidate]
            if len(candidate_sweeps) == 1:
                if candidate.final_residual_ratio <= max_residual_ratio:
                    return candidate
                break
            try:
                return _combine_projection_sweeps(
                    position,
                    circulation,
                    radius,
                    volume,
                    candidate_sweeps,
                    grid_spacing=grid_spacing,
                    regularization=regularization,
                    max_grid_nodes=max_grid_nodes,
                    max_correction_norm=max_correction_norm,
                    max_residual_ratio=max_residual_ratio,
                    energy_tolerance=energy_tolerance,
                    enstrophy_tolerance=enstrophy_tolerance,
                    helicity_tolerance=helicity_tolerance,
                    variation_tolerance=variation_tolerance,
                    spectral_convergence_fraction=spectral_convergence_fraction,
                    reference_scales=reference_scales,
                    reference_tolerances=reference_tolerances,
                    target_moments=target_moments,
                )
            except DivergenceRelaxationError as error:
                last_error = error
                if error.gate == "residual ratio":
                    break
                if error.gate not in amplitude_retryable_gates:
                    raise
        assert full_sweep is not None
        if sweep_index == max_projection_sweeps - 1:
            assert last_error is not None
            raise last_error
        sweeps.append(full_sweep)
        working_circulation = full_sweep.circulation
    raise AssertionError("projection sweep loop terminated without a result")
