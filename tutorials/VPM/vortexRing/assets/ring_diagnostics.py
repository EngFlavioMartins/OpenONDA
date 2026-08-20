"""Compact particle diagnostics sampled during a vortex-ring run."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


CSV_COLUMNS = (
    "time",
    "step",
    "group_id",
    "x_centroid",
    "y_centroid",
    "z_centroid",
    "major_radius",
    "tube_circulation",
    "impulse_x",
    "impulse_norm",
    "impulse_radius",
    "length_strength",
    "vector_circulation_x",
    "vector_circulation_y",
    "vector_circulation_z",
    "max_strength",
)

MODE_CSV_COLUMNS = (
    "time",
    "step",
    "group_id",
    "mode",
    "radial_amplitude",
    "axial_amplitude",
    "combined_amplitude",
    "radial_phase",
    "axial_phase",
    "major_radius",
    "azimuthal_coverage",
)


class RingDiagnosticsSampler:
    """Sample ring motion and circulation without writing a particle backup."""

    file_name = "ring_diagnostics"

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        positions = np.asarray(solver.particles_positions, dtype=np.float64)
        circulation = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        group_ids = np.asarray(solver.particles_group_ids, dtype=np.int32)

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(CSV_COLUMNS)
            for group_id in np.unique(group_ids):
                selected = group_ids == group_id
                row = self._sample_group(positions[selected], circulation[selected])
                writer.writerow([time, step, int(group_id), *row])

    @staticmethod
    def _sample_group(
        positions: np.ndarray,
        circulation: np.ndarray,
    ) -> tuple[float, ...]:
        strength = np.linalg.norm(circulation, axis=1)
        length_strength = float(strength.sum())
        if length_strength <= np.finfo(float).tiny:
            return (np.nan,) * (len(CSV_COLUMNS) - 3)

        centroid = np.einsum("i,ij->j", strength, positions) / length_strength
        centered = positions - centroid
        covariance = (centered * strength[:, None]).T @ centered / length_strength
        eigenvalues = np.linalg.eigvalsh(covariance)
        major_radius = float(np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)))
        tube_circulation = (
            length_strength / (2.0 * np.pi * major_radius)
            if major_radius > np.finfo(float).eps
            else np.nan
        )

        vector_circulation = circulation.sum(axis=0)
        impulse = 0.5 * np.sum(np.cross(positions, circulation), axis=0)
        impulse_norm = float(np.linalg.norm(impulse))
        impulse_radius = 2.0 * impulse_norm / length_strength
        return (
            *centroid,
            major_radius,
            tube_circulation,
            float(impulse[0]),
            impulse_norm,
            impulse_radius,
            length_strength,
            *vector_circulation,
            float(strength.max(initial=0.0)),
        )


class RingModeDiagnosticsSampler:
    """Measure centreline bending modes of a ring whose nominal axis is x.

    Particle strength is used as the cross-section weight.  Radial and axial
    Fourier moments are integrated directly over particle angles, avoiding an
    arbitrary dependence on the number of occupied diagnostic bins. Amplitudes
    are normalized by ``reference_radius``. A perfect perturbation

        r(theta) / R = 1 + epsilon cos(m theta + phi)

    therefore reports ``radial_amplitude = epsilon`` at mode ``m``.

    The diagnostic intentionally reports radial and axial components
    separately: a physical Widnall bending wave contains both, whereas random
    particle noise need not have their coherent modal structure.
    """

    file_name = "ring_modes"

    def __init__(
        self,
        *,
        maximum_mode: int = 40,
        azimuthal_bins: int = 128,
        reference_radius: float = 1.0,
        transverse_origin: tuple[float, float] | None = None,
    ) -> None:
        if maximum_mode < 1:
            raise ValueError("maximum_mode must be positive")
        if azimuthal_bins < 2 * maximum_mode + 1:
            raise ValueError("azimuthal_bins must exceed twice maximum_mode")
        if reference_radius <= 0.0:
            raise ValueError("reference_radius must be positive")
        self.maximum_mode = maximum_mode
        self.azimuthal_bins = azimuthal_bins
        self.reference_radius = reference_radius
        self.transverse_origin = transverse_origin

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        positions = np.asarray(solver.particles_positions, dtype=np.float64)
        circulation = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        group_ids = np.asarray(solver.particles_group_ids, dtype=np.int32)

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(MODE_CSV_COLUMNS)
            for group_id in np.unique(group_ids):
                selected = group_ids == group_id
                rows = self._sample_group(positions[selected], circulation[selected])
                for row in rows:
                    writer.writerow([time, step, int(group_id), *row])

    def _sample_group(
        self,
        positions: np.ndarray,
        circulation: np.ndarray,
    ) -> list[tuple[float, ...]]:
        strength = np.linalg.norm(circulation, axis=1)
        total_strength = float(strength.sum())
        if total_strength <= np.finfo(float).tiny:
            return []

        rough_centroid = np.einsum("i,ij->j", strength, positions) / total_strength
        if self.transverse_origin is None:
            transverse_origin = rough_centroid[1:]
        else:
            transverse_origin = np.asarray(self.transverse_origin, dtype=float)
        centered = positions.copy()
        centered[:, 1:] -= transverse_origin
        theta = np.mod(np.arctan2(centered[:, 2], centered[:, 1]), 2.0 * np.pi)
        radial_position = np.hypot(centered[:, 1], centered[:, 2])
        tangent = np.column_stack(
            (
                np.zeros_like(theta),
                -np.sin(theta),
                np.cos(theta),
            )
        )
        # Gamma_theta contains the cylindrical volume Jacobian rho. Dividing
        # by rho recovers the cross-sectional vorticity measure needed for a
        # centreline moment and avoids a false bias toward the outer side of
        # the torus. The radial circulation added by the solenoidal Widnall
        # initialization is intentionally excluded from this weight.
        tangential_strength = np.abs(np.einsum("ij,ij->i", circulation, tangent))
        cross_section_weight = tangential_strength / np.maximum(
            radial_position,
            np.finfo(float).eps,
        )
        axial_centroid = float(
            np.sum(cross_section_weight * centered[:, 0]) / np.sum(cross_section_weight)
        )
        axial_position = centered[:, 0] - axial_centroid

        bin_index = np.floor(theta * self.azimuthal_bins / (2.0 * np.pi)).astype(int)
        bin_index = np.minimum(bin_index, self.azimuthal_bins - 1)
        bin_weight = np.bincount(
            bin_index,
            weights=cross_section_weight,
            minlength=self.azimuthal_bins,
        )
        radial_sum = np.bincount(
            bin_index,
            weights=cross_section_weight * radial_position,
            minlength=self.azimuthal_bins,
        )
        occupied = bin_weight > np.finfo(float).tiny
        coverage = float(np.mean(occupied))
        if np.count_nonzero(occupied) < 2 * self.maximum_mode + 1:
            return []

        major_radius = float(np.sum(radial_sum) / np.sum(bin_weight))
        radial_displacement = radial_position - major_radius

        rows: list[tuple[float, ...]] = []
        for mode in range(1, self.maximum_mode + 1):
            phase_factor = np.exp(-1j * mode * theta)
            radial_coefficient = np.sum(
                cross_section_weight * radial_displacement * phase_factor
            ) / np.sum(cross_section_weight)
            axial_coefficient = np.sum(
                cross_section_weight * axial_position * phase_factor
            ) / np.sum(cross_section_weight)
            radial_amplitude = 2.0 * abs(radial_coefficient) / self.reference_radius
            axial_amplitude = 2.0 * abs(axial_coefficient) / self.reference_radius
            rows.append(
                (
                    mode,
                    radial_amplitude,
                    axial_amplitude,
                    float(np.hypot(radial_amplitude, axial_amplitude)),
                    float(np.angle(radial_coefficient)),
                    float(np.angle(axial_coefficient)),
                    major_radius,
                    coverage,
                )
            )
        return rows
