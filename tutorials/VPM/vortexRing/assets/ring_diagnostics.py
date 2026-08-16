"""Compact particle diagnostics sampled during a vortex-ring run."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


CSV_COLUMNS = (
    "flow_time",
    "time_step",
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
    "flow_time",
    "time_step",
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
        circulation = np.asarray(solver.particles_circulation, dtype=np.float64)
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

    Particle strength is used as the cross-section weight.  The weighted
    centreline is first reconstructed on uniformly spaced azimuthal bins, then
    decomposed into radial and axial Fourier modes.  Amplitudes are normalized
    by ``reference_radius``.  A perfect perturbation

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

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        positions = np.asarray(solver.particles_positions, dtype=np.float64)
        circulation = np.asarray(solver.particles_circulation, dtype=np.float64)
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

        centroid = np.einsum("i,ij->j", strength, positions) / total_strength
        centered = positions - centroid
        theta = np.mod(np.arctan2(centered[:, 2], centered[:, 1]), 2.0 * np.pi)
        radial_position = np.hypot(centered[:, 1], centered[:, 2])
        axial_position = centered[:, 0]

        bin_index = np.floor(theta * self.azimuthal_bins / (2.0 * np.pi)).astype(int)
        bin_index = np.minimum(bin_index, self.azimuthal_bins - 1)
        bin_weight = np.bincount(bin_index, weights=strength, minlength=self.azimuthal_bins)
        radial_sum = np.bincount(
            bin_index,
            weights=strength * radial_position,
            minlength=self.azimuthal_bins,
        )
        axial_sum = np.bincount(
            bin_index,
            weights=strength * axial_position,
            minlength=self.azimuthal_bins,
        )
        occupied = bin_weight > np.finfo(float).tiny
        coverage = float(np.mean(occupied))
        if np.count_nonzero(occupied) < 2 * self.maximum_mode + 1:
            return []

        radial = radial_sum[occupied] / bin_weight[occupied]
        axial = axial_sum[occupied] / bin_weight[occupied]
        angles = (np.flatnonzero(occupied) + 0.5) * 2.0 * np.pi / self.azimuthal_bins
        radial -= radial.mean()
        axial -= axial.mean()
        major_radius = float(np.sum(radial_sum) / np.sum(bin_weight))

        rows: list[tuple[float, ...]] = []
        for mode in range(1, self.maximum_mode + 1):
            phase_factor = np.exp(-1j * mode * angles)
            # A bin average attenuates mode m by sinc(m / N_bins).  Remove
            # that known measurement transfer function so the reported value
            # can be compared directly with the prescribed seed amplitude.
            bin_transfer = np.sinc(mode / self.azimuthal_bins)
            radial_coefficient = np.mean(radial * phase_factor) / bin_transfer
            axial_coefficient = np.mean(axial * phase_factor) / bin_transfer
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
