"""Compact particle diagnostics sampled during a vortex-ring run."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


CSV_COLUMNS = (
    "time",
    "step",
    "group_id",
    "vortex_centroid_x",
    "vortex_centroid_y",
    "vortex_centroid_z",
    "major_radius",
    "tube_circulation",
    "linear_impulse_x",
    "linear_impulse_magnitude",
    "impulse_radius",
    "vortex_strength_magnitude_sum",
    "net_vortex_strength_x",
    "net_vortex_strength_y",
    "net_vortex_strength_z",
    "max_vortex_strength_magnitude",
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


def vortex_ring_mode_sampler(*, reference_radius: float, schedule) -> RingModeDiagnosticsSampler:
    """Build this tutorial's fixed-resolution Widnall-mode diagnostic.

    The bin and mode counts are analysis choices for the vortex-ring tutorial,
    not solver-wide VPM settings.  Keeping them here makes the case setup
    describe physics and run cadence without embedding post-processing detail.
    """
    return RingModeDiagnosticsSampler(
        max_mode=40,
        azimuthal_bins=128,
        reference_radius=reference_radius,
        transverse_origin=(0.0, 0.0),
        schedule=schedule,
    )


class RingDiagnosticsSampler:
    """Sample ring motion and vortex strength without writing a particle backup."""

    file_name = "ring_diagnostics"

    def __init__(self, *, schedule=None) -> None:
        self.schedule = schedule

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        position = np.asarray(solver.particle_position, dtype=np.float64)
        vortex_strength = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        particle_group_id = np.asarray(solver.particle_group_id, dtype=np.int32)

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(CSV_COLUMNS)
            for group_id in np.unique(particle_group_id):
                selected = particle_group_id == group_id
                row = self._sample_group(position[selected], vortex_strength[selected])
                writer.writerow([time, step, int(group_id), *row])

    @staticmethod
    def _sample_group(
        position: np.ndarray,
        vortex_strength: np.ndarray,
    ) -> tuple[float, ...]:
        vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
        vortex_strength_magnitude_sum = float(vortex_strength_magnitude.sum())
        if vortex_strength_magnitude_sum <= np.finfo(float).tiny:
            return (np.nan,) * (len(CSV_COLUMNS) - 3)

        vortex_centroid = (
            np.einsum("i,ij->j", vortex_strength_magnitude, position)
            / vortex_strength_magnitude_sum
        )
        centred_position = position - vortex_centroid
        covariance = (
            (centred_position * vortex_strength_magnitude[:, None]).T
            @ centred_position
            / vortex_strength_magnitude_sum
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        major_radius = float(np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)))
        tube_circulation = (
            vortex_strength_magnitude_sum / (2.0 * np.pi * major_radius)
            if major_radius > np.finfo(float).eps
            else np.nan
        )

        net_vortex_strength = vortex_strength.sum(axis=0)
        impulse = 0.5 * np.sum(np.cross(position, vortex_strength), axis=0)
        linear_impulse_magnitude = float(np.linalg.norm(impulse))
        impulse_radius = 2.0 * linear_impulse_magnitude / vortex_strength_magnitude_sum
        return (
            *vortex_centroid,
            major_radius,
            tube_circulation,
            float(impulse[0]),
            linear_impulse_magnitude,
            impulse_radius,
            vortex_strength_magnitude_sum,
            *net_vortex_strength,
            float(vortex_strength_magnitude.max(initial=0.0)),
        )


class RingModeDiagnosticsSampler:
    """Measure centreline bending modes of a ring whose nominal axis is x.

    Particle vortex-strength magnitude is used as the cross-section weight. Radial and axial
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
        max_mode: int = 40,
        azimuthal_bins: int = 128,
        reference_radius: float = 1.0,
        transverse_origin: tuple[float, float] | None = None,
        schedule=None,
    ) -> None:
        if max_mode < 1:
            raise ValueError("max_mode must be positive")
        if azimuthal_bins < 2 * max_mode + 1:
            raise ValueError("azimuthal_bins must exceed twice max_mode")
        if reference_radius <= 0.0:
            raise ValueError("reference_radius must be positive")
        self.schedule = schedule
        self.max_mode = max_mode
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
        position = np.asarray(solver.particle_position, dtype=np.float64)
        vortex_strength = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        particle_group_id = np.asarray(solver.particle_group_id, dtype=np.int32)

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(MODE_CSV_COLUMNS)
            for group_id in np.unique(particle_group_id):
                selected = particle_group_id == group_id
                rows = self._sample_group(position[selected], vortex_strength[selected])
                for row in rows:
                    writer.writerow([time, step, int(group_id), *row])

    def _sample_group(
        self,
        position: np.ndarray,
        vortex_strength: np.ndarray,
    ) -> list[tuple[float, ...]]:
        vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
        vortex_strength_magnitude_sum = float(vortex_strength_magnitude.sum())
        if vortex_strength_magnitude_sum <= np.finfo(float).tiny:
            return []

        approximate_vortex_centroid = (
            np.einsum("i,ij->j", vortex_strength_magnitude, position)
            / vortex_strength_magnitude_sum
        )
        if self.transverse_origin is None:
            transverse_origin = approximate_vortex_centroid[1:]
        else:
            transverse_origin = np.asarray(self.transverse_origin, dtype=float)
        centred_position = position.copy()
        centred_position[:, 1:] -= transverse_origin
        theta = np.mod(np.arctan2(centred_position[:, 2], centred_position[:, 1]), 2.0 * np.pi)
        radial_position = np.hypot(centred_position[:, 1], centred_position[:, 2])
        tangent = np.column_stack(
            (
                np.zeros_like(theta),
                -np.sin(theta),
                np.cos(theta),
            )
        )
        # circulation_theta contains the cylindrical particle_volume Jacobian rho. Dividing
        # by rho recovers the cross-sectional vorticity measure needed for a
        # centreline moment and avoids a false bias toward the outer side of
        # the torus. The radial vortex_strength added by the solenoidal Widnall
        # initialization is intentionally excluded from this weight.
        tangential_vortex_strength_magnitude = np.abs(
            np.einsum("ij,ij->i", vortex_strength, tangent)
        )
        cross_section_weight = tangential_vortex_strength_magnitude / np.maximum(
            radial_position,
            np.finfo(float).eps,
        )
        axial_vortex_centroid = float(
            np.sum(cross_section_weight * centred_position[:, 0]) / np.sum(cross_section_weight)
        )
        axial_position = centred_position[:, 0] - axial_vortex_centroid

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
        if np.count_nonzero(occupied) < 2 * self.max_mode + 1:
            return []

        major_radius = float(np.sum(radial_sum) / np.sum(bin_weight))
        radial_displacement = radial_position - major_radius

        rows: list[tuple[float, ...]] = []
        for mode in range(1, self.max_mode + 1):
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
