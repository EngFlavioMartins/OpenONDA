"""Strength-weighted ring diagnostics, one CSV row per particle group and sample time.

Group centroids and radii are ancestry contributions, not vorticity maxima:
after the rings merge, a group's centroid does not mark a separate core.
"""

from __future__ import annotations

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
    "linear_impulse_y",
    "linear_impulse_z",
    "linear_impulse_magnitude",
    "impulse_radius",
    "vortex_strength_magnitude_sum",
    "net_vortex_strength_x",
    "net_vortex_strength_y",
    "net_vortex_strength_z",
    "max_vortex_strength_magnitude",
)


class RingDiagnosticsSampler:
    """Sample grouped ring diagnostics into ``ring_diagnostics.csv``.

    Periodic and final samplers share one output key, hence one CSV file.
    """

    file_name = "ring_diagnostics"
    csv_columns = CSV_COLUMNS[2:]
    output_key = ("RingDiagnosticsSampler", file_name)

    def __init__(self, *, schedule=None, initial=False):
        self.schedule = schedule
        self.initial = initial

    def sample(self, solver):
        position = np.asarray(solver.particle_position, dtype=np.float64)
        vortex_strength = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        group_ids = np.asarray(solver.particle_group_id, dtype=np.int32)
        rows = [
            [
                int(group),
                *sample_group(position[group_ids == group], vortex_strength[group_ids == group]),
            ]
            for group in np.unique(group_ids)
        ]
        return {
            name: np.asarray(
                [row[index] for row in rows],
                dtype=np.int32 if name == "group_id" else np.float64,
            )
            for index, name in enumerate(self.csv_columns)
        }


def sample_group(position, vortex_strength):
    """Centroid, major radius, impulse and strength sums of one particle group."""
    magnitude = np.linalg.norm(vortex_strength, axis=1)
    magnitude_sum = float(magnitude.sum())
    if magnitude_sum <= np.finfo(float).tiny:
        return (np.nan,) * (len(CSV_COLUMNS) - 3)

    # Strength-weighted centroid and in-plane radius from the second moment
    centroid = np.einsum("i,ij->j", magnitude, position) / magnitude_sum
    centred = position - centroid
    covariance = (centred * magnitude[:, None]).T @ centred / magnitude_sum
    eigenvalues = np.linalg.eigvalsh(covariance)
    major_radius = float(np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)))
    tube_circulation = (
        magnitude_sum / (2.0 * np.pi * major_radius)
        if major_radius > np.finfo(float).eps
        else np.nan
    )

    # Linear impulse I = 1/2 sum(x × Gamma) and its equivalent ring radius
    impulse = 0.5 * np.sum(np.cross(position, vortex_strength), axis=0)
    impulse_norm = float(np.linalg.norm(impulse))
    return (
        *centroid,
        major_radius,
        tube_circulation,
        *impulse,
        impulse_norm,
        2.0 * impulse_norm / magnitude_sum,
        magnitude_sum,
        *vortex_strength.sum(axis=0),
        float(magnitude.max(initial=0.0)),
    )


AZIMUTHAL_MODES = np.arange(1, 17)


class AzimuthalModeSampler:
    """Strength-weighted azimuthal Fourier amplitudes of each ring, ``azimuthal_modes.csv``.

    ``A_m = |sum_p |Gamma_p| exp(i m theta_p)| / sum_p |Gamma_p|``, with ``theta`` the
    azimuth about the x axis. The rings start axisymmetric, so ``A_m`` measures the
    growth of three-dimensional disturbances, and its maximum over ``m`` marks breakdown.
    """

    file_name = "azimuthal_modes"
    csv_columns = ("group_id", *(f"mode_{m}" for m in AZIMUTHAL_MODES))

    def __init__(self, *, schedule=None, initial=False):
        self.schedule = schedule
        self.initial = initial

    def sample(self, solver):
        position = np.asarray(solver.particle_position, dtype=np.float64)
        magnitude = np.linalg.norm(np.asarray(solver.particle_vortex_strength, dtype=np.float64), axis=1)
        group_ids = np.asarray(solver.particle_group_id, dtype=np.int32)
        azimuth = np.arctan2(position[:, 2], position[:, 1])
        rows = []
        for group in np.unique(group_ids):
            selected = group_ids == group
            phase = np.exp(1j * np.outer(AZIMUTHAL_MODES, azimuth[selected]))
            amplitude = np.abs(phase @ magnitude[selected]) / magnitude[selected].sum()
            rows.append((group, amplitude))
        return {
            "group_id": np.asarray([group for group, _ in rows], dtype=np.int32),
            **{
                f"mode_{m}": np.asarray([amplitude[k] for _, amplitude in rows])
                for k, m in enumerate(AZIMUTHAL_MODES)
            },
        }
