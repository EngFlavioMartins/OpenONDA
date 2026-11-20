"""Strength-weighted ring diagnostics, one CSV row per sample time."""

from __future__ import annotations

import numpy as np

CSV_COLUMNS = (
    "time",
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
    """Sample grouped ring diagnostics into ``ring_diagnostics.csv``."""

    file_name = "ring_diagnostics"
    csv_columns = CSV_COLUMNS[1:]

    def __init__(self, *, schedule=None, initial=True):
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
        return (np.nan,) * (len(CSV_COLUMNS) - 2)

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
