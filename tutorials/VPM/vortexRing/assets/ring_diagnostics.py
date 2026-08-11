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
