"""Compact diagnostics for grouped vortex-ring particle clouds."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RING_DIAGNOSTIC_COLUMNS = (
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
    """Write one compact diagnostic row per particle group and sample time.

    The sampler is executed through :class:`SamplerExecutor`, so its cadence is
    the solver's ``logging_interval_steps`` and its output lives below the
    solver-managed ``samples/`` directory. Particle groups are interpreted as
    individual rings.
    """

    file_name = "ring_diagnostics"

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
                writer.writerow(RING_DIAGNOSTIC_COLUMNS)
            for group_id in np.unique(particle_group_id):
                selected = particle_group_id == group_id
                row = self._sample_group(position[selected], vortex_strength[selected])
                writer.writerow([time, step, int(group_id), *row])

    @staticmethod
    def _sample_group(
        position: np.ndarray,
        vortex_strength: np.ndarray,
    ) -> tuple[float, ...]:
        strength = np.linalg.norm(vortex_strength, axis=1)
        vortex_strength_magnitude_sum = float(strength.sum())
        if vortex_strength_magnitude_sum <= np.finfo(float).tiny:
            return (np.nan,) * (len(RING_DIAGNOSTIC_COLUMNS) - 3)

        centroid = np.einsum("i,ij->j", strength, position) / vortex_strength_magnitude_sum
        centred_position = position - centroid
        covariance = (
            (centred_position * strength[:, None]).T
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
        impulse_norm = float(np.linalg.norm(impulse))
        impulse_radius = 2.0 * impulse_norm / vortex_strength_magnitude_sum
        return (
            *centroid,
            major_radius,
            tube_circulation,
            *impulse,
            impulse_norm,
            impulse_radius,
            vortex_strength_magnitude_sum,
            *net_vortex_strength,
            float(strength.max(initial=0.0)),
        )
