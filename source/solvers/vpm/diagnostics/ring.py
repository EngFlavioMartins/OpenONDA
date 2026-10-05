"""Compact diagnostics for grouped vortex-ring particle clouds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..io.sampling.schedule import OutputSchedule

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

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
    """Sample one compact diagnostic row per particle group and accepted time.

    These are strength-weighted group centroids and radii, not vorticity maxima.
    A group remains an ancestry contribution after rings merge; its centroid
    does not establish a separate physical core. Remeshing must preserve group
    contributions for this interpretation (nearest-source relabelling does not).
    The sampler owns its schedule and canonical output name.
    """

    csv_columns = RING_DIAGNOSTIC_COLUMNS[2:]

    def __init__(
        self,
        *,
        schedule: OutputSchedule | None = None,
        file_name: str = "ring_diagnostics",
        initial: bool | None = None,
    ) -> None:
        """Configure grouped vortex-ring diagnostics output.

        Parameters
        ----------
        schedule : OutputSchedule or None, optional
            Optional time/step schedule. ``None`` leaves scheduling to the
            caller. The output manager owns atomic, monotonic persistence.
        file_name : str, default="ring_diagnostics"
            Stem of the CSV file written below the active output directory;
            ``.csv`` is appended by the output manager.
        initial : bool or None, default=None
            Include the initial state. ``None`` retains a subclass's policy.

        Raises
        ------
        ValueError
            If ``file_name`` is empty.

        Notes
        -----
        The constructor creates no files and does not retain particle arrays.
        """
        if not file_name:
            raise ValueError("RingDiagnosticsSampler file_name must not be empty")
        self.schedule = schedule
        self.file_name = file_name
        if initial is not None:
            self.initial = initial

    @property
    def output_identity(self) -> tuple[type, str]:
        """Periodic and final schedules share the same grouped CSV event.

        Ring diagnostics have no sampling options beyond their destination;
        the sampler type distinguishes alternative implementations.
        """
        return type(self), self.file_name

    def sample(self, solver: VPMSolver) -> dict[str, np.ndarray]:
        """Return grouped ring diagnostics for framework-owned CSV output."""
        position = np.asarray(solver.particle_position, dtype=np.float64)
        vortex_strength = np.asarray(solver.particle_vortex_strength, dtype=np.float64)
        particle_group_id = np.asarray(solver.particle_group_id, dtype=np.int32)
        rows = []
        for group_id in np.unique(particle_group_id):
            selected = particle_group_id == group_id
            rows.append(
                [int(group_id), *self._sample_group(position[selected], vortex_strength[selected])]
            )
        return {
            name: np.asarray(
                [row[index] for row in rows],
                dtype=np.int32 if name == "group_id" else np.float64,
            )
            for index, name in enumerate(self.csv_columns)
        }

    @staticmethod
    def _sample_group(
        position: np.ndarray,
        vortex_strength: np.ndarray,
    ) -> tuple[float, ...]:
        vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
        vortex_strength_magnitude_sum = float(vortex_strength_magnitude.sum())
        if vortex_strength_magnitude_sum <= np.finfo(float).tiny:
            return (np.nan,) * (len(RING_DIAGNOSTIC_COLUMNS) - 3)

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
        impulse_norm = float(np.linalg.norm(impulse))
        impulse_radius = 2.0 * impulse_norm / vortex_strength_magnitude_sum
        return (
            *vortex_centroid,
            major_radius,
            tube_circulation,
            *impulse,
            impulse_norm,
            impulse_radius,
            vortex_strength_magnitude_sum,
            *net_vortex_strength,
            float(vortex_strength_magnitude.max(initial=0.0)),
        )
