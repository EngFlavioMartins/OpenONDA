"""Sampler of the rigid motion of a moving FVM mesh."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.file_output import append_csv_rows
from openonda.core.sampling import Sampler
from openonda.core.schedules import EverySteps, Schedule

if TYPE_CHECKING:
    from ..core.solver import FVMSolver

MOTION_SAMPLE_COLUMNS = (
    "time",
    "time_step_size",
    "position_x",
    "position_y",
    "position_z",
    "displacement_x",
    "displacement_y",
    "displacement_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)


class MotionSampler(Sampler):
    """Record the position and velocity of the rotation centre of a moving mesh.

    One row per sampled time: the current position of the rotation centre in
    m, its displacement from the placed position, its velocity in m/s and the
    angular velocity of the mesh in rad/s. For an elastically mounted body
    this is the structural response.

    Examples
    --------
    >>> MotionSampler().name
    'motion'
    """

    def __init__(
        self,
        *,
        file_name: str | None = "motion_history",
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)

    def sample(self, context: FVMSolver) -> dict[str, np.ndarray]:
        """Return the rotation-centre position, displacement and velocities of the accepted state."""
        motion = context.mesh_motion
        if motion is None:
            raise ValueError("MotionSampler needs a case with mesh_motion")
        position = motion.rotation_centre()
        placed = np.asarray(motion.kinematics.rotation_centre, dtype=np.float64)
        return {
            "position": position,
            "displacement": position - placed,
            "velocity": np.asarray(motion.kinematics.get_velocity(context.time), dtype=np.float64),
            "angular_velocity": np.asarray(
                motion.kinematics.get_angular_velocity(context.time), dtype=np.float64
            ),
        }

    def write_csv(self, context: FVMSolver, samples_dir: str | Path, motion: dict) -> None:
        """Append one row for an already-sampled motion state."""
        row = [context.time, context._accepted_time_step_size]
        for name in ("position", "displacement", "velocity", "angular_velocity"):
            row.extend(float(value) for value in motion[name])
        append_csv_rows(Path(samples_dir) / f"{self.name}.csv", MOTION_SAMPLE_COLUMNS, [row])
