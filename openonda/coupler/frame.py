"""The frame translating with a moving FVM mesh, in which the VPM is solved.

A uniform frame translation leaves the vorticity equation unchanged: particle
strengths, induced velocities and diffusion are identical, and only the
particle drift changes by the frame velocity. The coupled VPM therefore runs
in the frame of a translating FVM mesh with the freestream ``U(t) - V(t)``. In
that frame the FVM box, its cells, walls and coupling faces stay fixed, so the
transfer lattice, its solid masks and the diffusion grid built for a static
mesh remain exact.

The FVM keeps absolute (inertial) velocities on its moving mesh. Interface
velocities are converted when they cross between the solvers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.kinematics import advance_pose

if TYPE_CHECKING:
    from openonda.fvm import FVMSolver


class MeshFrame:
    """Translation of the FVM mesh seen from the coupling.

    Parameters
    ----------
    fvm_solver : FVMSolver
        Solver whose ``mesh_motion`` defines the frame; a static mesh gives
        the inertial frame.

    Notes
    -----
    Positions in the frame are mesh coordinates at the start of the run (the
    placed reference mesh). Rotating meshes are rejected: the VPM has no
    rotating frame.
    """

    def __init__(self, fvm_solver: FVMSolver) -> None:
        self._motion = fvm_solver.mesh_motion

    @property
    def moving(self) -> bool:
        """Whether the FVM mesh moves."""
        return self._motion is not None

    def translation(self) -> np.ndarray:
        """Current mesh translation in m from its reference position."""
        if self._motion is None:
            return np.zeros(3)
        return np.array(self._motion.pose.translation, dtype=np.float64)

    def velocity(self, time: float) -> np.ndarray:
        """Mesh velocity in m/s at ``time``."""
        if self._motion is None:
            return np.zeros(3)
        return np.array(self._motion.kinematics.get_velocity(time), dtype=np.float64)

    def step_velocity(self, substep_times: np.ndarray) -> np.ndarray:
        """Mean mesh velocity in m/s from the accepted FVM time over the FVM substeps.

        ``substep_times`` are the end times of the coming FVM substeps. The
        VPM drifts with ``U - step_velocity`` over the exchange, so particles
        keep their exact position relative to the mesh the FVM will reach.
        """
        if self._motion is None:
            return np.zeros(3)
        pose = self._motion.accepted_pose
        start = time = self._motion.accepted_time
        for end in substep_times:
            pose = advance_pose(self._motion.kinematics, pose, time, end - time)
            time = end
        return (pose.translation - self._motion.accepted_pose.translation) / (time - start)

    def to_frame(self, points: np.ndarray) -> np.ndarray:
        """Map current FVM coordinates to frame coordinates."""
        return np.asarray(points, dtype=np.float64) - self.translation()

    def validate(self, start_time: float, end_time: float, time_step_size: float) -> None:
        """Reject mesh motion the translating frame cannot represent."""
        if self._motion is None:
            return
        if not np.allclose(self._motion.pose.rotation, np.eye(3), rtol=0.0, atol=1.0e-12):
            raise NotImplementedError(
                "FVM-VPM coupling follows translating meshes; this mesh has rotated"
            )
        n_steps = max(1, round((end_time - start_time) / time_step_size))
        for time in np.linspace(start_time, end_time, 4 * n_steps + 1):
            if np.any(self._motion.kinematics.get_angular_velocity(float(time)) != 0.0):
                raise NotImplementedError(
                    "FVM-VPM coupling follows translating meshes; the mesh kinematics "
                    f"rotate at t = {time:.6g} s, and the VPM has no rotating frame"
                )

    def maximum_relative_speed(
        self,
        freestream: Callable[[float], np.ndarray],
        start_time: float,
        end_time: float,
        time_step_size: float,
    ) -> float:
        """Largest freestream speed relative to the mesh over the coupled run.

        ``freestream`` maps a time in s to the absolute freestream in m/s.
        """
        n_steps = max(1, round((end_time - start_time) / time_step_size))
        times = np.linspace(start_time, end_time, 4 * n_steps + 1)
        return float(
            max(np.linalg.norm(freestream(time) - self.velocity(float(time))) for time in times)
        )


__all__ = ["MeshFrame"]
