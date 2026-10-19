"""The frame of a moving FVM mesh in which the VPM is solved, and the body axes.

A uniform frame translation leaves the vorticity equation unchanged: particle
strengths, induced velocities and diffusion are identical, and only the
particle drift changes by the frame velocity. The coupled VPM therefore runs
in the frame translating with the rotation centre of the FVM mesh, with the
freestream ``U(t) - V(t)``. The frame does not rotate, so the particles keep
the inertial axes and their absolute vorticity.

The coupling geometry built from the FVM mesh (transfer lattice, interpolation
stencils, walls) is kept in body coordinates, the mesh coordinates at the start
of the run. In the frame, body coordinates are a rotation ``R^T`` about the
rotation centre; for a translating mesh the two coincide. Boundary data and
transferred velocities are converted when they pass between the solvers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.kinematics import RigidPose, advance_pose

if TYPE_CHECKING:
    from openonda.fvm import FVMSolver


class MeshFrame:
    """Translation and rotation of the FVM mesh seen from the coupling.

    Parameters
    ----------
    fvm_solver : FVMSolver
        Solver whose ``mesh_motion`` defines the frame; a static mesh gives
        the inertial frame.

    Notes
    -----
    Frame coordinates are inertial coordinates less the displacement of the
    rotation centre since the start of the run. Body coordinates are the
    placed mesh coordinates at the start of the run. Methods taking ``pose``
    default to the current mesh pose.
    """

    def __init__(self, fvm_solver: FVMSolver) -> None:
        self._motion = fvm_solver.mesh_motion
        self.rotates = False
        self._centre = (
            np.zeros(3)
            if self._motion is None
            else np.asarray(self._motion.kinematics.rotation_centre, dtype=np.float64)
        )

    @property
    def moving(self) -> bool:
        """Whether the FVM mesh moves."""
        return self._motion is not None

    def _pose(self, pose: RigidPose | None) -> RigidPose:
        if pose is not None:
            return pose
        return RigidPose() if self._motion is None else self._motion.pose

    def offset(self, pose: RigidPose | None = None) -> np.ndarray:
        """Displacement in m of the rotation centre from its start position."""
        pose = self._pose(pose)
        return pose.rotation @ self._centre + pose.translation - self._centre

    def translation(self) -> np.ndarray:
        """Current translation in m of the frame from its start position."""
        return np.array(self.offset(), dtype=np.float64)

    def rotation(self, pose: RigidPose | None = None) -> np.ndarray:
        """Rotation from body axes to the frame (inertial) axes."""
        return np.array(self._pose(pose).rotation, dtype=np.float64)

    def velocity(self, time: float) -> np.ndarray:
        """Frame velocity, the rotation-centre velocity, in m/s at ``time``."""
        if self._motion is None:
            return np.zeros(3)
        return np.array(self._motion.kinematics.get_velocity(time), dtype=np.float64)

    def angular_velocity(self, time: float) -> np.ndarray:
        """Mesh angular velocity in rad/s at ``time``."""
        if self._motion is None:
            return np.zeros(3)
        return np.array(self._motion.kinematics.get_angular_velocity(time), dtype=np.float64)

    def pose_after(self, step_end_times: np.ndarray) -> RigidPose:
        """Mesh pose after the coming FVM steps ending at ``step_end_times``.

        The pose advances from the accepted pose step by step, as the FVM will
        move it.
        """
        if self._motion is None:
            return RigidPose()
        pose = self._motion.accepted_pose
        time = self._motion.accepted_time
        for end in step_end_times:
            pose = advance_pose(self._motion.kinematics, pose, time, end - time)
            time = end
        return pose

    def step_velocity(self, step_end_times: np.ndarray) -> np.ndarray:
        """Mean frame velocity in m/s from the accepted time over the coming FVM steps.

        ``step_end_times`` are the end times of the coming FVM steps. The VPM
        drifts with ``U - step_velocity`` over the shared time step, so
        particles keep their exact position relative to the rotation centre
        the FVM will reach.
        """
        if self._motion is None:
            return np.zeros(3)
        start = self._motion.accepted_time
        displacement = self.offset(self.pose_after(step_end_times)) - self.offset(
            self._motion.accepted_pose
        )
        return displacement / (step_end_times[-1] - start)

    def to_frame(self, points: np.ndarray) -> np.ndarray:
        """Map current FVM coordinates to frame coordinates."""
        return np.asarray(points, dtype=np.float64) - self.translation()

    def to_body(self, points: np.ndarray, pose: RigidPose | None = None) -> np.ndarray:
        """Map frame coordinates, shape ``(n, 3)``, to body coordinates."""
        points = np.asarray(points, dtype=np.float64)
        if not self.rotates:
            return points
        return (points - self._centre) @ self.rotation(pose) + self._centre

    def from_body(self, points: np.ndarray, pose: RigidPose | None = None) -> np.ndarray:
        """Map body coordinates, shape ``(n, 3)``, to frame coordinates."""
        points = np.asarray(points, dtype=np.float64)
        if not self.rotates:
            return points
        return (points - self._centre) @ self.rotation(pose).T + self._centre

    def vectors_to_body(self, vectors: np.ndarray, pose: RigidPose | None = None) -> np.ndarray:
        """Express frame-axis vectors, shape ``(n, 3)``, in body axes."""
        vectors = np.asarray(vectors, dtype=np.float64)
        return vectors @ self.rotation(pose) if self.rotates else vectors

    def vectors_from_body(self, vectors: np.ndarray, pose: RigidPose | None = None) -> np.ndarray:
        """Express body-axis vectors, shape ``(n, 3)``, in frame axes."""
        vectors = np.asarray(vectors, dtype=np.float64)
        return vectors @ self.rotation(pose).T if self.rotates else vectors

    def _sample_poses(self, start_time: float, end_time: float, time_step_size: float):
        """Times and poses over the coupled run, four samples per time step."""
        n_steps = max(1, round((end_time - start_time) / time_step_size))
        times = np.linspace(start_time, end_time, 4 * n_steps + 1)
        pose = self._motion.accepted_pose
        poses = [pose]
        for begin, end in zip(times[:-1], times[1:], strict=True):
            pose = advance_pose(self._motion.kinematics, pose, float(begin), float(end - begin))
            poses.append(pose)
        return times, poses

    def validate(
        self, start_time: float, end_time: float, time_step_size: float, *, planar: bool
    ) -> None:
        """Record whether the mesh rotates and reject rotations a planar flow cannot take.

        ``planar`` marks particles in one layer between slip planes normal to
        z; the mesh may then only rotate about z.
        """
        if self._motion is None:
            return
        times, poses = self._sample_poses(start_time, end_time, time_step_size)
        identity = np.eye(3)
        self.rotates = any(
            not np.allclose(pose.rotation, identity, rtol=0.0, atol=1.0e-12) for pose in poses
        ) or any(np.any(self.angular_velocity(float(time)) != 0.0) for time in times)
        if not planar:
            return
        for time, pose in zip(times, poses, strict=True):
            tilted = np.abs(pose.rotation[:, 2] - identity[:, 2]).max() > 1.0e-12
            if tilted or np.any(self.angular_velocity(float(time))[:2] != 0.0):
                raise ValueError(
                    "The particles of a planar coupled flow lie between slip planes "
                    f"normal to z; the FVM mesh may only rotate about z (t = {time:.6g} s)"
                )

    def swept_bounds(
        self, box: np.ndarray, start_time: float, end_time: float, time_step_size: float
    ) -> np.ndarray:
        """Frame bounds of a body-coordinate box over all poses of the coupled run."""
        box = np.asarray(box, dtype=np.float64).reshape(6)
        if not self.rotates:
            return box
        corners = np.array(np.meshgrid(*box.reshape(3, 2), indexing="ij")).reshape(3, -1).T
        _, poses = self._sample_poses(start_time, end_time, time_step_size)
        points = np.concatenate([self.from_body(corners, pose) for pose in poses])
        return np.ravel(np.column_stack((points.min(axis=0), points.max(axis=0))))

    def maximum_relative_speed(
        self,
        freestream: Callable[[float], np.ndarray],
        start_time: float,
        end_time: float,
        time_step_size: float,
        box: np.ndarray,
    ) -> float:
        """Largest freestream speed relative to the mesh in ``box`` over the coupled run.

        ``freestream`` maps a time in s to the absolute freestream in m/s;
        ``box`` holds body coordinates in m.
        """
        box = np.asarray(box, dtype=np.float64).reshape(6)
        corners = np.array(np.meshgrid(*box.reshape(3, 2), indexing="ij")).reshape(3, -1).T
        arm = float(np.max(np.linalg.norm(corners - self._centre, axis=1)))
        n_steps = max(1, round((end_time - start_time) / time_step_size))
        times = np.linspace(start_time, end_time, 4 * n_steps + 1)
        return float(
            max(
                np.linalg.norm(freestream(time) - self.velocity(float(time)))
                + arm * np.linalg.norm(self.angular_velocity(float(time)))
                for time in times
            )
        )


__all__ = ["MeshFrame"]
