"""Kinematics models for panel-coupled rigid-body kinematics."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

VectorLike = Sequence[float] | np.ndarray
ScalarOrCallable = float | Callable[[float], float]
VectorOrCallable = VectorLike | Callable[[float], VectorLike]


@dataclass
class BodyPose:
    """Complete rigid-body state for one panel body.

    ``translation`` is measured from the uploaded reference geometry.
    ``rotation_centre`` is fixed in that reference/world frame; its current
    location is ``rotation_centre + translation``.  Position is therefore
    ``R @ (x0 - c) + c + T`` and body velocity is
    ``V + omega x (x - (c + T))``.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation_centre: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def copy(self) -> BodyPose:
        """Return a deep copy suitable for a kinematics update."""
        return BodyPose(
            rotation=self.rotation.copy(),
            translation=self.translation.copy(),
            rotation_centre=self.rotation_centre.copy(),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


def _to_vec3(value: VectorLike, *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vec.size}.")
    return vec


def _eval_scalar(value: ScalarOrCallable, time: float) -> float:
    return float(value(time) if callable(value) else value)


def _eval_vec3(value: VectorOrCallable, time: float, *, name: str) -> np.ndarray:
    return _to_vec3(value(time) if callable(value) else value, name=name)


def _rotation_matrix_from_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    ax = _to_vec3(axis, name="axis")
    norm = np.linalg.norm(ax)
    if norm == 0.0:
        raise ValueError("axis must be non-zero.")
    ax = ax / norm
    x, y, z = ax
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


class PanelKinematics(abc.ABC):
    """Abstract base class for complete panel-body pose updates.

    A model first composes a :class:`BodyPose`; the solver then applies that
    pose exactly once.  Geometry mutation is deliberately absent from the
    subclasses so a rotation cannot overwrite a preceding translation.
    """

    @abc.abstractmethod
    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Return a new pose advanced by one physical time step.

        Parameters
        ----------
        time : float
            Start time of the step, in seconds.
        time_step_size : float
            Step size in seconds.
        pose : BodyPose
            Current complete pose.  Implementations must not mutate this
            object; stateful models may update their own integration state.

        Returns
        -------
        BodyPose
            Pose at ``time + time_step_size``.  Translation is in metres,
            linear velocity in m/s, angular velocity in rad/s, and rotation is
            a dimensionless ``(3, 3)`` matrix.

        Raises
        ------
        ValueError
            If a model parameter or vector has an invalid shape or magnitude.
        """

    def update(
        self, panel_solver, time: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        """Advance one body and apply its complete pose through the solver.

        Parameters
        ----------
        panel_solver : PanelSolver-like
            Object providing ``get_body_pose`` and ``apply_body_pose`` for the
            panel range.
        time : float
            Start time in seconds.
        time_step_size : float
            Physical step size in seconds.
        body_range : tuple[int, int]
            Half-open active panel range ``(start, end)`` identifying the body.

        Notes
        -----
        The solver's geometry and body-velocity fields are mutated exactly
        once.  The kinematics model may mutate its own accumulated position or
        angle; the supplied ``BodyPose`` is copied before it is returned.
        """
        pose = panel_solver.get_body_pose(body_range)
        panel_solver.apply_body_pose(
            self.advance_pose(time, time_step_size, pose),
            body_range,
        )


class StaticPanel(PanelKinematics):
    """No-kinematics model that preserves the incoming pose."""

    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Return an independent copy of ``pose`` without motion.

        ``time`` and ``time_step_size`` are accepted for protocol compatibility
        and are not used.  The returned arrays are copied, so callers can
        safely apply or modify the result without changing the input pose.
        """
        # This must be a no-op rather than an identity reset: StaticPanel can
        # be placed in a CompositePanel alongside a translating or rotating
        # component.
        return pose.copy()


class TranslatingPanel(PanelKinematics):
    """Rigid translation with a constant or time-varying linear velocity.

    ``velocity`` is sampled at both step endpoints and integrated with the
    trapezoidal rule.  The accumulated displacement is stateful and is used as
    the next pose translation.
    """

    def __init__(
        self, velocity: VectorOrCallable, initial_displacement: VectorLike = (0.0, 0.0, 0.0)
    ):
        """Create a translating body model.

        Parameters
        ----------
        velocity : array_like, shape (3,) or callable
            Linear velocity in m/s, or a function ``velocity(time)`` returning
            that vector.
        initial_displacement : array_like, shape (3,), default=(0, 0, 0)
            Initial translation from the uploaded geometry in metres.  The
            value is copied and the model's accumulated displacement is
            subsequently mutated by :meth:`advance_pose`.

        Raises
        ------
        ValueError
            If the initial displacement or a callable result is not a 3-vector.
        """
        self.velocity = velocity
        self.displacement = _to_vec3(initial_displacement, name="initial_displacement").copy()

    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Integrate translation over one step and return the new pose.

        The endpoint velocities are evaluated in m/s at ``time`` and
        ``time + time_step_size``; displacement is advanced by their trapezoid
        in m, and the returned pose stores the final velocity in m/s.
        """
        v0 = _eval_vec3(self.velocity, time, name="velocity")
        v1 = _eval_vec3(self.velocity, time + time_step_size, name="velocity")
        self.displacement += 0.5 * (v0 + v1) * time_step_size
        return replace(
            pose,
            translation=self.displacement.copy(),
            linear_velocity=v1,
        )


class RotatingPanel(PanelKinematics):
    """Rigid rotation around a fixed axis and rotation_centre.

    For time-varying angular speed, this class integrates angle using
    trapezoidal accumulation:

        angle(time + time_step_size) = angle(time) +
        0.5 * (angular_speed(time) + angular_speed(time + time_step_size))
        * time_step_size
    """

    def __init__(
        self,
        axis: VectorLike,
        angular_speed: ScalarOrCallable,
        rotation_centre: VectorLike = (0.0, 0.0, 0.0),
        initial_angle: float = 0.0,
    ):
        """Create a rigid rotation model.

        Parameters
        ----------
        axis : array_like, shape (3,)
            Rotation axis; it is normalized internally and is dimensionless.
        angular_speed : float or callable
            Angular speed in rad/s, or ``angular_speed(time)`` in rad/s.
        rotation_centre : array_like, shape (3,), default=(0, 0, 0)
            Fixed reference-frame point about which geometry rotates, in m.
        initial_angle : float, default=0.0
            Initial right-hand-rule angle in radians.

        Raises
        ------
        ValueError
            If ``axis`` or ``rotation_centre`` is not a 3-vector, or the axis
            is zero.
        """
        self.axis = _to_vec3(axis, name="axis")
        self.angular_speed = angular_speed
        self.rotation_centre = _to_vec3(rotation_centre, name="rotation_centre")
        self.angle = float(initial_angle)

    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Integrate angular speed and return the rotated body pose.

        The angle uses a trapezoidal update in radians.  The returned pose
        contains the rotation matrix, the final angular velocity in rad/s, and
        the configured rotation centre; the input pose is not mutated.
        """
        initial_angular_speed = _eval_scalar(self.angular_speed, time)
        final_angular_speed = _eval_scalar(self.angular_speed, time + time_step_size)
        self.angle += 0.5 * (initial_angular_speed + final_angular_speed) * time_step_size

        axis_unit = self.axis / np.linalg.norm(self.axis)
        return replace(
            pose,
            rotation=_rotation_matrix_from_axis_angle(axis_unit, self.angle),
            angular_velocity=axis_unit * final_angular_speed,
            rotation_centre=self.rotation_centre.copy(),
        )


class PitchingPanel(RotatingPanel):
    """Sinusoidal pitching around a fixed axis and rotation centre.

    The angle is ``bias + amplitude*sin(frequency*time + phase)`` in radians;
    therefore ``frequency`` is angular frequency in rad/s.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        bias: float = 0.0,
        axis: VectorLike = (0.0, 1.0, 0.0),
        rotation_centre: VectorLike = (0.0, 0.0, 0.0),
    ):
        """Create a sinusoidal pitching model.

        Parameters
        ----------
        amplitude : float
            Angular amplitude in radians.
        frequency : float
            Angular frequency in rad/s.
        phase : float, default=0.0
            Initial phase in radians.
        bias : float, default=0.0
            Constant angular offset in radians.
        axis : array_like, shape (3,), default=(0, 1, 0)
            Pitch axis, normalized internally.
        rotation_centre : array_like, shape (3,), default=(0, 0, 0)
            Rotation centre in metres.
        """
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.bias = float(bias)

        def angular_velocity_function(tau):
            return self.amplitude * self.frequency * np.cos(self.frequency * tau + self.phase)

        initial_angle = self.bias + self.amplitude * np.sin(self.phase)
        super().__init__(
            axis=axis,
            angular_speed=angular_velocity_function,
            rotation_centre=rotation_centre,
            initial_angle=initial_angle,
        )


class HeavingPanel(TranslatingPanel):
    """Sinusoidal heaving translation along a specified direction.

    The displacement is ``amplitude*sin(frequency*time + phase)*direction``;
    amplitude is in metres and frequency is in rad/s.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        direction: VectorLike = (0.0, 0.0, 1.0),
    ):
        """Create a sinusoidal heave model.

        Parameters
        ----------
        amplitude : float
            Displacement amplitude in metres.
        frequency : float
            Angular frequency in rad/s.
        phase : float, default=0.0
            Initial phase in radians.
        direction : array_like, shape (3,), default=(0, 0, 1)
            Non-zero translation direction; it is normalized internally.

        Raises
        ------
        ValueError
            If ``direction`` is not a non-zero 3-vector.
        """
        direction_vec = _to_vec3(direction, name="direction")
        norm = np.linalg.norm(direction_vec)
        if norm == 0.0:
            raise ValueError("direction must be non-zero.")
        self.direction = direction_vec / norm
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)

        def velocity_function(tau):
            return (
                self.direction
                * self.amplitude
                * self.frequency
                * np.cos(self.frequency * tau + self.phase)
            )

        super().__init__(velocity=velocity_function, initial_displacement=(0.0, 0.0, 0.0))


class ManeuverPanel(PanelKinematics):
    """Compose optional translation and rotation models into one pose update."""

    def __init__(
        self,
        translation: TranslatingPanel | None = None,
        rotation: RotatingPanel | None = None,
    ):
        """Create a combined rigid-body maneuver.

        Parameters
        ----------
        translation : TranslatingPanel, optional
            Translation model applied first.
        rotation : RotatingPanel, optional
            Rotation model applied second.

        Notes
        -----
        The component objects are retained, so their accumulated displacement
        and angle persist across steps.  At least one component is normally
        expected; with both omitted the pose is simply copied.
        """
        self.translation = translation
        self.rotation = rotation

    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Apply translation then rotation and return the composed pose."""
        result = pose
        if self.translation is not None:
            result = self.translation.advance_pose(time, time_step_size, result)
        if self.rotation is not None:
            result = self.rotation.advance_pose(time, time_step_size, result)
        return result


class CompositePanel(PanelKinematics):
    """Compose an ordered sequence of pose-producing kinematics models."""

    def __init__(self, components: Iterable[PanelKinematics]):
        """Create a composite model.

        Parameters
        ----------
        components : iterable of PanelKinematics
            Models applied in the supplied order.  The iterable is consumed
            immediately and stored as a list; component state remains shared.
        """
        self.components = list(components)

    def advance_pose(self, time: float, time_step_size: float, pose: BodyPose) -> BodyPose:
        """Apply every component in order and return the resulting pose."""
        result = pose
        for component in self.components:
            result = component.advance_pose(time, time_step_size, result)
        return result


class RampedRotatingPanel(RotatingPanel):
    """Rotation whose angular speed ramps linearly from zero to a target."""

    def __init__(
        self,
        axis: VectorLike,
        target_angular_speed: float,
        ramp_duration: float,
        rotation_centre: VectorLike = (0.0, 0.0, 0.0),
    ):
        """Create a linearly ramped rotation model.

        Parameters
        ----------
        axis : array_like, shape (3,)
            Rotation axis, normalized internally.
        target_angular_speed : float
            Final angular speed in rad/s.
        ramp_duration : float
            Time to reach the target in seconds.  Values at or below zero are
            clamped internally to a small positive duration.
        rotation_centre : array_like, shape (3,), default=(0, 0, 0)
            Rotation centre in metres.
        """
        self.target_angular_speed = float(target_angular_speed)
        self.ramp_duration = max(float(ramp_duration), 1e-12)

        def angular_velocity_function(time: float) -> float:
            ramp_fraction = min(max(time / self.ramp_duration, 0.0), 1.0)
            return ramp_fraction * self.target_angular_speed

        super().__init__(
            axis=axis,
            angular_speed=angular_velocity_function,
            rotation_centre=rotation_centre,
        )
