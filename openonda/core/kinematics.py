"""Prescribed rigid-body motion shared by the FVM mesh and the VLM surfaces.

Each motion gives the translational velocity of its rotation centre, the
angular velocity, and the rigid increment of the pose over one time step.
Motions with a closed form (translation, fixed-axis rotation, sinusoidal heave
and pitch, smooth ramp) advance the pose exactly; motions defined by velocity
functions (manoeuvre, composite) use the interval-midpoint rule.

The motion objects are immutable. The pose they advance belongs to the solver
that moves the body: :class:`~openonda.fvm.motion.rigid.RigidMeshMotion` for
the finite-volume mesh and :class:`~openonda.vpm.boundary_elements.vlm.VLMSolver`
for each lifting surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "CompositeMotion",
    "HeavingMotion",
    "LinearPeriodicMotion",
    "ManeuverMotion",
    "PitchingMotion",
    "RigidMotion",
    "RigidPose",
    "RotatingMotion",
    "SmoothRampMotion",
    "StaticMotion",
    "TranslatingMotion",
    "advance_pose",
    "axis_rotation_matrix",
    "finite_vector",
    "rotation_vector_matrix",
]


def finite_vector(name: str, value: ArrayLike, *, unit: bool = False) -> tuple[float, float, float]:
    """Return a finite three-component tuple, normalized when ``unit`` is set."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite components")
    if unit:
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            raise ValueError(f"{name} must be non-zero")
        array = array / norm
    return tuple(float(item) for item in array)


def _finite(name: str, value: float, *, minimum: float | None = None) -> float:
    """Return a finite float, optionally bounded below."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def axis_rotation_matrix(axis: ArrayLike, angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about a unit ``axis`` (Rodrigues' formula)."""
    a = np.asarray(axis, dtype=np.float64)
    skew = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def rotation_vector_matrix(rotation_vector: ArrayLike) -> np.ndarray:
    """Rotation matrix of a rotation vector ``angle * axis`` in radians."""
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-14:
        return np.eye(3)
    return axis_rotation_matrix(vector / angle, angle)


@dataclass(frozen=True, eq=False)
class RigidPose:
    """Rigid placement ``x = rotation @ x_reference + translation`` of a body.

    Attributes
    ----------
    rotation : ndarray, shape (3, 3)
        Proper rotation from the reference configuration to the current one.
    translation : ndarray, shape (3,)
        Translation in m applied after the rotation.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map reference points, shape ``(n, 3)``, to the current configuration."""
        return points @ self.rotation.T + self.translation

    def is_identity(self) -> bool:
        """Return whether the pose leaves every reference point in place."""
        return bool(np.array_equal(self.rotation, np.eye(3)) and not np.any(self.translation))


class RigidMotion(ABC):
    """Prescribed rigid motion of a body.

    The velocity of a point ``x`` of the body is

    ``u(x, t) = V(t) + Omega(t) x (x - c(t))``,

    with ``V`` from :meth:`get_velocity`, ``Omega`` from
    :meth:`get_angular_velocity` in rad/s, and ``c(t)`` the current position
    of :attr:`rotation_centre`, which translates with the body. Times are in
    seconds and the body as placed is the configuration at the run start.

    Subclasses provide :meth:`increment`, the rotation about the current
    rotation centre and the displacement of that centre over one interval.
    The default integrates the velocity functions at the interval midpoint.
    """

    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @abstractmethod
    def get_velocity(self, time: float) -> np.ndarray:
        """Return the rotation-centre velocity ``[Vx, Vy, Vz]`` in m/s at ``time``."""

    @abstractmethod
    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return the global angular velocity ``[Wx, Wy, Wz]`` in rad/s at ``time``."""

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the rotation and the centre displacement over ``[t, t + dt]``.

        Returns
        -------
        tuple[ndarray, ndarray]
            Rotation matrix, shape ``(3, 3)``, about the current rotation
            centre, and the displacement of that centre in m, shape ``(3,)``.
        """
        midpoint = time + 0.5 * time_step_size
        rotation = rotation_vector_matrix(self.get_angular_velocity(midpoint) * time_step_size)
        return rotation, self.get_velocity(midpoint) * time_step_size


class _ExactTranslation(RigidMotion):
    """Translation with a closed-form displacement ``D(t)``."""

    @abstractmethod
    def displacement(self, time: float) -> np.ndarray:
        """Return the closed-form centre displacement in m, up to a constant."""

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return zero angular velocity."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Translate by the exact displacement change over the interval."""
        return np.eye(3), self.displacement(time + time_step_size) - self.displacement(time)


class _ExactAxisRotation(RigidMotion):
    """Rotation about a fixed axis with a closed-form angle ``theta(t)``."""

    rotation_axis: tuple[float, float, float]

    @abstractmethod
    def rotation_angle(self, time: float) -> float:
        """Return the closed-form rotation angle in rad, up to a constant."""

    def get_velocity(self, time: float) -> np.ndarray:
        """Return zero translational velocity of the fixed rotation centre."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Rotate by the exact angle change over the interval."""
        angle = self.rotation_angle(time + time_step_size) - self.rotation_angle(time)
        return axis_rotation_matrix(self.rotation_axis, angle), np.zeros(3)


@dataclass(frozen=True, slots=True)
class StaticMotion(RigidMotion):
    """Keep the body fixed in its placed position.

    Examples
    --------
    >>> StaticMotion().get_velocity(time=1.0)
    array([0., 0., 0.])
    """

    def get_velocity(self, time: float) -> np.ndarray:
        """Return zero velocity."""
        return np.zeros(3)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return zero angular velocity."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the identity increment."""
        return np.eye(3), np.zeros(3)


@dataclass(frozen=True, slots=True)
class TranslatingMotion(_ExactTranslation):
    """Translate the body with constant velocity.

    Parameters
    ----------
    velocity : array-like, shape (3,)
        Constant global velocity in m/s.

    Examples
    --------
    >>> TranslatingMotion(velocity=(-1.0, 0.0, 0.0)).get_velocity(time=2.0)
    array([-1.,  0.,  0.])
    """

    velocity: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "velocity", finite_vector("TranslatingMotion.velocity", self.velocity)
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return the constant velocity in m/s."""
        return np.array(self.velocity)

    def displacement(self, time: float) -> np.ndarray:
        """Return ``velocity * time`` in m."""
        return np.array(self.velocity) * time


@dataclass(frozen=True, slots=True)
class RotatingMotion(_ExactAxisRotation):
    """Rotate the body about a fixed axis at constant angular speed.

    Parameters
    ----------
    angular_speed : float
        Signed angular speed in rad/s about ``rotation_axis``.
    rotation_axis : array-like, shape (3,)
        Non-zero global rotation axis, normalized internally.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Fixed pivot in m.
    acceleration_time : float, default=0
        Duration in s of a sin² spin-up from rest; zero starts at full speed.
    """

    angular_speed: float
    rotation_axis: tuple[float, float, float]
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angular_speed", _finite("RotatingMotion.angular_speed", self.angular_speed)
        )
        object.__setattr__(
            self,
            "rotation_axis",
            finite_vector("RotatingMotion.rotation_axis", self.rotation_axis, unit=True),
        )
        object.__setattr__(
            self,
            "rotation_centre",
            finite_vector("RotatingMotion.rotation_centre", self.rotation_centre),
        )
        object.__setattr__(
            self,
            "acceleration_time",
            _finite("RotatingMotion.acceleration_time", self.acceleration_time, minimum=0.0),
        )

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return the angular velocity in rad/s, including the smooth spin-up."""
        factor = (
            1.0
            if self.acceleration_time == 0.0
            else math.sin(0.5 * math.pi * min(max(time / self.acceleration_time, 0.0), 1.0)) ** 2
        )
        return factor * self.angular_speed * np.array(self.rotation_axis)

    def rotation_angle(self, time: float) -> float:
        """Integrated rotation angle in rad since ``t = 0``."""
        if self.acceleration_time == 0.0:
            return self.angular_speed * time
        ramp = min(max(time, 0.0), self.acceleration_time)
        spin_up = 0.5 * (
            ramp
            - self.acceleration_time / math.pi * math.sin(math.pi * ramp / self.acceleration_time)
        )
        return self.angular_speed * (spin_up + max(time - self.acceleration_time, 0.0))


@dataclass(frozen=True, slots=True)
class ManeuverMotion(RigidMotion):
    """Drive arbitrary rigid motion with user-supplied velocity functions.

    Parameters
    ----------
    velocity_function : callable or None
        ``f(time) -> array-like (3,)``, velocity of the rotation centre in m/s.
        ``None`` gives zero.
    angular_velocity_function : callable or None
        ``g(time) -> array-like (3,)``, global angular velocity in rad/s.
        ``None`` gives zero.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Initial pivot in m; it translates with the body.

    Notes
    -----
    Each step samples both functions at the interval midpoint. The functions
    must be deterministic for reproducible restarts.
    """

    velocity_function: Callable[[float], ArrayLike] | None = None
    angular_velocity_function: Callable[[float], ArrayLike] | None = None
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("velocity_function", "angular_velocity_function"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"ManeuverMotion.{name} must be callable or None")
        object.__setattr__(
            self,
            "rotation_centre",
            finite_vector("ManeuverMotion.rotation_centre", self.rotation_centre),
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Evaluate the velocity function in m/s."""
        if self.velocity_function is None:
            return np.zeros(3)
        return np.array(self.velocity_function(time), dtype=np.float64).reshape(3)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Evaluate the angular-velocity function in rad/s."""
        if self.angular_velocity_function is None:
            return np.zeros(3)
        return np.array(self.angular_velocity_function(time), dtype=np.float64).reshape(3)


@dataclass(frozen=True, slots=True)
class HeavingMotion(_ExactTranslation):
    """Sinusoidal heave ``z(t) = A sin(2 pi f t + phase)`` along ``direction``.

    Parameters
    ----------
    amplitude : float
        Heave amplitude in m.
    frequency : float
        Heave frequency in Hz.
    phase : float, default=0
        Phase in rad; ``0`` starts at the mean position with maximum velocity.
    direction : array-like, shape (3,), default=(0, 0, 1)
        Heave direction, normalized internally.
    """

    amplitude: float
    frequency: float
    phase: float = 0.0
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", _finite("HeavingMotion.amplitude", self.amplitude))
        object.__setattr__(
            self, "frequency", _finite("HeavingMotion.frequency", self.frequency, minimum=0.0)
        )
        object.__setattr__(self, "phase", _finite("HeavingMotion.phase", self.phase))
        object.__setattr__(
            self, "direction", finite_vector("HeavingMotion.direction", self.direction, unit=True)
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return ``A omega cos(omega t + phase)`` along the direction in m/s."""
        omega = 2.0 * math.pi * self.frequency
        speed = self.amplitude * omega * math.cos(omega * time + self.phase)
        return speed * np.array(self.direction)

    def displacement(self, time: float) -> np.ndarray:
        """Return ``A sin(omega t + phase)`` along the direction in m."""
        omega = 2.0 * math.pi * self.frequency
        return self.amplitude * math.sin(omega * time + self.phase) * np.array(self.direction)


@dataclass(frozen=True, slots=True)
class PitchingMotion(_ExactAxisRotation):
    """Sinusoidal pitch ``theta(t) = theta0 sin(2 pi f t + phase)`` about a fixed axis.

    Parameters
    ----------
    amplitude_degrees : float
        Peak pitch angle in degrees.
    frequency : float
        Oscillation frequency in Hz.
    phase : float, default=0
        Phase in rad.
    rotation_axis : array-like, shape (3,), default=(0, 1, 0)
        Pitch axis, normalized internally.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Pivot in m, for example the quarter chord.
    """

    amplitude_degrees: float
    frequency: float
    phase: float = 0.0
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "amplitude_degrees",
            _finite("PitchingMotion.amplitude_degrees", self.amplitude_degrees),
        )
        object.__setattr__(
            self, "frequency", _finite("PitchingMotion.frequency", self.frequency, minimum=0.0)
        )
        object.__setattr__(self, "phase", _finite("PitchingMotion.phase", self.phase))
        object.__setattr__(
            self,
            "rotation_axis",
            finite_vector("PitchingMotion.rotation_axis", self.rotation_axis, unit=True),
        )
        object.__setattr__(
            self,
            "rotation_centre",
            finite_vector("PitchingMotion.rotation_centre", self.rotation_centre),
        )

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return ``theta0 omega cos(omega t + phase)`` about the axis in rad/s."""
        omega = 2.0 * math.pi * self.frequency
        rate = math.radians(self.amplitude_degrees) * omega * math.cos(omega * time + self.phase)
        return rate * np.array(self.rotation_axis)

    def rotation_angle(self, time: float) -> float:
        """Return ``theta0 sin(omega t + phase)`` in rad."""
        omega = 2.0 * math.pi * self.frequency
        return math.radians(self.amplitude_degrees) * math.sin(omega * time + self.phase)


@dataclass(frozen=True, slots=True)
class LinearPeriodicMotion(_ExactTranslation):
    """Sinusoidal translation ``r(t) = r0 + A sin(2 pi f t + phase) d``.

    Parameters
    ----------
    amplitude : float
        Motion amplitude in m.
    frequency : float
        Motion frequency in Hz.
    direction : array-like, shape (3,)
        Motion direction, normalized internally.
    phase : float, default=0
        Phase in rad.
    """

    amplitude: float
    frequency: float
    direction: tuple[float, float, float]
    phase: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "amplitude", _finite("LinearPeriodicMotion.amplitude", self.amplitude)
        )
        object.__setattr__(
            self,
            "frequency",
            _finite("LinearPeriodicMotion.frequency", self.frequency, minimum=0.0),
        )
        object.__setattr__(
            self,
            "direction",
            finite_vector("LinearPeriodicMotion.direction", self.direction, unit=True),
        )
        object.__setattr__(self, "phase", _finite("LinearPeriodicMotion.phase", self.phase))

    def get_velocity(self, time: float) -> np.ndarray:
        """Return ``A omega cos(omega t + phase) d`` in m/s."""
        omega = 2.0 * math.pi * self.frequency
        speed = self.amplitude * omega * math.cos(omega * time + self.phase)
        return speed * np.array(self.direction)

    def displacement(self, time: float) -> np.ndarray:
        """Return ``A sin(omega t + phase) d`` in m."""
        omega = 2.0 * math.pi * self.frequency
        return self.amplitude * math.sin(omega * time + self.phase) * np.array(self.direction)


@dataclass(frozen=True, slots=True)
class CompositeMotion(RigidMotion):
    """Superpose several motions about one common rotation centre.

    Velocities and angular velocities of the components are summed; each
    component's own rotation centre is replaced by ``rotation_centre``, so a
    heave-plus-pitch motion pivots about one point. The pose advances with
    the interval-midpoint rule.

    Parameters
    ----------
    components : sequence[RigidMotion]
        Motions to combine.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Common pivot in m; it translates with the body.
    """

    components: tuple[RigidMotion, ...]
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or not all(isinstance(item, RigidMotion) for item in components):
            raise TypeError("CompositeMotion requires one or more RigidMotion components")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "rotation_centre",
            finite_vector("CompositeMotion.rotation_centre", self.rotation_centre),
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return the summed component velocity in m/s."""
        return np.sum([item.get_velocity(time) for item in self.components], axis=0)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return the summed component angular velocity in rad/s."""
        return np.sum([item.get_angular_velocity(time) for item in self.components], axis=0)


@dataclass(frozen=True, slots=True)
class SmoothRampMotion(_ExactTranslation):
    """Accelerate smoothly from rest to a constant velocity.

    ``V(t) = V_final sin²(pi (t - t0) / (2 T))`` for ``0 <= t - t0 < T`` and
    ``V_final`` afterwards, so velocity and acceleration are continuous at
    both ends of the ramp.

    Parameters
    ----------
    final_velocity : array-like, shape (3,)
        Terminal velocity in m/s.
    acceleration_time : float
        Positive ramp duration ``T`` in s.
    start_time : float, default=0
        Time ``t0`` in s at which the ramp starts.
    """

    final_velocity: tuple[float, float, float]
    acceleration_time: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "final_velocity",
            finite_vector("SmoothRampMotion.final_velocity", self.final_velocity),
        )
        duration = _finite("SmoothRampMotion.acceleration_time", self.acceleration_time)
        if duration <= 0.0:
            raise ValueError("SmoothRampMotion.acceleration_time must be positive")
        object.__setattr__(self, "acceleration_time", duration)
        object.__setattr__(
            self, "start_time", _finite("SmoothRampMotion.start_time", self.start_time)
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return the sin² ramp velocity in m/s."""
        elapsed = time - self.start_time
        if elapsed <= 0.0:
            return np.zeros(3)
        if elapsed >= self.acceleration_time:
            return np.array(self.final_velocity)
        return (
            np.array(self.final_velocity)
            * math.sin(0.5 * math.pi * elapsed / self.acceleration_time) ** 2
        )

    def displacement(self, time: float) -> np.ndarray:
        """Return the integrated ramp displacement in m."""
        elapsed = time - self.start_time
        final = np.array(self.final_velocity)
        if elapsed <= 0.0:
            return np.zeros(3)
        if elapsed >= self.acceleration_time:
            return final * (elapsed - 0.5 * self.acceleration_time)
        ramp = (
            self.acceleration_time / math.pi * math.sin(math.pi * elapsed / self.acceleration_time)
        )
        return 0.5 * final * (elapsed - ramp)


def advance_pose(
    motion: RigidMotion, pose: RigidPose, time: float, time_step_size: float
) -> RigidPose:
    """Advance ``pose`` at ``time`` over one interval of ``motion``.

    The increment rotates about the current position of the rotation centre
    and then translates that centre. The rotation is re-orthonormalized so
    round-off does not accumulate over long runs.
    """
    centre_reference = np.asarray(motion.rotation_centre, dtype=np.float64)
    centre = pose.rotation @ centre_reference + pose.translation
    increment_rotation, displacement = motion.increment(time, time_step_size)
    left, _, right = np.linalg.svd(increment_rotation @ pose.rotation)
    rotation = left @ right
    return RigidPose(rotation, centre + displacement - rotation @ centre_reference)
