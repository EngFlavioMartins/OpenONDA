"""
Prescribed rigid-body kinematics for a moving FVM mesh: static, translating,
rotating, manoeuvring, heaving, pitching, periodic and composite motion.

The classes mirror the VLM kinematics in
:mod:`openonda.vpm.boundary_elements.vlm.coupling.kinematics`. Each one gives
the translational velocity of its rotation centre, the angular velocity, and
the rigid increment of the mesh pose over one time step. Motions with a closed
form (translation, fixed-axis rotation, sinusoidal heave and pitch, smooth
ramp) advance the pose exactly; motions defined by velocity functions
(manoeuvre, composite) use the interval-midpoint rule of the VLM solver.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: October 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
import math

import numpy as np
from numpy.typing import ArrayLike


def _vector(name: str, value: ArrayLike, *, unit: bool = False) -> tuple[float, float, float]:
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
    """Rigid placement ``x = rotation @ x_reference + translation`` of the mesh.

    Attributes
    ----------
    rotation : ndarray, shape (3, 3)
        Proper rotation from the reference mesh to the current mesh.
    translation : ndarray, shape (3,)
        Translation in m applied after the rotation.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map reference points, shape ``(n, 3)``, to the current mesh."""
        return points @ self.rotation.T + self.translation

    def is_identity(self) -> bool:
        """Return whether the pose leaves every reference point in place."""
        return bool(np.array_equal(self.rotation, np.eye(3)) and not np.any(self.translation))


class MeshKinematics(ABC):
    """Prescribed rigid motion of a finite-volume mesh.

    The mesh velocity at a point ``x`` is

    ``u_mesh(x, t) = V(t) + Omega(t) x (x - c(t))``,

    with ``V`` from :meth:`get_velocity`, ``Omega`` from
    :meth:`get_angular_velocity` in rad/s, and ``c(t)`` the current position
    of :attr:`rotation_centre`, which translates with the mesh. Times are in
    seconds and the mesh as constructed is the configuration at the run start.

    Subclasses provide :meth:`increment`, the rotation about the current
    rotation centre and the displacement of that centre over one interval.
    The default integrates the velocity functions at the interval midpoint,
    as the VLM solver does.
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


class _ExactTranslation(MeshKinematics):
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


class _ExactAxisRotation(MeshKinematics):
    """Rotation about a fixed axis with a closed-form angle ``theta(t)``."""

    @property
    @abstractmethod
    def rotation_axis_vector(self) -> tuple[float, float, float]:
        """Unit rotation axis."""

    @abstractmethod
    def rotation_angle(self, time: float) -> float:
        """Return the closed-form rotation angle in rad, up to a constant."""

    def get_velocity(self, time: float) -> np.ndarray:
        """Return zero translational velocity of the fixed rotation centre."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Rotate by the exact angle change over the interval."""
        angle = self.rotation_angle(time + time_step_size) - self.rotation_angle(time)
        return axis_rotation_matrix(self.rotation_axis_vector, angle), np.zeros(3)


@dataclass(frozen=True)
class StaticMotion(MeshKinematics):
    """Keep the mesh fixed in its constructed position.

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


@dataclass(frozen=True)
class TranslatingMotion(_ExactTranslation):
    """Translate the mesh with constant velocity.

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
        object.__setattr__(self, "velocity", _vector("TranslatingMotion.velocity", self.velocity))

    def get_velocity(self, time: float) -> np.ndarray:
        """Return the constant velocity in m/s."""
        return np.array(self.velocity)

    def displacement(self, time: float) -> np.ndarray:
        """Return ``velocity * time`` in m."""
        return np.array(self.velocity) * time


@dataclass(frozen=True)
class RotatingMotion(_ExactAxisRotation):
    """Rotate the mesh about a fixed axis at constant angular speed.

    Parameters
    ----------
    angular_speed : float
        Signed angular speed in rad/s about ``axis``.
    axis : array-like, shape (3,)
        Non-zero global rotation axis, normalized internally.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Fixed pivot in m.
    acceleration_time : float, default=0
        Duration in s of a sin² spin-up from rest; zero starts at full speed.
    """

    angular_speed: float
    axis: tuple[float, float, float]
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angular_speed", _finite("RotatingMotion.angular_speed", self.angular_speed)
        )
        object.__setattr__(self, "axis", _vector("RotatingMotion.axis", self.axis, unit=True))
        object.__setattr__(
            self, "rotation_centre", _vector("RotatingMotion.rotation_centre", self.rotation_centre)
        )
        object.__setattr__(
            self,
            "acceleration_time",
            _finite("RotatingMotion.acceleration_time", self.acceleration_time, minimum=0.0),
        )

    @property
    def rotation_axis_vector(self) -> tuple[float, float, float]:
        """Unit rotation axis."""
        return self.axis

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return the angular velocity in rad/s, including the smooth spin-up."""
        factor = (
            1.0
            if self.acceleration_time == 0.0
            else math.sin(0.5 * math.pi * min(max(time / self.acceleration_time, 0.0), 1.0)) ** 2
        )
        return factor * self.angular_speed * np.array(self.axis)

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


@dataclass(frozen=True)
class ManeuverMotion(MeshKinematics):
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
        Initial pivot in m; it translates with the mesh.

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
            self, "rotation_centre", _vector("ManeuverMotion.rotation_centre", self.rotation_centre)
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


@dataclass(frozen=True)
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
            self, "direction", _vector("HeavingMotion.direction", self.direction, unit=True)
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


@dataclass(frozen=True)
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
            _vector("PitchingMotion.rotation_axis", self.rotation_axis, unit=True),
        )
        object.__setattr__(
            self, "rotation_centre", _vector("PitchingMotion.rotation_centre", self.rotation_centre)
        )

    @property
    def rotation_axis_vector(self) -> tuple[float, float, float]:
        """Unit pitch axis."""
        return self.rotation_axis

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return ``theta0 omega cos(omega t + phase)`` about the axis in rad/s."""
        omega = 2.0 * math.pi * self.frequency
        rate = math.radians(self.amplitude_degrees) * omega * math.cos(omega * time + self.phase)
        return rate * np.array(self.rotation_axis)

    def rotation_angle(self, time: float) -> float:
        """Return ``theta0 sin(omega t + phase)`` in rad."""
        omega = 2.0 * math.pi * self.frequency
        return math.radians(self.amplitude_degrees) * math.sin(omega * time + self.phase)


@dataclass(frozen=True)
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
            self, "direction", _vector("LinearPeriodicMotion.direction", self.direction, unit=True)
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


@dataclass(frozen=True)
class CompositeMotion(MeshKinematics):
    """Superpose several motions about one common rotation centre.

    Velocities and angular velocities of the components are summed; each
    component's own rotation centre is replaced by ``rotation_centre``, so a
    heave-plus-pitch motion pivots about one point. The pose advances with
    the interval-midpoint rule.

    Parameters
    ----------
    kinematics_components : sequence[MeshKinematics]
        Motions to combine.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Common pivot in m; it translates with the mesh.
    """

    kinematics_components: tuple[MeshKinematics, ...]
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        components = tuple(self.kinematics_components)
        if not components or not all(isinstance(item, MeshKinematics) for item in components):
            raise TypeError("CompositeMotion requires one or more MeshKinematics components")
        object.__setattr__(self, "kinematics_components", components)
        object.__setattr__(
            self,
            "rotation_centre",
            _vector("CompositeMotion.rotation_centre", self.rotation_centre),
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return the summed component velocity in m/s."""
        return np.sum([item.get_velocity(time) for item in self.kinematics_components], axis=0)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return the summed component angular velocity in rad/s."""
        return np.sum(
            [item.get_angular_velocity(time) for item in self.kinematics_components], axis=0
        )


@dataclass(frozen=True)
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
            _vector("SmoothRampMotion.final_velocity", self.final_velocity),
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
    kinematics: MeshKinematics, pose: RigidPose, time: float, time_step_size: float
) -> RigidPose:
    """Advance ``pose`` at ``time`` over one interval of the kinematics.

    The increment rotates about the current position of the rotation centre
    and then translates that centre. The rotation is re-orthonormalized so
    round-off does not accumulate over long runs.
    """
    centre_reference = np.asarray(kinematics.rotation_centre, dtype=np.float64)
    centre = pose.rotation @ centre_reference + pose.translation
    increment_rotation, displacement = kinematics.increment(time, time_step_size)
    left, _, right = np.linalg.svd(increment_rotation @ pose.rotation)
    rotation = left @ right
    return RigidPose(rotation, centre + displacement - rotation @ centre_reference)


def placement_pose(
    translation: Sequence[float] | None,
    rotation_degrees: Sequence[float] | None,
    rotation_centre: Sequence[float] | None,
) -> RigidPose:
    """Return the initial placement used by :class:`MeshMotion`.

    The rotations about global x, then y, then z (degrees) act about
    ``rotation_centre`` and are followed by ``translation``, the convention of
    :class:`~openonda.vpm.boundary_elements.vlm.config.VLMSurfaceSetup`.
    """
    rotation = np.eye(3)
    if rotation_degrees is not None:
        for axis, angle in zip(np.eye(3), rotation_degrees, strict=True):
            rotation = axis_rotation_matrix(axis, math.radians(angle)) @ rotation
    centre = np.zeros(3) if rotation_centre is None else np.asarray(rotation_centre, dtype=float)
    shift = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    return RigidPose(rotation, centre - rotation @ centre + shift)


@dataclass(frozen=True)
class MeshMotion:
    """Prescribed rigid-body motion of the whole FVM mesh.

    The mesh, every boundary patch on it, and every moment centre on it move
    with ``kinematics``. Wall patches become moving no-slip walls; inlet,
    outlet and freestream values stay in the absolute (inertial) frame. Cell
    velocity and pressure are absolute-frame quantities.

    Parameters
    ----------
    kinematics : MeshKinematics
        Motion of the mesh after its initial placement.
    translation : array-like, shape (3,) or None
        Initial translation of the constructed mesh in m.
    rotation_degrees : array-like, shape (3,) or None
        Initial rotations about global x, y and z in degrees.
    rotation_centre : array-like, shape (3,) or None
        Pivot of the initial rotation in m, in mesh coordinates.

    Notes
    -----
    The rotation centre of ``kinematics`` is given in the placed (global)
    coordinates. A mesh with ``empty`` patches (a 2-D model) may only move in
    their plane.
    """

    kinematics: MeshKinematics
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kinematics, MeshKinematics):
            raise TypeError("MeshMotion.kinematics must be a MeshKinematics instance")
        for name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(f"MeshMotion.{name}", value))

    def placement(self) -> RigidPose:
        """Return the initial placement pose of the constructed mesh."""
        return placement_pose(self.translation, self.rotation_degrees, self.rotation_centre)


def _kinematics_to_dict(kinematics: MeshKinematics) -> dict[str, object]:
    """Describe one kinematics object by class name and constructor fields."""
    if isinstance(kinematics, ManeuverMotion) and (
        kinematics.velocity_function is not None or kinematics.angular_velocity_function is not None
    ):
        raise TypeError("ManeuverMotion velocity functions cannot be written to a setup file")
    data: dict[str, object] = {"type": type(kinematics).__name__}
    for item in fields(kinematics):
        value = getattr(kinematics, item.name)
        if item.name == "kinematics_components":
            value = [_kinematics_to_dict(component) for component in value]
        data[item.name] = value
    return data


def _kinematics_from_dict(data: dict[str, object]) -> MeshKinematics:
    """Rebuild one kinematics object written by :func:`_kinematics_to_dict`."""
    values = dict(data)
    cls = _KINEMATICS_TYPES.get(str(values.pop("type", "")))
    if cls is None:
        raise ValueError(f"Unknown mesh kinematics type {data.get('type')!r}")
    if "kinematics_components" in values:
        values["kinematics_components"] = tuple(
            _kinematics_from_dict(item) for item in values["kinematics_components"]
        )
    return cls(**values)


def mesh_motion_to_dict(motion: MeshMotion) -> dict[str, object]:
    """Return a JSON-compatible description of a mesh motion."""
    return {
        "kinematics": _kinematics_to_dict(motion.kinematics),
        "translation": motion.translation,
        "rotation_degrees": motion.rotation_degrees,
        "rotation_centre": motion.rotation_centre,
    }


def mesh_motion_from_dict(data: dict[str, object]) -> MeshMotion:
    """Rebuild a mesh motion written by :func:`mesh_motion_to_dict`."""
    values = dict(data)
    values["kinematics"] = _kinematics_from_dict(values["kinematics"])
    return MeshMotion(**values)


_KINEMATICS_TYPES = {
    cls.__name__: cls
    for cls in (
        StaticMotion,
        TranslatingMotion,
        RotatingMotion,
        ManeuverMotion,
        HeavingMotion,
        PitchingMotion,
        LinearPeriodicMotion,
        CompositeMotion,
        SmoothRampMotion,
    )
}


__all__ = [
    "CompositeMotion",
    "HeavingMotion",
    "LinearPeriodicMotion",
    "ManeuverMotion",
    "MeshKinematics",
    "MeshMotion",
    "PitchingMotion",
    "RigidPose",
    "RotatingMotion",
    "SmoothRampMotion",
    "StaticMotion",
    "TranslatingMotion",
    "advance_pose",
    "axis_rotation_matrix",
    "mesh_motion_from_dict",
    "mesh_motion_to_dict",
    "placement_pose",
    "rotation_vector_matrix",
]
