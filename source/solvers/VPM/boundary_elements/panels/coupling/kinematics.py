"""Kinematics models for panel-coupled rigid-body motion."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Sequence

import numpy as np

VectorLike = Sequence[float] | np.ndarray
ScalarOrCallable = float | Callable[[float], float]
VectorOrCallable = VectorLike | Callable[[float], VectorLike]


def _to_vec3(value: VectorLike, *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vec.size}.")
    return vec


def _eval_scalar(value: ScalarOrCallable, t: float) -> float:
    return float(value(t) if callable(value) else value)


def _eval_vec3(value: VectorOrCallable, t: float, *, name: str) -> np.ndarray:
    return _to_vec3(value(t) if callable(value) else value, name=name)


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


def _call_translation_update(
    panel_solver,
    *,
    body_range: tuple[int, int],
    displacement: np.ndarray,
    linear_velocity: np.ndarray,
) -> None:
    """
    Apply rigid translation to a body range via the standard PanelSolver API.

    Standard signature::
        panel_solver.apply_translation_update(displacement, linear_velocity, body_range)
    """
    panel_solver.apply_translation_update(displacement, linear_velocity, body_range)


def _call_rotation_update(
    panel_solver,
    *,
    body_range: tuple[int, int],
    rotation_matrix: np.ndarray,
    angular_velocity: np.ndarray,
    rotation_center: np.ndarray,
) -> None:
    """
    Apply rigid rotation to a body range via the standard PanelSolver API.

    Standard signature::
        panel_solver.apply_rotation_update(
            rotation_matrix, angular_velocity, rotation_center, body_range
        )
    """
    panel_solver.apply_rotation_update(
        rotation_matrix, angular_velocity, rotation_center, body_range
    )


class PanelKinematics(abc.ABC):
    """Abstract base class for panel body kinematics updates."""

    @abc.abstractmethod
    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        """Advance body kinematics and apply the motion update on ``panel_solver``."""


class StaticPanel(PanelKinematics):
    """No-motion model."""

    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        zero = np.zeros(3, dtype=float)
        _call_translation_update(
            panel_solver,
            body_range=body_range,
            displacement=zero,
            linear_velocity=zero,
        )
        _call_rotation_update(
            panel_solver,
            body_range=body_range,
            rotation_matrix=np.eye(3, dtype=float),
            angular_velocity=zero,
            rotation_center=zero,
        )


class TranslatingPanel(PanelKinematics):
    """Rigid translation with time-varying linear velocity."""

    def __init__(
        self, velocity: VectorOrCallable, initial_displacement: VectorLike = (0.0, 0.0, 0.0)
    ):
        self.velocity = velocity
        self.displacement = _to_vec3(initial_displacement, name="initial_displacement").copy()

    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        v0 = _eval_vec3(self.velocity, t, name="velocity")
        v1 = _eval_vec3(self.velocity, t + time_step_size, name="velocity")
        self.displacement += 0.5 * (v0 + v1) * time_step_size
        _call_translation_update(
            panel_solver,
            body_range=body_range,
            displacement=self.displacement,
            linear_velocity=v1,
        )


class RotatingPanel(PanelKinematics):
    """Rigid rotation around a fixed axis and center.

    For time-varying angular speed, this class integrates angle using
    trapezoidal accumulation:

        theta(t + dt) = theta(t) + 0.5 * (omega(t) + omega(t + dt)) * dt
    """

    def __init__(
        self,
        axis: VectorLike,
        omega: ScalarOrCallable,
        center: VectorLike = (0.0, 0.0, 0.0),
        initial_angle: float = 0.0,
    ):
        self.axis = _to_vec3(axis, name="axis")
        self.omega = omega
        self.center = _to_vec3(center, name="center")
        self.angle = float(initial_angle)

    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        w0 = _eval_scalar(self.omega, t)
        w1 = _eval_scalar(self.omega, t + time_step_size)
        self.angle += 0.5 * (w0 + w1) * time_step_size

        axis_unit = self.axis / np.linalg.norm(self.axis)
        omega_vec = axis_unit * w1
        rotation_matrix = _rotation_matrix_from_axis_angle(axis_unit, self.angle)

        _call_rotation_update(
            panel_solver,
            body_range=body_range,
            rotation_matrix=rotation_matrix,
            angular_velocity=omega_vec,
            rotation_center=self.center,
        )


class PitchingPanel(RotatingPanel):
    """Sinusoidal pitching around a fixed axis/center."""

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        bias: float = 0.0,
        axis: VectorLike = (0.0, 1.0, 0.0),
        center: VectorLike = (0.0, 0.0, 0.0),
    ):
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.bias = float(bias)

        def omega_fn(tau):
            return self.amplitude * self.frequency * np.cos(self.frequency * tau + self.phase)

        initial_angle = self.bias + self.amplitude * np.sin(self.phase)
        super().__init__(axis=axis, omega=omega_fn, center=center, initial_angle=initial_angle)


class HeavingPanel(TranslatingPanel):
    """Sinusoidal heaving translation along a specified direction."""

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        direction: VectorLike = (0.0, 0.0, 1.0),
    ):
        direction_vec = _to_vec3(direction, name="direction")
        norm = np.linalg.norm(direction_vec)
        if norm == 0.0:
            raise ValueError("direction must be non-zero.")
        self.direction = direction_vec / norm
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)

        def velocity_fn(tau):
            return (
                self.direction
                * self.amplitude
                * self.frequency
                * np.cos(self.frequency * tau + self.phase)
            )

        super().__init__(velocity=velocity_fn, initial_displacement=(0.0, 0.0, 0.0))


class ManeuverPanel(PanelKinematics):
    """Combined translation and rotation for generic maneuvers."""

    def __init__(
        self,
        translation: TranslatingPanel | None = None,
        rotation: RotatingPanel | None = None,
    ):
        self.translation = translation
        self.rotation = rotation

    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        if self.translation is not None:
            self.translation.update(panel_solver, t, time_step_size, body_range)
        if self.rotation is not None:
            self.rotation.update(panel_solver, t, time_step_size, body_range)


class CompositePanel(PanelKinematics):
    """Apply a sequence of kinematics updates in order."""

    def __init__(self, components: Iterable[PanelKinematics]):
        self.components = list(components)

    def update(
        self, panel_solver, t: float, time_step_size: float, body_range: tuple[int, int]
    ) -> None:
        for component in self.components:
            component.update(panel_solver, t, time_step_size, body_range)


class Static(StaticPanel):
    """Backward-compatible alias for `StaticPanel`."""


class Plunging(HeavingPanel):
    """Backward-compatible alias for `HeavingPanel`."""

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        axis: VectorLike = (0.0, 0.0, 1.0),
        phase: float = 0.0,
    ):
        super().__init__(
            amplitude=amplitude, frequency=2.0 * np.pi * frequency, phase=phase, direction=axis
        )


class RampedRotation(RotatingPanel):
    """Backward-compatible ramped angular-speed model."""

    def __init__(
        self,
        axis: VectorLike,
        target_omega: float,
        ramp_time: float,
        center: VectorLike = (0.0, 0.0, 0.0),
    ):
        self.target_omega = float(target_omega)
        self.ramp_time = max(float(ramp_time), 1e-12)

        def omega_fn(t: float) -> float:
            ramp = min(max(t / self.ramp_time, 0.0), 1.0)
            return ramp * self.target_omega

        super().__init__(axis=axis, omega=omega_fn, center=center)
