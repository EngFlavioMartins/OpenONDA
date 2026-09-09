"""
Rigid-body kinematics for VLM lattices: static, translating, rotating,
manoeuvring, heaving, pitching, and periodic motion drivers.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..solver.vlm_solver import VLMSolver


class VLMKinematics(ABC):
    """Abstract rigid-motion contract consumed by the VLM solver.

    Subclasses provide translational and angular velocity at physical time and
    mutate an assigned panel range over one accepted interval. Users normally
    instantiate one of the concrete motion classes and attach it through
    :class:`VLMSurfaceSetup`.

    Attributes
    ----------
    current_position : ndarray, shape (3,)
        Accumulated Cartesian translation metadata in m.
    current_orientation : ndarray, shape (3, 3)
        Accumulated proper rotation matrix from reference to current frame.

    Notes
    -----
    ``time`` denotes the start of an interval and ``time_step_size`` its
    duration in s. ``update`` mutates VLM lattice geometry; velocity queries do
    not. Angular velocity vectors use rad/s in the global Cartesian frame.
    """

    @abstractmethod
    def get_velocity(self, time: float) -> np.ndarray:
        """Return Cartesian translational velocity at ``time``.

        Parameters
        ----------
        time : float
            Physical query time in s.

        Returns
        -------
        ndarray, shape (3,)
            Global ``[Vx, Vy, Vz]`` in m/s.
        """
        pass

    @abstractmethod
    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return global angular velocity at ``time``.

        ``time`` is in s; the returned NumPy vector has shape ``(3,)`` and
        units rad/s.
        """
        pass

    def __init__(self) -> None:
        """Initialize zero translation and identity orientation metadata."""
        self.current_position = np.zeros(3)
        self.current_orientation = np.eye(3)

    @abstractmethod
    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Advance assigned VLM panel geometry over one time interval.

        Parameters
        ----------
        vlm_solver : VLMSolver
            Runtime whose lattice coordinates are modified in place.
        time : float
            Start time of the interval in s.
        time_step_size : float
            Positive interval duration in s.
        panel_range : tuple[int, int] or None, default=None
            Half-open panel-index range ``[start, stop)``; ``None`` updates all
            panels represented by the lattice.
        """
        pass


class StaticVLM(VLMKinematics):
    """Keep a VLM surface fixed in its reference geometry.

    The VLM surfaces remain fixed in space. This is appropriate for
    steady-state analysis or when studying wake development without
    surface motion.

    Examples
    --------
    >>> kinematics = StaticVLM()
    >>> kinematics.get_velocity(time=1.0)
    array([0., 0., 0.])
    """

    def __init__(self) -> None:
        """Initialize static motion with zero translation and identity pose."""
        super().__init__()

    def get_velocity(self, time: float) -> np.ndarray:
        """Return a new zero Cartesian velocity vector in m/s."""
        return np.zeros(3)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return a new zero angular-velocity vector in rad/s."""
        return np.zeros(3)

    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Leave lattice geometry and pose metadata unchanged."""
        pass


class TranslatingVLM(VLMKinematics):
    """Translate VLM panels with constant Cartesian velocity.

    Parameters
    ----------
    velocity : array-like, shape (3,)
        Constant global velocity in m/s. It is copied to ``float64``.

    Notes
    -----
    :meth:`update` applies ``delta_x = velocity * time_step_size`` in place and
    increments :attr:`current_position`. It does not rotate panels.

    Examples
    --------
    >>> kinematics = TranslatingVLM(velocity=[30.0, 0.0, 0.0])
    >>> kinematics.get_velocity(time=1.0)
    array([30.,  0.,  0.])
    """

    def __init__(self, velocity: np.ndarray) -> None:
        """Copy a constant shape-``(3,)`` velocity vector in m/s."""
        super().__init__()
        self.velocity = np.array(velocity, dtype=np.float64)

    def get_velocity(self, time: float) -> np.ndarray:
        """Return a copy of the constant shape-``(3,)`` velocity in m/s."""
        return self.velocity.copy()

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return a new zero angular-velocity vector in rad/s."""
        return np.zeros(3)

    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Translate selected panels by ``velocity * time_step_size``.

        Geometry and :attr:`current_position` are mutated in place. An
        uninitialized/empty lattice is left unchanged.
        """
        # Safety check: ensure lattice is initialized
        if vlm_solver.lattice is None or vlm_solver.lattice.n_panels == 0:
            return

        displacement_increment = self.velocity * time_step_size

        # Update current position (metadata)
        self.current_position += displacement_increment

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(displacement_increment)
        else:
            lattice.translate_panels(displacement_increment, panel_range[0], panel_range[1])


class RotatingVLM(VLMKinematics):
    """Rotate VLM panels about a fixed axis at constant angular speed.

    Parameters
    ----------
    angular_speed : float
        Signed angular speed in rad/s.
    axis : array-like, shape (3,)
        Non-zero global rotation axis, normalized internally.
    rotation_centre : array-like, shape (3,), optional
        Fixed pivot coordinates in m; ``None`` uses the origin.

    Notes
    -----
    Each update uses Rodrigues' formula for the incremental angle
    ``angular_speed * time_step_size`` and mutates selected lattice panels.
    """

    def __init__(
        self,
        angular_speed: float,
        axis: np.ndarray,
        rotation_centre: np.ndarray | None = None,
        acceleration_time: float = 0.0,
    ) -> None:
        """Normalize the axis and copy fixed-pivot rotation parameters."""
        super().__init__()
        self.angular_speed = angular_speed
        if acceleration_time < 0:
            raise ValueError("rotation acceleration_time must be nonnegative")
        self.acceleration_time = float(acceleration_time)
        axis = np.array(axis, dtype=np.float64)
        self.axis = axis / np.linalg.norm(axis)
        self.rotation_centre = np.array(
            rotation_centre if rotation_centre is not None else [0, 0, 0], dtype=np.float64
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Return a new zero translational-velocity vector in m/s."""
        return np.zeros(3)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return prescribed global angular velocity, including smooth spin-up."""
        factor = (
            1.0
            if self.acceleration_time == 0
            else np.sin(0.5 * np.pi * np.clip(time / self.acceleration_time, 0.0, 1.0)) ** 2
        )
        return factor * self.angular_speed * self.axis

    def rotation_angle(self, time: float) -> float:
        """Integrated angle of the optional sin-squared spin-up, in radians."""
        if self.acceleration_time == 0:
            return self.angular_speed * time
        ramp = np.clip(time, 0.0, self.acceleration_time)
        integral = 0.5 * (
            ramp - self.acceleration_time / np.pi * np.sin(np.pi * ramp / self.acceleration_time)
        )
        return self.angular_speed * (integral + max(time - self.acceleration_time, 0.0))

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        """
        Create rotation matrix for rotation about self.axis by angle.

        Uses Rodrigues' rotation formula.
        """
        skew_symmetric_matrix = np.array(
            [
                [0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0],
            ]
        )

        identity_matrix = np.eye(3)
        rotation_matrix = (
            identity_matrix
            + np.sin(angle) * skew_symmetric_matrix
            + (1 - np.cos(angle)) * skew_symmetric_matrix @ skew_symmetric_matrix
        )
        return rotation_matrix

    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Rotate selected panel geometry over one interval.

        The lattice coordinates, :attr:`current_orientation`, and accumulated
        position metadata are mutated in place.
        """
        angle = self.rotation_angle(time + time_step_size) - self.rotation_angle(time)
        rotation_matrix = self._rotation_matrix(angle)

        # Update metadata
        self.current_orientation = rotation_matrix @ self.current_orientation
        self.current_position = (
            rotation_matrix @ (self.current_position - self.rotation_centre) + self.rotation_centre
        )

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(rotation_matrix, self.rotation_centre, np.zeros(3))
        else:
            lattice.rotate_translate_panels(
                rotation_matrix, self.rotation_centre, np.zeros(3), panel_range[0], panel_range[1]
            )


class ManeuverVLM(VLMKinematics):
    """Drive arbitrary rigid motion with user-supplied velocity functions.

    Parameters
    ----------
    velocity_function : callable or None
        Function ``f(time_seconds) -> array-like shape (3,)`` returning global
        translation velocity in m/s. ``None`` supplies zero.
    angular_velocity_function : callable or None
        Function ``g(time_seconds) -> array-like shape (3,)`` returning global
        angular velocity in rad/s. ``None`` supplies zero.
    rotation_centre : array-like, shape (3,), optional
        Initial pivot coordinates in m; ``None`` uses the origin. Translation
        advects this pivot during updates.

    Notes
    -----
    Update samples both functions at the interval midpoint, advances translation and applies an incremental Rodrigues rotation. The functions
    must be deterministic for reproducible restart behavior.
    """

    def __init__(
        self,
        velocity_function: Callable[[float], np.ndarray] | None = None,
        angular_velocity_function: Callable[[float], np.ndarray] | None = None,
        rotation_centre: np.ndarray | None = None,
    ) -> None:
        """Retain motion callables and copy the initial rotation centre."""
        super().__init__()
        self.velocity_function = velocity_function or (lambda time: np.zeros(3))
        self.angular_velocity_function = angular_velocity_function or (lambda time: np.zeros(3))
        self.rotation_centre = np.array(
            rotation_centre if rotation_centre is not None else [0, 0, 0], dtype=np.float64
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """Evaluate and return a ``float64`` velocity vector in m/s."""
        return np.array(self.velocity_function(time), dtype=np.float64)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Evaluate and return a ``float64`` angular velocity in rad/s."""
        return np.array(self.angular_velocity_function(time), dtype=np.float64)

    def _rotation_matrix(self, angular_velocity: np.ndarray, time_step_size: float) -> np.ndarray:
        """Create rotation matrix from angular velocity and time step."""
        angle = np.linalg.norm(angular_velocity) * time_step_size
        if angle < 1e-12:
            return np.eye(3)

        axis = angular_velocity / np.linalg.norm(angular_velocity)
        skew_symmetric_matrix = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        identity_matrix = np.eye(3)
        rotation_matrix = (
            identity_matrix
            + np.sin(angle) * skew_symmetric_matrix
            + (1 - np.cos(angle)) * skew_symmetric_matrix @ skew_symmetric_matrix
        )
        return rotation_matrix

    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Apply midpoint translation and rotation to selected panels.

        Lattice geometry, accumulated pose metadata, and the advected rotation
        centre are mutated in place.
        """
        velocity = self.get_velocity(time + 0.5 * time_step_size)
        angular_velocity = self.get_angular_velocity(time + 0.5 * time_step_size)

        displacement_increment = velocity * time_step_size
        rotation_matrix = self._rotation_matrix(angular_velocity, time_step_size)

        # Update metadata
        self.current_orientation = rotation_matrix @ self.current_orientation
        self.current_position = (
            rotation_matrix @ (self.current_position - self.rotation_centre)
            + self.rotation_centre
            + displacement_increment
        )

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(
                rotation_matrix, self.rotation_centre, displacement_increment
            )
        else:
            lattice.rotate_translate_panels(
                rotation_matrix,
                self.rotation_centre,
                displacement_increment,
                panel_range[0],
                panel_range[1],
            )

        self.rotation_centre += displacement_increment


class HeavingVLM(ManeuverVLM):
    """
    Sinusoidal heaving motion kinematics.

    Models a plunging/heaving wing with specified amplitude and frequency.

    z(time) = h0 * sin(2π * f * time + φ)
    Vz(time) = h0 * 2π * f * cos(2π * f * time + φ)

    Example:
        >>> kinematics = HeavingVLM(
        ...     amplitude=0.1,  # 10 cm heave amplitude
        ...     frequency=1.0,  # 1 Hz
        ... )
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        direction: np.ndarray = None,
    ):
        """
        Initialize heaving kinematics.

        Args:
            amplitude: Heave amplitude (m)
            frequency: Heave frequency (Hz)
            phase: Initial phase (rad)
            direction: Heave direction unit vector (default: [0, 0, 1])
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.direction = np.array(
            direction if direction is not None else [0, 0, 1], dtype=np.float64
        )
        self.direction = self.direction / np.linalg.norm(self.direction)

        angular_frequency = 2 * np.pi * frequency

        def velocity_function(time):
            heave_velocity = (
                amplitude * angular_frequency * np.cos(angular_frequency * time + phase)
            )
            return heave_velocity * self.direction

        super().__init__(velocity_function=velocity_function)


class PitchingVLM(ManeuverVLM):
    """Configure sinusoidal pitching about a fixed global axis.

    Parameters
    ----------
    amplitude_degrees : float
        Peak pitch angle in degrees.
    frequency : float
        Oscillation frequency in Hz.
    phase : float, default=0.0
        Initial phase in radians.
    rotation_axis : array-like, shape (3,), optional
        Non-zero pivot axis; ``None`` uses global y.
    rotation_centre : array-like, shape (3,), optional
        Pivot coordinates in m; ``None`` uses the origin.

    Notes
    -----
    ``theta(t)=theta0*sin(2*pi*f*t+phase)`` and the returned angular velocity
    is its derivative along ``rotation_axis`` in rad/s. Geometry updates use
    the inherited start-time incremental integration.

    Example:
        >>> kinematics = PitchingVLM(
        ...     amplitude_degrees=5.0,  # ±5° pitch
        ...     frequency=1.0,  # 1 Hz
        ...     rotation_axis=[0, 1, 0],  # pitch about Y axis
        ...     rotation_centre=[0.25, 0, 0],  # quarter-chord pivot
        ... )
    """

    def __init__(
        self,
        amplitude_degrees: float,
        frequency: float,
        phase: float = 0.0,
        rotation_axis: np.ndarray | None = None,
        rotation_centre: np.ndarray | None = None,
    ) -> None:
        """Create the sinusoidal angular-velocity function and pivot."""
        self.amplitude = np.radians(amplitude_degrees)
        self.frequency = frequency
        self.phase = phase

        axis = np.array(
            rotation_axis if rotation_axis is not None else [0, 1, 0],
            dtype=np.float64,
        )
        axis = axis / np.linalg.norm(axis)

        rotation_centre = np.array(
            rotation_centre if rotation_centre is not None else [0, 0, 0],
            dtype=np.float64,
        )

        angular_frequency = 2 * np.pi * frequency
        amplitude_radians = self.amplitude

        def angular_velocity_function(time):
            pitching_angular_speed = (
                amplitude_radians * angular_frequency * np.cos(angular_frequency * time + phase)
            )
            return pitching_angular_speed * axis

        super().__init__(
            angular_velocity_function=angular_velocity_function,
            rotation_centre=rotation_centre,
        )


class LinearPeriodicVLM(VLMKinematics):
    """
    Linear periodic motion kinematics (heaving/plunging).

    Models a body performing sinusoidal linear motion in a specified direction.

    Position: r(time) = r0 + A * sin(2π * f * time) * d
    Velocity: velocity(time) = A * 2π * f * cos(2π * f * time) * d
    Where:
        A: Amplitude (m)
        f: Frequency (Hz)
        d: Direction unit vector

    Example:
        >>> kinematics = LinearPeriodicVLM(
        ...     amplitude=1.0,  # 1m amplitude
        ...     frequency=1.0,  # 1 Hz
        ...     direction=[0, 0, 1]  # Plunging in Z
        ... )
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        direction: np.ndarray,
        phase: float = 0.0,
    ):
        """
        Initialize linear periodic kinematics.

        Args:
            amplitude: Motion amplitude (m)
            frequency: Motion frequency (Hz)
            direction: Motion direction vector (will be normalized)
            phase: Initial phase (rad)
        """
        super().__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.direction = np.array(direction, dtype=np.float64)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.phase = phase

        self.angular_frequency = 2 * np.pi * frequency

        # Initial offset adjustment not handled here (assumes starting at mean position)
        # Note: If phase=0, starts at 0 displacement, max velocity.

    def get_velocity(self, time: float) -> np.ndarray:
        """Return instantaneous velocity vector."""
        # velocity(time) = A * ω * cos(ωtime + φ) * d
        velocity_magnitude = (
            self.amplitude
            * self.angular_frequency
            * np.cos(self.angular_frequency * time + self.phase)
        )
        return velocity_magnitude * self.direction

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return zero angular velocity."""
        return np.zeros(3)

    def update(self, vlm_solver, time: float, time_step_size: float, panel_range: tuple = None):
        """
        Update VLM geometry for linear periodic motion.
        """
        velocity = self.get_velocity(time)
        displacement_increment = velocity * time_step_size

        # Update metadata
        self.current_position += displacement_increment

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(displacement_increment)
        else:
            lattice.translate_panels(displacement_increment, panel_range[0], panel_range[1])


class CompositeVLM(VLMKinematics):
    """
    Composite kinematics combining multiple motion profiles.

    Allows superposition of multiple kinematics (e.g., heaving + pitching)
    by summing velocity and angular velocity from each component.

    This is useful for studying combined motions like:
    - Heaving + pitching (flapping flight, optimal thrust generation)
    - Translation + rotation (maneuvering aircraft)
    - Any combination of the available kinematics classes

    Example:
        >>> # Heaving + pitching with 90° phase shift
        >>> heaving = LinearPeriodicVLM(amplitude=0.1, frequency=1.0, direction=[0,0,1])
        >>> pitching = PitchingVLM(amplitude_degrees=5.0, frequency=1.0, phase=np.pi/2)
        >>> combined = CompositeVLM(
        ...     kinematics_components=[heaving, pitching],
        ...     rotation_centre=[0.25, 0, 0]  # CG or pivot point
        ... )
    """

    def __init__(
        self,
        kinematics_components: list[VLMKinematics],
        rotation_centre: np.ndarray = None,
    ):
        """
        Initialize composite kinematics.

        Args:
            kinematics_components: List of VLMKinematics objects to combine
            rotation_centre: Center for rotation transformations (default: origin).
                           This overrides individual rotation centers for consistent
                           combined motion about a single point (e.g., CG).
        """
        super().__init__()
        self.kinematics_components = kinematics_components
        self.rotation_centre = np.array(
            rotation_centre if rotation_centre is not None else [0, 0, 0], dtype=np.float64
        )

    def get_velocity(self, time: float) -> np.ndarray:
        """
        Get combined translational velocity (sum of all components).

        Args:
            time: Current time (s)

        Returns:
            Combined velocity vector [Vx, Vy, Vz] (m/s)
        """
        total_velocity = np.zeros(3)
        for kinematics in self.kinematics_components:
            total_velocity += kinematics.get_velocity(time)
        return total_velocity

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """
        Get combined angular velocity (sum of all components).

        Args:
            time: Current time (s)

        Returns:
            Combined angular velocity vector [Wx, Wy, Wz] (rad/s)
        """
        total_angular_velocity = np.zeros(3)
        for kinematics in self.kinematics_components:
            total_angular_velocity += kinematics.get_angular_velocity(time)
        return total_angular_velocity

    def _rotation_matrix(self, angular_velocity: np.ndarray, time_step_size: float) -> np.ndarray:
        """Create rotation matrix from angular velocity and time step."""
        angle = np.linalg.norm(angular_velocity) * time_step_size
        if angle < 1e-12:
            return np.eye(3)

        axis = angular_velocity / np.linalg.norm(angular_velocity)
        skew_symmetric_matrix = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        identity_matrix = np.eye(3)
        rotation_matrix = (
            identity_matrix
            + np.sin(angle) * skew_symmetric_matrix
            + (1 - np.cos(angle)) * skew_symmetric_matrix @ skew_symmetric_matrix
        )
        return rotation_matrix

    def update(self, vlm_solver, time: float, time_step_size: float, panel_range: tuple = None):
        """
        Update VLM geometry for composite motion.
        """
        velocity = self.get_velocity(time)
        angular_velocity = self.get_angular_velocity(time)

        displacement_increment = velocity * time_step_size
        rotation_matrix = self._rotation_matrix(angular_velocity, time_step_size)

        # Update metadata
        self.current_orientation = rotation_matrix @ self.current_orientation
        self.current_position = (
            rotation_matrix @ (self.current_position - self.rotation_centre)
            + self.rotation_centre
            + displacement_increment
        )

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(
                rotation_matrix, self.rotation_centre, displacement_increment
            )
        else:
            lattice.rotate_translate_panels(
                rotation_matrix,
                self.rotation_centre,
                displacement_increment,
                panel_range[0],
                panel_range[1],
            )

        self.rotation_centre += displacement_increment


class AcceleratingVLM(VLMKinematics):
    """
    Constant acceleration kinematics.

    The VLM surfaces accelerate at a constant rate from rest or an initial velocity.
    Models impulsive start or ramp-up maneuvers (Wagner problem, starting vortex).

    Motion:
        velocity(time) = V0 + a * time  (for time < t_accel)
        velocity(time) = V_final     (for time >= t_accel)
        X(time) = X0 + V0*time + 0.5*a*time²

    Example:
        >>> # Accelerating flat plate (Wagner problem) to 10 m/s over 0.5 seconds
        >>> kinematics = AcceleratingVLM(
        ...     final_velocity=[10.0, 0.0, 0.0],
        ...     acceleration_time=0.5
        ... )
        >>>
        >>> # Or specify acceleration directly
        >>> kinematics = AcceleratingVLM(
        ...     acceleration=[10.0, 0.0, 0.0],  # 10 m/s² in x-direction
        ...     initial_velocity=[0.0, 0.0, 0.0]  # Start from rest
        ... )
    """

    def __init__(
        self,
        acceleration: np.ndarray = None,
        initial_velocity: np.ndarray = None,
        final_velocity: np.ndarray = None,
        acceleration_time: float = None,
    ):
        """
        Initialize accelerating kinematics.

        Args:
            acceleration: Constant acceleration vector [ax, ay, az] (m/s²).
                         If not provided, computed from final_velocity and acceleration_time.
            initial_velocity: Initial velocity [V0x, V0y, V0z] (m/s). Default: zero
            final_velocity: Final velocity vector after acceleration completes.
                    Alternative way to specify acceleration when used with acceleration_time.
            acceleration_time: Time duration for acceleration phase (s).
                              After this time, velocity becomes constant at final_velocity.

        Note:
            Either provide (acceleration, initial_velocity) OR (final_velocity, acceleration_time).
        """
        super().__init__()

        # Parse arguments
        if final_velocity is not None and acceleration_time is not None:
            # Compute acceleration from final velocity and time
            self.initial_velocity = np.array(
                initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=np.float64
            )
            final_velocity_array = np.array(final_velocity, dtype=np.float64)
            self.acceleration = (final_velocity_array - self.initial_velocity) / acceleration_time
            self.acceleration_time = acceleration_time
            self.final_velocity = final_velocity_array
        elif acceleration is not None:
            # Direct acceleration specification
            self.acceleration = np.array(acceleration, dtype=np.float64)
            self.initial_velocity = np.array(
                initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=np.float64
            )
            self.acceleration_time = None  # Infinite acceleration (never reaches constant velocity)
            self.final_velocity = None
        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. acceleration (and optionally initial_velocity)\n"
                "  2. final_velocity and acceleration_time"
            )

        self.start_time = 0.0  # Time when acceleration starts

    def get_velocity(self, time: float) -> np.ndarray:
        """
        Return velocity at time time.

        If acceleration_time is set:
            velocity(time) = V0 + a*time  (for time < t_accel)
            velocity(time) = V_final   (for time >= t_accel)
        Otherwise:
            velocity(time) = V0 + a*time  (unbounded acceleration)
        """
        t_rel = time - self.start_time

        if self.acceleration_time is not None and t_rel >= self.acceleration_time:
            # After acceleration phase, maintain constant velocity
            return self.final_velocity.copy()
        else:
            # During acceleration phase
            return self.initial_velocity + self.acceleration * t_rel

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return zero angular velocity (pure translation)."""
        return np.zeros(3)

    def update(self, vlm_solver, time: float, time_step_size: float, panel_range: tuple = None):
        """
        Update VLM geometry with constant acceleration.

        Uses average velocity over time step: V_avg = velocity(time) + 0.5*a*dt
        """
        # Safety check: ensure lattice is initialized
        if vlm_solver.lattice is None or vlm_solver.lattice.n_panels == 0:
            return

        # Velocity at midpoint of time step (for trapezoidal integration)
        midpoint_velocity = self.get_velocity(time + 0.5 * time_step_size)
        displacement_increment = midpoint_velocity * time_step_size

        # Update current position
        self.current_position += displacement_increment

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(displacement_increment)
        else:
            lattice.translate_panels(displacement_increment, panel_range[0], panel_range[1])


class SmoothRampVLM(VLMKinematics):
    """Ramp translation smoothly from rest to a constant velocity.

    Parameters
    ----------
    final_velocity : array-like, shape (3,)
        Cartesian terminal velocity in m/s.
    acceleration_time : float
        Duration of the smooth ramp in s; expected positive.
    start_time : float, default=0.0
        Physical time in s at which acceleration begins.

    Motion (sin² profile):
        velocity(time) = final_velocity * sin²(π * time / (2 * t_accel))  (for time < t_accel)
        velocity(time) = final_velocity                             (for time >= t_accel)

    Properties:
        - velocity(0) = 0, a(0) = 0 (smooth start, CL=0 at time=0)
        - velocity(t_accel) = final_velocity, a(t_accel) = 0 (smooth end of acceleration)
    """

    def __init__(
        self, final_velocity: np.ndarray, acceleration_time: float, start_time: float = 0.0
    ) -> None:
        """Copy the terminal velocity and retain the ramp time parameters."""
        super().__init__()
        self.final_velocity = np.array(final_velocity, dtype=np.float64)
        self.acceleration_time = acceleration_time
        self.start_time = start_time

    def get_velocity(self, time: float) -> np.ndarray:
        """Return shape-``(3,)`` sin-squared ramp velocity in m/s."""
        t_rel = time - self.start_time

        if t_rel <= 0:
            return np.zeros(3)
        if t_rel >= self.acceleration_time:
            return self.final_velocity.copy()

        # velocity(time) = final_velocity * sin²(π * time / (2 * t_accel))
        # Note: sin²(x) = (1 - cos(2x)) / 2
        phase = (np.pi * t_rel) / (2.0 * self.acceleration_time)
        return self.final_velocity * (np.sin(phase) ** 2)

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """Return zero angular velocity (pure translation)."""
        return np.zeros(3)

    def update(
        self,
        vlm_solver: "VLMSolver",
        time: float,
        time_step_size: float,
        panel_range: tuple[int, int] | None = None,
    ) -> None:
        """Translate panels using the exact displacement change over the step.

        Distance traveled during step:
          displacement_increment = X(time+dt) - X(time)
        """
        if vlm_solver.lattice is None or vlm_solver.lattice.n_panels == 0:
            return

        def _get_displacement(time):
            t_rel = time - self.start_time
            if t_rel <= 0:
                return np.zeros(3)
            if t_rel >= self.acceleration_time:
                # Fully accelerated distance: 0.5 * final_velocity * t_accel
                # PLUS constant velocity distance: final_velocity * (t_rel - t_accel)
                dist_accel = 0.5 * self.final_velocity * self.acceleration_time
                dist_const = self.final_velocity * (t_rel - self.acceleration_time)
                return dist_accel + dist_const

            # Integrated sin² displacement
            # ∫ [0.5 * final_velocity * (1 - cos(π*time/t_accel))] dt
            # = 0.5 * final_velocity * [time - (t_accel/π) * sin(π*time/t_accel)]
            term1 = t_rel
            term2 = (self.acceleration_time / np.pi) * np.sin(
                (np.pi * t_rel) / self.acceleration_time
            )
            return 0.5 * self.final_velocity * (term1 - term2)

        X1 = _get_displacement(time + time_step_size)
        X0 = _get_displacement(time)
        displacement_increment = X1 - X0

        self.current_position += displacement_increment
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(displacement_increment)
        else:
            lattice.translate_panels(displacement_increment, panel_range[0], panel_range[1])
