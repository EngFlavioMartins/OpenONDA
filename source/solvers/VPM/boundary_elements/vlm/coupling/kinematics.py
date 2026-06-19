"""
Kinematics module for VPM solver.
==================
Kinematics module for VPM solver. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class VLMKinematics(ABC):
    """
    Abstract base class for VLM surface kinematics.

    Kinematics define how the VLM surfaces move through time,
    providing position and velocity information at each time step.

    Subclasses must implement:
        - get_velocity(t): Return velocity vector at time t
        - get_angular_velocity(t): Return angular velocity at time t
        - update(vlm_solver, t, dt): Update VLM geometry for new time step
    """

    @abstractmethod
    def get_velocity(self, t: float) -> np.ndarray:
        """
        Get translational velocity at time t.

        Args:
            t: Current time (s)

        Returns:
            Velocity vector [Vx, Vy, Vz] (m/s)
        """
        pass

    @abstractmethod
    def get_angular_velocity(self, t: float) -> np.ndarray:
        """
        Get angular velocity at time t.

        Args:
            t: Current time (s)

        Returns:
            Angular velocity vector [Wx, Wy, Wz] (rad/s)
        """
        pass

    def __init__(self):
        self.current_position = np.zeros(3)
        self.current_orientation = np.eye(3)

    @abstractmethod
    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM solver geometry for new time step.

        Args:
            vlm_solver: VLMSolver instance to update
            t: Current time (s)
            dt: Time step (s)
            panel_range: Optional tuple (start_idx, end_idx) of panels to update.
                        If None, updates all panels.
        """
        pass


class StaticVLM(VLMKinematics):
    """
    Static kinematics - no motion.

    The VLM surfaces remain fixed in space. This is appropriate for
    steady-state analysis or when studying wake development without
    surface motion.

    Example:
        >>> kinematics = StaticVLM()
        >>> V = kinematics.get_velocity(t=1.0)
        >>> print(V)  # [0, 0, 0]
    """

    def __init__(self):
        """Initialize static kinematics."""
        pass

    def get_velocity(self, t: float) -> np.ndarray:
        """Return zero velocity (static)."""
        return np.zeros(3)

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return zero angular velocity (static)."""
        return np.zeros(3)

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """No update needed for static case."""
        pass


class TranslatingVLM(VLMKinematics):
    """
    Constant translational velocity kinematics.

    The VLM surfaces translate at a constant velocity. This can
    represent a vehicle in steady forward flight.

    Example:
        >>> kinematics = TranslatingVLM(velocity=[30.0, 0.0, 0.0])
        >>> V = kinematics.get_velocity(t=1.0)
        >>> print(V)  # [30, 0, 0]
    """

    def __init__(self, velocity: np.ndarray):
        """
        Initialize translating kinematics.

        Args:
            velocity: Constant velocity vector [Vx, Vy, Vz] (m/s)
        """
        super().__init__()
        self.velocity = np.array(velocity, dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        """Return constant velocity."""
        return self.velocity.copy()

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return zero angular velocity."""
        return np.zeros(3)

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Translate VLM geometry.
        """
        # Safety check: ensure lattice is initialized
        if vlm_solver.lattice is None or vlm_solver.lattice.num_panels == 0:
            return

        dX = self.velocity * dt

        # Update current position (metadata)
        self.current_position += dX

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(dX)
        else:
            lattice.translate_panels(dX, panel_range[0], panel_range[1])


class RotatingVLM(VLMKinematics):
    """
    Constant angular velocity kinematics (rotation about a fixed axis).

    The VLM surfaces rotate about a specified axis at constant rate.
    This can represent a rotor blade or a pitching wing.

    Example:
        >>> kinematics = RotatingVLM(
        ...     omega=10.0,  # rad/s
        ...     axis=[0, 1, 0],  # rotate about Y axis
        ...     center=[0, 0, 0]  # rotation center
        ... )
    """

    def __init__(
        self,
        omega: float,
        axis: np.ndarray,
        center: np.ndarray = None,
    ):
        """
        Initialize rotating kinematics.

        Args:
            omega: Angular velocity magnitude (rad/s)
            axis: Unit vector defining rotation axis
            center: Center of rotation (default: origin)
        """
        super().__init__()
        self.omega = omega
        axis = np.array(axis, dtype=np.float64)
        self.axis = axis / np.linalg.norm(axis)
        self.center = np.array(center if center is not None else [0, 0, 0], dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        """Return zero translational velocity."""
        return np.zeros(3)

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return angular velocity vector."""
        return self.omega * self.axis

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        """
        Create rotation matrix for rotation about self.axis by angle.

        Uses Rodrigues' rotation formula.
        """
        K = np.array(
            [
                [0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0],
            ]
        )

        I = np.eye(3)  # noqa: E741
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Rotate VLM geometry.
        """
        angle = self.omega * dt
        R = self._rotation_matrix(angle)

        # Update metadata
        self.current_orientation = R @ self.current_orientation
        self.current_position = R @ (self.current_position - self.center) + self.center

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(R, self.center, np.zeros(3))
        else:
            lattice.rotate_translate_panels(
                R, self.center, np.zeros(3), panel_range[0], panel_range[1]
            )


class ManeuverVLM(VLMKinematics):
    """
    General time-varying kinematics defined by callable functions.

    Allows arbitrary motion profiles by specifying functions that
    return velocity and angular velocity as functions of time.

    Example:
        >>> # Sinusoidal heaving motion
        >>> def heave_velocity(t):
        ...     return np.array([0, 0, 0.5 * np.cos(2 * np.pi * t)])
        >>> kinematics = ManeuverVLM(velocity_fn=heave_velocity)
    """

    def __init__(
        self,
        velocity_fn: Callable[[float], np.ndarray] | None = None,
        angular_velocity_fn: Callable[[float], np.ndarray] | None = None,
        rotation_center: np.ndarray = None,
    ):
        """
        Initialize maneuver kinematics.

        Args:
            velocity_fn: Function returning velocity vector for time t
            angular_velocity_fn: Function returning angular velocity for time t
            rotation_center: Center of rotation for angular velocity
        """
        super().__init__()
        self.velocity_fn = velocity_fn or (lambda t: np.zeros(3))
        self.angular_velocity_fn = angular_velocity_fn or (lambda t: np.zeros(3))
        self.rotation_center = np.array(
            rotation_center if rotation_center is not None else [0, 0, 0], dtype=np.float64
        )

    def get_velocity(self, t: float) -> np.ndarray:
        """Return velocity from user function."""
        return np.array(self.velocity_fn(t), dtype=np.float64)

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return angular velocity from user function."""
        return np.array(self.angular_velocity_fn(t), dtype=np.float64)

    def _rotation_matrix(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Create rotation matrix from angular velocity and time step."""
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-12:
            return np.eye(3)

        axis = omega / np.linalg.norm(omega)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        I = np.eye(3)  # noqa: E741
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM geometry for maneuver (rotation + translation).
        """
        V = self.get_velocity(t)
        W = self.get_angular_velocity(t)

        dX = V * dt
        R = self._rotation_matrix(W, dt)

        # Update metadata
        self.current_orientation = R @ self.current_orientation
        self.current_position = (
            R @ (self.current_position - self.rotation_center) + self.rotation_center + dX
        )

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(R, self.rotation_center, dX)
        else:
            lattice.rotate_translate_panels(
                R, self.rotation_center, dX, panel_range[0], panel_range[1]
            )

        # KEY FIX: Advect the rotation center with the translation
        # This ensures that for the NEXT step, we rotate around the new pivot position.
        self.rotation_center += dX


class HeavingVLM(ManeuverVLM):
    """
    Sinusoidal heaving motion kinematics.

    Models a plunging/heaving wing with specified amplitude and frequency.

    z(t) = h0 * sin(2π * f * t + φ)
    Vz(t) = h0 * 2π * f * cos(2π * f * t + φ)

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

        omega = 2 * np.pi * frequency

        def velocity_fn(t):
            Vz = amplitude * omega * np.cos(omega * t + phase)
            return Vz * self.direction

        super().__init__(velocity_fn=velocity_fn)


class PitchingVLM(ManeuverVLM):
    """
    Sinusoidal pitching motion kinematics.

    Models a pitching wing with specified amplitude and frequency.

    θ(t) = θ0 * sin(2π * f * t + φ)
    ω(t) = θ0 * 2π * f * cos(2π * f * t + φ)

    Example:
        >>> kinematics = PitchingVLM(
        ...     amplitude_deg=5.0,  # ±5° pitch
        ...     frequency=1.0,  # 1 Hz
        ...     pitch_axis=[0, 1, 0],  # pitch about Y axis
        ...     pivot=[0.25, 0, 0],  # quarter-chord pivot
        ... )
    """

    def __init__(
        self,
        amplitude_deg: float,
        frequency: float,
        phase: float = 0.0,
        pitch_axis: np.ndarray = None,
        pivot: np.ndarray = None,
    ):
        """
        Initialize pitching kinematics.

        Args:
            amplitude_deg: Pitch amplitude (degrees)
            frequency: Pitch frequency (Hz)
            phase: Initial phase (rad)
            pitch_axis: Axis of rotation (default: [0, 1, 0])
            pivot: Pivot point (default: origin)
        """
        self.amplitude = np.radians(amplitude_deg)
        self.frequency = frequency
        self.phase = phase

        axis = np.array(pitch_axis if pitch_axis is not None else [0, 1, 0], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        pivot = np.array(pivot if pivot is not None else [0, 0, 0], dtype=np.float64)

        omega = 2 * np.pi * frequency
        amplitude_rad = self.amplitude

        def angular_velocity_fn(t):
            omega_pitch = amplitude_rad * omega * np.cos(omega * t + phase)
            return omega_pitch * axis

        super().__init__(
            angular_velocity_fn=angular_velocity_fn,
            rotation_center=pivot,
        )


class LinearPeriodicVLM(VLMKinematics):
    """
    Linear periodic motion kinematics (heaving/plunging).

    Models a body performing sinusoidal linear motion in a specified direction.

    Position: r(t) = r0 + A * sin(2π * f * t) * d
    Velocity: V(t) = A * 2π * f * cos(2π * f * t) * d
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

        self.omega = 2 * np.pi * frequency

        # Initial offset adjustment not handled here (assumes starting at mean position)
        # Note: If phase=0, starts at 0 displacement, max velocity.

    def get_velocity(self, t: float) -> np.ndarray:
        """Return instantaneous velocity vector."""
        # V(t) = A * ω * cos(ωt + φ) * d
        V_mag = self.amplitude * self.omega * np.cos(self.omega * t + self.phase)
        return V_mag * self.direction

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return zero angular velocity."""
        return np.zeros(3)

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM geometry for linear periodic motion.
        """
        V = self.get_velocity(t)
        dX = V * dt

        # Update metadata
        self.current_position += dX

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(dX)
        else:
            lattice.translate_panels(dX, panel_range[0], panel_range[1])


class CompositeVLM(VLMKinematics):
    """
    Composite kinematics combining multiple motion profiles.

    Allows superposition of multiple kinematics (e.g., heaving + pitching)
    by summing velocities and angular velocities from each component.

    This is useful for studying combined motions like:
    - Heaving + pitching (flapping flight, optimal thrust generation)
    - Translation + rotation (maneuvering aircraft)
    - Any combination of the available kinematics classes

    Example:
        >>> # Heaving + pitching with 90° phase shift
        >>> heaving = LinearPeriodicVLM(amplitude=0.1, frequency=1.0, direction=[0,0,1])
        >>> pitching = PitchingVLM(amplitude_deg=5.0, frequency=1.0, phase=np.pi/2)
        >>> combined = CompositeVLM(
        ...     kinematics_list=[heaving, pitching],
        ...     rotation_center=[0.25, 0, 0]  # CG or pivot point
        ... )
    """

    def __init__(
        self,
        kinematics_list: list[VLMKinematics],
        rotation_center: np.ndarray = None,
    ):
        """
        Initialize composite kinematics.

        Args:
            kinematics_list: List of VLMKinematics objects to combine
            rotation_center: Center for rotation transformations (default: origin).
                           This overrides individual rotation centers for consistent
                           combined motion about a single point (e.g., CG).
        """
        super().__init__()
        self.kinematics_list = kinematics_list
        self.rotation_center = np.array(
            rotation_center if rotation_center is not None else [0, 0, 0], dtype=np.float64
        )

    def get_velocity(self, t: float) -> np.ndarray:
        """
        Get combined translational velocity (sum of all components).

        Args:
            t: Current time (s)

        Returns:
            Combined velocity vector [Vx, Vy, Vz] (m/s)
        """
        V_total = np.zeros(3)
        for kinematics in self.kinematics_list:
            V_total += kinematics.get_velocity(t)
        return V_total

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """
        Get combined angular velocity (sum of all components).

        Args:
            t: Current time (s)

        Returns:
            Combined angular velocity vector [Wx, Wy, Wz] (rad/s)
        """
        W_total = np.zeros(3)
        for kinematics in self.kinematics_list:
            W_total += kinematics.get_angular_velocity(t)
        return W_total

    def _rotation_matrix(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Create rotation matrix from angular velocity and time step."""
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-12:
            return np.eye(3)

        axis = omega / np.linalg.norm(omega)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        I = np.eye(3)  # noqa: E741
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM geometry for composite motion.
        """
        V = self.get_velocity(t)
        W = self.get_angular_velocity(t)

        dX = V * dt
        R = self._rotation_matrix(W, dt)

        # Update metadata
        self.current_orientation = R @ self.current_orientation
        self.current_position = (
            R @ (self.current_position - self.rotation_center) + self.rotation_center + dX
        )

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.rotate_translate_panels(R, self.rotation_center, dX)
        else:
            lattice.rotate_translate_panels(
                R, self.rotation_center, dX, panel_range[0], panel_range[1]
            )

        # KEY FIX: Advect the rotation center with the translation
        # This ensures that for the NEXT step, we rotate around the new pivot position.
        self.rotation_center += dX


class AcceleratingVLM(VLMKinematics):
    """
    Constant acceleration kinematics.

    The VLM surfaces accelerate at a constant rate from rest or an initial velocity.
    Models impulsive start or ramp-up maneuvers (Wagner problem, starting vortex).

    Motion:
        V(t) = V0 + a * t  (for t < t_accel)
        V(t) = V_final     (for t >= t_accel)
        X(t) = X0 + V0*t + 0.5*a*t²

    Example:
        >>> # Accelerating flat plate (Wagner problem) to 10 m/s over 0.5 seconds
        >>> kinematics = AcceleratingVLM(
        ...     U_final=[10.0, 0.0, 0.0],
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
        U_final: np.ndarray = None,
        acceleration_time: float = None,
    ):
        """
        Initialize accelerating kinematics.

        Args:
            acceleration: Constant acceleration vector [ax, ay, az] (m/s²).
                         If not provided, computed from U_final and acceleration_time.
            initial_velocity: Initial velocity [V0x, V0y, V0z] (m/s). Default: zero
            U_final: Final velocity vector after acceleration completes.
                    Alternative way to specify acceleration when used with acceleration_time.
            acceleration_time: Time duration for acceleration phase (s).
                              After this time, velocity becomes constant at U_final.

        Note:
            Either provide (acceleration, initial_velocity) OR (U_final, acceleration_time).
        """
        super().__init__()

        # Parse arguments
        if U_final is not None and acceleration_time is not None:
            # Compute acceleration from final velocity and time
            self.initial_velocity = np.array(
                initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=np.float64
            )
            U_final_arr = np.array(U_final, dtype=np.float64)
            self.acceleration = (U_final_arr - self.initial_velocity) / acceleration_time
            self.acceleration_time = acceleration_time
            self.U_final = U_final_arr
        elif acceleration is not None:
            # Direct acceleration specification
            self.acceleration = np.array(acceleration, dtype=np.float64)
            self.initial_velocity = np.array(
                initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=np.float64
            )
            self.acceleration_time = None  # Infinite acceleration (never reaches constant velocity)
            self.U_final = None
        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. acceleration (and optionally initial_velocity)\n"
                "  2. U_final and acceleration_time"
            )

        self.t_start = 0.0  # Time when acceleration starts

    def get_velocity(self, t: float) -> np.ndarray:
        """
        Return velocity at time t.

        If acceleration_time is set:
            V(t) = V0 + a*t  (for t < t_accel)
            V(t) = V_final   (for t >= t_accel)
        Otherwise:
            V(t) = V0 + a*t  (unbounded acceleration)
        """
        t_rel = t - self.t_start

        if self.acceleration_time is not None and t_rel >= self.acceleration_time:
            # After acceleration phase, maintain constant velocity
            return self.U_final.copy()
        else:
            # During acceleration phase
            return self.initial_velocity + self.acceleration * t_rel

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return zero angular velocity (pure translation)."""
        return np.zeros(3)

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM geometry with constant acceleration.

        Uses average velocity over time step: V_avg = V(t) + 0.5*a*dt
        """
        # Safety check: ensure lattice is initialized
        if vlm_solver.lattice is None or vlm_solver.lattice.num_panels == 0:
            return

        # Velocity at midpoint of time step (for trapezoidal integration)
        V_mid = self.get_velocity(t + 0.5 * dt)
        dX = V_mid * dt

        # Update current position
        self.current_position += dX

        # Update lattice via NumPy (robust, avoids Taichi field dimension bugs)
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(dX)
        else:
            lattice.translate_panels(dX, panel_range[0], panel_range[1])


class SmoothRampVLM(VLMKinematics):
    """
    Smooth acceleration kinematics using a sinusoidal velocity profile.

    The VLM surfaces accelerate smoothly from rest to a final velocity
    over a ramp period of acceleration_time.

    Motion (sin² profile):
        V(t) = U_final * sin²(π * t / (2 * t_accel))  (for t < t_accel)
        V(t) = U_final                             (for t >= t_accel)

    Properties:
        - V(0) = 0, a(0) = 0 (smooth start, CL=0 at t=0)
        - V(t_accel) = U_final, a(t_accel) = 0 (smooth end of acceleration)
    """

    def __init__(self, U_final: np.ndarray, acceleration_time: float, t_start: float = 0.0):
        """
        Initialize smooth ramp kinematics.

        Args:
            U_final: Final velocity vector [Vx, Vy, Vz] (m/s).
            acceleration_time: Duration of the acceleration phase (s).
            t_start: Time when acceleration begins (s).
        """
        super().__init__()
        self.U_final = np.array(U_final, dtype=np.float64)
        self.acceleration_time = acceleration_time
        self.t_start = t_start

    def get_velocity(self, t: float) -> np.ndarray:
        """Return velocity at time t using sin² ramp."""
        t_rel = t - self.t_start

        if t_rel <= 0:
            return np.zeros(3)
        if t_rel >= self.acceleration_time:
            return self.U_final.copy()

        # V(t) = U_final * sin²(π * t / (2 * t_accel))
        # Note: sin²(x) = (1 - cos(2x)) / 2
        phase = (np.pi * t_rel) / (2.0 * self.acceleration_time)
        return self.U_final * (np.sin(phase) ** 2)

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return zero angular velocity (pure translation)."""
        return np.zeros(3)

    def update(self, vlm_solver, t: float, dt: float, panel_range: tuple = None):
        """
        Update VLM geometry using sin² velocity integration.

        Distance traveled during step:
          dX = X(t+dt) - X(t)
        """
        if vlm_solver.lattice is None or vlm_solver.lattice.num_panels == 0:
            return

        def _get_displacement(time):
            t_rel = time - self.t_start
            if t_rel <= 0:
                return np.zeros(3)
            if t_rel >= self.acceleration_time:
                # Fully accelerated distance: 0.5 * U_final * t_accel
                # PLUS constant velocity distance: U_final * (t_rel - t_accel)
                dist_accel = 0.5 * self.U_final * self.acceleration_time
                dist_const = self.U_final * (t_rel - self.acceleration_time)
                return dist_accel + dist_const

            # Integrated sin² displacement
            # ∫ [0.5 * U_final * (1 - cos(π*t/t_accel))] dt
            # = 0.5 * U_final * [t - (t_accel/π) * sin(π*t/t_accel)]
            term1 = t_rel
            term2 = (self.acceleration_time / np.pi) * np.sin(
                (np.pi * t_rel) / self.acceleration_time
            )
            return 0.5 * self.U_final * (term1 - term2)

        X1 = _get_displacement(t + dt)
        X0 = _get_displacement(t)
        dX = X1 - X0

        self.current_position += dX
        lattice = vlm_solver.lattice
        if panel_range is None:
            lattice.translate_panels(dX)
        else:
            lattice.translate_panels(dX, panel_range[0], panel_range[1])
